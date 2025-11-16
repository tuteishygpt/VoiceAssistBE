import asyncio
import base64
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ValidationError

# Важна: мадэль XTTS, кэшы латэнтаў і ўсе залежнасці
# усталёўваюцца пры імпарце app.py. Менавіта там адбываецца
# кланаванне мадыфікаванай бібліятэкі, загрузка ваг і запуск
# генератара text_to_speech(...). Тут мы толькі карыстаемся
# ўжо ініцыялізаванай функцыяй як звычайнай бібліятэкай.
from app import (
    INITIAL_MIN_BUFFER_S,
    MIN_BUFFER_S,
    sampling_rate,
    text_to_speech,
)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY павінен быць зададзены ў асяроддзі.")
MODEL_ID = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent"
SYSTEM_PROMPT = (
    "Ты — кароткі і дакладны беларускі галасавы асістэнт. Адказвай сцісла,"\
    " сяброўска і толькі па-беларуску."
)

INDEX_PATH = Path(__file__).with_name("index.html")


class ASRResponse(BaseModel):
    transcript: str


class LLMRequest(BaseModel):
    transcript: str


class LLMResponse(BaseModel):
    reply: str


class TTSInitPayload(BaseModel):
    text: str
    speaker_audio: Optional[str] = None
    initial_buffer_s: float = Field(default=INITIAL_MIN_BUFFER_S)
    subsequent_buffer_s: float = Field(default=MIN_BUFFER_S)
    session_id: Optional[str] = None


class ConfigResponse(BaseModel):
    sample_rate: int
    default_initial_buffer_s: float
    default_subsequent_buffer_s: float


class GeminiClient:
    def __init__(self, *, api_key: str, base_url: str) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=30.0))

    async def close(self) -> None:
        await self._client.aclose()

    def _extract_text(self, data: Dict[str, Any]) -> str:
        for candidate in data.get("candidates", []):
            content = candidate.get("content") or {}
            parts = content.get("parts", [])
            texts = [part.get("text", "") for part in parts if part.get("text")]
            if texts:
                return "\n".join(texts).strip()
        return ""

    async def _generate(self, payload: Dict[str, Any]) -> str:
        response = await self._client.post(
            self._base_url,
            params={"key": self._api_key},
            json=payload,
        )
        response.raise_for_status()
        return self._extract_text(response.json())

    async def transcribe(self, *, audio_b64: str, mime_type: str) -> str:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": audio_b64}},
                        {"text": "Transcribe this audio to Belarusian text."},
                    ],
                }
            ],
            "responseModalities": ["text"],
        }
        return await self._generate(payload)

    async def complete(self, *, transcript: str) -> str:
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": transcript.strip()}],
                }
            ],
        }
        return await self._generate(payload)


gemini_client = GeminiClient(api_key=API_KEY, base_url=BASE_URL)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
active_tts_sessions: Dict[str, WebSocket] = {}


@app.on_event("shutdown")
async def _shutdown() -> None:
    await gemini_client.close()


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html не знойдзены")
    return INDEX_PATH.read_text(encoding="utf-8")


@app.get("/api/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    return ConfigResponse(
        sample_rate=sampling_rate,
        default_initial_buffer_s=INITIAL_MIN_BUFFER_S,
        default_subsequent_buffer_s=MIN_BUFFER_S,
    )


@app.post("/api/asr", response_model=ASRResponse)
async def asr_endpoint(audio: UploadFile = File(...)) -> ASRResponse:
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Пустое аўдыя")
    mime_type = audio.content_type or "audio/webm"
    audio_b64 = base64.b64encode(data).decode("ascii")
    try:
        transcript = await gemini_client.transcribe(audio_b64=audio_b64, mime_type=mime_type)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:  # pragma: no cover - сеткавыя памылкі
        raise HTTPException(status_code=500, detail=str(exc))
    if not transcript:
        transcript = ""
    return ASRResponse(transcript=transcript)


@app.post("/api/llm", response_model=LLMResponse)
async def llm_endpoint(req: LLMRequest) -> LLMResponse:
    if not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Трэба перадаць transcript")
    try:
        reply = await gemini_client.complete(transcript=req.transcript)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))
    if not reply:
        reply = "Прабач, не атрымалася атрымаць адказ."
    return LLMResponse(reply=reply)


async def _run_tts_generator(payload: TTSInitPayload, websocket: WebSocket) -> None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def worker() -> None:
        try:
            for item in text_to_speech(
                text_input=payload.text,
                speaker_audio=payload.speaker_audio,
                initial_buffer_s=payload.initial_buffer_s,
                subsequent_buffer_s=payload.subsequent_buffer_s,
            ):
                asyncio.run_coroutine_threadsafe(queue.put(("data", item)), loop)
        except Exception as exc:  # pragma: no cover - GPU/IO памылкі
            asyncio.run_coroutine_threadsafe(queue.put(("error", exc)), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, item = await queue.get()
        if kind == "data":
            b64_chunk, final_path, final_audio, metrics_json = item
            try:
                if metrics_json:
                    try:
                        metrics = json.loads(metrics_json)
                    except json.JSONDecodeError:
                        metrics = {"raw": metrics_json}
                    await websocket.send_json({"type": "metrics", "data": metrics})
                if b64_chunk == "__STOP__":
                    await websocket.send_json({
                        "type": "stop",
                        "final_path": final_path,
                    })
                    break
                if b64_chunk:
                    await websocket.send_json({"type": "chunk", "data": b64_chunk})
            except WebSocketDisconnect:
                break
        elif kind == "error":
            await websocket.send_json({"type": "error", "message": str(item)})
        elif kind == "done":
            break


@app.websocket("/ws/tts")
async def tts_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        init_payload_raw = await websocket.receive_text()
        payload = TTSInitPayload.parse_raw(init_payload_raw)
    except (ValidationError, json.JSONDecodeError):
        await websocket.close(code=4400)
        return

    session_id = payload.session_id or websocket.headers.get("x-session-id") or "default"
    prev_ws = active_tts_sessions.get(session_id)
    if prev_ws and prev_ws is not websocket:
        try:
            await prev_ws.close(code=4401)
        except RuntimeError:
            pass
    active_tts_sessions[session_id] = websocket

    try:
        await _run_tts_generator(payload, websocket)
    except WebSocketDisconnect:
        pass
    finally:
        stored = active_tts_sessions.get(session_id)
        if stored is websocket:
            active_tts_sessions.pop(session_id, None)

