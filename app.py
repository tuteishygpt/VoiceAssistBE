import os
import sys
import re
import time
import json
import base64
import hashlib
import tempfile
import subprocess
from typing import Iterator, Iterable, Optional, Tuple, Any, List
from dataclasses import dataclass
import pathlib

import spaces
import gradio as gr
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from scipy.io.wavfile import write

# ---------------------------------------------------------
# Налады асяроддзя для прадухілення празмернага выкарыстання CPU
# ---------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ---------------------------------------------------------
# 1) Кланаванне і імпарт мадыфікаванай бібліятэкі coqui-ai-TTS
# ---------------------------------------------------------
REPO_URL = "https://github.com/tuteishygpt/coqui-ai-TTS.git"
REPO_DIR = "coqui-ai-TTS"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
repo_root = os.path.abspath(REPO_DIR)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, split_sentence

# ---------------------------------------------------------
# 2) Загрузка файлаў мадэлі з Hugging Face
# ---------------------------------------------------------
repo_id = "archivartaunik/BE_XTTS_V2_10ep250k"
model_dir = pathlib.Path("./model")
model_dir.mkdir(exist_ok=True)
for fname in ("model.pth", "config.json", "vocab.json", "voice.wav"):
    if not (model_dir / fname).exists():
        hf_hub_download(repo_id, filename=fname, local_dir=model_dir)

# ---------------------------------------------------------
# 3) Ініцыялізацыя і загрузка мадэлі TTS
# ---------------------------------------------------------
print("Загрузка мадэлі...")
config = XttsConfig()
config.load_json(str(model_dir / "config.json"))
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_path=str(model_dir / "model.pth"), vocab_path=str(model_dir / "vocab.json"), use_deepspeed=False)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device.startswith("cuda"):
    torch.cuda.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

XTTS_MODEL.to(device)
sampling_rate = int(XTTS_MODEL.config.audio["sample_rate"])
print(f"Мадэль загружана на {device}. Частата дыскрэтызацыі: {sampling_rate} Гц.")

# ---------------------------------------------------------
# 4) Канфігурацыя струменевай перадачы
# ---------------------------------------------------------
INITIAL_MIN_BUFFER_S = 0.40
MIN_BUFFER_S = 0.15
FADE_S = 0.005
ENABLE_TEXT_SPLITTING = True
FIRST_SEGMENT_LIMIT = 450 # Ліміт для першага сегмента, каб пазбегнуць занадта доўгага чакання

# ---------------------------------------------------------
# 5) Кэшаванне латэнтаў для кланавання голасу
# ---------------------------------------------------------
PERSIST_LATENTS_DIR = pathlib.Path("./latents_cache")
PERSIST_LATENTS_DIR.mkdir(parents=True, exist_ok=True)
@dataclass(frozen=True)
class LatentsMeta:
    model_id: str
    gpt_cond_len: int
    max_ref_len: int
    sound_norm_refs: bool

LATENT_CACHE: dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
GPU_LATENT_CACHE: dict[Tuple[str, str], Tuple[torch.Tensor, torch.Tensor]] = {}
default_voice_file = str(model_dir / "voice.wav")

def _latents_key(path: str | None, meta: LatentsMeta) -> str:
    base = f"{os.path.abspath(path)}:{os.path.getmtime(path)}:{os.path.getsize(path)}" if path and os.path.exists(path) else "default_voice"
    return hashlib.md5((base + "|" + json.dumps(meta.__dict__, sort_keys=True)).encode("utf-8")).hexdigest()

def _latents_for(path: str | None, *, to_device: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    meta = LatentsMeta(model_id=repo_id, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_len=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
    key = _latents_key(path, meta)
    g, s = LATENT_CACHE.get(key) or (None, None)
    if g is None:
        disk_path = PERSIST_LATENTS_DIR / f"{key}.pt"
        if disk_path.exists():
            data = torch.load(disk_path, map_location="cpu")
            g, s = data["gpt_cond_latent"], data["speaker_embedding"]
        else:
            print(f"Разлік латэнтаў для {path or 'стандартнага голасу'}...")
            with torch.inference_mode():
                g_cpu, s_cpu = XTTS_MODEL.get_conditioning_latents(audio_path=path)
            g, s = g_cpu.cpu(), s_cpu.cpu()
            torch.save({"gpt_cond_latent": g, "speaker_embedding": s}, disk_path)
            print("Латэнты захаваны ў кэш.")
        LATENT_CACHE[key] = (g, s)
    if to_device:
        dev_key = (key, to_device)
        if dev_key in GPU_LATENT_CACHE:
            return GPU_LATENT_CACHE[dev_key]
        g, s = g.to(to_device, non_blocking=True), s.to(to_device, non_blocking=True)
        GPU_LATENT_CACHE[dev_key] = (g, s)
    return g, s

try:
    _latents_for(default_voice_file, to_device=device)
    print("Стандартны голас паспяхова пракэшаваны.")
except Exception as e:
    print(f"Папярэджанне: не атрымалася папярэдне кэшаваць стандартны голас: {e}")

# ---------------------------------------------------------
# 6) Дапаможныя функцыі для аўдыя
# ---------------------------------------------------------
def _to_np_audio(x) -> np.ndarray:
    if isinstance(x, dict) and "wav" in x:
        x = x["wav"]
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().contiguous().view(-1).numpy()
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(-1)

def _seconds_to_samples(sec: float, sr: int) -> int:
    return max(1, int(sec * sr))

def _crossfade_concat(chunks: List[np.ndarray], sr: int, fade_s: float) -> np.ndarray:
    if not chunks:
        return np.array([], dtype=np.float32)
    result = chunks[0]
    for i in range(1, len(chunks)):
        b = chunks[i]
        fade_n = min(_seconds_to_samples(fade_s, sr), result.size, b.size)
        if fade_n <= 1:
            result = np.concatenate([result, b])
            continue
        fade_out = np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        tail = (result[-fade_n:] * fade_out) + (b[:fade_n] * fade_in)
        result = np.concatenate([result[:-fade_n], tail, b[fade_n:]])
    return result

def _chunker(chunks: Iterable[np.ndarray], sr: int, initial_target_s: float, target_s: float) -> Iterator[np.ndarray]:
    is_first, target_samples = True, _seconds_to_samples(initial_target_s, sr)
    buffer = np.array([], dtype=np.float32)
    for c_np in map(_to_np_audio, chunks):
        if c_np.size == 0:
            continue
        buffer = np.concatenate([buffer, c_np])
        if buffer.size >= target_samples:
            yield buffer
            buffer = np.array([], dtype=np.float32)
            if is_first:
                is_first = False
                target_samples = _seconds_to_samples(target_s, sr)
    if buffer.size > 0:
        yield buffer

def _pcm_f32_to_b64(x: np.ndarray) -> str:
    return base64.b64encode(x.tobytes()).decode("ascii")

# ---------------------------------------------------------
# 7) Падзел тэксту: хуткі + рэзервовы варыянт
# ---------------------------------------------------------
_SENT_END = re.compile(r"([\.!\?…]+[»\")\]]*\s+)")
_WS = re.compile(r"\s+")

def _fast_split(text: str, limit: int) -> List[str]:
    text = text.strip()
    if not text: return []
    parts = []
    start = 0
    for m in _SENT_END.finditer(text):
        end = m.end()
        parts.append(text[start:end].strip())
        start = end
    if start < len(text): parts.append(text[start:].strip())
    chunks = []
    cur = ""
    for s in parts:
        if len(cur) + 1 + len(s) <= limit:
            cur = (cur + " " + s).strip() if cur else s
        else:
            if cur: chunks.append(cur)
            if len(s) <= limit:
                cur = s
            else:
                w = _WS.split(s); acc = ""
                for tok in w:
                    if len(acc) + 1 + len(tok) <= limit:
                        acc = (acc + " " + tok).strip() if acc else tok
                    else:
                        if acc: chunks.append(acc)
                        acc = tok
                if acc: cur = acc
                else: cur = ""
    if cur: chunks.append(cur)
    return [c for c in chunks if c]

def _split_text_smart(text_in: str, lang_short: str, chunk_limit: int) -> List[str]:
    text_in = text_in.strip()
    if not text_in: return []
    parts: List[str] = []
    if len(text_in) > FIRST_SEGMENT_LIMIT:
        head = text_in[:FIRST_SEGMENT_LIMIT]
        m = re.search(r".*[\.!\?…»)]", head)
        if m and len(m.group(0)) > 30:
            head = m.group(0)
        tail = text_in[len(head):].lstrip()
        parts.append(head)
        text_for_rest = tail
    else:
        text_for_rest = text_in
    if not text_for_rest: return parts or [text_in]

    rest = _fast_split(text_for_rest, chunk_limit)
    if not rest or sum(len(x) for x in rest) < int(0.6 * len(text_for_rest)):
        try:
            rest2 = split_sentence(text_for_rest, lang=lang_short, text_split_length=chunk_limit)
            rest2 = [s.strip() for s in rest2 if s and s.strip()]
            if rest2: rest = rest2
        except Exception:
            pass
    return parts + (rest or [text_for_rest])

# ---------------------------------------------------------
# 8) Асноўная функцыя TTS
# ---------------------------------------------------------
@spaces.GPU(duration=120)
def text_to_speech(text_input, speaker_audio, initial_buffer_s, subsequent_buffer_s):
    t_start_req = time.perf_counter()
    if not text_input or not str(text_input).strip():
        raise gr.Error("Увядзі хоць нейкі тэкст 🙂")

    t_lat_0 = time.perf_counter()
    gpt_cond_latent, speaker_embedding = _latents_for(speaker_audio or default_voice_file, to_device=device)
    t_lat_1 = time.perf_counter()

    t_split_0 = time.perf_counter()
    char_limit = XTTS_MODEL.tokenizer.char_limits.get("be", 250)
    texts = _split_text_smart(str(text_input).strip(), "be", char_limit) if ENABLE_TEXT_SPLITTING else [str(text_input).strip()]
    t_split_1 = time.perf_counter()

    server_metrics = {
        "latents_s": t_lat_1 - t_lat_0,
        "text_split_s": t_split_1 - t_split_0,
        "initial_buffer_s": initial_buffer_s,
        "subsequent_buffer_s": subsequent_buffer_s
    }
    yield ("", None, None, json.dumps(server_metrics))

    full_audio_chunks, first_chunk_sent = [], False
    t_gen_start = time.perf_counter()

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        all_chunks_iterator = (
            _to_np_audio(chunk) for part in texts for chunk in XTTS_MODEL.inference_stream(
                text=part,
                language="be",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.2,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=20,
                top_p=0.85
            )
        )

        for audio_chunk in _chunker(all_chunks_iterator, sampling_rate, initial_buffer_s, subsequent_buffer_s):
            if not first_chunk_sent:
                t_first_chunk_ready = time.perf_counter()
                server_metrics["gen_init_to_first_chunk_s"] = t_first_chunk_ready - t_gen_start
                server_metrics["until_first_chunk_total_s"] = t_first_chunk_ready - t_start_req
                yield (_pcm_f32_to_b64(audio_chunk), None, None, json.dumps(server_metrics))
                first_chunk_sent = True
            else:
                yield (_pcm_f32_to_b64(audio_chunk), None, None, None)
            full_audio_chunks.append(audio_chunk)

    if not full_audio_chunks:
        yield ("__STOP__", None, None, json.dumps(server_metrics))
        return

    t_write_0 = time.perf_counter()
    full_audio = _crossfade_concat(full_audio_chunks, sampling_rate, FADE_S)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        write(tmp.name, sampling_rate, full_audio)
    server_metrics["file_write_s"] = time.perf_counter() - t_write_0
    yield ("__STOP__", tmp.name, tmp.name, json.dumps(server_metrics))

# ---------------------------------------------------------
# 9) Карыстальніцкі інтэрфейс (UI) на Gradio
# ---------------------------------------------------------
examples = [["Прывітанне! Гэта праверка жывога струменя беларускага TTS.", None, INITIAL_MIN_BUFFER_S, MIN_BUFFER_S]]

with gr.Blocks() as demo:
    gr.Markdown("## Belarusian TTS — Streaming (стабільны старт) + фінальны файл")
    with gr.Row():
        inp_text = gr.Textbox(lines=5, label="Тэкст на беларускай мове")
        inp_voice = gr.Audio(type="filepath", label="Прыклад голасу (6–10 сек)")
    with gr.Accordion("Дадатковыя налады стрымінгу", open=True):
        initial_buffer_slider = gr.Slider(minimum=0.1, maximum=1.5, value=INITIAL_MIN_BUFFER_S, step=0.05, label="Пачатковы буфер (с)")
        subsequent_buffer_slider = gr.Slider(minimum=0.05, maximum=0.5, value=MIN_BUFFER_S, step=0.01, label="Наступны буфер (с)")
    with gr.Row():
        run_btn = gr.Button("Згенераваць")
        gr.Markdown(f"**Частата дыскрэтызацыі:** {sampling_rate} Гц")

    log_panel = gr.HTML(value='<div id="wa-log" style="font-family:monospace;font-size:12px;white-space:pre-line">[лог пусты]</div>', label="Лагі плэера")
    stream_pipe, log_pipe, final_file, final_audio = gr.Textbox(visible=False), gr.Textbox(visible=False), gr.File(label="Згенераваны WAV"), gr.Audio(label="Фінальнае аўдыя", type="filepath")

    JS_CODE = f"""
function() {{
  const sampleRate = {sampling_rate};
  function initOrResetPlayer() {{
    if (window.__wa) {{
      window.__wa.reset();
      return;
    }}
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) {{ console.error("AudioContext is not supported."); return; }}
    const ctx = new AC({{ sampleRate }});
    const node = ctx.createScriptProcessor(4096, 1, 1);
    let queue = [], playing = false, eos = false;
    let meta = {{ t_click_ms: performance.now(), t_first_push_ms: null, t_first_audio_ms: null, chunk_durations: [], server: null }};

    node.onaudioprocess = (e) => {{
      const out = e.outputBuffer.getChannelData(0); let i = 0;
      while (i < out.length) {{
        if (queue.length === 0 || !playing) {{ out[i++] = 0.0; continue; }}
        let cur = queue[0];
        const take = Math.min(cur.length, out.length - i);
        if (meta.t_first_audio_ms === null) {{ meta.t_first_audio_ms = performance.now(); logUpdate(); }}
        out.set(cur.subarray(0, take), i); i += take;
        if (take === cur.length) queue.shift(); else queue[0] = cur.subarray(take);
      }}
      if (eos && queue.length === 0 && playing) {{ playing = false; logUpdate(); }}
    }};
    node.connect(ctx.destination);

    function fmtS(x) {{ return x === null || x === undefined ? "n/a" : x.toFixed(3) + " s"; }}
    function logUpdate() {{
      const el = document.getElementById('wa-log'); if (!el) return;
      const s = meta.server || {{}}; const lines = ["Клік (Згенераваць): 0.000 s"];
      if (meta.t_first_push_ms) {{
        lines.push("Першы чанк прыйшоў:   " + fmtS((meta.t_first_push_ms - meta.t_click_ms) / 1000));
        if (meta.t_first_audio_ms) {{
          lines.push("Пачатак прайгравання: " + fmtS((meta.t_first_audio_ms - meta.t_click_ms) / 1000));
          lines.push("Затрымка (чанк→аўдыя): " + fmtS((meta.t_first_audio_ms - meta.t_first_push_ms) / 1000));
        }}
      }}
      lines.push("\\n— Налады стрыму —", "Пачатковы буфер (запыт):  " + fmtS(s.initial_buffer_s), "Наступны буфер (запыт):   " + fmtS(s.subsequent_buffer_s));
      if (meta.chunk_durations.length > 0) {{ lines.push("Працягласць 1-га чанка:    " + meta.chunk_durations[0] + " s", "Атрымана чанкаў:          " + meta.chunk_durations.length); }}
      lines.push("\\n— Серверныя метрыкі —", "Латэнты (умоўны голас):  " + fmtS(s.latents_s), "Падзел тэксту:           " + fmtS(s.text_split_s), "Ініт→1-ы чанк:           " + fmtS(s.gen_init_to_first_chunk_s), "Усё да 1-га чанка:       " + fmtS(s.until_first_chunk_total_s));
      if (meta.t_first_push_ms && s.until_first_chunk_total_s) {{ lines.push("\\nАцэнка чаргі ZeroGPU + сеткі: " + fmtS(Math.max(0, (meta.t_first_push_ms - meta.t_click_ms) / 1000 - s.until_first_chunk_total_s))); }}
      lines.push("\\nСтатус стрыму: " + (playing ? "playing" : "stopped"));
      el.innerHTML = lines.join("\\n");
    }}

    window.__wa = {{
      push: (b64) => {{
        if (!b64 || b64 === "__STOP__") {{ eos = true; logUpdate(); return; }}
        const bin = atob(b64); const buf = new ArrayBuffer(bin.length); const view = new Uint8Array(buf);
        for (let i=0; i<bin.length; i++) view[i] = bin.charCodeAt(i);
        const f32 = new Float32Array(buf);
        if (meta.chunk_durations.length === 0 && f32.length > 0) meta.t_first_push_ms = performance.now();
        meta.chunk_durations.push((f32.length / ctx.sampleRate).toFixed(3));
        queue.push(f32);
        if (!playing && queue.length > 0) {{ playing = true; if(ctx.state === "suspended") ctx.resume(); }}
        logUpdate();
      }},
      update_server_metrics: (js) => {{ if(js) meta.server = JSON.parse(js); logUpdate(); }},
      reset: () => {{
        playing = false; eos = false; queue.length = 0;
        meta = {{ t_click_ms: performance.now(), t_first_push_ms: null, t_first_audio_ms: null, chunk_durations: [], server: null }}; logUpdate();
      }},
    }};
  }}
  initOrResetPlayer();
}}
"""
    run_btn.click(fn=None, js=JS_CODE)
    run_btn.click(
        fn=text_to_speech,
        inputs=[inp_text, inp_voice, initial_buffer_slider, subsequent_buffer_slider],
        outputs=[stream_pipe, final_file, final_audio, log_pipe],
    )
    stream_pipe.change(fn=None, inputs=[stream_pipe], js="(b64) => { if(window.__wa) window.__wa.push(b64); }")
    log_pipe.change(fn=None, inputs=[log_pipe], js="(js) => { if(window.__wa) window.__wa.update_server_metrics(js); }")

    gr.Examples(examples=examples, inputs=[inp_text, inp_voice, initial_buffer_slider, subsequent_buffer_slider], cache_examples=False)

if __name__ == "__main__":
    demo.launch()