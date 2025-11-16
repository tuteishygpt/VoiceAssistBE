# Imports

import json
from functools import lru_cache
from time import sleep

import requests  # pip install requests

# Classes

class GeminiClient(object):
    def __init__(self, model_id="gemini-2.5-flash", api_key_path="gemini_api_key"):
        self.url_template = "https://generativelanguage.googleapis.com/v1beta/models/" + model_id + ":generateContent?key=%s"
        with open(api_key_path) as f:
            self.api_key = f.read().strip()

    def _make_payload(self, prompt):
        return {"contents": [{"parts": [{"text": prompt}]}]}

    def _extract_response_text(self, dct):
        assert "candidates" in dct, dct
        assert len(dct["candidates"]) == 1, dct["candidates"]
        assert "content" in dct["candidates"][0], dct["candidates"][0]
        if "parts" not in dct["candidates"][0]["content"]:
            return None
        assert len(dct["candidates"][0]["content"]["parts"]) == 1, dct["candidates"][0]["content"]["parts"]
        assert "text" in dct["candidates"][0]["content"]["parts"][0], dct["candidates"][0]["content"]["parts"][0]
        text = dct["candidates"][0]["content"]["parts"][0]["text"].strip()
        return text

    @lru_cache(maxsize=2**20)
    def ask(self, prompt, timeout=5):
        counter = 0
        while True:
            r = requests.post(
                self.url_template % self.api_key,
                headers={"Content-Type": "application/json"},
                data=json.dumps(self._make_payload(prompt))
            )
            extracted_response = self._extract_response_text(r.json())
            if r.status_code == 200 and extracted_response is not None:
                break
            else:
                duration = timeout * 2**counter
                print(r.status_code, duration)
                sleep(duration)
                counter += 1
        return extracted_response
