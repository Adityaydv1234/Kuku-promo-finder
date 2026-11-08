"""One small interface for every model provider.

    client = make_client("gemini", "gemini-flash-latest")
    text = client.complete(system, prompt)          # returns raw text (JSON expected)

Swapping models = changing provider/model in config.yaml or the UI.
Parser, validation and ranking never import anything provider-specific.
Responses are cached on disk (keyed by provider+model+prompt) so re-runs are free
and reproducible. 429/5xx errors are retried with exponential backoff.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests

KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",   # any OpenAI-compatible endpoint (set OPENAI_BASE_URL)
}


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: str, model: str, api_key: str | None = None,
                 cache_dir: str | Path = ".cache", temperature: float = 0.0,
                 min_interval: float = 0.0, max_retries: int = 6, timeout: int = 120,
                 log=print):
        if provider not in KEY_ENV:
            raise ValueError(f"unknown provider '{provider}', choose from {list(KEY_ENV)}")
        self.provider, self.model, self.temperature = provider, model, temperature
        self.api_key = api_key or os.environ.get(KEY_ENV[provider], "")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval, self.max_retries, self.timeout = min_interval, max_retries, timeout
        self.log = log
        self._last_call = 0.0
        self.calls = 0          # real API calls made
        self.cache_hits = 0

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def _key(self, system: str, prompt: str) -> Path:
        h = hashlib.sha256(json.dumps([self.provider, self.model, self.temperature, system, prompt]).encode()).hexdigest()
        return self.cache_dir / f"{h[:32]}.json"

    def complete(self, system: str, prompt: str, use_cache: bool = True) -> str:
        path = self._key(system, prompt)
        if use_cache and path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text("utf-8"))["text"]
        if not self.api_key:
            raise LLMError(f"No API key. Set {KEY_ENV[self.provider]} or enter it in the app.")
        delay = 4.0
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            try:
                text = getattr(self, f"_call_{self.provider}")(system, prompt)
                self.calls += 1
                path.write_text(json.dumps({"model": self.name, "text": text}), "utf-8")
                return text
            except _Retryable as e:
                if attempt == self.max_retries:
                    raise LLMError(f"{self.name} failed after {attempt} attempts: {e}") from e
                sleep = e.retry_after or delay
                self.log(f"  [{self.name}] {e}; retry {attempt}/{self.max_retries - 1} in {sleep:.0f}s")
                time.sleep(sleep)
                delay = min(delay * 2, 90)
        raise LLMError("unreachable")

    # ---------------- providers ----------------
    def _post(self, url, headers, body):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        except requests.RequestException as e:
            raise _Retryable(f"network error: {e}")
        if r.status_code == 429 or r.status_code >= 500:
            ra = r.headers.get("retry-after")
            raise _Retryable(f"HTTP {r.status_code}", float(ra) if ra and ra.replace('.', '', 1).isdigit() else None)
        if r.status_code >= 400:
            raise LLMError(f"{self.name} HTTP {r.status_code}: {r.text[:400]}")
        return r.json()

    def _call_gemini(self, system, prompt):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": self.temperature, "responseMimeType": "application/json"},
        }
        data = self._post(url, {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}, body)
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError):
            raise _Retryable(f"empty/blocked response: {str(data)[:200]}")

    def _openai_style(self, base, system, prompt, json_mode=True):
        body = {"model": self.model, "temperature": self.temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = self._post(f"{base}/chat/completions",
                          {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, body)
        return data["choices"][0]["message"]["content"]

    def _call_groq(self, system, prompt):
        return self._openai_style("https://api.groq.com/openai/v1", system, prompt)

    def _call_openai(self, system, prompt):
        return self._openai_style(os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), system, prompt)

    def _call_anthropic(self, system, prompt):
        body = {"model": self.model, "max_tokens": 8000, "temperature": self.temperature,
                "system": system, "messages": [{"role": "user", "content": prompt}]}
        data = self._post("https://api.anthropic.com/v1/messages",
                          {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                           "content-type": "application/json"}, body)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


class _Retryable(Exception):
    def __init__(self, msg, retry_after=None):
        super().__init__(msg)
        self.retry_after = retry_after


def extract_json(text: str):
    """Parse model output as JSON, tolerating ```json fences or leading prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        s, e = t.find("{"), t.rfind("}")
        if s != -1 and e > s:
            return json.loads(t[s:e + 1])
        raise
