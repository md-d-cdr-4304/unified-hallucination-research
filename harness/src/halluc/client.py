"""NVIDIA NIM client with on-disk caching and retries.

Caching is keyed on (model, messages, sampling params). Deterministic calls
(temperature=0) are always served from cache when available; sampled calls
include the sample index in the key so repeated runs are reproducible without
re-spending API quota.
"""

import hashlib
import json
import os
import time
from pathlib import Path

from openai import OpenAI, APIError, APIStatusError

BASE_URL = "https://integrate.api.nvidia.com/v1"
CACHE_DIR = Path(__file__).resolve().parents[2] / "results" / ".cache"


class NimClient:
    def __init__(self, cache_dir: Path = CACHE_DIR, max_retries: int = 5):
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set")
        self.client = OpenAI(base_url=BASE_URL, api_key=api_key)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.n_calls = 0
        self.n_cache_hits = 0

    def _cache_path(self, key: dict) -> Path:
        digest = hashlib.sha256(
            json.dumps(key, sort_keys=True).encode()
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        sample_idx: int = 0,
        top_p: float = 1.0,
    ) -> str:
        key = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "sample_idx": sample_idx,
        }
        path = self._cache_path(key)
        if path.exists():
            self.n_cache_hits += 1
            return json.loads(path.read_text())["content"]

        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                content = resp.choices[0].message.content or ""
                self.n_calls += 1
                path.write_text(json.dumps({"content": content, "key": key}))
                return content
            except (APIError, APIStatusError) as e:
                status = getattr(e, "status_code", None)
                if status in (429, 500, 502, 503) and attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise
        raise RuntimeError("unreachable")

    def stats(self) -> dict:
        return {"api_calls": self.n_calls, "cache_hits": self.n_cache_hits}
