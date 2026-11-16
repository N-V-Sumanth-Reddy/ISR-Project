"""Local LLM clients. Nothing here talks to a remote service: the default host is localhost.

Backends
  ollama  - Ollama's native /api/chat endpoint (default)
  openai  - any OpenAI-compatible /v1/chat/completions server on this machine
            (llama.cpp `llama-server`, LM Studio, ...)
  scripted- deterministic stand-in used by the tests

All calls go through a small SQLite cache keyed by (backend, model, messages, options),
so with temperature 0 a re-run of an evaluation is free and an interrupted run resumes.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests

Message = Dict[str, str]

# Calls go to localhost; never route them through a proxy configured in the environment.
_HTTP = requests.Session()
_HTTP.trust_env = False


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cached: bool = False
    model: str = ""


class LLMError(RuntimeError):
    pass


class LLMCache:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, created REAL)")
        self._db.commit()

    @staticmethod
    def key(payload: Dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, json.dumps(value, ensure_ascii=False), time.time()))
            self._db.commit()


class LLM:
    backend = "base"

    def __init__(self, model: str, temperature: float = 0.0, seed: int = 42, max_tokens: int = 320,
                 cache: Optional[LLMCache] = None):
        self.model = model
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.max_tokens = int(max_tokens)
        self.cache = cache
        self.calls = 0
        self.cache_hits = 0
        self.total_seconds = 0.0

    def _options(self) -> Dict[str, Any]:
        return {"temperature": self.temperature, "seed": self.seed}

    def _call(self, messages: List[Message], max_tokens: int, json_mode: bool) -> LLMResponse:
        raise NotImplementedError

    def chat(self, messages: List[Message], max_tokens: Optional[int] = None, json_mode: bool = False,
             use_cache: bool = True) -> LLMResponse:
        max_tokens = int(max_tokens or self.max_tokens)
        key = None
        if self.cache is not None and use_cache:
            key = LLMCache.key({"backend": self.backend, "model": self.model, "messages": messages,
                                "options": self._options(), "max_tokens": max_tokens, "json": json_mode})
            hit = self.cache.get(key)
            if hit is not None:
                self.cache_hits += 1
                return LLMResponse(text=hit["text"], prompt_tokens=hit.get("prompt_tokens", 0),
                                   completion_tokens=hit.get("completion_tokens", 0), seconds=hit.get("seconds", 0.0),
                                   cached=True, model=self.model)
        resp = self._call(messages, max_tokens, json_mode)
        self.calls += 1
        self.total_seconds += resp.seconds
        if key is not None:
            self.cache.put(key, {"text": resp.text, "prompt_tokens": resp.prompt_tokens,
                                 "completion_tokens": resp.completion_tokens, "seconds": resp.seconds})
        return resp


def _assert_local(host: str, allow_remote: bool) -> None:
    name = (urlparse(host).hostname or "").lower()
    if not allow_remote and name not in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"):
        raise LLMError("LLM host %r is not local. This tool is designed to keep documents on this machine; "
                       "set llm.allow_remote=true only if you really mean to send passages elsewhere." % host)


class OllamaLLM(LLM):
    backend = "ollama"

    def __init__(self, model: str, host: str = "http://localhost:11434", num_ctx: int = 8192, keep_alive: str = "30m",
                 timeout_s: int = 600, allow_remote: bool = False, **kw):
        super().__init__(model, **kw)
        _assert_local(host, allow_remote)
        self.host = host.rstrip("/")
        self.num_ctx = int(num_ctx)
        self.keep_alive = keep_alive
        self.timeout_s = timeout_s

    def _options(self) -> Dict[str, Any]:
        return {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx}

    def _call(self, messages: List[Message], max_tokens: int, json_mode: bool) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": False, "keep_alive": self.keep_alive,
            "options": dict(self._options(), num_predict=max_tokens),
        }
        if json_mode:
            payload["format"] = "json"
        t0 = time.time()
        try:
            r = _HTTP.post(self.host + "/api/chat", json=payload, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise LLMError("cannot reach Ollama at %s (%s). Is `ollama serve` running?" % (self.host, exc)) from exc
        if r.status_code != 200:
            raise LLMError("Ollama returned %d for model %r: %s" % (r.status_code, self.model, r.text[:300]))
        data = r.json()
        text = (data.get("message") or {}).get("content", "")
        return LLMResponse(text=text.strip(), prompt_tokens=int(data.get("prompt_eval_count") or 0),
                           completion_tokens=int(data.get("eval_count") or 0), seconds=time.time() - t0, model=self.model)

    # ---- helpers used by the environment check and the UI
    def available_models(self) -> List[str]:
        try:
            r = _HTTP.get(self.host + "/api/tags", timeout=10)
            r.raise_for_status()
            return [m.get("name") or m.get("model") for m in r.json().get("models", [])]
        except requests.RequestException:
            return []

    def is_up(self) -> bool:
        try:
            return _HTTP.get(self.host + "/api/tags", timeout=5).status_code == 200
        except requests.RequestException:
            return False


class OpenAICompatLLM(LLM):
    backend = "openai"

    def __init__(self, model: str, host: str = "http://localhost:8080", timeout_s: int = 600,
                 allow_remote: bool = False, **kw):
        super().__init__(model, **kw)
        _assert_local(host, allow_remote)
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    def _call(self, messages: List[Message], max_tokens: int, json_mode: bool) -> LLMResponse:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature,
                                   "seed": self.seed, "max_tokens": max_tokens, "stream": False}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        t0 = time.time()
        try:
            r = _HTTP.post(self.host + "/v1/chat/completions", json=payload, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise LLMError("cannot reach the local LLM server at %s (%s)" % (self.host, exc)) from exc
        if r.status_code != 200:
            raise LLMError("LLM server returned %d: %s" % (r.status_code, r.text[:300]))
        data = r.json()
        usage = data.get("usage") or {}
        return LLMResponse(text=(data["choices"][0]["message"].get("content") or "").strip(),
                           prompt_tokens=int(usage.get("prompt_tokens") or 0),
                           completion_tokens=int(usage.get("completion_tokens") or 0),
                           seconds=time.time() - t0, model=self.model)

    def is_up(self) -> bool:
        try:
            return _HTTP.get(self.host + "/v1/models", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def available_models(self) -> List[str]:
        try:
            return [m["id"] for m in _HTTP.get(self.host + "/v1/models", timeout=10).json().get("data", [])]
        except Exception:
            return []


class ScriptedLLM(LLM):
    """Deterministic stand-in: ``fn(messages, json_mode) -> str``. Used by tests only."""

    backend = "scripted"

    def __init__(self, fn: Callable[[List[Message], bool], str], model: str = "scripted", **kw):
        super().__init__(model, **kw)
        self.fn = fn

    def _call(self, messages: List[Message], max_tokens: int, json_mode: bool) -> LLMResponse:
        text = self.fn(messages, json_mode)
        return LLMResponse(text=text, prompt_tokens=sum(len(m["content"].split()) for m in messages),
                           completion_tokens=len(text.split()), seconds=0.0, model=self.model)

    def is_up(self) -> bool:
        return True

    def available_models(self) -> List[str]:
        return [self.model]


def make_llm(cfg, model: Optional[str] = None, cache: Optional[LLMCache] = None, max_tokens: Optional[int] = None) -> LLM:
    c = cfg.llm
    common = dict(temperature=c.get("temperature", 0.0), seed=c.get("seed", 42),
                  max_tokens=max_tokens or c.get("max_tokens", 320), cache=cache)
    name = model or c.model
    if c.get("backend", "ollama") == "openai":
        return OpenAICompatLLM(name, host=c.host, timeout_s=c.get("timeout_s", 600),
                               allow_remote=c.get("allow_remote", False), **common)
    return OllamaLLM(name, host=c.host, num_ctx=c.get("num_ctx", 8192), keep_alive=c.get("keep_alive", "30m"),
                     timeout_s=c.get("timeout_s", 600), allow_remote=c.get("allow_remote", False), **common)


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a JSON object from a model reply."""
    text = text.strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1] if "{" in text and "}" in text else ""):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None
