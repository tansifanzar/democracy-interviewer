"""Thin LLM client.

One code path for open-source models served locally (Ollama, vLLM, llama.cpp
all expose an OpenAI-compatible endpoint) and hosted APIs. A deterministic
`mock` backend lets the whole pipeline run offline, e.g. in tests or CI.

Configuration via environment variables:
    LLM_BACKEND   "openai" (default; any OpenAI-compatible server) or "mock"
    LLM_BASE_URL  e.g. http://localhost:11434/v1 for Ollama (default)
    LLM_MODEL     e.g. llama3.1:8b (default) or gpt-4o-mini
    LLM_API_KEY   not needed for Ollama
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "openai"))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.1:8b"))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "ollama"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.3")))


class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        self._client = None
        if self.config.backend != "mock":
            from openai import OpenAI  # imported lazily so mock mode needs no deps

            self._client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)

    @property
    def is_mock(self) -> bool:
        return self.config.backend == "mock"

    def chat(self, system: str, user: str, temperature: float | None = None) -> str:
        if self.is_mock:
            raise RuntimeError("chat() called in mock mode; agents handle mock themselves")
        resp = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature if temperature is None else temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()

    def chat_json(self, system: str, user: str, default: dict | list) -> dict | list:
        """Ask for JSON and parse it robustly; fall back to `default` on failure."""
        raw = self.chat(system + "\nRespond with valid JSON only, no prose.", user, temperature=0)
        return parse_json(raw, default)


def parse_json(raw: str, default):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    return default
