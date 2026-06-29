"""Provider-agnostic LLM wrapper.

One interface, three backends. Default is local Ollama (Qwen2.5-3B); set
`LLM_PROVIDER=openai` (or `anthropic`) with a key to upgrade the heavy steps.

    from tenk.llm import get_llm
    llm = get_llm()
    text = llm.complete("Summarise this filing section.", system="You are a financial analyst.")
    data = llm.json('Return {"sentiment": "..."} for: ...')   # parsed dict
"""
from __future__ import annotations

import json
import re
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from tenk.config import settings


class LLM:
    """Thin abstraction over Ollama / OpenAI / Anthropic chat completions."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or settings.llm_provider).lower()
        self._client = None
        if self.provider == "ollama":
            self.model = model or settings.ollama_model
        elif self.provider == "openai":
            self.model = model or settings.openai_model
        elif self.provider == "anthropic":
            self.model = model or settings.anthropic_model
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider!r}")

    # ---- public API ---------------------------------------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
        if self.provider == "ollama":
            return self._ollama(prompt, system, temperature)
        if self.provider == "openai":
            return self._openai(prompt, system, temperature)
        return self._anthropic(prompt, system, temperature)

    def json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        """Complete and parse the first JSON object/array found in the reply."""
        sys = (system or "") + "\nRespond with ONLY valid JSON, no prose, no code fences."
        raw = self.complete(prompt, system=sys.strip())
        return _extract_json(raw)

    # ---- backends -----------------------------------------------------------
    def _ollama(self, prompt: str, system: str | None, temperature: float) -> str:
        import ollama

        if self._client is None:
            self._client = ollama.Client(host=settings.ollama_host)
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        resp = self._client.chat(
            model=self.model, messages=messages, options={"temperature": temperature}
        )
        return resp["message"]["content"]

    def _openai(self, prompt: str, system: str | None, temperature: float) -> str:
        from openai import OpenAI

        if self._client is None:
            if not settings.openai_api_key:
                raise RuntimeError("LLM_PROVIDER=openai but OPENAI_API_KEY is empty.")
            self._client = OpenAI(api_key=settings.openai_api_key)
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature
        )
        return resp.choices[0].message.content or ""

    def _anthropic(self, prompt: str, system: str | None, temperature: float) -> str:
        import anthropic

        if self._client is None:
            if not settings.anthropic_api_key:
                raise RuntimeError("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty.")
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=temperature,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


def _extract_json(raw: str) -> dict[str, Any]:
    """Best-effort: strip code fences and grab the first {...} or [...] block."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"No JSON found in LLM reply: {raw[:200]!r}") from None


_DEFAULT: LLM | None = None


def get_llm() -> LLM:
    """Process-wide default LLM (cached)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LLM()
    return _DEFAULT
