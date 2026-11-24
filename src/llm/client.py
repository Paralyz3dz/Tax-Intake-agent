"""
client.py: the only place in the project that talks to an LLM provider.

Everything else calls chat_json() and never imports a provider SDK. Any OpenAI-compatible
server works (OpenAI, OpenRouter, DeepSeek, Azure, a local Ollama or vLLM): set LLM_BASE_URL,
LLM_API_KEY and LLM_EXTRACT_MODEL (see src/config.py; the old OPENAI_* names still work).

A call guard (set_call_guard) runs before every model call: the web service installs the daily
budget there (src/app/limits.py). Offline scripts install none, so they are never limited.
"""
from __future__ import annotations

from base64 import b64encode
from collections.abc import Callable, Sequence
import json

from src.config import llm_settings

_call_guard: Callable[[], None] | None = None


class LLMUnavailable(RuntimeError):
    """Raised for every model failure: no key, SDK missing, network, quota, bad JSON.

    Callers handle one error type and fall back to rules-only behaviour.
    """


def is_configured() -> bool:
    """True if a key, or a self-hosted base URL, is set (in the environment or .env)."""
    return llm_settings().configured


def set_call_guard(guard: Callable[[], None] | None) -> None:
    """Install a function to run before every model call (None removes it).

    The guard raises LLMUnavailable to refuse a call; the request is then never sent.
    """
    global _call_guard
    _call_guard = guard


def check_call_guard() -> None:
    """Run the installed guard, if any. chat_json() and the chat agent's model call this first."""
    if _call_guard is not None:
        _call_guard()


def chat_json(system: str, user: str, model: str | None = None, timeout: float = 30, *,
              images: Sequence[tuple[str, bytes]] = (), response_format: dict | None = None,
              max_completion_tokens: int | None = None) -> dict:
    """One system + one user message -> the reply as a dict (temperature 0, JSON mode).

    openai is imported only here, at call time, so tests and rules-only runs do not need it.
    Every failure, including a call refused by the guard, is raised as LLMUnavailable.
    """
    check_call_guard()
    settings = llm_settings()
    if not settings.configured:
        raise LLMUnavailable("No model configured: set LLM_API_KEY (or OPENAI_API_KEY) or LLM_BASE_URL")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMUnavailable("pip install openai") from exc

    content = user
    if images:
        content = [{"type": "text", "text": user}]
        for mime_type, data in images:
            encoded = b64encode(data).decode("ascii")
            content.append({"type": "image_url", "image_url": {
                "url": f"data:{mime_type};base64,{encoded}",
                "detail": "high",
            }})
    try:
        client = OpenAI(api_key=settings.client_key, base_url=settings.base_url, timeout=timeout)
        response = client.chat.completions.create(
            model=model or settings.extract_model,
            temperature=0,
            response_format=response_format or {"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            **({"max_completion_tokens": max_completion_tokens} if max_completion_tokens is not None else {}),
        )
        return json.loads(response.choices[0].message.content or "{}")
    except Exception as exc:
        raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
