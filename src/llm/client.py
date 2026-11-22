"""
client.py: the only place in the project that talks to an LLM provider.

Everything else calls chat_json() and never imports a provider SDK. Any OpenAI-compatible
server works (OpenAI, OpenRouter, DeepSeek, Azure, a local Ollama or vLLM): set LLM_BASE_URL,
LLM_API_KEY and LLM_EXTRACT_MODEL (see src/config.py; the old OPENAI_* names still work).
"""
from __future__ import annotations

from base64 import b64encode
from collections.abc import Sequence
import json

from src.config import llm_settings


class LLMUnavailable(RuntimeError):
    """Raised for every model failure: no key, SDK missing, network, quota, bad JSON.

    Callers handle one error type and fall back to rules-only behaviour.
    """


def is_configured() -> bool:
    """True if a key, or a self-hosted base URL, is set (in the environment or .env)."""
    return llm_settings().configured


def chat_json(system: str, user: str, model: str | None = None, timeout: float = 30, *,
              images: Sequence[tuple[str, bytes]] = (), response_format: dict | None = None,
              max_completion_tokens: int | None = None) -> dict:
    """Send one system + one user message and return the model's reply as a dict.

    The request uses temperature 0 and JSON mode, so the reply must be a JSON object.
    The openai package is imported only here, at call time, so tests and rules-only
    runs do not need it installed. Any failure is raised as LLMUnavailable.
    """
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
