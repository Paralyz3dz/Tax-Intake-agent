"""
normalize.py: let a model turn the assistant's free-text note into ONE plain value.

The engine calls the normalizer only after the exact parser (engine.parse_value) has
failed. The model's answer is only a proposal: parse_value() checks it again, so a
wrong or invented value is rejected instead of being written into the interview state.
"""
from __future__ import annotations

import json

from src.llm.client import LLMUnavailable, chat_json, is_configured

SYSTEM = """You convert a tax assistant's note about a client's answer into one plain value.
Rules:
- Use only what the note says. Do not guess, do not calculate, do not give tax advice.
- bool  -> "yes" or "no"
- money -> digits only, e.g. "20000" ("about 20k" -> "20000")
- enum  -> exactly one of the allowed options
- list  -> comma-separated items, or "none"
- If the note does not clearly answer the question, return null.
Return JSON: {"value": "<string or null>", "reason": "<short reason>"}"""


def make_normalizer(chat=None):
    """None when no model is configured, so the engine runs rules-only.
    `chat` replaces chat_json in tests (same (system, user) -> dict signature).
    """
    if chat is None:
        if not is_configured():
            return None
        chat = chat_json

    def normalizer(fact: dict, question: dict, note: str) -> str:
        """One plain value for this fact as a string ("yes", "20000", "single").
        ValueError (shown to the assistant) if the model is unavailable or the note does not answer.
        """
        user = json.dumps({
            "fact": fact["name"],
            "type": fact["type"],
            "allowed_options": fact.get("options", []),
            "question_asked": question["question"],
            "assistant_note": note,
        }, ensure_ascii=False)
        try:
            result = chat(SYSTEM, user)
        except LLMUnavailable as exc:
            raise ValueError(f"{fact['name']}: could not read the note automatically ({exc}); "
                             "please enter the value directly") from None
        value = result.get("value")
        if value in (None, "", "null"):
            raise ValueError(f"{fact['name']}: the note does not clearly answer this question - "
                             f"{result.get('reason', 'please ask again')}")
        return str(value)

    return normalizer
