"""
limits.py: a daily cap on model calls, per visitor and for the whole site.

DailyBudget.charge is installed as the call guard of src/llm/client.py, so it runs before
every model call the web service makes: chat_json() (note reader, tax questions, document
reading, listwise reranking in live retrieval) and the chat agent's model. Past either cap it
raises LLMUnavailable, which every caller already handles by falling back (rules-only answers,
unranked retrieval), so a spent budget never breaks an interview.

Who the visitor is comes from current_visitor, which api.py sets per request (the browser's
interview cookie, or its IP address). Counters live in this process's memory: run the web
service as one worker (docs/deploy.md). A restart resets them, which only ever errs toward
allowing calls; moving them to the database or Redis later changes this file only.
"""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from threading import Lock
import time

from src.llm.client import LLMUnavailable

current_visitor: ContextVar[str] = ContextVar("current_visitor", default="anonymous")


class DailyBudget:
    def __init__(self, per_visitor: int, per_day: int, clock=time.time):
        self.per_visitor = per_visitor
        self.per_day = per_day
        self._clock = clock
        self._lock = Lock()
        self._day: str | None = None
        self._total = 0
        self._by_visitor: dict[str, int] = {}

    def _roll_day(self) -> None:
        """Start counting again at 0 when the UTC date changes (call with the lock held)."""
        today = datetime.fromtimestamp(self._clock(), timezone.utc).date().isoformat()
        if today != self._day:
            self._day, self._total, self._by_visitor = today, 0, {}

    def allows(self, visitor: str) -> bool:
        """Whether visitor could make one more call now, without counting it."""
        with self._lock:
            self._roll_day()
            return self._total < self.per_day and self._by_visitor.get(visitor, 0) < self.per_visitor

    def take(self, visitor: str) -> bool:
        """Count one call for visitor; False (and nothing counted) if either cap is reached."""
        with self._lock:
            self._roll_day()
            used = self._by_visitor.get(visitor, 0)
            if self._total >= self.per_day or used >= self.per_visitor:
                return False
            self._total += 1
            self._by_visitor[visitor] = used + 1
            return True

    def charge(self) -> None:
        """The call guard: count one call for current_visitor, or raise LLMUnavailable."""
        if not self.take(current_visitor.get()):
            raise LLMUnavailable("daily model limit reached")


_budget: DailyBudget | None = None
_budget_lock = Lock()


def get_budget() -> DailyBudget:
    """This process's budget, created from settings on first use.

    In the web app it is shared by the interview service and the chat page. The MCP server
    runs in its own process and so has its own budget with the same limits.
    """
    global _budget
    with _budget_lock:
        if _budget is None:
            from src.config import app_settings
            s = app_settings()
            _budget = DailyBudget(s.llm_calls_per_visitor, s.llm_calls_per_day)
        return _budget
