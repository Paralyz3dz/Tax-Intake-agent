"""
config.py: the one place the web service reads its settings from the environment (.env).

Values are read at call time, not import time, so tests and a restarted service see the
current environment. Offline data scripts (src/rag, src/db loaders) keep their own settings.

LLM provider (any OpenAI-compatible server: OpenAI, OpenRouter, DeepSeek, Azure, Ollama, vLLM)
    LLM_API_KEY          falls back to OPENAI_API_KEY; may be empty when LLM_BASE_URL is local
    LLM_BASE_URL         falls back to OPENAI_BASE_URL; empty = api.openai.com
    LLM_MODEL            falls back to OPENAI_MODEL, default gpt-4o-mini (chat agent)
    LLM_EXTRACT_MODEL    falls back to OPENAI_EXTRACT_MODEL, then LLM_MODEL (JSON calls)
    LLM_TEMPERATURE      falls back to OPENAI_TEMPERATURE, default 0.5 (chat agent only)

Web service
    TAX_AGENT_MODE                           local (default: no sign-in, for your own machine)
                                             | public (sign-in required, see src/app/auth.py)
    TAX_AGENT_PREPARERS                      "name:hash,name2:hash2" (python -m src.app.auth hash)
    TAX_AGENT_INVITE_CODES                   "code1,code2" for the client pages
    TAX_AGENT_SESSION_SECRET                 signs sign-in cookies (python -m src.app.auth secret)
    TAX_AGENT_LLM_CALLS_PER_VISITOR          model calls per visitor per day (default 200)
    TAX_AGENT_LLM_CALLS_PER_DAY              model calls for the whole site per day (default 2000)
    TAX_AGENT_RETENTION_DAYS                 delete interviews not updated for this many days (0 = keep)
    TAX_AGENT_LLM / TAX_AGENT_RETRIEVAL      on | off
    TAX_AGENT_INTERVIEW_DB_PATH              SQLite file (default logs/interviews.sqlite3)
    TAX_AGENT_INTERVIEW_DATABASE_URL         PostgreSQL instead of SQLite
    TAX_AGENT_GUIDE_EVIDENCE                 prebuilt guide passages
                                             (default data/processed/guide_evidence.json)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _switch(name: str, default: bool = True) -> bool:
    return _env(name, default="on" if default else "off").lower() not in {"off", "0", "false", "no"}


def _int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        value = int(raw) if raw else default
    except ValueError:
        raise ValueError(f"{name} must be a whole number (got {raw!r})") from None
    if value < 0:
        raise ValueError(f"{name} must not be negative (got {value})")
    return value


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str | None
    model: str
    extract_model: str
    temperature: float

    @property
    def configured(self) -> bool:
        """A key, or a self-hosted server that needs none."""
        return bool(self.api_key or self.base_url)

    @property
    def client_key(self) -> str:
        """What to hand the SDK: it refuses an empty key even for servers that ignore it."""
        return self.api_key or "not-needed"


def llm_settings() -> LLMSettings:
    load_dotenv(ROOT / ".env")
    model = _env("LLM_MODEL", "OPENAI_MODEL", default="gpt-4o-mini")
    return LLMSettings(
        api_key=_env("LLM_API_KEY", "OPENAI_API_KEY"),
        base_url=_env("LLM_BASE_URL", "OPENAI_BASE_URL") or None,
        model=model,
        extract_model=_env("LLM_EXTRACT_MODEL", "OPENAI_EXTRACT_MODEL", default=model),
        temperature=float(_env("LLM_TEMPERATURE", "OPENAI_TEMPERATURE", default="0.5")),
    )


@dataclass(frozen=True)
class AppSettings:
    llm_enabled: bool
    retrieval_enabled: bool
    interview_db_path: Path
    interview_database_url: str | None
    guide_evidence: Path
    public: bool = False
    preparers: dict[str, str] = field(default_factory=dict)
    invite_codes: tuple[str, ...] = ()
    session_secret: str = ""
    llm_calls_per_visitor: int = 200
    llm_calls_per_day: int = 2000
    retention_days: int = 0


def _accounts(raw: str) -> dict[str, str]:
    """"alice:hash, bob:hash" -> {"alice": "hash", "bob": "hash"} (hashes contain no colon)."""
    accounts = {}
    for entry in filter(None, (part.strip() for part in raw.split(","))):
        name, sep, hashed = entry.partition(":")
        if not sep or not name.strip() or not hashed.strip():
            raise ValueError("TAX_AGENT_PREPARERS entries must look like name:hash "
                             "(create a hash with: python -m src.app.auth hash)")
        accounts[name.strip()] = hashed.strip()
    return accounts


def app_settings() -> AppSettings:
    load_dotenv(ROOT / ".env")
    return AppSettings(
        public=_env("TAX_AGENT_MODE", default="local").lower() == "public",
        preparers=_accounts(_env("TAX_AGENT_PREPARERS")),
        invite_codes=tuple(filter(None, (c.strip() for c in _env("TAX_AGENT_INVITE_CODES").split(",")))),
        session_secret=_env("TAX_AGENT_SESSION_SECRET"),
        llm_calls_per_visitor=_int("TAX_AGENT_LLM_CALLS_PER_VISITOR", 200),
        llm_calls_per_day=_int("TAX_AGENT_LLM_CALLS_PER_DAY", 2000),
        retention_days=_int("TAX_AGENT_RETENTION_DAYS", 0),
        guide_evidence=Path(_env("TAX_AGENT_GUIDE_EVIDENCE",
                                 default=str(ROOT / "data" / "processed" / "guide_evidence.json"))),
        llm_enabled=_switch("TAX_AGENT_LLM"),
        retrieval_enabled=_switch("TAX_AGENT_RETRIEVAL"),
        interview_db_path=Path(_env("TAX_AGENT_INTERVIEW_DB_PATH", default=str(ROOT / "logs" / "interviews.sqlite3"))),
        interview_database_url=_env("TAX_AGENT_INTERVIEW_DATABASE_URL") or None,
    )
