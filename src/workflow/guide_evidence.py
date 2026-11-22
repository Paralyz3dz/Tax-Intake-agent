"""
guide_evidence.py: official passages for every interview question, retrieved once ahead of time.

A question's retrieval_query, wording and why_ask come from the playbook, never from the client,
so the passages a guide shows are the same in every interview. Retrieving them live costs an
embedding (Ollama), a database search and a model rerank per question per visitor; building them
once here means the web service needs none of that to show the passages.

    python -m src.workflow.guide_evidence build      # needs DATABASE_URL, Ollama, the LLM key
    python -m src.workflow.guide_evidence check      # how many questions the file covers

Entries are keyed by fingerprint(question). When a playbook question is edited, its fingerprint
changes, lookup() misses, and the service falls back to live retrieval until the file is rebuilt.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

VERSION = 1
KEPT = ("citation", "text", "additional_citations")


def fingerprint(question: dict) -> str:
    """Hash of everything retrieval and reranking see about a question."""
    parts = [question.get("id"), question.get("retrieval_query"), question.get("question"), question.get("why_ask")]
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:24]


def all_questions(scenarios: list[str]) -> list[dict]:
    """Every question of these scenarios, once each (the modular intake shares many)."""
    from src.workflow.engine import load_scenario
    seen, out = set(), []
    for scenario in scenarios:
        _, questions = load_scenario(scenario)
        for q in questions:
            key = fingerprint(q)
            if key not in seen:
                seen.add(key)
                out.append(q)
    return out


def build(scenarios: list[str], search, meta: dict | None = None) -> dict:
    """Retrieve passages for every question. A failed question is listed, not fatal.

    Each match keeps only the fields build_guide() reads (KEPT). A result marked
    rerank_fallback (unranked candidates) counts as a failure, so it is not frozen into the file.
    """
    entries, failures = {}, []
    questions = all_questions(scenarios)
    for i, q in enumerate(questions, 1):
        try:
            result = search(q["retrieval_query"], question=q)
        except Exception as exc:
            failures.append({"question_id": q["id"], "error": f"{type(exc).__name__}: {exc}"})
            logger.warning("[%d/%d] %s failed: %s", i, len(questions), q["id"], exc)
            continue
        if result.get("rerank_fallback"):
            failures.append({"question_id": q["id"], "error": "rerank fallback"})
            continue
        matches = [{k: m[k] for k in KEPT if m.get(k)} for m in result.get("matches", [])]
        entries[fingerprint(q)] = {"question_id": q["id"], "matches": matches}
        logger.info("[%d/%d] %s: %d passages", i, len(questions), q["id"], len(matches))
    return {"version": VERSION,
            "meta": {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(meta or {})},
            "entries": entries, "failures": failures}


def save(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


class GuideEvidence:
    """Read side: the passages for a question, or None if this question was not prebuilt."""

    def __init__(self, data: dict):
        self.meta = data.get("meta", {})
        self._entries = data.get("entries", {})

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, question: dict) -> dict | None:
        entry = self._entries.get(fingerprint(question))
        return {"matches": entry["matches"], "precomputed": True} if entry else None


def load(path: Path) -> GuideEvidence | None:
    """The prebuilt file, or None if it does not exist or is unreadable (live retrieval then)."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable guide evidence %s: %s", path, exc)
        return None
    if data.get("version") != VERSION:
        logger.warning("Ignoring guide evidence %s: version %r, expected %r", path, data.get("version"), VERSION)
        return None
    return GuideEvidence(data)


def _scenarios() -> list[str]:
    from src.app.interviews import list_playbooks
    return [p["id"] for p in list_playbooks()]


def main(argv: list[str]) -> int:
    from src.config import app_settings
    path = app_settings().guide_evidence
    command = argv[0] if argv else "check"
    if command == "build":
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        from src.workflow.engine import _default_search
        settings = json.loads((Path(__file__).resolve().parents[2] / "config" / "settings.json").read_text(encoding="utf-8"))
        data = build(_scenarios(), _default_search,
                     meta={k: settings.get(k) for k in ("tax_year", "corpus_version", "rerank_method", "listwise_model")})
        save(data, path)
        print(f"Wrote {len(data['entries'])} questions to {path}; {len(data['failures'])} failed.")
        return 1 if data["failures"] else 0
    if command == "check":
        evidence = load(path)
        questions = all_questions(_scenarios())
        covered = sum(evidence.lookup(q) is not None for q in questions) if evidence else 0
        print(f"{path}: {covered}/{len(questions)} questions prebuilt"
              + (f" (built {evidence.meta.get('built_at')})" if evidence else " (file missing)"))
        return 0 if covered == len(questions) else 1
    print("usage: python -m src.workflow.guide_evidence build|check")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
