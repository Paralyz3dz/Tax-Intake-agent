"""
tools.py: the interview as MCP tools, in plain Python (no MCP import), so it is tested without the SDK.

Who does what when an MCP host (Claude Desktop, Cursor, ...) drives the interview:
    host model (the user's own subscription)  talks to the client, rephrases the question,
                                              turns the reply into an answer in the expected format,
                                              passes anything else the client said as client_note
    this server (engine + playbook)           decides the next question, validates every answer
                                              (parse_value), raises escalations, writes the summary

The engine never trusts the host model: an answer it cannot parse comes back as an error and the
question is asked again. Our own model note reader is switched off here (TAX_AGENT_LLM=off).
Retrieval of official text follows TAX_AGENT_RETRIEVAL and can use OpenAI for Listwise ranking.

Interviews are saved in the web server's interview store (src/db/interview_store.py) under the owner
MCP_OWNER, so the preparer reviews them on the /preparer page. A copy of each finished
summary is also written to TAX_AGENT_SUMMARY_DIR (default: <repo>/interview_summaries/).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.app.interviews import MCP_OWNER, InterviewClosed, InterviewNotFound, InterviewService
from src.app.limits import get_budget
from src.forms import f8843
from src.llm.client import set_call_guard
from src.workflow.engine import INTAKE, ROOT, load_scenario, sections

ANSWER_FORMAT = {
    "bool": 'Answer "yes" or "no".',
    "enum": "Answer with exactly one of the options.",
    "money": "Answer with a number only (no $ sign or text), for example 2500.",
    "list": "Answer with a comma-separated list, or \"none\".",
}


RULES = [
    "Ask the client exactly next.question, one question at a time; rephrase only in plain words.",
    "Call submit_answer with the reply in next.answer_format; if the reply is unclear, ask again, never guess.",
    "Put anything else the client mentions in client_note, never in answer.",
    "Give no tax advice (taxable, deductible, which forms, amounts owed): the preparer decides.",
    "Never ask for, repeat or pass on full SSN, ITIN, bank, routing or passport numbers, or the client's name.",
]
REMINDER = "One question at a time; no tax advice; no full ID or account numbers."

_service: InterviewService | None = None


def get_service() -> InterviewService:
    """The process-wide interview service, built on first use with our own model reader turned off.

    The daily budget is installed as the model call guard of this process too, so the Listwise
    reranking in live retrieval counts against TAX_AGENT_LLM_CALLS_PER_DAY.
    """
    global _service
    if _service is None:
        os.environ["TAX_AGENT_LLM"] = "off"
        set_call_guard(get_budget().charge)
        _service = InterviewService.from_env()
    return _service


def set_service(service: InterviewService | None) -> None:
    """Replace the service (tests inject one with a stub search)."""
    global _service
    _service = service


def _summary_dir() -> Path:
    return Path(os.getenv("TAX_AGENT_SUMMARY_DIR") or ROOT / "interview_summaries")


def _shape(out: dict, state: dict) -> dict:
    result = {"interview_id": state["interview_id"], "status": state["status"],
              "answered": state["answered"], "escalations": [f["reason"] for f in state["flags"]]}
    if out["type"] == "question":
        result["next"] = {
            "question": out["question"],
            "answer_format": ANSWER_FORMAT.get(out["input_type"], ""),
            "options": out["options"],
            "section": out.get("section"),
            "why_ask": out["why_ask"],
            "documents_to_request": out["required_docs"],
            "asked_now_because_client_mentioned": out.get("triggered_by"),
            "official_text": [{"citation": c["citation"], "text": (c["text"] or "")[:800]} for c in out["citations"]],
            "source_to_cite": [r["reference"] for r in out["references"]],
        }
        result["reminder"] = REMINDER
    elif out["type"] == "error":
        result["answer_not_accepted"] = out["message"]
        result["next"] = _shape(out["guide"], state)["next"]
    else:
        path = _summary_dir() / f"{state['interview_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        result["summary"] = out
        result["summary_file"] = str(path)
        result["preparer_page"] = "/preparer (web server) lists this interview for review"
        result["tell_the_client"] = ("The interview is complete. A preparer will review the summary; "
                                     "nothing has been filed and no tax advice has been given.")
    return result


def _with_count(result: dict) -> dict:
    """Count the client's answers only (the stored history also holds the background and notes)."""
    turns = get_service().get(result["interview_id"], owner=MCP_OWNER)["turns"]
    return result | {"answered": len(turns)}


def list_topics() -> dict:
    definition, _ = load_scenario(INTAKE)
    return {"topics": sections(definition),
            "how_to_use": "Pass topic ids you are unsure about as `priority` to start_interview; they are asked "
                          "first. A short client background note (for example 'student on an F-1 visa, works as "
                          "a TA') moves the matching follow-up questions to the front. Every base question is "
                          "still asked."}


def start_interview(priority: list[str] | None = None, background: str = "") -> dict:
    try:
        r = get_service().start(INTAKE, priority or [], background, owner=MCP_OWNER)
    except ValueError as exc:
        return {"error": str(exc), "valid_topics": [s["id"] for s in list_topics()["topics"]]}
    return {"rules": RULES, **_with_count(_shape(r["output"], r["state"]))}


def submit_answer(interview_id: str, answer: str, client_note: str = "") -> dict:
    try:
        service = get_service()
        if client_note.strip():
            service.add_note(interview_id, client_note, owner=MCP_OWNER)
        r = service.answer(interview_id, answer, owner=MCP_OWNER)
        return _with_count(_shape(r["output"], r["state"]))
    except InterviewNotFound:
        return {"error": "No interview with this id. Start a new one with start_interview."}
    except InterviewClosed as exc:
        return {"error": str(exc)}


def get_interview(interview_id: str) -> dict:
    try:
        r = get_service().get(interview_id, owner=MCP_OWNER)
    except InterviewNotFound:
        return {"error": "No interview with this id in this session."}
    return {"state": r["state"]}


def lookup_official_text(query: str) -> dict:
    try:
        matches = get_service()._cached_search(query).get("matches", [])[:3]
    except Exception as exc:
        return {"error": f"Knowledge base unavailable: {type(exc).__name__}"}
    return {"matches": [{"citation": m.get("citation"), "text": (m.get("text") or "")[:800]} for m in matches],
            "note": "Use these passages to explain the question. They are not advice for this client."}


def draft_form_8843(interview_id: str) -> dict:
    """No names, addresses, dates or passport numbers pass through the assistant: the client
    types those on /form-8843, which builds the PDF.
    """
    try:
        state = get_service().get(interview_id, owner=MCP_OWNER)["state"]
    except InterviewNotFound:
        return {"error": "No interview with this id. Start a new one with start_interview."}
    plan = f8843.build_plan(state["facts"], None, state["unsure"])
    return {
        "applicable": plan["applicable"],
        "needs_preparer": [s["reason"] for s in plan["stops"]],
        "settled_by_interview": [{"line": f["line"], "label": f["label"], "answer": f["value"]}
                                 for f in plan["fields"] if f["status"] == f8843.FILLED and f["value"]],
        "open_from_interview": [{"line": f["line"], "label": f["label"], "why": f["note"]}
                                for f in plan["fields"] if f["id"] in ("l12", "l13") and f["status"] == f8843.MISSING],
        "client_fills_on_form_page": sorted({f"line {f['line']}: {f['label']}" for f in plan["fields"]
                                             if f["status"] in (f8843.MISSING, f8843.USER_ONLY)
                                             and f["id"] not in ("l12", "l13")}),
        "tell_the_client": ("Form 8843 is a statement, not a tax return. You check, sign and mail it yourself; "
                            "nothing is filed for you. Do not tell me your passport number or SSN."),
    }
