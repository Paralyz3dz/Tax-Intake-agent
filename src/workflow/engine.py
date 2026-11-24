"""
engine.py: the guided-interview workflow (the preparer's "digital clone").

Playbooks: playbooks/<scenario>/ or playbooks/modules/*/ (facts, questions, escalation rules).

One turn, step(state, answer):
    1. extract_facts   answer -> typed value (rules first, model only as a fallback)
    2. review          escalation rules -> flags; required documents -> missing_docs
    3. next_question   first required, unknown, askable fact
    4. build_guide     question + why + documents + retrieved official text
    5. summarize       nothing left: summary for the preparer

Principles:
    - Code picks the next question from the playbook, never the model.
    - A model may propose a value, but parse_value() validates it before it enters the state.
    - The summary only claims passages that retrieval actually returned.
    - Retrieval and the model are passed in, so tests need neither.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from src.workflow.conditions import check
from src.workflow.state import State

ROOT = Path(__file__).resolve().parents[2]
PLAYBOOKS = ROOT / "playbooks"
MODULES = PLAYBOOKS / "modules"
INTAKE = "intake"
BASE_MODULE = "base_intake"
BASE_PARTS = {"about_you": "About you", "household": "Household", "income": "Income",
              "expenses": "Expenses", "events": "Life events"}
logger = logging.getLogger(__name__)

YES = {"yes", "y", "true", "1", "是", "有", "对"}
NO = {"no", "n", "false", "0", "否", "没有", "无", "不是"}
NONE_WORDS = {"none", "no", "n/a", "无", "没有"}
UNSURE = {"not sure", "i don't know", "unsure", "don't know", "不确定", "不知道", "不清楚"}


def load_scenario(scenario: str) -> tuple[dict, list[dict]]:
    """"intake" -> the modular playbook (compose_modules); any other name -> playbooks/<scenario>/."""
    if scenario == INTAKE:
        return compose_modules()
    folder = PLAYBOOKS / scenario
    definition = json.loads((folder / "scenario.json").read_text(encoding="utf-8"))
    questions = json.loads((folder / "questions.json").read_text(encoding="utf-8"))
    return definition, questions


def compose_modules() -> tuple[dict, list[dict]]:
    """Merge playbooks/modules/*/ into one playbook, base_intake first.

    A topic module is switched on by its activate_when (usually a yes to a base question). The gate is
    prepended to each module fact's ask_when, so next_question needs no special case.
    """
    folders = sorted(p for p in MODULES.iterdir() if (p / "module.json").exists())
    folders.sort(key=lambda p: p.name != BASE_MODULE)
    facts, questions, rules, reminders, modules, triggers = [], [], [], [], [], []
    for folder in folders:
        m = json.loads((folder / "module.json").read_text(encoding="utf-8"))
        gate = m.get("activate_when") or []
        modules.append({"id": m["id"], "title": m["title"], "activate_when": gate,
                        "keywords": m.get("keywords", [])})
        for f in m.get("facts", []):
            f = dict(f, module=m["id"])
            if gate:
                f["ask_when"] = gate + (f.get("ask_when") or [])
            facts.append(f)
        rules += [_gated(r, gate, m["id"]) for r in m.get("escalation_rules", [])]
        reminders += [_gated(r, gate, m["id"]) for r in m.get("reminders", [])]
        triggers += [dict(t, module=m["id"]) for t in m.get("triggers", [])]
        qs = json.loads((folder / "questions.json").read_text(encoding="utf-8"))
        questions += [dict(q, module=m["id"]) for q in qs]
    definition = {"id": INTAKE, "title": "Client intake (Form 13614-C) with topic modules",
                  "facts": facts, "escalation_rules": rules, "reminders": reminders, "modules": modules,
                  "triggers": triggers}
    return definition, questions


def _gated(item: dict, gate: list[dict], module: str) -> dict:
    """A module rule may test only base facts, so without the gate it would fire with the module off."""
    when = list(item.get("when") or [])
    return dict(item, module=module, when=[c for c in gate if c not in when] + when)


def sections(definition: dict) -> list[dict]:
    """The choices for "ask these first": base_intake parts plus topic modules."""
    out = [{"id": k, "label": v} for k, v in BASE_PARTS.items()
           if any(f.get("part") == k for f in definition["facts"])]
    out += [{"id": m["id"], "label": m["title"]} for m in definition.get("modules", []) if m["id"] != BASE_MODULE]
    return out


def section_of(fact: dict) -> str | None:
    module = fact.get("module")
    if module and module != BASE_MODULE:
        return module
    return fact.get("part")


def _position(text: str, keyword: str) -> int | None:
    """Index of keyword in text as a whole word or phrase (case ignored), or None."""
    m = re.search(r"(?<![a-z0-9_])" + re.escape(keyword.lower()) + r"(?![a-z0-9_])", text.lower())
    return m.start() if m else None


def _mentions(text: str, keyword: str) -> bool:
    return _position(text, keyword) is not None


def apply_triggers(state: State, definition: dict, text: str) -> list[str]:
    """Trigger words in free text move follow-up questions to the front.

    The practitioner's method: a keyword the client says ("kids", "crypto") opens deeper questions at once.
      - trigger keyword -> its ask_first facts go to state.boosted, in the order mentioned
      - module keyword  -> the module is added to state.priority
    Returns the ids that fired, for the history record.
    """
    if not text:
        return []
    fired = []
    names = {f["name"] for f in definition["facts"]}
    done = {b["trigger"] for b in state.boosted}
    hits = []
    for t in definition.get("triggers", []):
        if t["id"] in done:
            continue
        found = [(_position(text, k), k) for k in t["keywords"]]
        found = [x for x in found if x[0] is not None]
        if found:
            hits.append((min(found), t))
    for (_, hit), t in sorted(hits, key=lambda x: x[0][0]):
        fired.append(t["id"])
        for fact in t["ask_first"]:
            if fact in names and state.facts.get(fact) is None:
                state.boosted.append({"fact": fact, "trigger": t["id"], "keyword": hit})
    for m in definition.get("modules", []):
        if m["id"] != BASE_MODULE and m["id"] not in state.priority and any(_mentions(text, k) for k in m["keywords"]):
            state.priority.append(m["id"])
            fired.append(m["id"])
    return fired


def ordered_facts(definition: dict, priority: list[str], boosted: list[dict] | None = None) -> list[dict]:
    """Asking order: triggered facts, then priority sections, then playbook order.

    A prioritized module also pulls up the base question that switches it on (e.g. has_self_employment
    for side_gig), so choosing "Self-employment" starts with that question.
    """
    boost = {}
    for b in boosted or []:
        boost.setdefault(b["fact"], len(boost))
    if not priority and not boost:
        return definition["facts"]
    rank = {sec: i for i, sec in enumerate(priority)}
    for m in definition.get("modules", []):
        if m["id"] in rank:
            for cond in m["activate_when"]:
                rank.setdefault("gate:" + cond["fact"], rank[m["id"]] - 0.5)
    last = len(priority)

    def key(item):
        i, f = item
        if f["name"] in boost:
            return (-1, boost[f["name"]], i)
        r = rank.get("gate:" + f["name"], rank.get(section_of(f), last))
        return (r, 0, i)

    return [f for _, f in sorted(enumerate(definition["facts"]), key=key)]


def start_interview(scenario: str, priority: list[str] | None = None, background: str = "") -> State:
    """priority: section ids to ask first, in order (unknown ids raise ValueError).
    background: client note whose trigger words reorder the interview before the first question.
    """
    definition, _ = load_scenario(scenario)
    priority = list(priority or [])
    allowed = {s["id"] for s in sections(definition)}
    unknown = [p for p in priority if p not in allowed]
    if unknown:
        raise ValueError(f"Unknown priority section(s): {', '.join(unknown)}")
    facts = {f["name"]: None for f in definition["facts"]}
    state = State(scenario=scenario, facts=facts, priority=priority)
    if background:
        fired = apply_triggers(state, definition, background)
        state.history.append({"background": background, "triggers": fired})
    return state


def next_question(state: State) -> dict | None:
    """Pure code, no model. None means nothing is left to ask, which ends the interview."""
    definition, questions = load_scenario(state.scenario)
    by_fact = {q["fact"]: q for q in questions}
    for f in ordered_facts(definition, state.priority, state.boosted):
        name = f["name"]
        if not f.get("required"):
            continue
        if state.facts.get(name) is not None:
            continue
        if name in state.unsure:
            continue
        if not check(f.get("ask_when"), state.facts):
            continue
        q = by_fact.get(name)
        if q is None:
            continue
        return q
    return None


def review(state: State) -> State:
    """After every accepted answer: add matching escalation rules to flags (once), and the
    required_docs of every truthy answer to missing_docs (once). A "no" needs no documents.
    """
    definition, questions = load_scenario(state.scenario)

    flagged = [f["rule"] for f in state.flags]
    for rule in definition["escalation_rules"]:
        hit = check(rule["when"], state.facts)
        if hit and rule["id"] not in flagged:
            state.flags.append({"rule": rule["id"], "reason": rule["reason"]})

    for q in questions:
        value = state.facts.get(q["fact"])
        if not value:
            continue
        for doc in q["required_docs"]:
            if doc not in state.missing_docs:
                state.missing_docs.append(doc)
    return state


def fact_definition(definition: dict, name: str) -> dict:
    """KeyError means questions.json and scenario.json disagree."""
    for f in definition["facts"]:
        if f["name"] == name:
            return f
    raise KeyError(f"Fact {name!r} is not defined in scenario {definition['id']!r}")


def parse_value(fact: dict, text: str):
    """Answer text -> the fact's type, or ValueError telling the assistant what to enter. No model.

        bool   yes / no (also y, true, 1, 是, 有, 没有 ...)  -> True / False
        money  "$12,000", "99.5"; negative rejected       -> int or float
        enum   one of options; "Head of household" ok     -> "head_of_household"
        list   comma-separated, or "none"                 -> list of str
    """
    raw = text.strip()
    low = raw.lower()
    kind = fact["type"]

    if kind == "bool":
        if low in YES:
            return True
        if low in NO:
            return False
        raise ValueError(f"{fact['name']}: please answer yes or no (got {raw!r})")

    if kind == "money":
        cleaned = low.replace("$", "").replace(",", "").replace(" ", "")
        try:
            amount = float(cleaned)
        except ValueError:
            raise ValueError(f"{fact['name']}: please enter an amount such as 12000 (got {raw!r})") from None
        if amount < 0:
            raise ValueError(f"{fact['name']}: amount cannot be negative")
        return int(amount) if amount.is_integer() else round(amount, 2)

    if kind == "enum":
        key = low.replace("-", "_").replace(" ", "_")
        if key in fact["options"]:
            return key
        raise ValueError(f"{fact['name']}: choose one of {', '.join(fact['options'])} (got {raw!r})")

    if kind == "list":
        if low in NONE_WORDS:
            return []
        return [item.strip() for item in raw.split(",") if item.strip()]

    raise ValueError(f"{fact['name']}: unknown fact type {kind!r} in scenario.json")


def extract_facts(state: State, question: dict, answer_text: str, normalizer=None) -> dict:
    """Rules first; if they fail and a normalizer is given, the model rewrites free text
    ("about 20k from her IRA" -> "20000") and that goes through parse_value() again.
    If the model fails too, the original ValueError is raised: nothing unvalidated reaches the state.
    """
    definition, _ = load_scenario(state.scenario)
    fact = fact_definition(definition, question["fact"])
    try:
        return {"facts": {fact["name"]: parse_value(fact, answer_text)}, "parsed_by": "rules"}
    except ValueError as rule_error:
        if normalizer is None:
            raise
        proposal = normalizer(fact, question, answer_text)
        try:
            value = parse_value(fact, proposal)
        except ValueError:
            raise rule_error from None
        return {"facts": {fact["name"]: value}, "parsed_by": "model"}


def _default_search(query: str, *, question: dict | None = None) -> dict:
    """Imported here so the engine and its tests load without the DB or Ollama."""
    from src.rag.retriever import retrieve
    return retrieve(query, question=question)


def _doc_key(text: str) -> str:
    """Document name for comparing references: "IRS Pub 17 (2025), 'Filing status', p. 23" -> "irs pub 17"."""
    head = re.sub(r"\s*\(\d{4}\)", "", text).split(",")[0]
    return " ".join(head.lower().split())


def check_references(source: str, citations: list[dict], retrieval_ok: bool) -> list[dict]:
    """Mark each playbook reference (question["source"], ";"-separated) by retrieval result:
    True found, False not found (likely missing from the knowledge base), None retrieval was down.
    """
    found = {_doc_key(c["citation"]) for c in citations if c.get("citation")}
    found.update(_doc_key(alias) for c in citations for alias in c.get("additional_citations", []))
    refs = []
    for part in source.split(";"):
        ref = part.strip()
        if ref:
            refs.append({"reference": ref,
                         "retrieved": (_doc_key(ref) in found) if retrieval_ok else None})
    return refs


def _section_label(definition: dict, fact: dict) -> str | None:
    sec = section_of(fact)
    for s in sections(definition):
        if s["id"] == sec:
            return s["label"]
    return None


def build_guide(state: State, question: dict, search=None, top_n: int = 5, *, question_search=None) -> dict:
    """What the assistant sees for one question.

    Retrieval errors are logged and the guide is returned without passages: RAG never stops the interview.
    Two kinds of source are kept apart and saved in state.evidence for summarize():
        citations   passages retrieval really returned
        references  the playbook's pointers, marked by check_references()
    """
    definition, _ = load_scenario(state.scenario)
    fact = fact_definition(definition, question["fact"])
    citations, retrieval_error = [], None
    try:
        if question_search is not None:
            result = question_search(question)
        elif search is not None:
            result = search(question["retrieval_query"])
        else:
            result = _default_search(question["retrieval_query"], question=question)
        for m in result.get("matches", [])[:top_n]:
            passage = {"citation": m.get("citation"), "text": (m.get("text") or "")}
            if m.get("additional_citations"):
                passage["additional_citations"] = m["additional_citations"]
            citations.append(passage)
    except Exception as exc:
        retrieval_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Retrieval failed for %r: %s", question["retrieval_query"], retrieval_error)

    references = check_references(question["source"], citations, retrieval_error is None)
    state.evidence[question["id"]] = {"citations": [c["citation"] for c in citations],
                                      "references": references}

    return {
        "type": "question",
        "question_id": question["id"],
        "fact": question["fact"],
        "input_type": fact["type"],
        "options": fact.get("options", []),
        "step_number": len(state.asked),
        "section": _section_label(definition, fact),
        "triggered_by": next((b["keyword"] for b in state.boosted if b["fact"] == question["fact"]), None),
        "question": question["question"],
        "why_ask": question["why_ask"],
        "required_docs": question["required_docs"],
        "references": references,
        "citations": citations,
        "retrieval_error": retrieval_error,
        "flags": list(state.flags),
    }


def summarize(state: State) -> dict:
    """Summary for the preparer.

    references_not_in_kb lists references retrieval never returned for any question in this interview
    (possible knowledge-base gaps); one that missed for a question but came back for another is not listed.
    """
    definition, questions = load_scenario(state.scenario)
    by_fact = {q["fact"]: q for q in questions}
    answered = {k: v for k, v in state.facts.items() if v is not None}
    unanswered = [f["name"] for f in definition["facts"]
                  if f.get("required") and state.facts.get(f["name"]) is None
                  and check(f.get("ask_when"), state.facts)
                  and f["name"] not in state.unsure]
    follow_up = [{"fact": name, "question": by_fact[name]["question"], "why_ask": by_fact[name]["why_ask"]}
                 for name in state.unsure]

    answered_ids = [by_fact[k]["id"] for k in answered if k in by_fact]
    retrieved, verify, missing, unchecked = set(), set(), set(), set()
    for qid in answered_ids:
        ev = state.evidence.get(qid, {})
        retrieved.update(c for c in ev.get("citations", []) if c)
        for r in ev.get("references", []):
            verify.add(r["reference"])
            if r["retrieved"] is False:
                missing.add(r["reference"])
            elif r["retrieved"] is None:
                unchecked.add(r["reference"])

    seen_docs = {_doc_key(c) for c in retrieved}
    missing = {ref for ref in missing if _doc_key(ref) not in seen_docs}

    reminders = [r["text"] for r in definition.get("reminders", []) if check(r.get("when"), state.facts)]

    return {
        "type": "summary",
        "interview_id": state.interview_id,
        "scenario": definition["title"],
        "facts": answered,
        "not_yet_covered": unanswered,
        "needs_preparer_follow_up": follow_up,
        "flags": list(state.flags),
        "needs_preparer_attention": bool(state.flags) or bool(follow_up),
        "missing_docs": list(state.missing_docs),
        "citations_retrieved": sorted(retrieved),
        "references_to_verify": sorted(verify),
        "references_not_in_kb": sorted(missing),
        "references_unchecked": sorted(unchecked),
        "reminders": reminders,
        "priority": list(state.priority),
        "turns": sum("tax_qa" not in entry and "submitted" not in entry for entry in state.history),
        "status": state.status,
    }


def _question_by_id(state: State, question_id: str) -> dict:
    _, questions = load_scenario(state.scenario)
    for q in questions:
        if q["id"] == question_id:
            return q
    raise KeyError(question_id)


def _show(state: State, question: dict, search, question_search) -> dict:
    if not state.asked or state.asked[-1] != question["id"]:
        state.asked.append(question["id"])
    return build_guide(state, question, search=search, question_search=question_search)


def record_answer(state: State, answer_text: str, normalizer=None) -> str | None:
    current = _question_by_id(state, state.asked[-1])
    definition, _ = load_scenario(state.scenario)
    if answer_text.strip().lower() in UNSURE:
        state.unsure.append(current["fact"])
        state.history.append({"question_id": current["id"], "answer_text": answer_text,
                              "facts": {}, "parsed_by": "unsure"})
        return None
    try:
        extracted = extract_facts(state, current, answer_text, normalizer=normalizer)
    except ValueError as exc:
        apply_triggers(state, definition, answer_text)
        return str(exc)

    fired = apply_triggers(state, definition, answer_text) if extracted["parsed_by"] != "rules" else []
    new_facts = extracted["facts"]
    state.facts.update(new_facts)
    state.history.append({"question_id": current["id"], "answer_text": answer_text,
                          "facts": new_facts, "parsed_by": extracted["parsed_by"], "triggers": fired})
    review(state)
    return None


def step(state: State, answer_text: str | None = None, search=None,
         normalizer=None, *, question_search=None, auto_submit: bool = True) -> tuple[State, dict]:
    """One turn. step(state) -> first question; step(state, answer) -> answer the last asked question.

    output["type"]: "question" (next guide), "error" (answer not understood; state unchanged,
    output["guide"] repeats the question) or "summary" (done).
    Only a free-text answer (not a button or exact value) can fire trigger words.
    Raises RuntimeError if the interview is no longer in progress.
    """
    if state.status != "in_progress":
        raise RuntimeError(f"Interview is {state.status}; nothing to step.")

    if answer_text is not None and state.asked:
        error = record_answer(state, answer_text, normalizer)
        if error is not None:
            current = _question_by_id(state, state.asked[-1])
            guide = build_guide(state, current, search=search, question_search=question_search)
            return state, {"type": "error", "message": error, "guide": guide}

    question = next_question(state)
    if question is None:
        if auto_submit:
            state.status = "waiting_for_preparer"
        return state, summarize(state)
    return state, _show(state, question, search, question_search)


def undo(state: State, search=None, *, question_search=None) -> tuple[State, dict]:
    """Take back the last manual answer and ask it again.

    Facts from uploaded documents and tax questions are left alone. Flags and missing_docs are rebuilt by
    review(). RuntimeError once submitted or decided (the preparer must return it); ValueError if nothing to undo.
    """
    if state.status != "in_progress":
        raise RuntimeError(f"Interview is {state.status}; answers can no longer be changed.")

    index = None
    for i in reversed(range(len(state.history))):
        if "answer_text" in state.history[i]:
            index = i
            break
    if index is None:
        raise ValueError("There is no answer to undo.")

    entry = state.history.pop(index)
    question = _question_by_id(state, entry["question_id"])
    for name in entry["facts"]:
        state.facts[name] = None
    if question["fact"] in state.unsure:
        state.unsure.remove(question["fact"])

    state.flags.clear()
    state.missing_docs.clear()
    review(state)

    while state.asked and state.asked[-1] != question["id"]:
        state.asked.pop()
    return state, _show(state, question, search, question_search)
