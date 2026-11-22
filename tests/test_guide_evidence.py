"""Precomputed guide passages (src/workflow/guide_evidence.py).

A question's retrieval query comes from the playbook, never from the client, so its passages
are the same for every interview: they are retrieved once at build time and served from a file.
"""
from src.app.interviews import InterviewService
from src.workflow import guide_evidence
from src.workflow.engine import load_scenario

SCENARIO = "ca_w2_side_gig"


def fake_search(query, *, question=None):
    return {"matches": [{"citation": f"IRS Pub 17 (2025), p. {len(query)}", "text": "passage for " + query,
                         "chunk_id": "c1", "score": 0.9}]}


def failing_search(query, *, question=None):
    raise AssertionError("live retrieval must not run when the passage is precomputed")


def test_build_covers_every_question_and_keeps_only_what_the_guide_shows():
    data = guide_evidence.build([SCENARIO], search=fake_search, meta={"corpus_version": "test"})
    _, questions = load_scenario(SCENARIO)
    assert len(data["entries"]) == len(questions)
    assert data["meta"]["corpus_version"] == "test"
    entry = data["entries"][guide_evidence.fingerprint(questions[0])]
    assert entry["matches"] == [{"citation": f"IRS Pub 17 (2025), p. {len(questions[0]['retrieval_query'])}",
                                 "text": "passage for " + questions[0]["retrieval_query"]}]


def test_lookup_misses_when_the_question_wording_changed():
    _, questions = load_scenario(SCENARIO)
    q = questions[0]
    evidence = guide_evidence.GuideEvidence(guide_evidence.build([SCENARIO], search=fake_search))
    assert evidence.lookup(q) is not None
    assert evidence.lookup({**q, "retrieval_query": q["retrieval_query"] + " changed"}) is None


def test_build_records_failures_without_stopping():
    def flaky(query, *, question=None):
        raise RuntimeError("Ollama is off")
    data = guide_evidence.build([SCENARIO], search=flaky)
    assert data["entries"] == {} and data["failures"]


def test_build_does_not_freeze_unranked_fallback_results():
    def unranked(query, *, question=None):
        return {"matches": [{"citation": "x", "text": "y"}], "rerank_fallback": True}
    data = guide_evidence.build([SCENARIO], search=unranked)
    assert data["entries"] == {} and data["failures"][0]["error"] == "rerank fallback"


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "evidence.json"
    guide_evidence.save(guide_evidence.build([SCENARIO], search=fake_search), path)
    loaded = guide_evidence.load(path)
    assert loaded is not None and len(loaded) > 0
    assert guide_evidence.load(tmp_path / "missing.json") is None


def test_unreadable_or_old_files_are_ignored(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    old = tmp_path / "old.json"
    old.write_text('{"version": 0, "entries": {}}', encoding="utf-8")
    assert guide_evidence.load(bad) is None and guide_evidence.load(old) is None


def test_service_serves_precomputed_passages_without_live_retrieval():
    evidence = guide_evidence.GuideEvidence(guide_evidence.build([SCENARIO], search=fake_search))
    svc = InterviewService(search=failing_search, evidence=evidence)
    out = svc.start(SCENARIO)["output"]
    assert out["retrieval_error"] is None
    assert out["citations"][0]["text"].startswith("passage for ")


def test_service_falls_back_to_live_retrieval_for_unknown_questions():
    calls = []

    def live(query, *, question=None):
        calls.append(query)
        return {"matches": []}

    svc = InterviewService(search=live, evidence=guide_evidence.GuideEvidence({"meta": {}, "entries": {}}))
    svc.start(SCENARIO)
    assert calls
