"""Public-site protections: sign-in (src/app/auth.py), the model-call cap (src/app/limits.py),
data retention, and the access rules api.py applies when TAX_AGENT_MODE=public."""
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
import pytest

from src.app import auth, limits
from src.app.api import app
from src.app.interviews import InterviewService
from src.app.limits import DailyBudget
from src.config import AppSettings, app_settings
from src.llm.client import LLMUnavailable
from src.llm.normalize import make_normalizer

SECRET = "s" * 40
PASSWORD = "correct horse battery"
FAST_HASH = auth.hash_password(PASSWORD, iterations=1_000)


def settings(**overrides) -> AppSettings:
    """Public-mode settings with one preparer (alice) and one invite code (demo-1)."""
    base = app_settings()
    values = dict(llm_enabled=False, retrieval_enabled=False, interview_db_path=base.interview_db_path,
                  interview_database_url=None, guide_evidence=base.guide_evidence,
                  public=True, preparers={"alice": FAST_HASH}, invite_codes=("demo-1",), session_secret=SECRET)
    return AppSettings(**(values | overrides))


def test_password_hash_round_trip():
    assert auth.verify_password(PASSWORD, FAST_HASH)
    assert not auth.verify_password("wrong", FAST_HASH)
    assert not auth.verify_password(PASSWORD, "garbage")


def test_session_cookie_is_signed_and_expires():
    """A cookie stops working when it expires, the secret changes, or its payload is edited."""
    s = settings()
    token = auth.sign(s, auth.Session("preparer", "alice"), now=1000)
    assert auth.session_from(s, token, now=1001) == auth.Session("preparer", "alice")
    assert auth.session_from(s, token, now=1000 + auth.SESSION_SECONDS + 1) is None
    assert auth.session_from(settings(session_secret="x" * 40), token, now=1001) is None
    body, _, mac = token.partition(".")
    assert auth.session_from(s, body[:-2] + "AA." + mac, now=1001) is None


def test_removing_an_invite_code_or_account_signs_people_out():
    s = settings()
    client = auth.sign(s, auth.check_invite(s, "demo-1"))
    preparer = auth.sign(s, auth.Session("preparer", "alice"))
    assert auth.session_from(s, client) is not None
    assert auth.session_from(settings(invite_codes=("demo-2",)), client) is None
    assert auth.session_from(settings(preparers={"bob": FAST_HASH}), preparer) is None


def test_public_mode_refuses_unsafe_settings():
    """Public mode needs a long secret and an account; local mode needs neither."""
    auth.check_public_settings(settings())
    auth.check_public_settings(settings(public=False, preparers={}, session_secret=""))
    with pytest.raises(RuntimeError, match="SESSION_SECRET"):
        auth.check_public_settings(settings(session_secret="short"))
    with pytest.raises(RuntimeError, match="PREPARERS"):
        auth.check_public_settings(settings(preparers={}))


def test_login_attempts_block_after_repeated_failures():
    now = [0.0]
    a = auth.LoginAttempts(max_failures=2, window=60, clock=lambda: now[0])
    a.fail("ip")
    a.fail("ip")
    assert a.blocked("ip") and not a.blocked("other")
    now[0] += 61
    assert not a.blocked("ip")


def test_budget_caps_each_visitor_and_the_whole_site_and_resets_daily():
    """Visitor a stops at its own cap of 2; c is refused by the site cap of 3; the next day resets."""
    now = [0.0]
    b = DailyBudget(per_visitor=2, per_day=3, clock=lambda: now[0])
    assert [b.take("a"), b.take("a"), b.take("a")] == [True, True, False]
    assert [b.take("b"), b.take("c")] == [True, False]
    now[0] += 24 * 60 * 60
    assert b.take("c")


def test_metered_chat_raises_llm_unavailable_past_the_cap():
    """Past the cap the wrapper raises LLMUnavailable and never calls the provider."""
    calls = []
    chat = DailyBudget(per_visitor=1, per_day=10).metered(lambda *a: calls.append(a) or {"ok": 1})
    token = limits.current_visitor.set("v1")
    try:
        assert chat("sys", "user") == {"ok": 1}
        with pytest.raises(LLMUnavailable):
            chat("sys", "user")
    finally:
        limits.current_visitor.reset(token)
    assert len(calls) == 1


def test_spent_budget_falls_back_to_rules_instead_of_breaking_the_interview():
    """With no budget left, free text is refused but exact answers still move the interview on."""
    spent = DailyBudget(per_visitor=0, per_day=0)
    svc = InterviewService(search=lambda q: {"matches": []},
                           normalizer=make_normalizer(chat=spent.metered(lambda s, u: {"value": "single"})))
    iid = svc.start("ca_w2_side_gig")["state"]["interview_id"]
    out = svc.answer(iid, "she is not married")["output"]
    assert out["type"] == "error" and "enter the value directly" in out["message"]
    assert svc.answer(iid, "single")["output"]["type"] == "question"


def test_retention_deletes_only_stale_interviews():
    """Only the interview last updated 40 days ago is removed at 30 days; 0 days keeps everything."""
    svc = InterviewService(search=lambda q: {"matches": []})
    old = svc.start("ca_w2_side_gig")["state"]["interview_id"]
    new = svc.start("ca_w2_side_gig")["state"]["interview_id"]
    stale = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat(timespec="microseconds")
    with svc._store._connection() as conn:
        conn.execute("UPDATE guided_interviews SET updated_at = ? WHERE interview_id = ?", (stale, old))
    assert svc.purge_older_than(0) == 0
    assert svc.purge_older_than(30) == 1
    assert svc._store.find(old) is None and svc._store.find(new) is not None


@pytest.fixture
def site(monkeypatch):
    """The app in public mode, with a fake chat agent, no retrieval or model,
    and a fresh budget of 2 model calls per visitor."""
    monkeypatch.setattr("src.app.api.app_settings", lambda: settings())
    monkeypatch.setattr(limits, "_budget", DailyBudget(per_visitor=2, per_day=100))
    agent = MagicMock()
    agent.invoke.side_effect = lambda payload, **kw: {"messages": [*payload["messages"], AIMessage(content="ok")]}
    with patch("src.app.api.create_tax_agent", return_value=agent), \
         patch.object(InterviewService, "from_env", classmethod(lambda cls: cls(search=lambda q: {"matches": []}))):
        with TestClient(app) as client:
            yield client


def test_signed_out_visitors_are_sent_to_sign_in(site):
    r = site.get("/interview", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login?next=/interview"
    assert site.post("/api/interviews", json={"scenario": "ca_w2_side_gig"}).status_code == 401
    assert site.get("/api/preparer/interviews").status_code == 401
    assert site.get("/login").status_code == 200
    assert site.get("/healthz").json()["mode"] == "public"


def test_invite_code_opens_client_pages_but_not_the_preparer_queue(site):
    assert site.post("/api/login", json={"invite_code": "wrong"}).status_code == 401
    r = site.post("/api/login", json={"invite_code": "demo-1"})
    assert r.status_code == 200 and r.json()["role"] == "client"
    assert site.get("/interview").status_code == 200
    assert site.post("/api/interviews", json={"scenario": "ca_w2_side_gig"}).status_code == 200
    assert site.get("/api/preparer/interviews").status_code == 403
    assert site.get("/preparer", follow_redirects=False).status_code == 303


def test_preparer_decision_records_who_decided(site):
    site.post("/api/login", json={"invite_code": "demo-1"})
    iid = site.post("/api/interviews", json={"scenario": "ca_w2_side_gig"}).json()["state"]["interview_id"]
    for answer in ("single", "no", "no", "no", "no", "no", "no", "no"):
        out = site.post(f"/api/interviews/{iid}/answer", json={"answer": answer}).json()["output"]
        if out["type"] == "summary":
            break
    assert site.post(f"/api/interviews/{iid}/submit").status_code == 200

    assert site.post("/api/login", json={"username": "alice", "password": "nope"}).status_code == 401
    assert site.post("/api/login", json={"username": "alice", "password": PASSWORD}).json()["role"] == "preparer"
    assert site.post(f"/api/preparer/interviews/{iid}/decision", json={"approve": True}).status_code == 200
    service = app.state.interviews
    assert service.preparer_get(iid)["state"]["status"] == "approved"
    owner = service._store.find(iid)["owner"]
    assert service._get(iid, owner).state.history[-1] == {"decision": "approved", "note": "", "by": "alice"}


def test_repeated_failed_sign_ins_are_blocked(site):
    for _ in range(10):
        site.post("/api/login", json={"invite_code": "guess"})
    assert site.post("/api/login", json={"invite_code": "demo-1"}).status_code == 429


def test_chat_stops_at_the_visitor_budget(site):
    site.post("/api/login", json={"invite_code": "demo-1"})
    codes = [site.post("/api/chat", json={"message": "hi"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_logout_ends_the_session(site):
    site.post("/api/login", json={"invite_code": "demo-1"})
    assert site.get("/api/session").json()["role"] == "client"
    site.post("/api/logout")
    assert site.get("/api/session").json()["role"] is None


def test_local_mode_needs_no_sign_in(monkeypatch):
    monkeypatch.setattr("src.app.api.app_settings", lambda: settings(public=False))
    with patch("src.app.api.create_tax_agent", return_value=MagicMock()), \
         patch.object(InterviewService, "from_env", classmethod(lambda cls: cls(search=lambda q: {"matches": []}))):
        with TestClient(app) as client:
            assert client.get("/interview").status_code == 200
            assert client.get("/api/preparer/interviews").status_code == 200
