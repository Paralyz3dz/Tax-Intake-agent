"""
api.py: FastAPI app.

    /                 free-form tax chat (LangChain agent, src/agent.py)
    /interview        guided client interview page
    /api/chat         chat endpoint
    /api/playbooks, /api/interviews/...   guided interview endpoints (src/app/interviews.py)
    /preparer, /api/preparer/...         preparer review of every interview (web page and MCP)
    /form-8843, /api/interviews/{id}/forms/8843/...   Form 8843 draft from an interview (src/forms/f8843.py)
    /login, /api/login, /api/logout, /api/session   sign-in when TAX_AGENT_MODE=public (src/app/auth.py)
    /healthz          liveness and which optional parts are on
"""
import asyncio
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import logging
import re
from secrets import token_urlsafe
from pathlib import Path
from threading import Lock
from typing import Annotated
from urllib.parse import quote
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from src.agent import create_tax_agent
from src.app import auth
from src.app.documents import MAX_FILE_BYTES, MAX_FILES
from src.app.interviews import (DocumentExtractionUnavailable, InvalidInterviewAnswer, InterviewClosed,
                                InterviewNotFound, InterviewService, StagedDocumentUnavailable,
                                list_playbooks)
from src.app.interview_pdf import render_interview_pdf, render_review_pdf
from src.app.limits import current_visitor, get_budget
from src.forms import f8843
from src.llm.client import LLMUnavailable, set_call_guard
from src.config import app_settings
from src.workflow.engine import INTAKE


logger = logging.getLogger(__name__)

INTERVIEW_COOKIE = "guided_interview_browser"


def interview_owner(request: Request, response: Response | None = None) -> str:
    token = request.cookies.get(INTERVIEW_COOKIE, "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
        if response is None:
            return ""
        token = token_urlsafe(32)
        response.set_cookie(INTERVIEW_COOKIE, token, max_age=365 * 24 * 60 * 60,
                            httponly=True, samesite="lax", secure=request.url.scheme == "https")
    return sha256(token.encode()).hexdigest()


@dataclass
class Session:
    """One chat conversation: its message history and a lock so one session handles one request at a time."""
    messages: list = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create the chat agent and the interview service at startup.

    If the chat agent cannot be created (e.g. no OPENAI_API_KEY), only /api/chat is
    disabled (503); the guided interview still works. A public site with missing sign-in
    settings refuses to start (auth.check_public_settings). The daily budget is installed as
    the model call guard for as long as the app runs.
    """
    app.state.settings = app_settings()
    auth.check_public_settings(app.state.settings)
    app.state.login_attempts = auth.LoginAttempts()
    set_call_guard(get_budget().charge)
    try:
        app.state.agent = create_tax_agent()
    except Exception as exc:
        logger.warning("Chat agent unavailable (%s); /api/chat will return 503", exc)
        app.state.agent = None
    app.state.interviews = InterviewService.from_env()
    app.state.sessions = {}
    app.state.sessions_lock = Lock()
    retention = asyncio.create_task(_retention_loop(app.state.interviews, app.state.settings.retention_days))
    yield
    set_call_guard(None)
    retention.cancel()
    with suppress(asyncio.CancelledError):
        await retention


async def _retention_loop(service: InterviewService, days: int, every: float = 6 * 60 * 60):
    """Delete old interviews at startup and every few hours (TAX_AGENT_RETENTION_DAYS; 0 = never)."""
    if days <= 0:
        return
    while True:
        try:
            await run_in_threadpool(service.purge_older_than, days)
        except Exception:
            logger.exception("Retention cleanup failed")
        await asyncio.sleep(every)


app = FastAPI(title="Tax Agent Chat", lifespan=lifespan)

STATIC = Path(__file__).parent / "static"
OPEN_PATHS = {"/login", "/api/login", "/api/logout", "/api/session", "/healthz", "/workspace.css",
              "/interview/inter-variable.woff2", "/interview/instrument-serif.ttf"}


def _needs_preparer(path: str) -> bool:
    return path == "/preparer" or path.startswith("/api/preparer/")


@app.middleware("http")
async def access_control(request: Request, call_next):
    """Public mode: every page and API needs a session; /preparer needs a preparer session.

    OPEN_PATHS (sign-in, health check, fonts, stylesheet) are reachable without signing in.

    Pages redirect to /login; API calls get 401 (not signed in) or 403 (not a preparer).
    The session is kept on request.state.session (None in local mode). Every request is also
    tagged with its visitor, whom model calls are counted against (src/app/limits.py).
    """
    settings = request.app.state.settings
    path = request.url.path
    request.state.session = None
    if settings.public and path not in OPEN_PATHS:
        session = auth.session_from(settings, request.cookies.get(auth.SESSION_COOKIE))
        if session is None or (_needs_preparer(path) and not session.is_preparer):
            if path.startswith("/api/"):
                if session is None:
                    return JSONResponse({"detail": "Sign in to continue."}, status_code=401)
                return JSONResponse({"detail": "Preparer sign-in required."}, status_code=403)
            return RedirectResponse("/login?next=" + quote(path), status_code=303)
        request.state.session = session
    token = current_visitor.set(_visitor(request))
    try:
        return await call_next(request)
    finally:
        current_visitor.reset(token)


def _visitor(request: Request) -> str:
    """This browser (its interview cookie), else its address."""
    return interview_owner(request) or "ip:" + (request.client.host if request.client else "unknown")


class LoginRequest(BaseModel):
    """Body of POST /api/login: an invite code, or a preparer's username and password."""
    invite_code: str = Field(default="", max_length=200)
    username: str = Field(default="", max_length=100)
    password: str = Field(default="", max_length=200)


LOCAL_SESSION = {"mode": "local", "role": "preparer", "name": "local"}


@app.get("/login", response_class=FileResponse, include_in_schema=False)
def login_page():
    return FileResponse(STATIC / "login.html", headers={"Cache-Control": "no-store"})


@app.post("/api/login")
def login(payload: LoginRequest, request: Request, response: Response):
    """Sign in (public mode); sets the session cookie. 401 wrong code or password, 429 too many failures."""
    settings = request.app.state.settings
    if not settings.public:
        return LOCAL_SESSION
    ip = request.client.host if request.client else "unknown"
    attempts = request.app.state.login_attempts
    if attempts.blocked(ip):
        raise HTTPException(429, "Too many failed sign-ins. Try again in 15 minutes.")
    if payload.username:
        session = auth.check_password(settings, payload.username, payload.password)
    else:
        session = auth.check_invite(settings, payload.invite_code)
    if session is None:
        attempts.fail(ip)
        raise HTTPException(401, "That sign-in did not work.")
    response.set_cookie(auth.SESSION_COOKIE, auth.sign(settings, session), max_age=auth.SESSION_SECONDS,
                        httponly=True, samesite="lax", secure=request.url.scheme == "https")
    return {"mode": "public", "role": session.role, "name": session.name}


@app.post("/api/logout", status_code=204)
def logout():
    response = Response(status_code=204)
    response.delete_cookie(auth.SESSION_COOKIE)
    return response


@app.get("/api/session")
def current_session(request: Request):
    """Who is signed in: {"mode", "role", "name"} (role None if nobody)."""
    settings = request.app.state.settings
    if not settings.public:
        return LOCAL_SESSION
    session = auth.session_from(settings, request.cookies.get(auth.SESSION_COOKIE))
    return {"mode": "public", "role": session.role if session else None, "name": session.name if session else None}


@app.get("/healthz", include_in_schema=False)
def healthz(request: Request):
    service = request.app.state.interviews
    return {"status": "ok", "mode": "public" if request.app.state.settings.public else "local",
            "llm": service.normalizer is not None, "chat": request.app.state.agent is not None,
            "guide_passages": len(service.evidence) if service.evidence is not None else 0}


class ChatRequest(BaseModel):
    """Body of POST /api/chat: {"message": text, "session_id": optional UUID of an existing chat}."""
    message: str = Field(min_length=1, max_length=10000)
    session_id: UUID | None = None

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        """Trim spaces; reject a message that is only whitespace (422)."""
        value = value.strip()
        if not value:
            raise ValueError("Message must not be blank.")
        return value


class ChatResponse(BaseModel):
    """Reply of POST /api/chat: the session id to send next time and the agent's answer."""
    session_id: UUID
    answer: str


@app.get("/", include_in_schema=False)
def index():
    """Home: the interview start page. Tax questions are answered inside the interview (no separate chat page)."""
    return RedirectResponse("/interview", status_code=307)


DAILY_LIMIT = "The daily limit for questions has been reached. Please try again tomorrow."


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request):
    """Free-form tax Q&A: run the LangChain agent (retrieval tool + OpenAI) on the session's history.

    503 if the agent is not configured, 404 for an unknown session, 502 if the agent fails,
    429 if the daily model-call budget is spent (before or during the answer; each model call
    the agent makes is counted by the call guard). A failed turn does not change the saved history."""
    state = request.app.state
    if state.agent is None:
        raise HTTPException(503, "Chat is not configured. Set LLM_API_KEY and restart.")
    if not get_budget().allows(current_visitor.get()):
        raise HTTPException(429, DAILY_LIMIT)
    session_id = payload.session_id or uuid4()
    with state.sessions_lock:
        if payload.session_id is not None:
            session = state.sessions.get(session_id)
            if session is None:
                raise HTTPException(404, "Session expired. Start a new conversation.")
        else:
            session = Session()

    with session.lock:
        messages = deepcopy(session.messages)
        messages.append({"role": "user", "content": payload.message})
        try:
            result = state.agent.invoke(
                {"messages": messages}, config={"recursion_limit": 20}
            )
            final = result["messages"][-1]
            if final.type != "ai" or getattr(final, "tool_calls", []):
                raise ValueError("Agent did not produce a final answer.")
            content = final.content
            answer = content if isinstance(content, str) else "\n".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
            if not answer.strip():
                raise ValueError("Agent returned an empty answer.")
        except LLMUnavailable:
            raise HTTPException(429, DAILY_LIMIT) from None
        except Exception:
            logger.exception("Chat agent invocation failed")
            raise HTTPException(502, "Unable to get an answer. Please try again.") from None

        session.messages = result["messages"]
        with state.sessions_lock:
            state.sessions[session_id] = session
    return ChatResponse(session_id=session_id, answer=answer)


class StartRequest(BaseModel):
    """Body of POST /api/interviews: {"scenario", "priority": [section ids asked first], "background": client note}."""
    scenario: str = Field(min_length=1, max_length=100)
    priority: list[str] = Field(default_factory=list, max_length=20)
    background: str = Field(default="", max_length=2000)
    document_ids: list[UUID] = Field(default_factory=list, max_length=MAX_FILES)


class AnswerRequest(BaseModel):
    answer: str | None = Field(default=None, min_length=1, max_length=2000)
    question_id: str | None = Field(default=None, max_length=100)
    document_ids: list[UUID] = Field(default_factory=list, max_length=MAX_FILES)

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str | None) -> str | None:
        """Trim spaces; reject an answer that is only whitespace (422)."""
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("Answer must not be blank.")
        return value

    @model_validator(mode="after")
    def require_content(self):
        if self.answer is None and not self.document_ids:
            raise ValueError("Send an answer or attach a file.")
        return self


class RenameInterviewRequest(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100,
                                           pattern=r"^[^\x00-\x1f\x7f]+$")]


class SubmitInterviewRequest(BaseModel):
    note: str = Field(default="", max_length=2000)


class DecisionRequest(SubmitInterviewRequest):
    approve: bool


class SyncDocumentsRequest(BaseModel):
    document_ids: list[UUID] = Field(max_length=MAX_FILES)


@app.get("/interview", response_class=FileResponse)
def interview_page(request: Request):
    """Serve the assistant's guided-interview page."""
    response = FileResponse(Path(__file__).parent / "static" / "interview.html",
                            headers={"Cache-Control": "no-store"})
    interview_owner(request, response)
    return response


@app.get("/interview/inter-variable.woff2", response_class=FileResponse, include_in_schema=False)
def interview_font():
    font = Path(__file__).parent / "static" / "fonts" / "InterVariable.woff2"
    return FileResponse(font, media_type="font/woff2")


@app.get("/workspace.css", response_class=FileResponse, include_in_schema=False)
def workspace_stylesheet():
    return FileResponse(Path(__file__).parent / "static" / "workspace.css", media_type="text/css")


@app.get("/interview/instrument-serif.ttf", response_class=FileResponse, include_in_schema=False)
def interview_title_font():
    font = Path(__file__).parent / "static" / "fonts" / "InstrumentSerif-Regular.ttf"
    return FileResponse(font, media_type="font/ttf")


@app.get("/api/playbooks")
def playbooks():
    """List available scenarios for the page's dropdown."""
    return {"playbooks": list_playbooks()}


@app.get("/api/interviews")
def interview_list(request: Request):
    owner = interview_owner(request)
    interviews = request.app.state.interviews.list(owner) if owner else []
    return JSONResponse({"interviews": interviews}, headers={"Cache-Control": "no-store"})


@app.post("/api/interviews")
def start(payload: StartRequest, request: Request, response: Response):
    """Start an interview; returns the first question and a sidebar snapshot (404 unknown scenario, 422 bad priority)."""
    try:
        owner = interview_owner(request, response)
        result = request.app.state.interviews.start(
            payload.scenario, payload.priority, payload.background,
            [str(document_id) for document_id in payload.document_ids], owner=owner,
        )
        response.headers["Cache-Control"] = "no-store"
        return result
    except StagedDocumentUnavailable as exc:
        raise HTTPException(409, str(exc)) from None
    except ValueError as exc:
        raise HTTPException(404 if "scenario" in str(exc) else 422, str(exc)) from None


@app.post("/api/interviews/{interview_id}/answer")
def answer(interview_id: str, payload: AnswerRequest, request: Request):
    """Submit an answer; returns the next question, an error, or the summary (404 / 409)."""
    try:
        result = request.app.state.interviews.answer(
            interview_id, payload.answer, payload.question_id, owner=interview_owner(request),
            document_ids=[str(document_id) for document_id in payload.document_ids],
            tax_questions=True,
        )
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found. Start a new one.") from None
    except (InterviewClosed, StagedDocumentUnavailable) as exc:
        raise HTTPException(409, str(exc)) from None
    except InvalidInterviewAnswer as exc:
        raise HTTPException(422, str(exc)) from None


@app.post("/api/interviews/{interview_id}/undo")
def undo_answer(interview_id: str, request: Request):
    """Take back the last answer; returns that question again (404 unknown, 409 nothing to undo / decided)."""
    try:
        result = request.app.state.interviews.undo(interview_id, owner=interview_owner(request))
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found. Start a new one.") from None
    except InterviewClosed as exc:
        raise HTTPException(409, str(exc)) from None


@app.post("/api/interview-documents")
async def stage_documents(request: Request, response: Response, scenario: str = INTAKE):
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "multipart/form-data":
        raise HTTPException(415, "Send the files as multipart form data.")

    async def limited_stream():
        remaining = MAX_FILES * MAX_FILE_BYTES + 64 * 1024
        async for chunk in request.stream():
            remaining -= len(chunk)
            if remaining < 0:
                raise HTTPException(413, "This upload is too large. Choose smaller files.")
            yield chunk

    try:
        form = await MultiPartParser(request.headers, limited_stream(), max_files=MAX_FILES,
                                     max_fields=0).parse()
    except MultiPartException:
        raise HTTPException(400, f"Attach up to {MAX_FILES} files in the files field.") from None
    try:
        files = form.getlist("files")
        if not files or len(files) != len(form.multi_items()) or not all(
            isinstance(file, UploadFile) and file.filename for file in files
        ):
            raise HTTPException(422, "Attach one or more files in the files field.")
        uploads = []
        for file in files:
            filename = Path(file.filename.replace("\\", "/")).name
            uploads.append((filename, await file.read(MAX_FILE_BYTES + 1)))
    finally:
        await form.close()

    try:
        owner = interview_owner(request, response)
        result = await run_in_threadpool(request.app.state.interviews.stage_documents, scenario, uploads, owner=owner)
        response.headers["Cache-Control"] = "no-store"
        return result
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from None
    except DocumentExtractionUnavailable as exc:
        raise HTTPException(503, str(exc)) from None


@app.delete("/api/interview-documents/{document_id}", status_code=204)
def remove_document(document_id: UUID, request: Request):
    request.app.state.interviews.remove_document(str(document_id), owner=interview_owner(request))
    return Response(status_code=204)


@app.put("/api/interviews/{interview_id}/documents")
def sync_documents(interview_id: str, payload: SyncDocumentsRequest, request: Request):
    try:
        result = request.app.state.interviews.sync_documents(
            interview_id, [str(document_id) for document_id in payload.document_ids], owner=interview_owner(request),
        )
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    except (InterviewClosed, StagedDocumentUnavailable) as exc:
        raise HTTPException(409, str(exc)) from None


@app.get("/api/interviews/{interview_id}")
def get_interview(interview_id: str, request: Request):
    """Return the interview state, history, and current question or summary."""
    try:
        result = request.app.state.interviews.get(interview_id, owner=interview_owner(request))
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None


@app.patch("/api/interviews/{interview_id}")
def rename_interview(interview_id: str, payload: RenameInterviewRequest, request: Request):
    try:
        result = request.app.state.interviews.rename(interview_id, payload.title, owner=interview_owner(request))
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None


@app.delete("/api/interviews/{interview_id}", status_code=204)
def delete_interview(interview_id: str, request: Request):
    try:
        request.app.state.interviews.delete(interview_id, owner=interview_owner(request))
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    return Response(status_code=204)


@app.post("/api/interviews/{interview_id}/submit")
def submit_interview(interview_id: str, request: Request, payload: SubmitInterviewRequest | None = None):
    try:
        result = request.app.state.interviews.submit_for_review(
            interview_id, payload.note.strip() if payload else "", owner=interview_owner(request),
        )
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    except InterviewClosed as exc:
        raise HTTPException(409, str(exc)) from None


PREPARER_STATUSES = {"in_progress", "waiting_for_preparer", "approved", "returned"}


@app.get("/preparer", response_class=FileResponse)
def preparer_page():
    return FileResponse(Path(__file__).parent / "static" / "preparer.html", headers={"Cache-Control": "no-store"})


@app.get("/api/preparer/interviews")
def preparer_list(request: Request, status: str | None = None, limit: int = 100):
    """Every interview (web page and MCP), newest first; status filters by state."""
    if status is not None and status not in PREPARER_STATUSES:
        raise HTTPException(422, f"status must be one of {sorted(PREPARER_STATUSES)}")
    items = request.app.state.interviews.preparer_list(status, max(1, min(limit, 500)))
    return JSONResponse({"interviews": items}, headers={"Cache-Control": "no-store"})


@app.get("/api/preparer/interviews/{interview_id}")
def preparer_get(interview_id: str, request: Request):
    try:
        result = request.app.state.interviews.preparer_get(interview_id)
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@app.post("/api/preparer/interviews/{interview_id}/decision")
def preparer_decide(interview_id: str, payload: DecisionRequest, request: Request):
    """Approve or return any interview waiting for the preparer (409 otherwise)."""
    try:
        session = request.state.session
        result = request.app.state.interviews.preparer_decide(interview_id, payload.approve, payload.note,
                                                              reviewer=session.name if session else "")
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    except InterviewClosed as exc:
        raise HTTPException(409, str(exc)) from None
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@app.get("/api/interviews/{interview_id}/summary.pdf")
def interview_summary_pdf(interview_id: str, request: Request):
    try:
        interview = request.app.state.interviews.summary_export(interview_id, owner=interview_owner(request))
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    except InterviewClosed as exc:
        raise HTTPException(409, str(exc)) from None
    state = interview["state"]
    interview_id = state["interview_id"]
    pdf = render_interview_pdf(
        state["facts"], interview_id=interview_id, title=interview["title"], created_at=interview["created_at"],
    )
    return pdf_download_response(pdf, "interview-summary", interview_id)


@app.get("/api/preparer/interviews/{interview_id}/summary.pdf")
def preparer_summary_pdf(interview_id: str, request: Request):
    try:
        review = request.app.state.interviews.preparer_get(interview_id)
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    return pdf_download_response(render_review_pdf(review), "interview-review", review["state"]["interview_id"])


def pdf_download_response(pdf: bytes, name: str, interview_id: str) -> Response:
    short_id = re.sub(r"[^a-zA-Z0-9]", "", interview_id)[:8]
    return Response(pdf, media_type="application/pdf", headers={
        "Cache-Control": "no-store",
        "Content-Disposition": f'attachment; filename="{name}-{short_id}.pdf"',
    })


class Form8843Request(BaseModel):
    """Body of the Form 8843 endpoints: what the student typed on the form page, and the items they ticked.

    details is checked by f8843.clean_details (422 on a bad value). It is used for this one request only:
    never stored, logged or sent to a model.
    """
    details: dict = Field(default_factory=dict)
    confirmed: list[Annotated[str, StringConstraints(max_length=40)]] = Field(default_factory=list, max_length=50)

    @field_validator("details")
    @classmethod
    def limit_details(cls, value: dict) -> dict:
        if len(value) > 30:
            raise ValueError("Too many detail fields.")
        return value


@app.get("/form-8843", response_class=FileResponse)
def form_8843_page(request: Request):
    """Serve the Form 8843 draft page (opened from a finished interview: /form-8843?interview=<id>)."""
    response = FileResponse(STATIC / "form8843.html", headers={"Cache-Control": "no-store"})
    interview_owner(request, response)
    return response


def _form_8843_plan(interview_id: str, payload: Form8843Request, request: Request) -> dict:
    try:
        interview = request.app.state.interviews.get(interview_id, owner=interview_owner(request))
    except InterviewNotFound:
        raise HTTPException(404, "Interview not found.") from None
    try:
        details = f8843.clean_details(payload.details)
    except f8843.FormDetailsError as exc:
        raise HTTPException(422, str(exc)) from None
    state = interview["state"]
    return f8843.build_plan(state["facts"], details, state["unsure"])


@app.post("/api/interviews/{interview_id}/forms/8843/plan")
def form_8843_plan(interview_id: str, payload: Form8843Request, request: Request):
    """Every Form 8843 field with its value and status (filled / confirm / user_only / missing)."""
    return JSONResponse(_form_8843_plan(interview_id, payload, request), headers={"Cache-Control": "no-store"})


@app.post("/api/interviews/{interview_id}/forms/8843/review.pdf")
def form_8843_review_pdf(interview_id: str, payload: Form8843Request, request: Request):
    """The marked review copy with its checklist page (stamped "do not mail")."""
    plan = _form_8843_plan(interview_id, payload, request)
    return pdf_download_response(f8843.render_pdf(plan, "review"), "form-8843-review", interview_id)


@app.post("/api/interviews/{interview_id}/forms/8843/final.pdf")
def form_8843_final_pdf(interview_id: str, payload: Form8843Request, request: Request):
    """Pages 1-2 ready to print, sign and mail; 409 with the open items until every check is done."""
    plan = _form_8843_plan(interview_id, payload, request)
    try:
        pdf = f8843.render_pdf(plan, "final", confirmed=payload.confirmed)
    except f8843.FormNotReady as exc:
        return JSONResponse({"detail": "The final form is not ready yet.", "reasons": exc.reasons},
                            status_code=409, headers={"Cache-Control": "no-store"})
    return pdf_download_response(pdf, "form-8843", interview_id)
