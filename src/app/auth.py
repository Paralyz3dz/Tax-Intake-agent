"""
auth.py: who may use the site when TAX_AGENT_MODE=public. Local mode needs no sign-in.

Two roles, one signed session cookie:
    client     entered a valid invite code (TAX_AGENT_INVITE_CODES); may use the chat and
               interview pages. Removing a code from the list signs its visitors out.
    preparer   signed in with a TAX_AGENT_PREPARERS account; may also use /preparer.

Swapping this for real accounts (a users table, OAuth) later means replacing check_invite(),
check_password() and the session payload here; api.py only calls session_from() and friends.

    python -m src.app.auth hash        # make a password hash for TAX_AGENT_PREPARERS
    python -m src.app.auth secret      # make a TAX_AGENT_SESSION_SECRET
"""
from __future__ import annotations

from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections import deque
from dataclasses import dataclass
import hashlib
import hmac
import json
import secrets
import sys
from threading import Lock
import time

from src.config import AppSettings

SESSION_COOKIE = "tax_agent_session"
SESSION_SECONDS = 12 * 60 * 60
ITERATIONS = 310_000


def _b64(data: bytes) -> str:
    return urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str, *, salt: bytes | None = None, iterations: int = ITERATIONS) -> str:
    """"pbkdf2_sha256$<iterations>$<salt>$<digest>": what goes after "name:" in TAX_AGENT_PREPARERS.

    PBKDF2-SHA256 with ITERATIONS rounds (OWASP guidance) and a random 16-byte salt.
    """
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt, digest = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), _unb64(salt), int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, _unb64(digest))


def _code_id(code: str) -> str:
    """Invite codes are not put in cookies; a short hash records which one was used."""
    return hashlib.sha256(code.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Session:
    """Who is signed in.

    role   "client" or "preparer"
    name   the preparer's account name, or "invite" for clients
    code   _code_id of the invite code used (clients only)
    """
    role: str
    name: str
    code: str = ""

    @property
    def is_preparer(self) -> bool:
        return self.role == "preparer"


def check_invite(settings: AppSettings, code: str) -> Session | None:
    code = code.strip()
    if code and any(hmac.compare_digest(code, c) for c in settings.invite_codes):
        return Session("client", "invite", _code_id(code))
    return None


_DUMMY_HASH = hash_password("unused", salt=b"0" * 16)


def check_password(settings: AppSettings, name: str, password: str) -> Session | None:
    """A preparer session if the account exists and the password matches, else None.

    An unknown name is checked against a dummy hash, so the response time does not reveal
    which account names exist.
    """
    stored = settings.preparers.get(name.strip())
    ok = verify_password(password, stored or _DUMMY_HASH)
    return Session("preparer", name.strip()) if stored and ok else None


def sign(settings: AppSettings, session: Session, *, now: float | None = None) -> str:
    """Cookie value: base64(JSON payload) + "." + HMAC-SHA256 of it with the session secret."""
    payload = {"r": session.role, "n": session.name, "c": session.code,
               "exp": int((now or time.time()) + SESSION_SECONDS)}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode())
    mac = hmac.new(settings.session_secret.encode(), body.encode(), hashlib.sha256).digest()
    return f"{body}.{_b64(mac)}"


def session_from(settings: AppSettings, token: str | None, *, now: float | None = None) -> Session | None:
    """The session in a cookie, or None if missing, forged, expired, or its code/account was removed."""
    if not token or "." not in token or not settings.session_secret:
        return None
    body, _, mac = token.partition(".")
    expected = hmac.new(settings.session_secret.encode(), body.encode(), hashlib.sha256).digest()
    try:
        if not hmac.compare_digest(_unb64(mac), expected):
            return None
        data = json.loads(_unb64(body))
    except (ValueError, TypeError):
        return None
    if data.get("exp", 0) < (now or time.time()):
        return None
    session = Session(data.get("r", ""), data.get("n", ""), data.get("c", ""))
    if session.role == "client":
        return session if any(_code_id(c) == session.code for c in settings.invite_codes) else None
    if session.role == "preparer":
        return session if session.name in settings.preparers else None
    return None


class LoginAttempts:
    """Slows password and invite-code guessing: at most max_failures per address per window."""

    def __init__(self, max_failures: int = 10, window: float = 15 * 60, clock=time.monotonic):
        self.max_failures = max_failures
        self.window = window
        self._clock = clock
        self._lock = Lock()
        self._failures: dict[str, deque] = {}

    def _recent(self, key: str) -> deque:
        q = self._failures.setdefault(key, deque())
        cutoff = self._clock() - self.window
        while q and q[0] < cutoff:
            q.popleft()
        return q

    def blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._recent(key)) >= self.max_failures

    def fail(self, key: str) -> None:
        with self._lock:
            self._recent(key).append(self._clock())


def check_public_settings(settings: AppSettings) -> None:
    """Refuse to start a public site that anyone could administer."""
    if not settings.public:
        return
    problems = []
    if len(settings.session_secret) < 32:
        problems.append("TAX_AGENT_SESSION_SECRET must be at least 32 characters (python -m src.app.auth secret)")
    if not settings.preparers:
        problems.append("TAX_AGENT_PREPARERS must list at least one account (python -m src.app.auth hash)")
    if problems:
        raise RuntimeError("TAX_AGENT_MODE=public is not safe to start: " + "; ".join(problems))


def main(argv: list[str]) -> int:
    command = argv[0] if argv else ""
    if command == "hash":
        from getpass import getpass
        password = getpass("Preparer password: ")
        if len(password) < 10 or password != getpass("Again: "):
            print("Passwords must match and be at least 10 characters.")
            return 1
        print(hash_password(password))
        return 0
    if command == "secret":
        print(secrets.token_urlsafe(48))
        return 0
    print("usage: python -m src.app.auth hash|secret")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
