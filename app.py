"""CMU Canvas study assistant.

A local, read-only chat assistant that answers questions about your Canvas
courses, assignments and due dates. Run it with:

    streamlit run app.py

Everything is local: the Streamlit UI, the SQLite conversation history and the
Canvas API calls. The only outbound traffic is to the Canvas host you configure
and to the DeepSeek API.

Safety: every Canvas request goes through ReadOnlySession, which refuses any
HTTP method other than GET and refuses any host other than the configured
Canvas origin. Nothing in this file can submit an assignment, send a message or
change Canvas state.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Secret handling
# ---------------------------------------------------------------------------

REDACTION_PLACEHOLDER = "***REDACTED***"

_secret_values: set[str] = set()


def register_secret(value: str | None) -> None:
    """Remember a secret so that redact() can scrub it from any output."""
    if value and len(value) >= 8:
        _secret_values.add(value)


def redact(text: Any) -> str:
    """Replace every registered secret in ``text`` with a placeholder."""
    result = text if isinstance(text, str) else str(text)
    for secret in sorted(_secret_values, key=len, reverse=True):
        result = result.replace(secret, REDACTION_PLACEHOLDER)
    return result


class RedactingFilter(logging.Filter):
    """Scrub secrets from log records before they reach a handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


logger = logging.getLogger("canvas_study_assistant")
logger.addFilter(RedactingFilter())

# ---------------------------------------------------------------------------
# Configuration (environment variables only)
# ---------------------------------------------------------------------------

DEFAULT_CANVAS_BASE_URL = "https://canvas.cmu.edu"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_DB_PATH = "canvas_assistant.db"

REQUIRED_ENV_VARS = ("DEEPSEEK_API_KEY", "CANVAS_API_TOKEN")


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    canvas_api_token: str
    canvas_base_url: str = DEFAULT_CANVAS_BASE_URL
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    model: str = DEEPSEEK_MODEL
    db_path: str = DEFAULT_DB_PATH

    @property
    def missing(self) -> list[str]:
        values = {
            "DEEPSEEK_API_KEY": self.deepseek_api_key,
            "CANVAS_API_TOKEN": self.canvas_api_token,
        }
        return [name for name in REQUIRED_ENV_VARS if not values[name].strip()]


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Build Settings from environment variables. Secrets are never defaulted."""
    source = os.environ if env is None else env
    settings = Settings(
        deepseek_api_key=source.get("DEEPSEEK_API_KEY", "").strip(),
        canvas_api_token=source.get("CANVAS_API_TOKEN", "").strip(),
        canvas_base_url=(source.get("CANVAS_BASE_URL") or DEFAULT_CANVAS_BASE_URL).strip().rstrip("/"),
        deepseek_base_url=(source.get("DEEPSEEK_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL).strip().rstrip("/"),
        model=(source.get("DEEPSEEK_MODEL") or DEEPSEEK_MODEL).strip(),
        db_path=(source.get("CANVAS_ASSISTANT_DB") or DEFAULT_DB_PATH).strip(),
    )
    register_secret(settings.deepseek_api_key)
    register_secret(settings.canvas_api_token)
    return settings


# ---------------------------------------------------------------------------
# Read-only HTTP transport
# ---------------------------------------------------------------------------


class ReadOnlyViolation(RuntimeError):
    """Raised when code attempts a non-GET or off-host Canvas request."""


def origin_of(url: str) -> str:
    """Return the lowercased scheme://host:port for ``url``."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ReadOnlyViolation(f"Canvas URL is not absolute: {url!r}")
    host = parsed.hostname or ""
    if parsed.scheme != "https" and host not in {"localhost", "127.0.0.1"}:
        raise ReadOnlyViolation(f"Canvas requests must use https, got {parsed.scheme!r}")
    netloc = host.lower()
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return f"{parsed.scheme.lower()}://{netloc}"


class ReadOnlySession(requests.Session):
    """A requests Session that can only ever issue GETs to one Canvas origin.

    Both hooks matter: ``request`` blocks direct calls such as ``session.post``,
    and ``send`` blocks anything that reaches the transport another way,
    including redirects, which are re-checked one hop at a time.
    """

    def __init__(self, allowed_base_url: str) -> None:
        super().__init__()
        self.allowed_origin = origin_of(allowed_base_url)

    def _check(self, method: str | None, url: str) -> None:
        if (method or "").upper() != "GET":
            raise ReadOnlyViolation(
                f"Blocked {method!r} request to Canvas: this app is read-only and may only issue GET."
            )
        if origin_of(url) != self.allowed_origin:
            raise ReadOnlyViolation(
                f"Blocked Canvas request to {origin_of(url)}: only {self.allowed_origin} is allowed."
            )

    def request(self, method, url, *args, **kwargs):  # type: ignore[override]
        self._check(method, url)
        return super().request(method, url, *args, **kwargs)

    def send(self, request, **kwargs):  # type: ignore[override]
        self._check(request.method, request.url)
        return super().send(request, **kwargs)


# ---------------------------------------------------------------------------
# Canvas client (read-only)
# ---------------------------------------------------------------------------


class CanvasError(RuntimeError):
    """A Canvas request failed in a way worth showing the user."""


def parse_canvas_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def describe_due(due: datetime | None, now: datetime | None = None) -> dict[str, Any]:
    """Render a due date in UTC, in the machine's local time, and as a delta."""
    if due is None:
        return {"due_at": None, "due_at_local": None, "days_until": None}
    now = now or datetime.now(timezone.utc)
    return {
        "due_at": due.isoformat().replace("+00:00", "Z"),
        "due_at_local": due.astimezone().strftime("%a %b %d, %Y %I:%M %p %Z").strip(),
        "days_until": round((due - now).total_seconds() / 86400, 2),
    }


class CanvasClient:
    """Read-only wrapper around the Canvas LMS REST API."""

    def __init__(
        self,
        base_url: str,
        api_token: str,
        timeout: float = 30.0,
        max_pages: int = 10,
        session: ReadOnlySession | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_pages = max_pages
        self._token = api_token
        self.session = session or ReadOnlySession(self.base_url)
        self.session.headers.update(
            {"Accept": "application/json", "User-Agent": "cmu-canvas-study-assistant/1.0"}
        )

    # -- plumbing ---------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/v1/{path.lstrip('/')}"

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _get(self, url: str, params: dict[str, Any] | None = None) -> requests.Response:
        try:
            response = self.session.get(
                url, params=params, headers=self._auth_headers(), timeout=self.timeout
            )
        except ReadOnlyViolation:
            raise
        except requests.RequestException as exc:
            raise CanvasError(f"Could not reach Canvas at {self.base_url}: {redact(exc)}") from exc

        if response.status_code == 401:
            raise CanvasError(
                "Canvas rejected the API token (401). Generate a new token and update CANVAS_API_TOKEN in your .env."
            )
        if response.status_code == 403:
            raise CanvasError("Canvas denied access to this data (403).")
        if response.status_code == 404:
            raise CanvasError(f"Canvas has no such resource (404): {urlparse(url).path}")
        if response.status_code >= 400:
            raise CanvasError(f"Canvas returned HTTP {response.status_code} for {urlparse(url).path}.")
        return response

    @staticmethod
    def _next_link(response: requests.Response) -> str | None:
        """Pull the rel="next" URL out of a Canvas Link header."""
        header = response.headers.get("Link") or response.headers.get("link")
        if not header:
            return None
        for part in header.split(","):
            segments = part.split(";")
            if len(segments) < 2:
                continue
            url = segments[0].strip().strip("<>")
            if any(seg.strip().replace('"', "").replace(" ", "") == "rel=next" for seg in segments[1:]):
                return url
        return None

    def get_list(self, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """GET a paginated Canvas collection and return every row we fetched."""
        query = {"per_page": 100}
        query.update(params or {})
        url: str | None = self._url(path)
        rows: list[dict[str, Any]] = []
        pages = 0
        while url and pages < self.max_pages:
            response = self._get(url, params=query if pages == 0 else None)
            try:
                payload = response.json()
            except ValueError as exc:
                raise CanvasError(f"Canvas returned a non-JSON response for {path}.") from exc
            if isinstance(payload, dict):
                payload = payload.get("assignments") or payload.get("courses") or []
            rows.extend(item for item in payload if isinstance(item, dict))
            url = self._next_link(response)
            pages += 1
        return rows

    # -- data -------------------------------------------------------------

    def list_courses(self, include_concluded: bool = False) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"include[]": "term"}
        if include_concluded:
            params["state[]"] = ["available", "completed"]
        else:
            params["enrollment_state"] = "active"
        rows = self.get_list("courses", params)
        courses = []
        for row in rows:
            if row.get("access_restricted_by_date"):
                continue
            term = row.get("term") or {}
            courses.append(
                {
                    "course_id": row.get("id"),
                    "name": row.get("name") or row.get("course_code") or f"Course {row.get('id')}",
                    "course_code": row.get("course_code"),
                    "term": term.get("name"),
                    "html_url": f"{self.base_url}/courses/{row.get('id')}",
                }
            )
        return courses

    def list_assignments(
        self,
        course_id: int,
        bucket: str | None = None,
        search_term: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"include[]": "submission", "order_by": "due_at"}
        if bucket:
            params["bucket"] = bucket
        if search_term:
            params["search_term"] = search_term
        return self.get_list(f"courses/{int(course_id)}/assignments", params)

    def _course_lookup(self, course_id: int | None) -> list[dict[str, Any]]:
        courses = self.list_courses()
        if course_id is None:
            return courses
        return [c for c in courses if c["course_id"] == int(course_id)]

    @staticmethod
    def _normalize_assignment(row: dict[str, Any], course: dict[str, Any]) -> dict[str, Any]:
        due = parse_canvas_timestamp(row.get("due_at"))
        submission = row.get("submission") or {}
        return {
            "assignment_id": row.get("id"),
            "title": row.get("name"),
            "course_id": course.get("course_id"),
            "course_name": course.get("name"),
            "points_possible": row.get("points_possible"),
            "submitted": bool(submission.get("submitted_at")) if submission else None,
            "html_url": row.get("html_url"),
            **describe_due(due),
        }

    def list_upcoming_assignments(
        self,
        days_ahead: int = 14,
        course_id: int | None = None,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(days=max(1, int(days_ahead)))
        found: list[dict[str, Any]] = []
        for course in self._course_lookup(course_id):
            for row in self.list_assignments(course["course_id"], bucket="upcoming"):
                due = parse_canvas_timestamp(row.get("due_at"))
                if due is None or due < now or due > horizon:
                    continue
                found.append(self._normalize_assignment(row, course))
        found.sort(key=lambda item: item["due_at"] or "")
        return found[: max(1, int(limit))]

    def find_due_dates(
        self,
        query: str,
        course_id: int | None = None,
        include_past: bool = False,
    ) -> list[dict[str, Any]]:
        term = (query or "").strip()
        if len(term) < 2:
            raise CanvasError("Search text must be at least 2 characters long.")
        now = datetime.now(timezone.utc)
        matches: list[dict[str, Any]] = []
        for course in self._course_lookup(course_id):
            for row in self.list_assignments(course["course_id"], search_term=term):
                if term.lower() not in (row.get("name") or "").lower():
                    continue
                due = parse_canvas_timestamp(row.get("due_at"))
                if not include_past and due is not None and due < now:
                    continue
                matches.append(self._normalize_assignment(row, course))
        matches.sort(key=lambda item: item["due_at"] or "9999")
        return matches


# ---------------------------------------------------------------------------
# Tools exposed to the model (all read-only)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_my_courses",
            "description": "List the Canvas courses the student is enrolled in.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_concluded": {
                        "type": "boolean",
                        "description": "Include finished courses from past terms. Defaults to false.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_upcoming_assignments",
            "description": (
                "List assignments that are due soon, sorted by due date. "
                "Use this for questions like 'what is due this week?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days_ahead": {
                        "type": "integer",
                        "description": "How many days ahead to look. Defaults to 14.",
                    },
                    "course_id": {
                        "type": "integer",
                        "description": "Restrict to one Canvas course id. Omit to search every active course.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of assignments to return. Defaults to 25.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_due_dates",
            "description": (
                "Find the due date of specific assignments by searching their titles, "
                "for example 'Homework 4' or 'midterm'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for in assignment titles (at least 2 characters).",
                    },
                    "course_id": {
                        "type": "integer",
                        "description": "Restrict to one Canvas course id. Omit to search every active course.",
                    },
                    "include_past": {
                        "type": "boolean",
                        "description": "Include assignments whose due date has already passed. Defaults to false.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_NAMES = tuple(schema["function"]["name"] for schema in TOOL_SCHEMAS)


def dispatch_tool(client: CanvasClient, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one read-only Canvas tool and return a JSON-serializable result."""
    args = arguments or {}
    try:
        if name == "list_my_courses":
            courses = client.list_courses(include_concluded=bool(args.get("include_concluded", False)))
            return {"courses": courses, "count": len(courses)}
        if name == "list_upcoming_assignments":
            assignments = client.list_upcoming_assignments(
                days_ahead=int(args.get("days_ahead", 14) or 14),
                course_id=args.get("course_id"),
                limit=int(args.get("limit", 25) or 25),
            )
            return {"assignments": assignments, "count": len(assignments)}
        if name == "find_due_dates":
            assignments = client.find_due_dates(
                query=str(args.get("query", "")),
                course_id=args.get("course_id"),
                include_past=bool(args.get("include_past", False)),
            )
            return {"assignments": assignments, "count": len(assignments)}
    except (CanvasError, ReadOnlyViolation, ValueError, TypeError) as exc:
        return {"error": redact(exc)}
    return {"error": f"Unknown tool {name!r}. Available tools: {', '.join(TOOL_NAMES)}."}


# ---------------------------------------------------------------------------
# DeepSeek client
# ---------------------------------------------------------------------------


class DeepSeekError(RuntimeError):
    """The DeepSeek chat completion call failed."""


class DeepSeekClient:
    """Minimal OpenAI-compatible chat completions client for DeepSeek.

    This talks to DeepSeek, not Canvas, so it uses a plain session and POSTs.
    The read-only guarantee applies to Canvas, whose traffic never touches this
    session.
    """

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 120.0) -> None:
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise DeepSeekError(f"Could not reach DeepSeek: {redact(exc)}") from exc

        if response.status_code == 401:
            raise DeepSeekError("DeepSeek rejected the API key (401). Check DEEPSEEK_API_KEY in your .env.")
        if response.status_code >= 400:
            raise DeepSeekError(f"DeepSeek returned HTTP {response.status_code}: {redact(response.text)[:300]}")
        try:
            data = response.json()
            return data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError) as exc:
            raise DeepSeekError("DeepSeek returned an unexpected response shape.") from exc


# ---------------------------------------------------------------------------
# SQLite conversation history
# ---------------------------------------------------------------------------

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        title      TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role            TEXT NOT NULL,
        content         TEXT,
        tool_calls      TEXT,
        tool_call_id    TEXT,
        name            TEXT,
        created_at      TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id, id)",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ConversationStore:
    """Conversation history in a local SQLite file."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        parent = Path(db_path).expanduser().parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _create_schema(self) -> None:
        with self._connect() as conn:
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)

    def create_conversation(self, title: str = "New chat") -> int:
        now = _utcnow()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
                (title, now, now),
            )
            return int(cursor.lastrowid)

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_conversation(self, conversation_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def set_title(self, conversation_id: int, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title[:80], _utcnow(), conversation_id),
            )

    def add_message(self, conversation_id: int, message: dict[str, Any]) -> int:
        tool_calls = message.get("tool_calls")
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, tool_calls, tool_call_id, name, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    message.get("role", "user"),
                    message.get("content"),
                    json.dumps(tool_calls) if tool_calls else None,
                    message.get("tool_call_id"),
                    message.get("name"),
                    _utcnow(),
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?", (_utcnow(), conversation_id)
            )
            return int(cursor.lastrowid)

    def get_messages(self, conversation_id: int) -> list[dict[str, Any]]:
        """Return stored messages in the shape the chat API expects."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, tool_calls, tool_call_id, name FROM messages "
                "WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            message: dict[str, Any] = {"role": row["role"], "content": row["content"]}
            if row["tool_calls"]:
                message["tool_calls"] = json.loads(row["tool_calls"])
            if row["tool_call_id"]:
                message["tool_call_id"] = row["tool_call_id"]
            if row["name"]:
                message["name"] = row["name"]
            messages.append(message)
        return messages


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a study assistant for a Carnegie Mellon student. You help them keep \
track of their Canvas courses, assignments and deadlines, and you help them plan their study time.

You have read-only access to Canvas through the provided tools. You cannot submit work, post \
messages or change anything in Canvas; if the student asks for that, say so plainly and suggest \
they do it themselves in Canvas.

Guidelines:
- Call a tool whenever the answer depends on the student's actual Canvas data. Never invent course \
  names, assignment titles or due dates.
- Today is {today}. The student's local timezone is {timezone}.
- Tool results give due dates both in UTC ("due_at") and in local time ("due_at_local"). Always \
  quote local time to the student.
- Be concise. Use short lists for multiple assignments, and mention the course for each one.
- If a tool returns an error, explain it briefly and suggest a next step."""

MAX_TOOL_ROUNDS = 4


def build_system_prompt(now: datetime | None = None) -> str:
    now = now or datetime.now().astimezone()
    return SYSTEM_PROMPT.format(
        today=now.strftime("%A, %B %d, %Y"),
        timezone=now.strftime("%Z") or "the machine's local timezone",
    )


def run_agent_turn(
    deepseek: DeepSeekClient,
    canvas: CanvasClient,
    history: list[dict[str, Any]],
    on_message: Callable[[dict[str, Any]], None] | None = None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> list[dict[str, Any]]:
    """Drive the tool-calling loop and return the messages produced this turn.

    ``history`` is the stored conversation (no system message). ``on_message``
    is called with each new message so the caller can persist and render it.
    """
    working = [{"role": "system", "content": build_system_prompt()}] + list(history)
    produced: list[dict[str, Any]] = []

    def emit(message: dict[str, Any]) -> None:
        produced.append(message)
        working.append(message)
        if on_message:
            on_message(message)

    for _ in range(max_tool_rounds):
        reply = deepseek.complete(working, tools=TOOL_SCHEMAS)
        tool_calls = reply.get("tool_calls") or []
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": reply.get("content") or "",
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        emit(assistant_message)

        if not tool_calls:
            return produced

        for call in tool_calls:
            function = call.get("function") or {}
            name = function.get("name", "")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result = dispatch_tool(canvas, name, arguments)
            emit(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": name,
                    "content": json.dumps(result, default=str),
                }
            )

    emit(
        {
            "role": "assistant",
            "content": (
                "I looked things up several times but could not settle on an answer. "
                "Try asking a narrower question, for example about a single course."
            ),
        }
    )
    return produced


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

PAGE_TITLE = "CMU Canvas Study Assistant"


@st.cache_resource(show_spinner=False)
def get_store(db_path: str) -> ConversationStore:
    return ConversationStore(db_path)


@st.cache_resource(show_spinner=False)
def get_canvas_client(_settings: Settings, cache_key: str) -> CanvasClient:
    return CanvasClient(_settings.canvas_base_url, _settings.canvas_api_token)


@st.cache_resource(show_spinner=False)
def get_deepseek_client(_settings: Settings, cache_key: str) -> DeepSeekClient:
    return DeepSeekClient(_settings.deepseek_api_key, _settings.deepseek_base_url, _settings.model)


def render_tool_message(message: dict[str, Any]) -> None:
    with st.expander(f"Canvas lookup: {message.get('name', 'tool')}", expanded=False):
        try:
            st.json(json.loads(message.get("content") or "{}"))
        except json.JSONDecodeError:
            st.code(message.get("content") or "")


def render_message(message: dict[str, Any]) -> None:
    role = message.get("role")
    if role == "tool":
        render_tool_message(message)
        return
    if role == "assistant":
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            st.caption(f"Calling `{function.get('name')}` with `{function.get('arguments')}`")
        if message.get("content"):
            with st.chat_message("assistant"):
                st.markdown(message["content"])
        return
    if role == "user" and message.get("content"):
        with st.chat_message("user"):
            st.markdown(message["content"])


def render_sidebar(settings: Settings, store: ConversationStore) -> None:
    with st.sidebar:
        st.subheader("Configuration")
        for name, value in (
            ("DEEPSEEK_API_KEY", settings.deepseek_api_key),
            ("CANVAS_API_TOKEN", settings.canvas_api_token),
        ):
            st.write(f"{'OK' if value else 'MISSING'} — `{name}`")
        st.write(f"Canvas: `{settings.canvas_base_url}`")
        st.write(f"Model: `{settings.model}`")
        st.caption("Token values are never shown here, logged, or sent anywhere except Canvas.")

        st.divider()
        st.subheader("Conversations")
        if st.button("New chat", width="stretch"):
            st.session_state.conversation_id = store.create_conversation()
            st.rerun()

        for conversation in store.list_conversations():
            active = conversation["id"] == st.session_state.get("conversation_id")
            columns = st.columns([5, 1])
            label = ("● " if active else "") + conversation["title"]
            if columns[0].button(label, key=f"open-{conversation['id']}", width="stretch"):
                st.session_state.conversation_id = conversation["id"]
                st.rerun()
            if columns[1].button("✕", key=f"delete-{conversation['id']}", help="Delete"):
                store.delete_conversation(conversation["id"])
                if active:
                    st.session_state.pop("conversation_id", None)
                st.rerun()

        st.divider()
        st.caption("Read-only: this app only ever sends GET requests to Canvas.")


def main() -> None:
    load_dotenv()
    settings = load_settings()

    st.set_page_config(page_title=PAGE_TITLE, page_icon="📚", layout="centered")
    st.title(PAGE_TITLE)
    st.caption("Ask about your courses, upcoming work and due dates. Canvas access is read-only.")

    store = get_store(settings.db_path)
    render_sidebar(settings, store)

    if settings.missing:
        st.error(
            "Missing environment variables: "
            + ", ".join(f"`{name}`" for name in settings.missing)
            + ". Copy `.env.example` to `.env`, fill in your tokens, then restart the app."
        )
        st.stop()

    if "conversation_id" not in st.session_state:
        conversations = store.list_conversations()
        st.session_state.conversation_id = (
            conversations[0]["id"] if conversations else store.create_conversation()
        )
    conversation_id = st.session_state.conversation_id

    canvas = get_canvas_client(settings, settings.canvas_base_url)
    deepseek = get_deepseek_client(settings, f"{settings.deepseek_base_url}:{settings.model}")

    history = store.get_messages(conversation_id)
    for message in history:
        render_message(message)

    prompt = st.chat_input("What's due this week?")
    if not prompt:
        return

    user_message = {"role": "user", "content": prompt}
    store.add_message(conversation_id, user_message)
    if not history:
        store.set_title(conversation_id, prompt.strip().splitlines()[0])
    render_message(user_message)
    history.append(user_message)

    def persist_and_render(message: dict[str, Any]) -> None:
        store.add_message(conversation_id, message)
        render_message(message)

    with st.spinner("Checking Canvas..."):
        try:
            run_agent_turn(deepseek, canvas, history, on_message=persist_and_render)
        except (DeepSeekError, CanvasError, ReadOnlyViolation) as exc:
            st.error(redact(exc))


if __name__ == "__main__":
    main()
