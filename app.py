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

import io
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitAPIException

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
USER_AGENT = "cmu-canvas-study-assistant/1.0"

# Files larger than this are described but not downloaded, so a stray 2 GB
# lecture recording cannot wedge the app.
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024

PREVIEWABLE_TYPES = ("application/pdf", "image/", "text/")

# Assignment prompts can be long; keep enough to be useful in a prompt budget.
MAX_DESCRIPTION_CHARS = 8_000

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

    ``allow_offsite_redirects`` exists only for file downloads: Canvas answers a
    file download with a redirect to its storage backend (S3), which is a
    different host. Those hops stay GET-only, must be https, and have the
    Authorization header stripped, so the Canvas token never leaves the Canvas
    origin. A download still has to *start* at the Canvas origin.
    """

    def __init__(self, allowed_base_url: str, allow_offsite_redirects: bool = False) -> None:
        super().__init__()
        self.allowed_origin = origin_of(allowed_base_url)
        self.allow_offsite_redirects = allow_offsite_redirects

    def _check(self, method: str | None, url: str) -> None:
        if (method or "").upper() != "GET":
            raise ReadOnlyViolation(
                f"Blocked {method!r} request to Canvas: this app is read-only and may only issue GET."
            )
        origin = origin_of(url)
        if origin != self.allowed_origin and not self.allow_offsite_redirects:
            raise ReadOnlyViolation(
                f"Blocked Canvas request to {origin}: only {self.allowed_origin} is allowed."
            )

    def request(self, method, url, *args, **kwargs):  # type: ignore[override]
        self._check(method, url)
        if origin_of(url) != self.allowed_origin:
            raise ReadOnlyViolation(
                f"Blocked Canvas request to {origin_of(url)}: requests must start at {self.allowed_origin}."
            )
        return super().request(method, url, *args, **kwargs)

    def send(self, request, **kwargs):  # type: ignore[override]
        self._check(request.method, request.url)
        if origin_of(request.url) != self.allowed_origin:
            request.headers.pop("Authorization", None)
            request.headers.pop("Cookie", None)
        return super().send(request, **kwargs)


# ---------------------------------------------------------------------------
# Canvas client (read-only)
# ---------------------------------------------------------------------------


class CanvasError(RuntimeError):
    """A Canvas request failed in a way worth showing the user."""


class CanvasAccessError(CanvasError):
    """Canvas refused this particular resource (401/403).

    Canvas answers "you may not list this" with 401 just as it answers "your
    token is bad", so callers that have already made a successful request treat
    this as a permission problem for one endpoint rather than a dead token.
    """


def parse_canvas_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def matches_all_words(haystack: str, query: str) -> bool:
    """Loose search: every word in the query must appear somewhere."""
    text = (haystack or "").lower()
    words = [word for word in (query or "").lower().split() if word]
    return all(word in text for word in words)


class _HtmlExtractor(HTMLParser):
    """Turn a Canvas HTML description into plain text plus its links."""

    BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._link_text: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip_depth += 1
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")
        if tag == "a":
            self._href = dict(attrs).get("href")
            self._link_text = []

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "a":
            if self._href:
                self.links.append((self._href, "".join(self._link_text).strip()))
            self._href = None
            self._link_text = []

    def handle_data(self, data):
        if self._skip_depth:
            return
        self.parts.append(data)
        if self._href is not None:
            self._link_text.append(data)


FILE_LINK_PATTERN = re.compile(r"/files/(\d+)")


def html_to_text_and_links(html: str) -> tuple[str, list[tuple[int, str]]]:
    """Return readable text and any Canvas file links found in ``html``."""
    if not html:
        return "", []
    parser = _HtmlExtractor()
    parser.feed(html)
    parser.close()

    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines()).strip()
    if len(text) > MAX_DESCRIPTION_CHARS:
        text = text[:MAX_DESCRIPTION_CHARS] + "\n[truncated]"

    files: list[tuple[int, str]] = []
    seen: set[int] = set()
    for href, label in parser.links:
        match = FILE_LINK_PATTERN.search(href or "")
        if not match:
            continue
        file_id = int(match.group(1))
        if file_id in seen:
            continue
        seen.add(file_id)
        files.append((file_id, label or f"file {file_id}"))
    return text, files


def human_size(num_bytes: Any) -> str | None:
    if not isinstance(num_bytes, (int, float)) or num_bytes < 0:
        return None
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
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
            {"Accept": "application/json", "User-Agent": USER_AGENT}
        )
        # Separate session for file bodies, which Canvas serves via a redirect
        # to its storage backend. Still GET-only; the token is dropped offsite.
        self.download_session = ReadOnlySession(self.base_url, allow_offsite_redirects=True)
        self.download_session.headers.update({"User-Agent": USER_AGENT})

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

        if response.status_code in (401, 403):
            raise CanvasAccessError(
                f"Canvas refused access ({response.status_code}) to {urlparse(url).path}. "
                "Either the API token is invalid or this part of the course is hidden from students."
            )
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
        """GET a paginated Canvas collection and return every row we fetched.

        ``path`` may be an API path or a full Canvas URL, since Canvas hands
        back absolute URLs in places like a module's ``items_url``.
        """
        query = {"per_page": 100}
        query.update(params or {})
        url: str | None = path if path.startswith("http") else self._url(path)
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
        # Canvas rejects search_term shorter than 3 characters, so short
        # queries are filtered locally instead.
        if search_term and len(search_term) >= 3:
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
            "submission_types": row.get("submission_types") or [],
            "submitted": bool(submission.get("submitted_at")) if submission else None,
            "has_description": bool((row.get("description") or "").strip()),
            "html_url": row.get("html_url"),
            **describe_due(due),
        }

    def get_assignment(self, course_id: int, assignment_id: int) -> dict[str, Any]:
        response = self._get(
            self._url(f"courses/{int(course_id)}/assignments/{int(assignment_id)}"),
            {"include[]": "submission"},
        )
        try:
            return response.json()
        except ValueError as exc:
            raise CanvasError(f"Canvas returned a non-JSON response for assignment {assignment_id}.") from exc

    def get_assignment_details(
        self, assignment_id: int, course_id: int | None = None
    ) -> dict[str, Any]:
        """Full detail for one assignment: the prompt, how to submit, attachments."""
        courses = self._course_lookup(course_id)
        if course_id is not None and not courses:
            raise CanvasError(f"Course {course_id} is not one of your active courses.")

        row: dict[str, Any] | None = None
        course: dict[str, Any] = {}
        for candidate in courses:
            try:
                row = self.get_assignment(candidate["course_id"], assignment_id)
            except CanvasError:
                continue  # wrong course when we are scanning; try the next one
            course = candidate
            break
        if row is None:
            raise CanvasError(
                f"Could not find assignment {assignment_id}. Pass the course_id from "
                "list_upcoming_assignments or find_due_dates."
            )

        details = self._normalize_assignment(row, course)
        description = row.get("description") or ""
        text, links = html_to_text_and_links(description)
        submission = row.get("submission") or {}
        details.update(
            {
                "instructions": text or None,
                "submission_types": row.get("submission_types") or [],
                "allowed_extensions": row.get("allowed_extensions") or [],
                "allowed_attempts": row.get("allowed_attempts"),
                "grading_type": row.get("grading_type"),
                "unlock_at": (describe_due(parse_canvas_timestamp(row.get("unlock_at"))))["due_at_local"],
                "lock_at": (describe_due(parse_canvas_timestamp(row.get("lock_at"))))["due_at_local"],
                "attached_files": [
                    {"file_id": file_id, "filename": name} for file_id, name in links
                ],
                "submission_status": {
                    "submitted_at": submission.get("submitted_at"),
                    "workflow_state": submission.get("workflow_state"),
                    "attempt": submission.get("attempt"),
                    "late": submission.get("late"),
                    "missing": submission.get("missing"),
                    "score": submission.get("score"),
                },
                "rubric": [
                    {
                        "criterion": item.get("description"),
                        "points": item.get("points"),
                    }
                    for item in (row.get("rubric") or [])
                ],
            }
        )
        return details

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

    # -- files ------------------------------------------------------------

    def _normalize_file(self, row: dict[str, Any], course: dict[str, Any] | None = None) -> dict[str, Any]:
        """Describe a file for the model. The download URL is deliberately
        omitted: it carries a verifier that acts like a bearer token."""
        updated = parse_canvas_timestamp(row.get("updated_at") or row.get("created_at"))
        size = row.get("size")
        return {
            "file_id": row.get("id"),
            "filename": row.get("display_name") or row.get("filename"),
            "content_type": row.get("content-type") or row.get("content_type"),
            "size_bytes": size,
            "size_readable": human_size(size),
            "course_id": (course or {}).get("course_id"),
            "course_name": (course or {}).get("name"),
            "updated_at": updated.isoformat().replace("+00:00", "Z") if updated else None,
            "locked": bool(row.get("locked_for_user") or row.get("locked")),
            "previewable": bool(
                (row.get("content-type") or row.get("content_type") or "").startswith(PREVIEWABLE_TYPES)
            ),
            "found_in": row.get("module") or row.get("source"),
        }

    def _files_from_files_tab(self, course_id: int) -> list[dict[str, Any]]:
        # No search_term or sort parameters here on purpose: Canvas rejects
        # short search terms outright, and a rejected request used to look
        # exactly like a course with no files. Filter and sort locally instead.
        return [dict(row, source="files") for row in self.get_list(f"courses/{course_id}/files")]

    def _files_from_modules(self, course_id: int) -> list[dict[str, Any]]:
        """Find files through Modules, which stay visible when Files is hidden."""
        found: list[dict[str, Any]] = []
        for module in self.get_list(f"courses/{course_id}/modules", {"include[]": "items"}):
            items = module.get("items")
            if items is None and module.get("items_url"):
                try:
                    items = self.get_list(module["items_url"])
                except CanvasError:
                    items = []
            for item in items or []:
                if item.get("type") != "File" or not item.get("content_id"):
                    continue
                found.append(
                    {
                        "id": item["content_id"],
                        "display_name": item.get("title"),
                        "source": "modules",
                        "module": module.get("name"),
                    }
                )
        return found

    def collect_course_files(self, course: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
        """Gather a course's files from every place a student can see them."""
        course_id = course["course_id"]
        name = course.get("name") or f"course {course_id}"
        candidates: dict[Any, dict[str, Any]] = {}
        notes: list[str] = []

        try:
            for row in self._files_from_files_tab(course_id):
                candidates[row.get("id")] = row
        except CanvasAccessError:
            notes.append(f"{name}: the Files tab is hidden, so only files posted in Modules are visible.")
        except CanvasError as exc:
            notes.append(f"{name}: could not list Files ({exc})")

        # Modules are the fallback when Files is hidden or empty, which is how
        # most courses that "have no files" actually publish their handouts.
        if not candidates:
            try:
                for row in self._files_from_modules(course_id):
                    candidates.setdefault(row["id"], row)
            except CanvasAccessError:
                notes.append(f"{name}: Modules are not visible either.")
            except CanvasError as exc:
                notes.append(f"{name}: could not read Modules ({exc})")

        return [row for row in candidates.values() if row.get("id")], notes

    def find_course_files(
        self,
        query: str | None = None,
        course_id: int | None = None,
        limit: int = 20,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Search the student's course files. Returns (files, notes).

        Notes explain any course that could not be searched, so a zero-result
        answer can say why instead of implying the course has no files.
        """
        term = (query or "").strip()
        notes: list[str] = []
        matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
        courses = self._course_lookup(course_id)
        if not courses:
            return [], ["No active courses were returned by Canvas."]

        for course in courses:
            candidates, course_notes = self.collect_course_files(course)
            notes.extend(course_notes)
            for row in candidates:
                haystack = " ".join(
                    str(part)
                    for part in (
                        row.get("display_name"),
                        row.get("filename"),
                        row.get("module"),
                    )
                    if part
                )
                if term and not matches_all_words(haystack, term):
                    continue
                matched.append((row, course))

        matched.sort(key=lambda pair: str(pair[0].get("updated_at") or ""), reverse=True)
        files: list[dict[str, Any]] = []
        for row, course in matched[: max(1, int(limit))]:
            # Module items carry only a title and an id; fill in the rest so the
            # model can tell a PDF from a video before opening it.
            if not (row.get("content-type") or row.get("content_type")):
                try:
                    row = dict(
                        self.get_file_metadata(row["id"]),
                        source=row.get("source"),
                        module=row.get("module"),
                    )
                except CanvasError:
                    pass
            files.append(self._normalize_file(row, course))
        return files, notes

    def get_file_metadata(self, file_id: int) -> dict[str, Any]:
        response = self._get(self._url(f"files/{int(file_id)}"))
        try:
            return response.json()
        except ValueError as exc:
            raise CanvasError(f"Canvas returned a non-JSON response for file {file_id}.") from exc

    def describe_file(self, file_id: int) -> dict[str, Any]:
        return self._normalize_file(self.get_file_metadata(file_id))

    def download_file(self, file_id: int, max_bytes: int = MAX_DOWNLOAD_BYTES) -> tuple[bytes, dict[str, Any]]:
        """Fetch a file's bytes along with its description. GET only."""
        raw = self.get_file_metadata(file_id)
        described = self._normalize_file(raw)
        if described["locked"]:
            raise CanvasError(f"Canvas has locked {described['filename']!r}, so it cannot be opened.")

        size = described.get("size_bytes")
        if isinstance(size, int) and size > max_bytes:
            raise CanvasError(
                f"{described['filename']!r} is {described['size_readable']}, larger than the "
                f"{human_size(max_bytes)} preview limit. Open it in Canvas instead."
            )

        url = raw.get("url")
        if not url:
            raise CanvasError(f"Canvas did not provide a download link for file {file_id}.")

        try:
            response = self.download_session.get(
                url, headers=self._auth_headers(), timeout=self.timeout, stream=True
            )
        except ReadOnlyViolation:
            raise
        except requests.RequestException as exc:
            raise CanvasError(f"Could not download {described['filename']!r}: {redact(exc)}") from exc

        with response:
            if response.status_code >= 400:
                raise CanvasError(
                    f"Canvas returned HTTP {response.status_code} while downloading "
                    f"{described['filename']!r}."
                )
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise CanvasError(
                        f"{described['filename']!r} exceeds the {human_size(max_bytes)} preview limit."
                    )
                chunks.append(chunk)

        described["size_bytes"] = total
        described["size_readable"] = human_size(total)
        if not described["content_type"]:
            described["content_type"] = response.headers.get("Content-Type", "application/octet-stream")
        return b"".join(chunks), described


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
    {
        "type": "function",
        "function": {
            "name": "find_course_files",
            "description": (
                "Find files posted in the student's courses — lecture slides, PDFs, handouts, "
                "syllabi — searching both the Files area and Modules. Returns file ids and "
                "metadata, not the file contents. Use open_file afterwards to show a file. "
                "Call with no query to list everything available. If it returns nothing, read "
                "the 'notes' field: it says which courses could not be searched and why. Files "
                "attached to a specific assignment appear in get_assignment_details instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search for in file names, for example 'syllabus' or 'lecture 5'.",
                    },
                    "course_id": {
                        "type": "integer",
                        "description": "Restrict to one Canvas course id. Omit to search every active course.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of files to return. Defaults to 20.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_file",
            "description": (
                "Display a Canvas file to the student in the chat. PDFs, images and text files are "
                "shown inline; anything else is offered as a download. Get the file_id from "
                "find_course_files first. Returns the file's metadata, and for text files its text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "integer",
                        "description": "The Canvas file id to display.",
                    }
                },
                "required": ["file_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_assignment_details",
            "description": (
                "Get everything Canvas knows about one assignment: the full instructions/prompt, "
                "how it must be submitted (file upload, text entry, URL, on paper), allowed file "
                "extensions, attempts, rubric, attached files, and the student's submission "
                "status. Use this whenever the student asks what an assignment actually requires "
                "or how to turn it in. Get assignment_id and course_id from "
                "list_upcoming_assignments or find_due_dates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "assignment_id": {
                        "type": "integer",
                        "description": "The Canvas assignment id.",
                    },
                    "course_id": {
                        "type": "integer",
                        "description": "The Canvas course id the assignment belongs to.",
                    },
                },
                "required": ["assignment_id"],
            },
        },
    },
]

TOOL_NAMES = tuple(schema["function"]["name"] for schema in TOOL_SCHEMAS)

MAX_EXTRACTED_CHARS = 20_000


def extract_text(content: bytes, content_type: str) -> tuple[str | None, bool]:
    """Pull readable text out of a downloaded file so the model can discuss it.

    Returns (text, truncated). Text is None for formats we cannot read, in
    which case the student still sees the file itself in the UI.
    """
    text: str | None = None
    if content_type.startswith("text/"):
        text = content.decode("utf-8", errors="replace")
    elif content_type == "application/pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return None, False
        try:
            reader = PdfReader(io.BytesIO(content))
            pages = [page.extract_text() or "" for page in reader.pages[:50]]
        except Exception:  # pypdf raises a wide range of parse errors
            return None, False
        text = "\n\n".join(page.strip() for page in pages if page.strip())

    if not text or not text.strip():
        return None, False
    if len(text) > MAX_EXTRACTED_CHARS:
        return text[:MAX_EXTRACTED_CHARS], True
    return text, False


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
        if name == "find_course_files":
            files, notes = client.find_course_files(
                query=args.get("query"),
                course_id=args.get("course_id"),
                limit=int(args.get("limit", 20) or 20),
            )
            payload: dict[str, Any] = {"files": files, "count": len(files)}
            if notes:
                payload["notes"] = notes
            if not files:
                payload["hint"] = (
                    "No files matched. Try again with no query to see everything, or tell the "
                    "student the file may be attached to an assignment (check "
                    "get_assignment_details) or posted somewhere this app cannot read."
                )
            return payload
        if name == "get_assignment_details":
            if args.get("assignment_id") in (None, ""):
                return {"error": "get_assignment_details needs an assignment_id."}
            return {
                "assignment": client.get_assignment_details(
                    assignment_id=int(args["assignment_id"]),
                    course_id=int(args["course_id"]) if args.get("course_id") else None,
                )
            }
        if name == "open_file":
            if args.get("file_id") in (None, ""):
                return {"error": "open_file needs a file_id from find_course_files."}
            content, described = client.download_file(int(args["file_id"]))
            result: dict[str, Any] = {
                "file": described,
                "displayed_to_student": True,
                "note": "The file is now shown in the chat; do not try to paste its contents.",
            }
            text, truncated = extract_text(content, described.get("content_type") or "")
            if text:
                result["text_excerpt"] = text
                result["text_truncated"] = truncated
            return result
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

You have read-only access to Canvas through the provided tools. You can read course files and show \
them to the student, but you cannot submit work, post messages, upload files or change anything in \
Canvas; if the student asks for that, say so plainly and suggest they do it themselves in Canvas.

Guidelines:
- Call a tool whenever the answer depends on the student's actual Canvas data. Never invent course \
  names, assignment titles or due dates.
- To show a file, call find_course_files to locate it, then open_file with its file_id. The file \
  itself is rendered in the chat, so introduce it in one short sentence instead of describing every \
  page. If open_file returns a text excerpt you may use it to answer questions about the contents, \
  and say so if the excerpt was truncated.
- If find_course_files returns nothing, do not simply say "no files". Retry once with no query, and \
  check get_assignment_details for the relevant assignment, since attachments often live on the \
  assignment rather than in Files. Report anything in the "notes" field, such as a course whose \
  Files tab is hidden.
- For "what does this assignment want?", "how do I submit?", or anything about instructions, \
  formats or rubrics, call get_assignment_details. The list tools only carry titles and due dates.
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


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def fetch_file(_canvas: CanvasClient, file_id: int) -> tuple[bytes, dict[str, Any]]:
    return _canvas.download_file(file_id)


def render_file_preview(canvas: CanvasClient, payload: dict[str, Any]) -> None:
    """Show a Canvas file inline. Called on every rerun, hence the cache."""
    described = payload.get("file") or {}
    file_id = described.get("file_id")
    if not file_id:
        return
    try:
        content, described = fetch_file(canvas, int(file_id))
    except (CanvasError, ReadOnlyViolation) as exc:
        st.warning(redact(exc))
        return

    filename = described.get("filename") or f"file-{file_id}"
    content_type = described.get("content_type") or ""
    st.caption(f"{filename} — {described.get('size_readable') or ''} from Canvas")
    if content_type == "application/pdf":
        try:
            st.pdf(io.BytesIO(content), height=600)
        except StreamlitAPIException:
            # st.pdf needs the streamlit[pdf] extra; without it, still hand
            # over the file rather than blowing up the chat.
            st.info(
                "Install the PDF viewer to see this inline: `pip install -r requirements.txt` "
                "(or `pip install \"streamlit[pdf]\"`). You can download the file below."
            )
    elif content_type.startswith("image/"):
        st.image(content, caption=filename)
    elif content_type.startswith("text/"):
        st.code(content.decode("utf-8", errors="replace")[:20_000])
    else:
        st.info("This file type cannot be previewed here — download it to open it.")
    st.download_button(
        "Download",
        data=content,
        file_name=filename,
        mime=content_type or "application/octet-stream",
        key=f"download-{file_id}-{abs(hash(filename)) % 10_000}",
    )


def render_tool_message(message: dict[str, Any], canvas: CanvasClient | None = None) -> None:
    try:
        payload = json.loads(message.get("content") or "{}")
    except json.JSONDecodeError:
        payload = None

    if canvas and message.get("name") == "open_file" and isinstance(payload, dict) and payload.get("file"):
        render_file_preview(canvas, payload)

    with st.expander(f"Canvas lookup: {message.get('name', 'tool')}", expanded=False):
        if payload is None:
            st.code(message.get("content") or "")
        else:
            st.json(payload)


def render_message(message: dict[str, Any], canvas: CanvasClient | None = None) -> None:
    role = message.get("role")
    if role == "tool":
        render_tool_message(message, canvas)
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
    st.caption(
        "Ask about your courses, upcoming work, due dates and course files. "
        "Canvas access is read-only."
    )

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
        render_message(message, canvas)

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
        render_message(message, canvas)

    with st.spinner("Checking Canvas..."):
        try:
            run_agent_turn(deepseek, canvas, history, on_message=persist_and_render)
        except (DeepSeekError, CanvasError, ReadOnlyViolation) as exc:
            st.error(redact(exc))


if __name__ == "__main__":
    main()
