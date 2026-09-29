"""Deterministic Canvas resource matching.

Course codes and Canvas course ids stay separate from the student's search
query. Title matching is done in Python so the model cannot stuff a course_id
into ``query``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_COURSE_KEY_STRIP = re.compile(r"[^a-z0-9]+")
_COURSE_CODE_RE = re.compile(r"\b(\d{2,3}-?\d{3})\b")
_LETTER_DIGIT = re.compile(r"([a-zA-Z])(\d)")
_DIGIT_LETTER = re.compile(r"(\d)([a-zA-Z])")

_LEADING_VERBS = {
    "open",
    "show",
    "find",
    "get",
    "display",
    "fetch",
    "pull",
    "look",
    "lookup",
    "search",
    "bring",
    "give",
    "please",
}

_FILLER_WORDS = {
    "the",
    "a",
    "an",
    "me",
    "my",
    "up",
    "to",
    "and",
    "then",
}

_TITLE_SYNONYMS = {
    "hw": "homework",
    "hws": "homework",
    "homework": "homework",
    "homeworks": "homework",
    "ps": "problemset",
    "pset": "problemset",
    "psets": "problemset",
    "problemset": "problemset",
    "problemsets": "problemset",
}

RESOURCE_SEARCH_TOOLS = {
    "find_course_files",
    "find_course_modules",
    "find_assignments_by_title",
    "find_course_resource",
    "find_due_dates",
}


def compact_key(value: str) -> str:
    return _COURSE_KEY_STRIP.sub("", (value or "").lower())


def normalize_title_tokens(text: str) -> list[str]:
    """Split a title into comparable tokens: HW4 → ['homework', '4']."""
    raw = (text or "").strip().lower()
    if not raw:
        return []
    raw = _LETTER_DIGIT.sub(r"\1 \2", raw)
    raw = _DIGIT_LETTER.sub(r"\1 \2", raw)
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    tokens: list[str] = []
    for part in raw.split():
        if part.isdigit():
            tokens.append(str(int(part)))
        else:
            tokens.append(part)
    expanded: list[str] = []
    index = 0
    while index < len(tokens):
        current = tokens[index]
        nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
        if current in {"problem", "problems"} and nxt in {"set", "sets"}:
            expanded.append("problemset")
            index += 2
            continue
        expanded.append(_TITLE_SYNONYMS.get(current, current))
        index += 1
    return expanded


def normalize_title_text(text: str) -> str:
    return " ".join(normalize_title_tokens(text))


def compact_title_key(text: str) -> str:
    return "".join(normalize_title_tokens(text))


def title_match_score(query: str, title: str) -> int:
    """Higher is better. 0 means no match.

    3 — same token sequence (hw 4 == Homework 4)
    2 — same compact form (hw4 == Homework 4)
    1 — every query token appears in the title
    """
    query_tokens = normalize_title_tokens(query)
    title_tokens = normalize_title_tokens(title)
    if not query_tokens or not title_tokens:
        return 0
    if query_tokens == title_tokens:
        return 3
    if "".join(query_tokens) == "".join(title_tokens):
        return 2
    if all(token in title_tokens for token in query_tokens):
        return 1
    return 0


def titles_match(query: str, title: str) -> bool:
    return title_match_score(query, title) > 0


def looks_like_course_id_query(
    query: str | None,
    *,
    course_id: int | None = None,
    course: str | None = None,
    course_code: str | None = None,
) -> bool:
    """True when ``query`` is actually a course id/code, not a resource name."""
    text = str(query or "").strip()
    if not text.isdigit() or len(text) < 4:
        return False
    candidates = [
        str(course_id) if course_id is not None else "",
        compact_key(str(course or "")),
        compact_key(str(course_code or "")),
    ]
    compact = compact_key(text)
    return any(compact and compact == compact_key(item) for item in candidates if item)


def extract_course_code(text: str) -> str | None:
    match = _COURSE_CODE_RE.search(text or "")
    return match.group(1) if match else None


def parse_resource_request(user_text: str) -> dict[str, str | None]:
    """Split a student request into the resource query and a course code.

    ``Open hw 4 for 21128`` → query ``hw 4``, course code ``21128``.
    The Canvas ``course_id`` is not inferred here; that is a later lookup.
    """
    original = (user_text or "").strip()
    course_code = extract_course_code(original)
    remainder = original
    if course_code:
        remainder = re.sub(
            rf"\b(?:for|in|from|of)\s+{re.escape(course_code)}\b",
            " ",
            remainder,
            flags=re.IGNORECASE,
        )
        remainder = re.sub(rf"\b{re.escape(course_code)}\b", " ", remainder)
    words = [word for word in re.split(r"\s+", remainder) if word]
    while words and words[0].lower().strip(".,!?") in _LEADING_VERBS | _FILLER_WORDS:
        words.pop(0)
    while words and words[-1].lower().strip(".,!?") in _FILLER_WORDS:
        words.pop()
    cleaned = " ".join(words).strip(" .,!?")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return {
        "original_user_query": original,
        "normalized_search_query": cleaned or None,
        "course_code": course_code,
    }


def search_attempt_key(endpoint: str, course_id: int | None, query: str | None) -> tuple[str, str, str]:
    return (
        str(endpoint or "").strip(),
        str(course_id) if course_id is not None else "",
        normalize_title_text(query or ""),
    )


@dataclass
class ResourceLookupSession:
    """Turn-scoped ids that must stay distinct through the tool loop."""

    original_user_query: str = ""
    normalized_search_query: str | None = None
    course_code: str | None = None
    course_id: int | None = None
    course_name: str | None = None
    attempted: set[tuple[str, str, str]] = field(default_factory=set)
    files_tab_hidden: set[int] = field(default_factory=set)
    prefer_modules: set[int] = field(default_factory=set)

    def bind_user_text(self, user_text: str) -> None:
        parsed = parse_resource_request(user_text)
        self.original_user_query = str(parsed["original_user_query"] or "")
        if parsed["normalized_search_query"]:
            self.normalized_search_query = str(parsed["normalized_search_query"])
        if parsed["course_code"]:
            self.course_code = str(parsed["course_code"])

    def remember_course(self, course: dict[str, Any] | None) -> None:
        if not isinstance(course, dict):
            return
        if course.get("course_id") not in (None, ""):
            try:
                self.course_id = int(course["course_id"])
            except (TypeError, ValueError):
                pass
        name = str(course.get("name") or "").strip()
        if name:
            self.course_name = name
        code = str(course.get("course_code") or "").strip()
        if code:
            self.course_code = self.course_code or code

    def mark_files_hidden(self, course_id: int) -> None:
        self.files_tab_hidden.add(int(course_id))
        self.prefer_modules.add(int(course_id))

    def files_are_hidden(self, course_id: int | None) -> bool:
        if course_id is None:
            return False
        return int(course_id) in self.files_tab_hidden or int(course_id) in self.prefer_modules

    def already_attempted(self, endpoint: str, course_id: int | None, query: str | None) -> bool:
        return search_attempt_key(endpoint, course_id, query) in self.attempted

    def record_attempt(self, endpoint: str, course_id: int | None, query: str | None) -> None:
        self.attempted.add(search_attempt_key(endpoint, course_id, query))

    def sanitize_query(
        self,
        query: str | None,
        *,
        course_id: int | None = None,
        course: str | None = None,
    ) -> str | None:
        """Keep the resource name. Never replace it with a course_id."""
        text = str(query).strip() if query not in (None, "") else None
        resolved_id = course_id if course_id is not None else self.course_id
        if looks_like_course_id_query(
            text,
            course_id=resolved_id,
            course=course or self.course_code,
            course_code=self.course_code,
        ):
            return self.normalized_search_query
        if text:
            if self.normalized_search_query is None:
                parsed = parse_resource_request(text)
                if parsed["normalized_search_query"]:
                    self.normalized_search_query = str(parsed["normalized_search_query"])
            return text
        return self.normalized_search_query

    def as_lookup_fields(self) -> dict[str, Any]:
        return {
            "original_user_query": self.original_user_query or None,
            "normalized_search_query": self.normalized_search_query,
            "course_code": self.course_code,
            "course_id": self.course_id,
            "course_name": self.course_name,
        }


def human_search_query(
    lookup: dict[str, Any] | None,
    *,
    user_text: str = "",
    fallback: str | None = None,
) -> str:
    """The student-facing search string, never a stuffed course_id."""
    data = lookup or {}
    course_id = data.get("course_id")
    candidates = [
        data.get("query"),
        data.get("normalized_search_query"),
        fallback,
        parse_resource_request(str(data.get("original_user_query") or user_text or "")).get(
            "normalized_search_query"
        ),
        user_text,
    ]
    for item in candidates:
        text = str(item or "").strip()
        if not text:
            continue
        if looks_like_course_id_query(text, course_id=course_id, course=str(data.get("course_code") or "")):
            continue
        return text
    return ""
