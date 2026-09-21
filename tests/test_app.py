"""Unit tests that run without any real Canvas or DeepSeek credentials.

Canvas HTTP traffic is faked at the transport adapter, so everything above it
(the read-only session, pagination, parsing) is the real code path.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402

BASE_URL = "https://canvas.example.edu"
TOKEN = "canvas-token-abcdef123456"


def make_response(payload, status=200, headers=None, url=BASE_URL) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = url
    response.headers.update(headers or {})
    response.headers.setdefault("Content-Type", "application/json")
    response.raw = io.BytesIO(json.dumps(payload).encode())
    return response


class FakeCanvas:
    """Serves canned responses in order and records the requests it saw."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def send(self, adapter, request, **kwargs):
        self.requests.append(request)
        if not self.responses:
            return make_response([], url=request.url)
        payload = self.responses.pop(0)
        if isinstance(payload, requests.Response):
            payload.url = request.url
            return payload
        return make_response(payload, url=request.url)


@pytest.fixture
def client_factory():
    def build(responses=()):
        fake = FakeCanvas(responses)
        client = app.CanvasClient(BASE_URL, TOKEN)
        patcher = mock.patch.object(
            requests.adapters.HTTPAdapter,
            "send",
            autospec=True,
            side_effect=lambda adapter, request, **kw: fake.send(adapter, request, **kw),
        )
        patcher.start()
        client._test_patcher = patcher  # type: ignore[attr-defined]
        return client, fake

    yield build
    mock.patch.stopall()


def iso_in(days: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat().replace("+00:00", "Z")


# --- read-only enforcement -------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
def test_session_blocks_non_get_methods(method):
    session = app.ReadOnlySession(BASE_URL)
    with pytest.raises(app.ReadOnlyViolation):
        session.request(method, f"{BASE_URL}/api/v1/courses")


def test_session_blocks_convenience_write_helpers():
    session = app.ReadOnlySession(BASE_URL)
    for call in (
        lambda: session.post(f"{BASE_URL}/api/v1/courses/1/assignments/2/submissions"),
        lambda: session.put(f"{BASE_URL}/api/v1/courses/1"),
        lambda: session.delete(f"{BASE_URL}/api/v1/courses/1"),
        lambda: session.patch(f"{BASE_URL}/api/v1/courses/1"),
    ):
        with pytest.raises(app.ReadOnlyViolation):
            call()


def test_session_blocks_other_hosts():
    session = app.ReadOnlySession(BASE_URL)
    with pytest.raises(app.ReadOnlyViolation):
        session.get("https://evil.example.com/api/v1/courses")


def test_session_blocks_prepared_request_sent_directly():
    session = app.ReadOnlySession(BASE_URL)
    prepared = requests.Request("POST", f"{BASE_URL}/api/v1/courses").prepare()
    with pytest.raises(app.ReadOnlyViolation):
        session.send(prepared)


def test_session_requires_https():
    with pytest.raises(app.ReadOnlyViolation):
        app.ReadOnlySession("http://canvas.example.edu")


def test_client_only_issues_gets(client_factory):
    client, fake = client_factory([[{"id": 1, "name": "Course"}]])
    client.list_courses()
    assert [request.method for request in fake.requests] == ["GET"]


def test_pagination_follows_rel_next(client_factory):
    page_one = make_response(
        [{"id": 1, "name": "A"}],
        headers={"Link": f'<{BASE_URL}/api/v1/courses?page=2>; rel="next", <{BASE_URL}/api/v1/courses?page=3>; rel="last"'},
    )
    client, fake = client_factory([page_one, [{"id": 2, "name": "B"}]])
    courses = client.list_courses()
    assert [course["course_id"] for course in courses] == [1, 2]
    assert len(fake.requests) == 2
    assert fake.requests[1].url.endswith("page=2")


def test_pagination_stops_at_max_pages(client_factory):
    looping = [
        make_response([{"id": i}], headers={"Link": f'<{BASE_URL}/api/v1/courses?page={i}>; rel="next"'})
        for i in range(1, 30)
    ]
    client, fake = client_factory(looping)
    client.max_pages = 3
    client.get_list("courses")
    assert len(fake.requests) == 3


# --- auth and secret handling ---------------------------------------------


def test_token_is_sent_as_bearer_header_only(client_factory):
    client, fake = client_factory([[]])
    client.list_courses()
    request = fake.requests[0]
    assert request.headers["Authorization"] == f"Bearer {TOKEN}"
    assert TOKEN not in request.url


def test_redact_scrubs_registered_secrets():
    app.register_secret(TOKEN)
    message = f"request failed with token {TOKEN}"
    assert TOKEN not in app.redact(message)
    assert app.REDACTION_PLACEHOLDER in app.redact(message)


def test_error_messages_do_not_leak_the_token(client_factory):
    app.register_secret(TOKEN)
    client, _ = client_factory([make_response({"errors": [{"message": "bad"}]}, status=401)])
    with pytest.raises(app.CanvasError) as excinfo:
        client.list_courses()
    assert TOKEN not in app.redact(excinfo.value)


def test_settings_report_missing_env_vars():
    settings = app.load_settings({"CANVAS_API_TOKEN": "x" * 20})
    assert settings.missing == ["DEEPSEEK_API_KEY"]
    assert settings.canvas_base_url == app.DEFAULT_CANVAS_BASE_URL
    assert settings.model == "deepseek-flash"


def test_settings_read_base_url_from_env():
    settings = app.load_settings(
        {
            "DEEPSEEK_API_KEY": "sk-test-key-123456",
            "CANVAS_API_TOKEN": "canvas-test-token-123456",
            "CANVAS_BASE_URL": "https://canvas.other.edu/",
        }
    )
    assert settings.missing == []
    assert settings.canvas_base_url == "https://canvas.other.edu"


# --- Canvas data shaping ---------------------------------------------------


def test_list_courses_normalizes_and_skips_restricted(client_factory):
    client, fake = client_factory(
        [
            [
                {"id": 7, "name": "15-213 Intro to Computer Systems", "course_code": "15213", "term": {"name": "Fall 2026"}},
                {"id": 8, "access_restricted_by_date": True},
            ]
        ]
    )
    courses = client.list_courses()
    assert courses == [
        {
            "course_id": 7,
            "name": "15-213 Intro to Computer Systems",
            "course_code": "15213",
            "term": "Fall 2026",
            "html_url": f"{BASE_URL}/courses/7",
        }
    ]
    assert "enrollment_state=active" in fake.requests[0].url


def test_list_upcoming_assignments_filters_window_and_sorts(client_factory):
    client, _ = client_factory(
        [
            [{"id": 1, "name": "Course A"}, {"id": 2, "name": "Course B"}],
            [
                {"id": 11, "name": "Later lab", "due_at": iso_in(5), "html_url": "u1"},
                {"id": 12, "name": "Way out", "due_at": iso_in(90), "html_url": "u2"},
                {"id": 13, "name": "No due date", "due_at": None},
                {"id": 14, "name": "Already past", "due_at": iso_in(-3)},
            ],
            [{"id": 21, "name": "Soon quiz", "due_at": iso_in(1), "submission": {"submitted_at": iso_in(-1)}}],
        ]
    )
    upcoming = client.list_upcoming_assignments(days_ahead=14)
    assert [item["title"] for item in upcoming] == ["Soon quiz", "Later lab"]
    assert upcoming[0]["course_name"] == "Course B"
    assert upcoming[0]["submitted"] is True
    assert upcoming[0]["due_at_local"]
    assert 0 < upcoming[0]["days_until"] < 2


def test_list_upcoming_assignments_can_target_one_course(client_factory):
    client, fake = client_factory(
        [
            [{"id": 1, "name": "Course A"}, {"id": 2, "name": "Course B"}],
            [{"id": 21, "name": "Quiz", "due_at": iso_in(2)}],
        ]
    )
    upcoming = client.list_upcoming_assignments(course_id=2)
    assert [item["course_id"] for item in upcoming] == [2]
    assert "/courses/2/assignments" in fake.requests[1].url
    assert "bucket=upcoming" in fake.requests[1].url


def test_find_due_dates_matches_titles_and_hides_past(client_factory):
    client, fake = client_factory(
        [
            [{"id": 1, "name": "Course A"}],
            [
                {"id": 31, "name": "Homework 4", "due_at": iso_in(3)},
                {"id": 32, "name": "Homework 1", "due_at": iso_in(-10)},
                {"id": 33, "name": "Reading response", "due_at": iso_in(3)},
            ],
        ]
    )
    matches = client.find_due_dates("homework")
    assert [item["title"] for item in matches] == ["Homework 4"]
    assert "search_term=homework" in fake.requests[1].url


def test_find_due_dates_rejects_short_queries(client_factory):
    client, _ = client_factory([])
    with pytest.raises(app.CanvasError):
        client.find_due_dates("a")


def test_describe_due_handles_missing_date():
    assert app.describe_due(None) == {"due_at": None, "due_at_local": None, "days_until": None}


def test_parse_canvas_timestamp_handles_z_suffix_and_garbage():
    parsed = app.parse_canvas_timestamp("2026-09-21T03:59:00Z")
    assert parsed == datetime(2026, 9, 21, 3, 59, tzinfo=timezone.utc)
    assert app.parse_canvas_timestamp("not a date") is None
    assert app.parse_canvas_timestamp(None) is None


# --- tool layer ------------------------------------------------------------


def test_tool_schemas_are_well_formed():
    assert set(app.TOOL_NAMES) == {"list_my_courses", "list_upcoming_assignments", "find_due_dates"}
    for schema in app.TOOL_SCHEMAS:
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] and function["description"]
        parameters = function["parameters"]
        assert parameters["type"] == "object"
        for required in parameters.get("required", []):
            assert required in parameters["properties"]
        for spec in parameters["properties"].values():
            assert spec["type"] in {"string", "integer", "boolean"}
            assert spec["description"]
        json.dumps(schema)


def test_dispatch_tool_returns_serializable_results(client_factory):
    client, _ = client_factory([[{"id": 1, "name": "Course A"}]])
    result = app.dispatch_tool(client, "list_my_courses", {})
    assert result["count"] == 1
    json.dumps(result)


def test_dispatch_tool_reports_unknown_tools(client_factory):
    client, _ = client_factory([])
    result = app.dispatch_tool(client, "submit_assignment", {"assignment_id": 1})
    assert "Unknown tool" in result["error"]


def test_dispatch_tool_converts_canvas_errors_to_messages(client_factory):
    app.register_secret(TOKEN)
    client, _ = client_factory([make_response({}, status=403)])
    result = app.dispatch_tool(client, "list_my_courses", {})
    assert "403" in result["error"]
    assert TOKEN not in result["error"]


# --- conversation history --------------------------------------------------


def test_store_roundtrips_messages(tmp_path):
    store = app.ConversationStore(str(tmp_path / "nested" / "history.db"))
    conversation_id = store.create_conversation("First chat")
    store.add_message(conversation_id, {"role": "user", "content": "what is due?"})
    store.add_message(
        conversation_id,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "list_my_courses", "arguments": "{}"}}],
        },
    )
    store.add_message(
        conversation_id,
        {"role": "tool", "tool_call_id": "call_1", "name": "list_my_courses", "content": "{\"count\": 0}"},
    )

    messages = store.get_messages(conversation_id)
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"][0]["id"] == "call_1"
    assert messages[2]["tool_call_id"] == "call_1"
    assert messages[2]["name"] == "list_my_courses"


def test_store_schema_tables_exist(tmp_path):
    import sqlite3

    db_path = tmp_path / "history.db"
    app.ConversationStore(str(db_path))
    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
    assert {"conversations", "messages"} <= tables
    assert {"conversation_id", "role", "content", "tool_calls", "tool_call_id", "name", "created_at"} <= columns


def test_store_lists_and_deletes_conversations(tmp_path):
    store = app.ConversationStore(str(tmp_path / "history.db"))
    first = store.create_conversation("one")
    second = store.create_conversation("two")
    store.set_title(second, "renamed")
    titles = [c["title"] for c in store.list_conversations()]
    assert "renamed" in titles and "one" in titles

    store.delete_conversation(first)
    assert [c["id"] for c in store.list_conversations()] == [second]
    assert store.get_messages(first) == []


# --- agent loop ------------------------------------------------------------


class ScriptedDeepSeek:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, tools=None, temperature=0.2):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self.replies.pop(0)


def test_agent_turn_executes_tool_calls_then_answers(client_factory):
    client, _ = client_factory([[{"id": 1, "name": "Course A"}]])
    deepseek = ScriptedDeepSeek(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "list_my_courses", "arguments": "{}"}}
                ],
            },
            {"role": "assistant", "content": "You are enrolled in Course A."},
        ]
    )
    produced = app.run_agent_turn(deepseek, client, [{"role": "user", "content": "my courses?"}])

    assert [m["role"] for m in produced] == ["assistant", "tool", "assistant"]
    assert json.loads(produced[1]["content"])["count"] == 1
    assert produced[-1]["content"] == "You are enrolled in Course A."
    assert deepseek.calls[0]["messages"][0]["role"] == "system"
    assert deepseek.calls[0]["tools"] == app.TOOL_SCHEMAS


def test_agent_turn_stops_after_max_tool_rounds(client_factory):
    client, _ = client_factory([[] for _ in range(10)])
    loop_reply = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_x", "type": "function", "function": {"name": "list_my_courses", "arguments": "{}"}}
        ],
    }
    deepseek = ScriptedDeepSeek([dict(loop_reply) for _ in range(5)])
    produced = app.run_agent_turn(deepseek, client, [{"role": "user", "content": "loop"}], max_tool_rounds=2)
    assert len(deepseek.calls) == 2
    assert "could not settle" in produced[-1]["content"]


def test_agent_turn_survives_malformed_tool_arguments(client_factory):
    client, _ = client_factory([[{"id": 1, "name": "Course A"}]])
    deepseek = ScriptedDeepSeek(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "list_my_courses", "arguments": "{not json"}}
                ],
            },
            {"role": "assistant", "content": "done"},
        ]
    )
    produced = app.run_agent_turn(deepseek, client, [{"role": "user", "content": "hi"}])
    assert json.loads(produced[1]["content"])["count"] == 1


def test_system_prompt_states_read_only_and_date():
    prompt = app.build_system_prompt(datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc))
    assert "read-only" in prompt
    assert "September 21, 2026" in prompt
