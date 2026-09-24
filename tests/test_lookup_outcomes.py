"""Structured Canvas lookup outcomes and the deterministic empty-model composer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402
from test_app import (  # noqa: E402
    BASE_URL,
    FILE_ROW,
    TOKEN,
    FakeCanvas,
    ScriptedDeepSeek,
    iso_in,
    make_binary_response,
    make_response,
)


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


GENERIC = "Here's what I found. Ask if you want more detail"


def _lookup(result: dict) -> dict:
    assert "lookup" in result
    lookup = result["lookup"]
    assert lookup["outcome"] in app.LOOKUP_OUTCOMES
    assert lookup["tool"]
    assert lookup["endpoint"]
    dumped = json.dumps(result)
    assert TOKEN not in dumped
    assert "Traceback" not in dumped
    assert 'File "' not in dumped
    return lookup


def test_lookup_outcomes_are_the_agreed_names():
    assert app.LOOKUP_OUTCOMES == (
        "course_not_found",
        "course_ambiguous",
        "no_matching_resource",
        "resource_found_but_could_not_open",
        "resource_opened_without_readable_text",
        "canvas_request_failed",
        "successful",
    )


@pytest.mark.parametrize(
    "outcome",
    app.LOOKUP_OUTCOMES,
)
def test_compose_lookup_reply_covers_every_outcome(outcome):
    lookup = app.build_lookup(
        outcome=outcome,
        tool="find_course_files",
        course="Putnam",
        query="p4 putnam handut",
        result_count=2 if outcome in {"successful", "course_ambiguous"} else 0,
        selected_resource_title="Putnam Problem Set 4.pdf",
        opened=outcome == "successful",
        readable_text=outcome == "successful",
        matching_courses=["21-128 Putnam", "21-301 Putnam"],
        suggested_alternatives=["Putnam Problem Set 4", "Putnam handout"],
        resource_types_searched=["files", "modules"],
    )
    text = app.compose_lookup_reply([lookup], user_text="show p4 putnam handut")
    assert text
    assert GENERIC not in text
    assert "Here's what I found" not in text
    if outcome != "successful":
        assert lookup["error_category"] == outcome
    else:
        assert lookup["error_category"] is None


def test_compose_course_not_found_quotes_query_and_suggests_without_claiming_search():
    lookup = app.build_lookup(
        outcome=app.LOOKUP_OUTCOME_COURSE_NOT_FOUND,
        tool="list_my_courses",
        course="Putnamm",
        query="Putnamm",
        suggested_alternatives=["21-128 Putnam Seminar"],
        resource_types_searched=["courses"],
    )
    text = app.compose_lookup_reply([lookup])
    assert "Putnamm" in text
    assert "21-128 Putnam Seminar" in text
    assert "searched" not in text.lower() or "You may want to try" in text
    assert "You may want to try" in text
    assert GENERIC not in text


def test_compose_course_ambiguous_asks_the_student_to_choose():
    lookup = app.build_lookup(
        outcome=app.LOOKUP_OUTCOME_COURSE_AMBIGUOUS,
        tool="list_my_courses",
        course="Putnam",
        query="Putnam",
        result_count=2,
        matching_courses=["21-128 Putnam Seminar", "21-301 Combinatorics"],
        resource_types_searched=["courses"],
    )
    text = app.compose_lookup_reply([lookup])
    assert "Several courses match 'Putnam'" in text
    assert "21-128 Putnam Seminar" in text
    assert "Which one did you mean?" in text


def test_compose_no_matching_resource_lists_types_actually_searched():
    lookup = app.build_lookup(
        outcome=app.LOOKUP_OUTCOME_NO_MATCHING_RESOURCE,
        tool="find_course_files",
        course="Putnam",
        query="p4 putnam handut",
        resource_types_searched=["files", "modules"],
        suggested_alternatives=["Putnam Problem Set 4", "Putnam handout"],
    )
    text = app.compose_lookup_reply([lookup], user_text="show p4 putnam handut")
    assert "p4 putnam handut" in text
    assert "files, modules" in text
    assert "pages" not in text
    assert "You may want to try" in text
    assert "Putnam Problem Set 4" in text
    assert GENERIC not in text


def test_dispatch_list_my_courses_successful(client_factory):
    client, _ = client_factory([[{"id": 1, "name": "Course A", "course_code": "21-128"}]])
    result = app.dispatch_tool(client, "list_my_courses", {})
    lookup = _lookup(result)
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_SUCCESSFUL
    assert lookup["tool"] == "list_my_courses"
    assert lookup["endpoint"] == "GET /api/v1/courses"
    assert lookup["result_count"] == 1
    assert lookup["selected_resource_title"] == "Course A"
    assert lookup["error_category"] is None
    assert result["count"] == 1


def test_dispatch_list_my_courses_course_not_found(client_factory):
    client, _ = client_factory(
        [[{"id": 1, "name": "21-128 Putnam Seminar", "course_code": "21-128"}]]
    )
    result = app.dispatch_tool(client, "list_my_courses", {"course": "Calc 3"})
    lookup = _lookup(result)
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_COURSE_NOT_FOUND
    assert lookup["course"] == "Calc 3"
    assert lookup["query"] == "Calc 3"
    assert result["count"] == 0
    assert lookup["error_category"] == "course_not_found"
    text = app.compose_lookup_reply([lookup])
    assert "Calc 3" in text
    assert GENERIC not in text


def test_dispatch_list_my_courses_course_ambiguous(client_factory):
    client, _ = client_factory(
        [
            [
                {"id": 1, "name": "21-128 Putnam Seminar", "course_code": "21-128"},
                {"id": 2, "name": "21-301 Putnam Combinatorics", "course_code": "21-301"},
            ]
        ]
    )
    result = app.dispatch_tool(client, "list_my_courses", {"course": "Putnam"})
    lookup = _lookup(result)
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_COURSE_AMBIGUOUS
    assert lookup["result_count"] == 2
    assert "21-128 Putnam Seminar" in lookup["matching_courses"]
    assert "21-301 Putnam Combinatorics" in lookup["matching_courses"]
    text = app.compose_lookup_reply([lookup])
    assert "Which one did you mean?" in text
    assert GENERIC not in text


def test_dispatch_find_course_files_no_matching_resource(client_factory):
    client, _ = client_factory(
        [
            [{"id": 1, "name": "21-128 Putnam Seminar"}],
            [FILE_ROW],
        ]
    )
    result = app.dispatch_tool(
        client, "find_course_files", {"query": "p4 putnam handut", "course": "Putnam"}
    )
    lookup = _lookup(result)
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_NO_MATCHING_RESOURCE
    assert lookup["query"] == "p4 putnam handut"
    assert lookup["course"] == "Putnam"
    assert lookup["result_count"] == 0
    assert lookup["resource_types_searched"] == ["files", "modules"]
    text = app.compose_lookup_reply([lookup])
    assert "p4 putnam handut" in text
    assert "files, modules" in text
    assert GENERIC not in text


def test_dispatch_find_course_files_course_not_found(client_factory):
    client, fake = client_factory(
        [[{"id": 1, "name": "21-128 Putnam Seminar", "course_code": "21-128"}]]
    )
    result = app.dispatch_tool(
        client, "find_course_files", {"query": "handout", "course": "Underwater Basketweaving"}
    )
    lookup = _lookup(result)
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_COURSE_NOT_FOUND
    assert lookup["course"] == "Underwater Basketweaving"
    assert "error" in result
    assert fake.requests  # listed courses, did not invent files


def test_dispatch_find_course_files_course_ambiguous(client_factory):
    client, _ = client_factory(
        [
            [
                {"id": 1, "name": "21-128 Putnam Seminar"},
                {"id": 2, "name": "21-301 Putnam Combinatorics"},
            ]
        ]
    )
    result = app.dispatch_tool(
        client, "find_course_files", {"query": "handout", "course": "Putnam"}
    )
    lookup = _lookup(result)
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_COURSE_AMBIGUOUS
    assert len(lookup["matching_courses"]) == 2


def test_dispatch_open_file_resource_found_but_could_not_open(client_factory):
    client, _ = client_factory([dict(FILE_ROW, locked_for_user=True)])
    result = app.dispatch_tool(client, "open_file", {"file_id": 9001})
    lookup = _lookup(result)
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_RESOURCE_FOUND_BUT_COULD_NOT_OPEN
    assert lookup["opened"] is False
    assert "error" in result
    assert "locked" in result["error"].lower()
    assert lookup["selected_resource_title"] == "15213-syllabus.pdf"
    text = app.compose_lookup_reply([lookup])
    assert "15213-syllabus.pdf" in text
    assert "could not open" in text
    assert GENERIC not in text


def test_dispatch_open_url_without_readable_text(client_factory, monkeypatch):
    monkeypatch.setattr(
        app.socket,
        "getaddrinfo",
        lambda *a, **k: [(app.socket.AF_INET, app.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
    )
    client, _ = client_factory([make_binary_response(b"PK\x03\x04binaryzip", "application/zip")])
    result = app.dispatch_tool(client, "open_url", {"url": "https://cs.example.edu/slides.zip"})
    lookup = _lookup(result)
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_RESOURCE_OPENED_WITHOUT_READABLE_TEXT
    assert lookup["opened"] is True
    assert lookup["readable_text"] is False
    assert "text_excerpt" not in result
    text = app.compose_lookup_reply([lookup])
    assert "readable text" in text
    assert GENERIC not in text


def test_dispatch_list_my_courses_canvas_request_failed(client_factory):
    app.register_secret(TOKEN)
    client, _ = client_factory([make_response({}, status=403)])
    result = app.dispatch_tool(client, "list_my_courses", {})
    lookup = _lookup(result)
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_CANVAS_REQUEST_FAILED
    assert "403" in result["error"]
    assert TOKEN not in result["error"]
    assert TOKEN not in json.dumps(lookup)
    text = app.compose_lookup_reply([lookup])
    assert "Canvas could not be reached or returned an error" in text
    assert "does not exist" not in text
    assert GENERIC not in text


def test_dispatch_open_file_successful(client_factory):
    text_file = dict(FILE_ROW, id=9002, display_name="notes.txt", **{"content-type": "text/plain"})
    client, _ = client_factory(
        [text_file, make_binary_response(b"Homework 4 is due Friday.", "text/plain")]
    )
    result = app.dispatch_tool(client, "open_file", {"file_id": 9002})
    lookup = _lookup(result)
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_SUCCESSFUL
    assert lookup["opened"] is True
    assert lookup["readable_text"] is True
    assert lookup["selected_resource_title"] == "notes.txt"
    assert lookup["tool"] == "open_file"
    assert lookup["endpoint"] == "GET /api/v1/files/{file_id}"
    text = app.compose_lookup_reply([lookup])
    assert "notes.txt" in text
    assert GENERIC not in text


def test_dispatch_find_due_dates_successful_includes_lookup(client_factory):
    client, _ = client_factory(
        [
            [{"id": 1, "name": "Course A"}],
            [{"id": 31, "name": "Homework 4", "due_at": iso_in(3)}],
        ]
    )
    result = app.dispatch_tool(client, "find_due_dates", {"query": "4"})
    lookup = _lookup(result)
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_SUCCESSFUL
    assert lookup["query"] == "4"
    assert lookup["result_count"] == 1
    assert lookup["selected_resource_title"] == "Homework 4"
    assert lookup["resource_types_searched"] == ["assignments"]


def test_dispatch_get_assignment_details_no_matching_resource(client_factory):
    client, _ = client_factory([[{"id": 1, "name": "Course"}], make_response({}, status=404)])
    result = app.dispatch_tool(client, "get_assignment_details", {"assignment_id": 999, "course_id": 1})
    lookup = _lookup(result)
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_NO_MATCHING_RESOURCE
    assert "error" in result


def test_generic_fallback_is_not_used_when_a_specific_outcome_exists(client_factory):
    client, _ = client_factory([[{"id": 1, "name": "Course A"}]])
    deepseek = ScriptedDeepSeek(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "list_my_courses", "arguments": "{}"},
                    }
                ],
            },
            {"role": "assistant", "content": ""},
        ]
    )
    produced = app.run_agent_turn(
        deepseek, client, [{"role": "user", "content": "my courses?"}]
    )
    payload = json.loads(produced[1]["content"])
    lookup = payload["lookup"]
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_SUCCESSFUL
    assert produced[-1]["content"] == app.compose_lookup_reply([lookup])
    assert GENERIC not in produced[-1]["content"]
    assert produced[-1]["content"] != app.GENERIC_LOOKUP_FALLBACK
    assert produced[-1]["content"] != app.MAX_TOOL_ROUNDS_WITH_EVIDENCE


def test_empty_model_uses_no_matching_resource_composer(client_factory):
    client, _ = client_factory([[{"id": 1, "name": "21-128 Putnam Seminar"}], [FILE_ROW]])
    deepseek = ScriptedDeepSeek(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "find_course_files",
                            "arguments": json.dumps({"query": "p4 putnam handut", "course": "Putnam"}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": ""},
        ]
    )
    produced = app.run_agent_turn(
        deepseek,
        client,
        [{"role": "user", "content": "show p4 putnam handut"}],
    )
    lookup = json.loads(produced[1]["content"])["lookup"]
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_NO_MATCHING_RESOURCE
    reply = produced[-1]["content"]
    assert reply == app.compose_lookup_reply([lookup], user_text="show p4 putnam handut")
    assert "p4 putnam handut" in reply
    assert GENERIC not in reply


def test_empty_model_uses_canvas_failure_composer(client_factory):
    app.register_secret(TOKEN)
    client, _ = client_factory([make_response({}, status=503)])
    deepseek = ScriptedDeepSeek(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "list_my_courses", "arguments": "{}"},
                    }
                ],
            },
            {"role": "assistant", "content": ""},
        ]
    )
    produced = app.run_agent_turn(
        deepseek, client, [{"role": "user", "content": "what courses do I have?"}]
    )
    lookup = json.loads(produced[1]["content"])["lookup"]
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_CANVAS_REQUEST_FAILED
    reply = produced[-1]["content"]
    assert "Canvas could not be reached or returned an error" in reply
    assert GENERIC not in reply
    assert TOKEN not in reply


def test_every_canvas_tool_attaches_lookup(client_factory):
    """Smoke: each tool name gets a lookup object, including unknown tools."""
    client, _ = client_factory([[{"id": 1, "name": "Course A"}]])
    listed = app.dispatch_tool(client, "list_my_courses", {})
    assert _lookup(listed)["outcome"] == app.LOOKUP_OUTCOME_SUCCESSFUL

    unknown = app.dispatch_tool(client, "submit_assignment", {"assignment_id": 1})
    lookup = _lookup(unknown)
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_CANVAS_REQUEST_FAILED
    assert "Unknown tool" in unknown["error"]


def test_query_wording_suggestions_do_not_claim_search():
    suggestions = app.query_wording_suggestions("p4 putnam handut")
    assert suggestions
    assert "Problem Set 4" in suggestions[0]
    assert "handout" in suggestions[0]
    merged = app.merge_suggested_alternatives(suggestions, ["Putnam handout"])
    assert len(merged) <= 2
