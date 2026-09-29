"""Query vs course_id stay separate; hidden Files uses modules; flexible titles."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402
import resource_lookup  # noqa: E402
from test_app import (  # noqa: E402
    ASSIGNMENT_ROW,
    BASE_URL,
    FILE_ROW,
    TOKEN,
    FakeCanvas,
    ScriptedDeepSeek,
    make_binary_response,
    make_response,
)


COURSE_21128 = {
    "id": 55658,
    "name": "Mathematical Concepts and Proofs",
    "course_code": "21-128",
}


@pytest.fixture
def client_factory():
    def build(responses=(), router=None):
        fake = router or FakeCanvas(responses)
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


class PathRouter:
    """Serve Canvas payloads by URL path instead of call order."""

    def __init__(self, handlers: dict):
        self.handlers = handlers
        self.requests = []
        self.files_tab_hits = 0

    def _payload_for(self, path: str):
        if "/download" in path and "/download" in self.handlers:
            return self.handlers["/download"]
        if path.endswith("/files") and "/courses/" in path:
            self.files_tab_hits += 1
            if "/files" in self.handlers:
                return self.handlers["/files"]
            return make_response({"status": "unauthorized"}, status=401)
        ordered = sorted(self.handlers.items(), key=lambda item: len(item[0]), reverse=True)
        for key, payload in ordered:
            if key in {"/files", "/api/v1/courses", "/download"}:
                continue
            if key in path:
                return payload
        if path.rstrip("/").endswith("/courses"):
            return self.handlers.get("/api/v1/courses")
        return []

    def send(self, adapter, request, **kwargs):
        self.requests.append(request)
        path = urlparse(request.url).path
        body = self._payload_for(path)
        body = body() if callable(body) else body
        if isinstance(body, requests.Response):
            body.url = request.url
            return body
        return make_response(body, url=request.url)


def _paths(fake) -> list[str]:
    return [urlparse(request.url).path for request in fake.requests]


def test_21128_resolves_to_course_id_55658(client_factory):
    client, _ = client_factory([[COURSE_21128]])
    matched = client._course_lookup(course="21128")
    assert len(matched) == 1
    assert matched[0]["course_id"] == 55658
    assert matched[0]["name"] == "Mathematical Concepts and Proofs"
    assert client.lookup_session.course_id == 55658


def test_parse_keeps_hw4_query_separate_from_course_code():
    parsed = app.parse_resource_request("Open hw 4 for 21128")
    assert parsed["course_code"] == "21128"
    assert parsed["normalized_search_query"] == "hw 4"
    assert "55658" not in parsed["normalized_search_query"]
    assert parsed["original_user_query"] == "Open hw 4 for 21128"


@pytest.mark.parametrize(
    "query,title",
    [
        ("hw 4", "Homework 4"),
        ("hw4", "Homework 4"),
        ("homework 4", "HW4"),
        ("homework 04", "Homework 4"),
        ("ps4", "Problem Set 4"),
        ("hw 4", "HW4"),
        ("problem set 4", "Problem Set 4"),
    ],
)
def test_hw4_flexible_title_match(query, title):
    assert app.titles_match(query, title)


def test_problem_set_aliases_match_ps4():
    assert app.titles_match("problem set 4", "Problem Set 4")
    assert app.titles_match("ps4", "PS4")
    assert app.titles_match("hw 4", "HW4")
    assert app.titles_match("homework 04", "Homework 4")
    assert not app.titles_match("hw 4", "Homework 5")


def test_stuffed_course_id_query_is_replaced_with_hw4(client_factory):
    router = PathRouter(
        {
            "/api/v1/courses": [COURSE_21128],
            "/assignments": [],
            "/modules": [],
            "/files": make_response({"status": "unauthorized"}, status=401),
        }
    )
    client, fake = client_factory(router=router)
    client.reset_lookup_turn("Open hw 4 for 21128")
    result = app.dispatch_tool(
        client,
        "find_course_files",
        {"query": "55658", "course_id": 55658},
    )
    lookup = result["lookup"]
    assert lookup["query"] == "hw 4"
    assert lookup["course_id"] == 55658
    assert "55658" not in str(lookup["query"])
    text = app.compose_lookup_reply([lookup], user_text="Open hw 4 for 21128")
    assert "hw 4" in text
    assert "55658" not in text


def test_find_course_resource_keeps_query_and_course_id_separate(client_factory):
    router = PathRouter(
        {
            "/api/v1/courses": [COURSE_21128],
            "/assignments": [dict(ASSIGNMENT_ROW, id=3100, name="Homework 4")],
            "/assignments/3100": dict(ASSIGNMENT_ROW, id=3100, name="Homework 4"),
        }
    )
    client, fake = client_factory(router=router)
    client.reset_lookup_turn("Open hw 4 for 21128")
    result = app.dispatch_tool(
        client,
        "find_course_resource",
        {"query": "hw 4", "course": "21128"},
    )
    lookup = result["lookup"]
    assert lookup["query"] == "hw 4"
    assert lookup["course_id"] == 55658
    assert lookup["course_name"] == "Mathematical Concepts and Proofs"
    assert result["assignment"]["title"] == "Homework 4"
    assert not any(path.endswith("/files") and "/courses/" in path for path in _paths(fake))


def test_hidden_files_tab_prioritizes_modules_and_does_not_hammer_files(client_factory):
    modules = [
        {
            "id": 9,
            "name": "Week 4",
            "items": [
                {"type": "File", "title": "Homework 4", "content_id": 9001},
            ],
        }
    ]
    router = PathRouter(
        {
            "/api/v1/courses": [COURSE_21128],
            "/assignments": [],
            "/modules": modules,
            "/files/9001": dict(FILE_ROW, id=9001, display_name="Homework 4.pdf"),
            "/download": make_binary_response(b"HW4 body", "text/plain"),
        }
    )
    client, fake = client_factory(router=router)
    client.reset_lookup_turn("Open hw 4 for 21128")
    result = app.dispatch_tool(
        client, "find_course_resource", {"query": "hw 4", "course_id": 55658}
    )
    assert result["lookup"]["query"] == "hw 4"
    assert result["file"]["file_id"] == 9001
    assert fake.files_tab_hits == 0
    app.dispatch_tool(client, "find_course_files", {"query": "hw 4", "course_id": 55658})
    app.dispatch_tool(client, "find_course_files", {"query": "hw 4", "course_id": 55658})
    assert fake.files_tab_hits <= 1


def test_open_file_module_item(client_factory):
    router = PathRouter(
        {
            "/api/v1/courses": [COURSE_21128],
            "/assignments": [],
            "/modules": [
                {
                    "id": 1,
                    "name": "Handouts",
                    "items": [{"type": "File", "title": "HW4", "content_id": 9001}],
                }
            ],
            "/files/9001": dict(FILE_ROW, id=9001, display_name="HW4.pdf", **{"content-type": "text/plain"}),
            "/download": make_binary_response(b"problems", "text/plain"),
        }
    )
    client, _ = client_factory(router=router)
    result = app.dispatch_tool(
        client, "find_course_resource", {"query": "hw 4", "course": "21128"}
    )
    assert result["file"]["file_id"] == 9001
    assert result["selected_resource"]["type"] == "File"
    assert result["lookup"]["outcome"] == app.LOOKUP_OUTCOME_SUCCESSFUL


def test_open_assignment_module_item(client_factory):
    router = PathRouter(
        {
            "/api/v1/courses": [COURSE_21128],
            "/assignments": [{"id": 99, "name": " unrelated quiz"}],
            "/modules": [
                {
                    "id": 2,
                    "name": "Work",
                    "items": [
                        {"type": "Assignment", "title": "Homework 4", "content_id": 3100},
                    ],
                }
            ],
            "/assignments/3100": dict(ASSIGNMENT_ROW, id=3100, name="Homework 4", description="<p>Prove it.</p>"),
        }
    )
    client, fake = client_factory(router=router)
    result = app.dispatch_tool(
        client, "find_course_resource", {"query": "hw 4", "course_id": 55658}
    )
    assert result["assignment"]["title"] == "Homework 4"
    assert result["selected_resource"]["type"] == "Assignment"
    assert any("/assignments/3100" in urlparse(req.url).path for req in fake.requests)


def test_no_repeated_identical_searches(client_factory):
    client, fake = client_factory(
        [
            [COURSE_21128],
            [],  # files tab empty
            [],  # modules empty
        ]
    )
    client.reset_lookup_turn("Open hw 4 for 21128")
    first = app.dispatch_tool(
        client, "find_course_files", {"query": "hw 4", "course_id": 55658}
    )
    before = len(fake.requests)
    second = app.dispatch_tool(
        client, "find_course_files", {"query": "hw 4", "course_id": 55658}
    )
    assert second.get("skipped") is True
    assert len(fake.requests) == before
    assert first["lookup"]["query"] == "hw 4"
    assert second["lookup"]["query"] == "hw 4"


def test_error_messages_use_search_query_not_course_id(client_factory):
    router = PathRouter(
        {
            "/api/v1/courses": [COURSE_21128],
            "/assignments": [],
            "/modules": [],
            "/files": [],
        }
    )
    client, _ = client_factory(router=router)
    client.reset_lookup_turn("Open hw 4 for 21128")
    result = app.dispatch_tool(
        client,
        "find_course_resource",
        {"query": "55658", "course_id": 55658},
    )
    lookup = result["lookup"]
    assert lookup["query"] == "hw 4"
    assert lookup["course_id"] == 55658
    assert lookup["outcome"] == app.LOOKUP_OUTCOME_NO_MATCHING_RESOURCE
    text = app.compose_lookup_reply([lookup], user_text="Open hw 4 for 21128")
    assert "hw 4" in text
    assert "Mathematical Concepts and Proofs" in text
    assert "55658" not in text
    assert "I couldn't find a Canvas resource matching 'hw 4' in Mathematical Concepts and Proofs." in text


def test_agent_loop_does_not_search_files_for_course_id(client_factory):
    router = PathRouter(
        {
            "/api/v1/courses": [COURSE_21128],
            "/assignments": [dict(ASSIGNMENT_ROW, id=3100, name="Homework 4")],
            "/assignments/3100": dict(ASSIGNMENT_ROW, id=3100, name="Homework 4", description="Solve 4."),
            "/modules": [],
            "/files": make_response({"status": "unauthorized"}, status=401),
        }
    )
    client, fake = client_factory(router=router)
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
                            "name": "list_my_courses",
                            "arguments": json.dumps({"course": "21128"}),
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "find_course_files",
                            "arguments": json.dumps({"query": "55658", "course_id": 55658}),
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
        [{"role": "user", "content": "Open hw 4 for 21128"}],
    )
    file_search = json.loads(produced[3]["content"])
    assert file_search["lookup"]["query"] == "hw 4"
    assert "55658" not in file_search["lookup"]["query"]
    reply = produced[-1]["content"]
    assert "55658" not in reply


def test_find_course_files_skips_files_tab_after_hidden(client_factory):
    router = PathRouter(
        {
            "/api/v1/courses": [COURSE_21128],
            "/modules": [
                {
                    "id": 1,
                    "name": "Week 1",
                    "items": [{"type": "File", "title": "Homework 4", "content_id": 9001}],
                }
            ],
            "/files/9001": dict(FILE_ROW, id=9001, display_name="Homework 4.pdf"),
            "/files": make_response({"status": "unauthorized"}, status=401),
        }
    )
    client, fake = client_factory(router=router)
    client.reset_lookup_turn("open homework 4")
    first = app.dispatch_tool(client, "find_course_files", {"query": "hw 4", "course_id": 55658})
    assert first["count"] == 1
    assert fake.files_tab_hits == 1
    second = app.dispatch_tool(client, "find_course_files", {"query": "lecture 5", "course_id": 55658})
    assert second["count"] == 0
    assert fake.files_tab_hits == 1


def test_looks_like_course_id_query_rejects_55658_not_hw4():
    assert resource_lookup.looks_like_course_id_query("55658", course_id=55658)
    assert not resource_lookup.looks_like_course_id_query("hw 4", course_id=55658)
    assert not resource_lookup.looks_like_course_id_query("4", course_id=55658)
