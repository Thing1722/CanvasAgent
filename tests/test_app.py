"""Unit tests that run without any real Canvas or DeepSeek credentials.

Canvas HTTP traffic is faked at the transport adapter, so everything above it
(the read-only session, pagination, parsing) is the real code path.
"""

from __future__ import annotations

import io
import json
import socket
import os
import sys
import time
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


def make_binary_response(content: bytes, content_type: str, status=200) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.raw = io.BytesIO(content)
    response.headers["Content-Type"] = content_type
    return response


def make_pdf(text: str) -> bytes:
    """Smallest valid PDF that carries a line of extractable text."""
    stream = b"BT /F1 12 Tf 20 100 Td (" + text.encode("ascii") + b") Tj ET"
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length %d>>stream\n" % len(stream) + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % index + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


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


def test_download_session_still_refuses_non_get():
    session = app.ReadOnlySession(BASE_URL, allow_offsite_redirects=True)
    with pytest.raises(app.ReadOnlyViolation):
        session.post(f"{BASE_URL}/api/v1/files/1")
    prepared = requests.Request("DELETE", "https://files.example.com/x").prepare()
    with pytest.raises(app.ReadOnlyViolation):
        session.send(prepared)


def test_download_session_must_start_at_canvas():
    session = app.ReadOnlySession(BASE_URL, allow_offsite_redirects=True)
    with pytest.raises(app.ReadOnlyViolation):
        session.get("https://files.example.com/leaked")


def test_download_session_strips_auth_on_offsite_redirect():
    session = app.ReadOnlySession(BASE_URL, allow_offsite_redirects=True)
    prepared = requests.Request(
        "GET", "https://storage.example.com/file.pdf", headers={"Authorization": f"Bearer {TOKEN}"}
    ).prepare()
    sent = {}

    def capture(self, request, **kwargs):
        sent["headers"] = dict(request.headers)
        return make_response([], url=request.url)

    with mock.patch.object(requests.Session, "send", autospec=True, side_effect=capture):
        session.send(prepared)
    assert "Authorization" not in sent["headers"]


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
    assert settings.timezone == "America/New_York"


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


def test_settings_read_timezone_from_env():
    settings = app.load_settings(
        {
            "DEEPSEEK_API_KEY": "sk-test-key-123456",
            "CANVAS_API_TOKEN": "canvas-test-token-123456",
            "CANVAS_ASSISTANT_TZ": "America/Los_Angeles",
        }
    )
    assert settings.timezone == "America/Los_Angeles"


def test_settings_invalid_timezone_falls_back_to_pittsburgh():
    settings = app.load_settings(
        {
            "DEEPSEEK_API_KEY": "sk-test-key-123456",
            "CANVAS_API_TOKEN": "canvas-test-token-123456",
            "CANVAS_ASSISTANT_TZ": "Not/A_Zone",
        }
    )
    assert settings.timezone == "America/New_York"


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


def test_find_due_dates_matches_a_single_character_query(client_factory):
    """The model often searches for '4' when asked about Homework 4."""
    client, fake = client_factory(
        [
            [{"id": 1, "name": "Course A"}],
            [
                {"id": 31, "name": "Homework 4", "due_at": iso_in(3)},
                {"id": 32, "name": "Homework 1", "due_at": iso_in(4)},
            ],
        ]
    )
    matches = client.find_due_dates("4")
    assert [item["title"] for item in matches] == ["Homework 4"]
    assert "search_term=" not in fake.requests[1].url


def test_find_due_dates_rejects_empty_queries(client_factory):
    client, _ = client_factory([])
    with pytest.raises(app.CanvasError, match="required"):
        client.find_due_dates("")
    with pytest.raises(app.CanvasError, match="required"):
        client.find_due_dates("   ")


def test_describe_due_handles_missing_date():
    assert app.describe_due(None) == {"due_at": None, "due_at_local": None, "days_until": None}


def test_parse_canvas_timestamp_handles_z_suffix_and_garbage():
    parsed = app.parse_canvas_timestamp("2026-09-21T03:59:00Z")
    assert parsed == datetime(2026, 9, 21, 3, 59, tzinfo=timezone.utc)
    assert app.parse_canvas_timestamp("not a date") is None
    assert app.parse_canvas_timestamp(None) is None


# --- student timezone (Pittsburgh default, never machine local) ------------

# Canvas 11:59 PM Eastern on Oct 8 2026 is 03:59 UTC on Oct 9.
PITTSBURGH_DUE = datetime(2026, 10, 9, 3, 59, tzinfo=timezone.utc)


def _force_machine_tz(name: str) -> str | None:
    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    if hasattr(time, "tzset"):
        time.tzset()
    return previous


def _restore_machine_tz(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = previous
    if hasattr(time, "tzset"):
        time.tzset()


@pytest.fixture
def machine_tz_shanghai():
    previous = _force_machine_tz("Asia/Shanghai")
    try:
        yield
    finally:
        _restore_machine_tz(previous)


def test_default_timezone_is_pittsburgh_eastern():
    assert app.DEFAULT_TIMEZONE == "America/New_York"
    assert app.resolve_timezone_name(None) == "America/New_York"
    assert app.resolve_timezone_name("") == "America/New_York"
    assert app.as_zoneinfo(None).key == "America/New_York"


def test_describe_due_defaults_to_pittsburgh_not_utc_offset():
    described = app.describe_due(PITTSBURGH_DUE)
    assert described["due_at"] == "2026-10-09T03:59:00Z"
    assert described["due_at_local"] == "Thu Oct 08, 2026 11:59 PM EDT"


def test_describe_due_follows_eastern_dst():
    winter = datetime(2026, 1, 15, 16, 59, tzinfo=timezone.utc)
    summer = datetime(2026, 7, 15, 15, 59, tzinfo=timezone.utc)
    assert "EST" in app.describe_due(winter)["due_at_local"]
    assert "EDT" in app.describe_due(summer)["due_at_local"]
    assert "11:59 AM" in app.describe_due(winter)["due_at_local"]
    assert "11:59 AM" in app.describe_due(summer)["due_at_local"]


def test_describe_due_switching_zone_changes_local_clock():
    pittsburgh = app.describe_due(PITTSBURGH_DUE, tz="America/New_York")
    beijing = app.describe_due(PITTSBURGH_DUE, tz="Asia/Shanghai")
    pacific = app.describe_due(PITTSBURGH_DUE, tz="America/Los_Angeles")
    assert pittsburgh["due_at"] == beijing["due_at"] == pacific["due_at"] == "2026-10-09T03:59:00Z"
    assert pittsburgh["due_at_local"] == "Thu Oct 08, 2026 11:59 PM EDT"
    assert beijing["due_at_local"] == "Fri Oct 09, 2026 11:59 AM CST"
    assert "PDT" in pacific["due_at_local"] or "PST" in pacific["due_at_local"]
    assert pacific["due_at_local"] != pittsburgh["due_at_local"]


def test_describe_due_ignores_machine_timezone_when_set_to_shanghai(machine_tz_shanghai):
    machine_local = PITTSBURGH_DUE.astimezone().strftime("%a %b %d, %Y %I:%M %p %Z").strip()
    assert "CST" in machine_local or "GMT+8" in machine_local or "Oct 09" in machine_local

    described = app.describe_due(PITTSBURGH_DUE)
    assert described["due_at_local"] == "Thu Oct 08, 2026 11:59 PM EDT"
    assert described["due_at_local"] != machine_local
    assert "CST" not in described["due_at_local"]
    prompt = app.build_system_prompt(PITTSBURGH_DUE)
    assert "America/New_York" in prompt
    assert "EDT" in prompt
    assert "Asia/Shanghai" not in prompt
    assert "CST" not in prompt


def test_days_until_is_relative_to_now_in_the_student_zone():
    # Naive "now" is interpreted in the student zone, so the same clock reading
    # is a different instant in Pittsburgh vs Beijing.
    noon = datetime(2026, 10, 8, 12, 0)
    pittsburgh = app.describe_due(PITTSBURGH_DUE, now=noon, tz="America/New_York")
    beijing = app.describe_due(PITTSBURGH_DUE, now=noon, tz="Asia/Shanghai")
    assert 0 < pittsburgh["days_until"] < 1
    assert beijing["days_until"] > pittsburgh["days_until"]
    aware_now = datetime(2026, 10, 8, 16, 0, tzinfo=timezone.utc)
    assert (
        app.describe_due(PITTSBURGH_DUE, now=aware_now, tz="America/New_York")["days_until"]
        == app.describe_due(PITTSBURGH_DUE, now=aware_now, tz="Asia/Shanghai")["days_until"]
    )


def test_system_prompt_uses_student_timezone_not_machine(machine_tz_shanghai):
    # 02:00 UTC on Sep 21 is still Sep 20 evening in Pittsburgh, already Sep 21 in Beijing.
    now = datetime(2026, 9, 21, 2, 0, tzinfo=timezone.utc)
    default_prompt = app.build_system_prompt(now)
    assert "Sunday, September 20, 2026" in default_prompt
    assert "America/New_York" in default_prompt
    assert "EDT" in default_prompt

    beijing_prompt = app.build_system_prompt(now, tz="Asia/Shanghai")
    assert "Monday, September 21, 2026" in beijing_prompt
    assert "Asia/Shanghai" in beijing_prompt
    assert beijing_prompt != default_prompt


def test_canvas_client_due_dates_follow_configured_timezone(client_factory):
    client, _ = client_factory(
        [
            [{"id": 1, "name": "76-101"}],
            [
                {
                    "id": 31,
                    "name": "Homework 4",
                    "due_at": "2026-10-09T03:59:00Z",
                    "submission": {},
                }
            ],
        ]
    )
    assert client.tz.key == "America/New_York"
    matches = client.find_due_dates("Homework 4")
    assert matches[0]["due_at"] == "2026-10-09T03:59:00Z"
    assert matches[0]["due_at_local"] == "Thu Oct 08, 2026 11:59 PM EDT"

    client.set_timezone("Asia/Shanghai")
    switched = client._describe_due(app.parse_canvas_timestamp("2026-10-09T03:59:00Z"))
    assert switched["due_at"] == "2026-10-09T03:59:00Z"
    assert switched["due_at_local"] == "Fri Oct 09, 2026 11:59 AM CST"


def test_store_persists_timezone_setting(tmp_path):
    store = app.ConversationStore(str(tmp_path / "history.db"))
    assert store.get_setting("timezone") is None
    store.set_setting("timezone", "Asia/Shanghai")
    assert store.get_setting("timezone") == "Asia/Shanghai"
    again = app.ConversationStore(str(tmp_path / "history.db"))
    assert again.get_setting("timezone") == "Asia/Shanghai"


# --- course files ----------------------------------------------------------

FILE_ROW = {
    "id": 9001,
    "display_name": "15213-syllabus.pdf",
    "filename": "15213-syllabus.pdf",
    "content-type": "application/pdf",
    "size": 240_000,
    "updated_at": "2026-08-25T14:00:00Z",
    "url": f"{BASE_URL}/files/9001/download?download_frd=1&verifier=supersecretverifier",
}


def test_find_course_files_normalizes_and_hides_download_url(client_factory):
    client, fake = client_factory(
        [
            [{"id": 1, "name": "15-213"}],
            [FILE_ROW, {"id": 9002, "display_name": "notes.txt", "content-type": "text/plain", "size": 12}],
        ]
    )
    files, notes = client.find_course_files("syllabus")
    assert notes == []
    assert len(files) == 1
    found = files[0]
    assert found["file_id"] == 9001
    assert found["filename"] == "15213-syllabus.pdf"
    assert found["course_name"] == "15-213"
    assert found["size_readable"] == "234.4 KB"
    assert found["previewable"] is True
    assert "verifier" not in json.dumps(found)


def test_file_search_never_sends_search_term_to_canvas(client_factory):
    """Canvas 400s on search terms under 3 characters, which used to look
    identical to a course with no files. Filtering happens locally now."""
    client, fake = client_factory([[{"id": 1, "name": "15-213"}], [FILE_ROW]])
    files, _ = client.find_course_files("CGA")
    assert all("search_term" not in request.url for request in fake.requests)
    assert all("sort=" not in request.url for request in fake.requests)
    assert files == []  # "CGA" genuinely does not match the syllabus


def test_file_search_matches_every_word_in_any_order(client_factory):
    client, _ = client_factory(
        [
            [{"id": 1, "name": "Comparative Genre Analysis"}],
            [{"id": 1, "display_name": "CGA-prompt-final.pdf", "content-type": "application/pdf"}],
        ]
    )
    files, _ = client.find_course_files("prompt cga")
    assert [f["filename"] for f in files] == ["CGA-prompt-final.pdf"]


def test_find_course_files_falls_back_to_modules_when_files_tab_is_hidden(client_factory):
    """The reported bug: Canvas answers 401 for a hidden Files tab, which the
    old code swallowed, so every search came back empty."""
    client, _ = client_factory(
        [
            [{"id": 1, "name": "76-101 Interp & Argument"}],
            make_response({"status": "unauthorized"}, status=401),
            [
                {
                    "id": 55,
                    "name": "Unit 2",
                    "items": [
                        {"type": "Page", "title": "Overview"},
                        {"type": "File", "title": "CGA assignment.pdf", "content_id": 4242},
                    ],
                }
            ],
            dict(FILE_ROW, id=4242, display_name="CGA assignment.pdf"),
        ]
    )
    files, notes = client.find_course_files("CGA")
    assert [f["file_id"] for f in files] == [4242]
    assert files[0]["filename"] == "CGA assignment.pdf"
    assert files[0]["found_in"] == "Unit 2"
    assert any("Files tab is hidden" in note for note in notes)


def test_find_course_files_follows_items_url_for_large_modules(client_factory):
    client, fake = client_factory(
        [
            [{"id": 1, "name": "Course"}],
            [],  # Files tab is readable but empty
            [{"id": 7, "name": "Week 1", "items_url": f"{BASE_URL}/api/v1/courses/1/modules/7/items"}],
            [{"type": "File", "title": "handout.pdf", "content_id": 99}],
            dict(FILE_ROW, id=99, display_name="handout.pdf"),
        ]
    )
    files, _ = client.find_course_files("handout")
    assert [f["file_id"] for f in files] == [99]
    assert any("/modules/7/items" in request.url for request in fake.requests)


def test_find_course_files_explains_a_course_it_could_not_read(client_factory):
    client, _ = client_factory(
        [
            [{"id": 1, "name": "Locked down"}],
            make_response({}, status=401),
            make_response({}, status=401),
        ]
    )
    files, notes = client.find_course_files()
    assert files == []
    assert any("Files tab is hidden" in note for note in notes)
    assert any("Modules are not visible" in note for note in notes)


def test_find_course_files_reports_when_there_are_no_courses(client_factory):
    client, _ = client_factory([[]])
    files, notes = client.find_course_files()
    assert files == []
    assert notes == ["No active courses were returned by Canvas."]


def test_file_listing_with_no_query_returns_everything(client_factory):
    client, _ = client_factory(
        [
            [{"id": 1, "name": "Course"}],
            [FILE_ROW, dict(FILE_ROW, id=9002, display_name="lecture-1.pdf")],
        ]
    )
    files, _ = client.find_course_files()
    assert {f["file_id"] for f in files} == {9001, 9002}


def test_download_file_returns_bytes_and_metadata(client_factory):
    pdf_bytes = make_pdf("Syllabus week 1")
    client, fake = client_factory([FILE_ROW, make_binary_response(pdf_bytes, "application/pdf")])

    content, described = client.download_file(9001)
    assert content == pdf_bytes
    assert described["filename"] == "15213-syllabus.pdf"
    assert described["size_bytes"] == len(pdf_bytes)
    assert [request.method for request in fake.requests] == ["GET", "GET"]


def test_download_file_refuses_oversized_files(client_factory):
    big = dict(FILE_ROW, size=200 * 1024 * 1024)
    client, fake = client_factory([big])
    with pytest.raises(app.CanvasError) as excinfo:
        client.download_file(9001)
    assert "preview limit" in str(excinfo.value)
    assert len(fake.requests) == 1  # never fetched the body


def test_download_file_stops_reading_past_the_cap(client_factory):
    client, _ = client_factory(
        [dict(FILE_ROW, size=None), make_binary_response(b"x" * 5000, "application/pdf")]
    )
    with pytest.raises(app.CanvasError):
        client.download_file(9001, max_bytes=1000)


def test_download_file_refuses_locked_files(client_factory):
    client, _ = client_factory([dict(FILE_ROW, locked_for_user=True)])
    with pytest.raises(app.CanvasError) as excinfo:
        client.download_file(9001)
    assert "locked" in str(excinfo.value)


def test_extract_text_reads_text_files():
    text, truncated = app.extract_text(b"hello notes", "text/plain")
    assert text == "hello notes"
    assert truncated is False


def test_extract_text_truncates_long_text():
    text, truncated = app.extract_text(b"a" * (app.MAX_EXTRACTED_CHARS + 50), "text/markdown")
    assert truncated is True
    assert len(text) == app.MAX_EXTRACTED_CHARS


def test_extract_text_reads_a_real_pdf():
    pytest.importorskip("pypdf")
    text, truncated = app.extract_text(make_pdf("Syllabus week 1"), "application/pdf")
    assert "Syllabus week 1" in text
    assert truncated is False


def test_extract_text_ignores_unreadable_formats():
    assert app.extract_text(b"\x00\x01binary", "application/zip") == (None, False)
    assert app.extract_text(b"not really a pdf", "application/pdf") == (None, False)


def test_human_size_formats_bytes():
    assert app.human_size(512) == "512 B"
    assert app.human_size(2048) == "2.0 KB"
    assert app.human_size(5 * 1024 * 1024) == "5.0 MB"
    assert app.human_size(None) is None


# --- tool layer ------------------------------------------------------------


def test_tool_schemas_are_well_formed():
    assert set(app.TOOL_NAMES) == {
        "list_my_courses",
        "list_upcoming_assignments",
        "find_due_dates",
        "find_course_files",
        "open_file",
        "get_assignment_details",
        "open_url",
    }
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


def test_dispatch_open_file_reports_the_file_and_its_text(client_factory):
    text_file = dict(FILE_ROW, id=9002, display_name="notes.txt", **{"content-type": "text/plain"})
    client, _ = client_factory(
        [text_file, make_binary_response(b"Homework 4 is due Friday.", "text/plain")]
    )

    result = app.dispatch_tool(client, "open_file", {"file_id": 9002})
    assert result["displayed_to_student"] is True
    assert result["file"]["filename"] == "notes.txt"
    assert result["text_excerpt"] == "Homework 4 is due Friday."
    assert "verifier" not in json.dumps(result)
    json.dumps(result)


def test_dispatch_open_file_requires_a_file_id(client_factory):
    client, fake = client_factory([])
    result = app.dispatch_tool(client, "open_file", {})
    assert "file_id" in result["error"]
    assert fake.requests == []


def test_dispatch_open_file_extracts_pdf_text_for_the_model(client_factory):
    pytest.importorskip("pypdf")
    client, _ = client_factory(
        [FILE_ROW, make_binary_response(make_pdf("Midterm is October 9"), "application/pdf")]
    )
    result = app.dispatch_tool(client, "open_file", {"file_id": 9001})
    assert "Midterm is October 9" in result["text_excerpt"]
    assert result["file"]["content_type"] == "application/pdf"


def test_dispatch_find_course_files(client_factory):
    client, _ = client_factory([[{"id": 1, "name": "15-213"}], [FILE_ROW]])
    result = app.dispatch_tool(client, "find_course_files", {"query": "syllabus"})
    assert result["count"] == 1
    assert result["files"][0]["file_id"] == 9001
    assert "hint" not in result


def test_dispatch_find_course_files_explains_an_empty_result(client_factory):
    client, _ = client_factory(
        [[{"id": 1, "name": "Course"}], make_response({}, status=401), make_response({}, status=401)]
    )
    result = app.dispatch_tool(client, "find_course_files", {"query": "CGA"})
    assert result["count"] == 0
    assert result["notes"]
    assert "get_assignment_details" in result["hint"]


# --- assignment details ----------------------------------------------------

ASSIGNMENT_ROW = {
    "id": 3100,
    "name": "Comparative Genre Analysis",
    "due_at": "2026-10-09T03:59:00Z",
    "unlock_at": "2026-09-20T04:00:00Z",
    "lock_at": None,
    "points_possible": 100,
    "grading_type": "points",
    "submission_types": ["online_upload", "online_text_entry"],
    "allowed_extensions": ["pdf", "docx"],
    "allowed_attempts": 2,
    "html_url": f"{BASE_URL}/courses/1/assignments/3100",
    "description": (
        "<p>Write a <strong>comparative genre analysis</strong> of two texts.</p>"
        "<ul><li>1200-1500 words</li><li>MLA format</li></ul>"
        "<p>See the "
        '<a href="/courses/1/files/4242/download?wrap=1">CGA prompt</a> for details.</p>'
        '<p>Read <a href="https://www.cs.cmu.edu/~handout.pdf">the extra PDF</a>.</p>'
        "<script>ignore me</script>"
    ),
    "rubric": [{"description": "Analysis", "points": 60}, {"description": "Prose", "points": 40}],
    "submission": {"submitted_at": None, "workflow_state": "unsubmitted", "missing": False},
}


def test_get_assignment_details_returns_prompt_and_submission_type(client_factory):
    client, _ = client_factory([[{"id": 1, "name": "76-101"}], ASSIGNMENT_ROW])
    details = client.get_assignment_details(3100, course_id=1)

    assert details["title"] == "Comparative Genre Analysis"
    assert details["submission_types"] == ["online_upload", "online_text_entry"]
    assert details["allowed_extensions"] == ["pdf", "docx"]
    assert details["allowed_attempts"] == 2
    assert "comparative genre analysis" in details["instructions"]
    assert "1200-1500 words" in details["instructions"]
    assert "ignore me" not in details["instructions"]
    assert "<p>" not in details["instructions"]
    assert details["attached_files"] == [{"file_id": 4242, "filename": "CGA prompt"}]
    assert details["linked_urls"] == [
        {"url": "https://www.cs.cmu.edu/~handout.pdf", "label": "the extra PDF"}
    ]
    assert details["submission_status"]["workflow_state"] == "unsubmitted"
    assert [item["criterion"] for item in details["rubric"]] == ["Analysis", "Prose"]
    assert details["due_at_local"] == "Thu Oct 08, 2026 11:59 PM EDT"


def test_get_assignment_details_scans_courses_when_course_id_is_missing(client_factory):
    client, fake = client_factory(
        [
            [{"id": 1, "name": "Wrong course"}, {"id": 2, "name": "Right course"}],
            make_response({}, status=404),
            ASSIGNMENT_ROW,
        ]
    )
    details = client.get_assignment_details(3100)
    assert details["course_name"] == "Right course"
    assert len(fake.requests) == 3


def test_get_assignment_details_reports_a_missing_assignment(client_factory):
    client, _ = client_factory([[{"id": 1, "name": "Course"}], make_response({}, status=404)])
    with pytest.raises(app.CanvasError) as excinfo:
        client.get_assignment_details(999)
    assert "Could not find assignment" in str(excinfo.value)


def test_dispatch_get_assignment_details(client_factory):
    client, _ = client_factory([[{"id": 1, "name": "76-101"}], ASSIGNMENT_ROW])
    result = app.dispatch_tool(client, "get_assignment_details", {"assignment_id": 3100, "course_id": 1})
    assert result["assignment"]["submission_types"] == ["online_upload", "online_text_entry"]
    json.dumps(result)


def test_dispatch_get_assignment_details_requires_an_id(client_factory):
    client, fake = client_factory([])
    result = app.dispatch_tool(client, "get_assignment_details", {})
    assert "assignment_id" in result["error"]
    assert fake.requests == []


def test_assignment_listings_now_include_submission_types(client_factory):
    client, _ = client_factory(
        [
            [{"id": 1, "name": "Course"}],
            [{"id": 11, "name": "Essay", "due_at": iso_in(2), "submission_types": ["online_upload"]}],
        ]
    )
    upcoming = client.list_upcoming_assignments()
    assert upcoming[0]["submission_types"] == ["online_upload"]


def test_html_to_text_and_links():
    text, links, urls = app.html_to_text_and_links(
        "<h2>Prompt</h2><p>Compare &amp; contrast.</p>"
        '<p><a href="https://canvas.example.edu/courses/1/files/77">handout.pdf</a></p>'
        '<p><a href="https://example.com/other">not a canvas file</a></p>'
    )
    assert "Prompt" in text and "Compare & contrast." in text
    assert links == [(77, "handout.pdf")]
    assert urls == [{"url": "https://example.com/other", "label": "not a canvas file"}]


def test_html_to_text_keeps_bullets_on_consecutive_lines():
    text, *_ = app.html_to_text_and_links("<ul><li>1200 words</li><li>MLA format</li></ul>")
    assert "- 1200 words\n- MLA format" in text


def test_html_to_text_truncates_very_long_prompts():
    text, *_ = app.html_to_text_and_links("<p>" + "word " * 5000 + "</p>")
    assert text.endswith("[truncated]")
    assert len(text) <= app.MAX_DESCRIPTION_CHARS + 20


def test_html_to_text_handles_empty_description():
    assert app.html_to_text_and_links("") == ("", [], [])
    assert app.html_to_text_and_links(None) == ("", [], [])


def test_matches_all_words():
    assert app.matches_all_words("CGA prompt final.pdf", "cga prompt")
    assert app.matches_all_words("CGA prompt final.pdf", "PROMPT cga")
    assert not app.matches_all_words("CGA prompt final.pdf", "cga rubric")
    assert app.matches_all_words("anything", "")


def test_dispatch_tool_converts_canvas_errors_to_messages(client_factory):
    app.register_secret(TOKEN)
    client, _ = client_factory([make_response({}, status=403)])
    result = app.dispatch_tool(client, "list_my_courses", {})
    assert "403" in result["error"]
    assert TOKEN not in result["error"]


def test_dispatch_find_due_dates_empty_query_is_a_tool_error_not_a_crash(client_factory):
    client, _ = client_factory([])
    result = app.dispatch_tool(client, "find_due_dates", {"query": ""})
    assert "error" in result
    assert "required" in result["error"]


def test_dispatch_find_due_dates_single_character_query(client_factory):
    client, _ = client_factory(
        [
            [{"id": 1, "name": "Course A"}],
            [{"id": 31, "name": "Homework 4", "due_at": iso_in(3)}],
        ]
    )
    result = app.dispatch_tool(client, "find_due_dates", {"query": "4"})
    assert result["count"] == 1
    assert result["assignments"][0]["title"] == "Homework 4"


def test_dispatch_tool_catches_stale_exception_classes():
    """Cached CanvasClient from a previous Streamlit run raises a CanvasError
    that is not isinstance of the CanvasError bound in this run."""

    class StaleCanvasError(RuntimeError):
        pass

    class StaleClient:
        def find_due_dates(self, **kwargs):
            raise StaleCanvasError("Search text must be at least 2 characters long.")

    result = app.dispatch_tool(StaleClient(), "find_due_dates", {"query": "4"})
    assert result["error"] == "Search text must be at least 2 characters long."


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
    assert {"conversations", "messages", "settings"} <= tables
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
    assert "America/New_York" in deepseek.calls[0]["messages"][0]["content"]
    assert deepseek.calls[0]["tools"] == app.TOOL_SCHEMAS


def test_agent_turn_system_prompt_follows_canvas_timezone(client_factory):
    client, _ = client_factory([[]])
    client.set_timezone("Asia/Shanghai")
    deepseek = ScriptedDeepSeek([{"role": "assistant", "content": "hi"}])
    app.run_agent_turn(deepseek, client, [{"role": "user", "content": "hi"}])
    prompt = deepseek.calls[0]["messages"][0]["content"]
    assert "Asia/Shanghai" in prompt
    assert "America/New_York" not in prompt


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


def test_agent_turn_does_not_crash_on_a_one_character_due_date_query(client_factory):
    client, _ = client_factory(
        [
            [{"id": 1, "name": "Course A"}],
            [{"id": 31, "name": "Homework 4", "due_at": iso_in(3)}],
        ]
    )
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
                            "name": "find_due_dates",
                            "arguments": '{"query": "4"}',
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "Homework 4 is due in three days."},
        ]
    )
    produced = app.run_agent_turn(
        deepseek, client, [{"role": "user", "content": "when is 4 due?"}]
    )
    payload = json.loads(produced[1]["content"])
    assert payload["count"] == 1
    assert payload["assignments"][0]["title"] == "Homework 4"
    assert produced[-1]["content"] == "Homework 4 is due in three days."


def test_agent_turn_converts_stale_tool_exceptions_into_tool_results():
    class StaleCanvasError(RuntimeError):
        pass

    class ExplodingClient:
        def find_due_dates(self, **kwargs):
            raise StaleCanvasError("Search text must be at least 2 characters long.")

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
                            "name": "find_due_dates",
                            "arguments": '{"query": ""}',
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "I need a more specific assignment name."},
        ]
    )
    produced = app.run_agent_turn(
        deepseek, ExplodingClient(), [{"role": "user", "content": "what's due?"}]
    )
    assert [m["role"] for m in produced] == ["assistant", "tool", "assistant"]
    payload = json.loads(produced[1]["content"])
    assert "error" in payload
    assert produced[-1]["content"] == "I need a more specific assignment name."


def test_system_prompt_states_read_only_and_date():
    prompt = app.build_system_prompt(datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc))
    assert "read-only" in prompt
    assert "September 21, 2026" in prompt
    assert "open_url" in prompt
    assert "$...$" in prompt


# --- math / LaTeX conversion ------------------------------------------------


def test_to_streamlit_math_converts_mathjax_delimiters():
    converted = app.to_streamlit_math(r"Inline \(E=mc^2\) and display \[\int_0^1 x\,dx\].")
    assert "$E=mc^2$" in converted
    assert "$$\n\\int_0^1 x\\,dx\n$$" in converted


def test_html_math_survives_conversion():
    text, files, urls = app.html_to_text_and_links(
        "<p>Compute \\(\\frac{a}{b}\\) then "
        "\\[\\sum_{n=1}^{N} n\\].</p>"
        '<script type="math/tex">E=mc^2</script>'
        '<script type="math/tex; mode=display">x^2+y^2=z^2</script>'
        '<span class="math">\\log n</span>'
        '<div class="math">\\int_0^1 x\\,dx</div>'
        '<p>Also $$\\sqrt{2}$$ in the prompt.</p>'
        "<script>alert(1)</script>"
    )
    assert files == [] and urls == []
    assert r"$\frac{a}{b}$" in text
    assert r"\sum_{n=1}^{N} n" in text
    assert "$$" in text
    assert "$E=mc^2$" in text
    assert r"$x^2+y^2=z^2$" not in text  # display math uses $$
    assert r"x^2+y^2=z^2" in text
    assert r"$\log n$" in text
    assert r"\int_0^1 x\,dx" in text
    assert r"\sqrt{2}" in text
    assert "alert" not in text


def test_extract_text_strips_html_and_keeps_math():
    html = (
        b"<html><body><h1>Syllabus</h1><p>Grade is \\(x^2\\).</p>"
        b'<a href="https://example.com/rubric">rubric</a></body></html>'
    )
    text, truncated = app.extract_text(html, "text/html; charset=utf-8", base_url="https://example.com/s")
    assert truncated is False
    assert "Syllabus" in text
    assert "<h1>" not in text
    assert "$x^2$" in text


def test_extract_text_reads_tex_as_text():
    source = rb"\frac{1}{2} + \alpha"
    text, truncated = app.extract_text(source, "text/x-tex", filename="snippet.tex")
    assert text == source.decode()
    assert truncated is False


# --- open_url --------------------------------------------------------------

PUBLIC_ADDRINFO = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]


@pytest.fixture
def public_dns():
    with mock.patch.object(app.socket, "getaddrinfo", return_value=PUBLIC_ADDRINFO):
        yield


def test_external_session_blocks_non_get():
    session = app.ExternalGetSession()
    with pytest.raises(app.ReadOnlyViolation):
        session.post("https://example.com/")
    prepared = requests.Request("PUT", "https://example.com/").prepare()
    with pytest.raises(app.ReadOnlyViolation):
        session.send(prepared)


def test_external_session_strips_authorization_header():
    session = app.ExternalGetSession()
    prepared = requests.Request(
        "GET", "https://example.com/notes.txt", headers={"Authorization": f"Bearer {TOKEN}"}
    ).prepare()
    sent = {}

    def capture(self, request, **kwargs):
        sent["headers"] = dict(request.headers)
        return make_response([], url=request.url)

    with mock.patch.object(requests.Session, "send", autospec=True, side_effect=capture):
        session.send(prepared)
    assert "Authorization" not in sent["headers"]
    assert "Cookie" not in sent["headers"]


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/x",
        "https://localhost/secret",
        "https://127.0.0.1/",
        "https://[::1]/",
        "https://10.0.0.5/admin",
        "https://192.168.1.8/",
        "https://172.16.0.1/",
        "https://169.254.169.254/latest/meta-data/",
        "https://metadata.google.internal/",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "https://user:pass@example.com/x",
    ],
)
def test_open_url_rejects_ssrf_and_non_https(client_factory, url):
    client, fake = client_factory([])
    result = app.dispatch_tool(client, "open_url", {"url": url})
    assert "error" in result
    assert fake.requests == []
    assert TOKEN not in json.dumps(result)


def test_open_url_rejects_hostname_that_resolves_privately(client_factory):
    client, fake = client_factory([])
    private = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 0))]
    with mock.patch.object(app.socket, "getaddrinfo", return_value=private):
        result = app.dispatch_tool(client, "open_url", {"url": "https://evil.example.com/flag"})
    assert "error" in result
    assert "private" in result["error"].lower() or "Blocked" in result["error"]
    assert fake.requests == []


def test_open_url_requires_a_url(client_factory):
    client, fake = client_factory([])
    result = app.dispatch_tool(client, "open_url", {})
    assert "https URL" in result["error"]
    assert fake.requests == []


def test_open_url_html(client_factory, public_dns):
    html = b"<html><body><h1>Syllabus</h1><p>Week 1: stacks.</p></body></html>"
    client, fake = client_factory([make_binary_response(html, "text/html")])
    result = app.dispatch_tool(client, "open_url", {"url": "https://cs.example.edu/syllabus.html"})
    assert result["displayed_to_student"] is True
    assert result["resource"]["content_type"] == "text/html"
    assert "Week 1: stacks." in result["text_excerpt"]
    assert "<h1>" not in result["text_excerpt"]
    assert "Authorization" not in fake.requests[0].headers
    assert TOKEN not in fake.requests[0].url
    json.dumps(result)


def test_open_url_plain_text(client_factory, public_dns):
    client, fake = client_factory([make_binary_response(b"Office hours: Friday", "text/plain")])
    result = app.dispatch_tool(client, "open_url", {"url": "https://cs.example.edu/hours.txt"})
    assert result["text_excerpt"] == "Office hours: Friday"
    assert "Authorization" not in fake.requests[0].headers


def test_open_url_pdf(client_factory, public_dns):
    pytest.importorskip("pypdf")
    pdf = make_pdf("Midterm is October 9")
    client, fake = client_factory([make_binary_response(pdf, "application/pdf")])
    result = app.dispatch_tool(client, "open_url", {"url": "https://cs.example.edu/midterm.pdf"})
    assert "Midterm is October 9" in result["text_excerpt"]
    assert result["resource"]["content_type"] == "application/pdf"
    assert "Authorization" not in fake.requests[0].headers


def test_open_url_other_types_do_not_dump_binary(client_factory, public_dns):
    client, _ = client_factory([make_binary_response(b"PK\x03\x04binaryzip", "application/zip")])
    result = app.dispatch_tool(client, "open_url", {"url": "https://cs.example.edu/slides.zip"})
    assert "text_excerpt" not in result
    dumped = json.dumps(result)
    assert "PK" not in dumped
    assert result["displayed_to_student"] is True


def test_open_url_canvas_file_routes_to_download(client_factory):
    pytest.importorskip("pypdf")
    pdf = make_pdf("Syllabus week 1")
    client, fake = client_factory([FILE_ROW, make_binary_response(pdf, "application/pdf")])
    result = app.dispatch_tool(
        client, "open_url", {"url": f"{BASE_URL}/courses/1/files/9001/download?wrap=1"}
    )
    assert result["opened_via"] == "open_file"
    assert result["file"]["file_id"] == 9001
    assert "Syllabus week 1" in result["text_excerpt"]
    assert "verifier" not in json.dumps(result)
    assert fake.requests[0].headers["Authorization"] == f"Bearer {TOKEN}"
    assert "/files/9001" in fake.requests[0].url


def test_open_url_canvas_assignment_url_uses_details(client_factory):
    client, fake = client_factory([[{"id": 1, "name": "76-101"}], ASSIGNMENT_ROW])
    result = app.dispatch_tool(
        client, "open_url", {"url": f"{BASE_URL}/courses/1/assignments/3100"}
    )
    assert result["opened_via"] == "get_assignment_details"
    assert result["assignment"]["title"] == "Comparative Genre Analysis"
    assert "comparative genre analysis" in result["assignment"]["instructions"]
    assert fake.requests[0].headers["Authorization"] == f"Bearer {TOKEN}"


def test_open_url_does_not_follow_redirect_onto_private_host(client_factory, public_dns):
    redirect = make_response({}, status=302, headers={"Location": "https://10.0.0.5/secret"})
    client, fake = client_factory([redirect])
    result = app.dispatch_tool(client, "open_url", {"url": "https://cs.example.edu/bounce"})
    assert "error" in result
    assert len(fake.requests) == 1
    assert "Authorization" not in fake.requests[0].headers


def test_open_url_redirect_does_not_attach_canvas_token(client_factory, public_dns):
    redirect = make_response(
        {}, status=302, headers={"Location": "https://cdn.example.edu/notes.txt"}
    )
    client, fake = client_factory(
        [redirect, make_binary_response(b"notes body", "text/plain")]
    )
    result = app.dispatch_tool(client, "open_url", {"url": "https://cs.example.edu/go"})
    assert result["text_excerpt"] == "notes body"
    assert len(fake.requests) == 2
    for request in fake.requests:
        assert "Authorization" not in request.headers
        assert TOKEN not in request.url


def test_agent_turn_open_url_does_not_crash(client_factory, public_dns):
    client, _ = client_factory([make_binary_response(b"<p>Hello</p>", "text/html")])
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
                            "name": "open_url",
                            "arguments": '{"url": "https://cs.example.edu/hello.html"}',
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "The page says Hello. $a=b$."},
        ]
    )
    produced = app.run_agent_turn(
        deepseek, client, [{"role": "user", "content": "open https://cs.example.edu/hello.html"}]
    )
    assert [m["role"] for m in produced] == ["assistant", "tool", "assistant"]
    payload = json.loads(produced[1]["content"])
    assert payload["displayed_to_student"] is True
    assert "Hello" in payload["text_excerpt"]
    assert produced[-1]["content"] == "The page says Hello. $a=b$."


def test_canvas_file_id_from_url_only_matches_configured_origin():
    assert app.canvas_file_id_from_url(f"{BASE_URL}/files/99/download", BASE_URL) == 99
    assert app.canvas_file_id_from_url("https://evil.example.com/files/99", BASE_URL) is None
    assert app.canvas_file_id_from_url("https://example.com/midterm.pdf", BASE_URL) is None

    assert "America/New_York" in prompt
    assert "the machine's local timezone" not in prompt
