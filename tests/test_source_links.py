"""Source links stay metadata until the final answer, and only relevant ones show."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402

HW4 = "https://canvas.cmu.edu/courses/1/assignments/3100"
HW5 = "https://canvas.cmu.edu/courses/1/assignments/3101"
HW6 = "https://canvas.cmu.edu/courses/1/assignments/3102"
HW7 = "https://canvas.cmu.edu/courses/1/assignments/3103"
SYLLABUS = "https://canvas.cmu.edu/courses/1/files/9001"
REVIEW = "https://cs.example.edu/exam-review.pdf"
COURSE = "https://canvas.cmu.edu/courses/1"


def _tool(name: str, payload: dict, call_id: str = "call_1") -> dict:
    return {
        "role": "tool",
        "name": name,
        "tool_call_id": call_id,
        "content": json.dumps(payload),
    }


def _list_assignments_message() -> dict:
    return _tool(
        "list_upcoming_assignments",
        {
            "count": 3,
            "assignments": [
                {"title": "Homework 4", "html_url": HW4},
                {"title": "Homework 5", "html_url": HW5},
                {"title": "Homework 6", "html_url": HW6},
            ],
        },
    )


def _assignment_details_message(url: str = HW4, title: str = "Homework 4") -> dict:
    return _tool(
        "get_assignment_details",
        {
            "assignment": {
                "title": title,
                "html_url": url,
                "assignment_id": 3100,
                "course_id": 1,
                "instructions": "Write 1200 words. See attached prompt.",
                "attached_files": [{"file_id": 4242, "filename": "CGA prompt"}],
            }
        },
        call_id="call_details",
    )


def _open_file_message(file_id: int = 9001, filename: str = "notes.txt", url: str | None = SYLLABUS) -> dict:
    file_dict = {"file_id": file_id, "filename": filename, "content_type": "text/plain"}
    if url:
        file_dict["url"] = url
    return _tool("open_file", {"file": file_dict}, call_id=f"call_file_{file_id}")


def test_source_record_keeps_title_type_and_url_separate():
    record = app.make_source_record(
        title="Homework 4",
        source_type=app.SOURCE_TYPE_ASSIGNMENT,
        canonical_url=HW4,
        explicitly_opened=True,
        used_in_answer=True,
    )
    for field in app.SOURCE_RECORD_FIELDS:
        assert field in record
    assert record["title"] == "Homework 4"
    assert record["source_type"] == "assignment"
    assert record["canonical_url"] == HW4
    mashed = f"[Homework 4]({HW4})"
    assert record["title"] != mashed
    assert record["canonical_url"] != mashed
    assert "source_type" in record and record["source_type"] != mashed


def test_list_search_html_urls_are_not_shown():
    turn = [
        _list_assignments_message(),
        {"role": "assistant", "content": f"Upcoming: HW4 {HW4}, HW5 {HW5}, HW6 {HW6}."},
    ]
    sources = app.displayed_sources_for_turn(turn, user_text="what's due?")
    assert sources == []
    text = app.assistant_content_with_sources(turn[-1]["content"], sources)
    assert HW4 not in text
    assert HW5 not in text
    assert HW6 not in text
    assert "Upcoming: HW4" in text


def test_find_due_dates_html_urls_are_not_shown():
    message = _tool(
        "find_due_dates",
        {"count": 1, "assignments": [{"title": "Lab 2", "html_url": HW7}]},
    )
    records = app.collect_source_records([message])
    assert records and records[0]["canonical_url"] == HW7
    assert records[0]["explicitly_opened"] is False
    displayed = app.select_displayed_sources(records, intent="general")
    assert displayed == []


def test_assignment_object_does_not_auto_render_a_page():
    payload = json.loads(_assignment_details_message()["content"])
    assert app.tool_result_should_render_assignment_page("get_assignment_details", payload) is False
    assert app.tool_result_should_render_assignment_page(
        "open_url", {"assignment": payload["assignment"], "url": HW4}
    ) is False
    assert app.tool_result_should_preview_file("get_assignment_details", payload) is False


def test_assignment_attached_files_are_not_previewed_until_opened():
    message = _assignment_details_message()
    assert app.opened_file_entries_from_tool_message(message) == []
    payload = json.loads(message["content"])
    assert app.tool_result_should_preview_file(message["name"], payload) is False
    records = app.source_records_from_tool_message(message)
    assert all(record["source_type"] != app.SOURCE_TYPE_FILE for record in records)


def test_file_preview_only_when_explicitly_opened():
    opened = _open_file_message()
    listed = _tool(
        "find_course_files",
        {"count": 1, "files": [{"file_id": 9001, "filename": "notes.txt", "url": SYLLABUS}]},
    )
    assert app.tool_result_should_preview_file(opened["name"], json.loads(opened["content"])) is True
    assert app.opened_file_entries_from_tool_message(opened)
    assert app.tool_result_should_preview_file(listed["name"], json.loads(listed["content"])) is False
    assert app.opened_file_entries_from_tool_message(listed) == []


def test_assignment_summary_shows_one_open_assignment_link():
    turn = [
        _list_assignments_message(),
        _assignment_details_message(),
        {
            "role": "assistant",
            "content": f"Write 1200 words. Also see {HW5} and {HW6}.",
        },
    ]
    sources = app.displayed_sources_for_turn(turn, user_text="/summarize what I need to do for HW4")
    assert len(sources) == 1
    assert sources[0]["source_type"] == app.SOURCE_TYPE_ASSIGNMENT
    assert sources[0]["canonical_url"] == HW4
    assert sources[0]["title"] == "Homework 4"
    text = app.assistant_content_with_sources(turn[-1]["content"], sources)
    assert text.count(app.OPEN_ASSIGNMENT_IN_CANVAS) == 1
    assert text.count(HW4) == 1
    assert HW5 not in text
    assert HW6 not in text
    assert f"[{app.OPEN_ASSIGNMENT_IN_CANVAS}]({HW4})" in text


def test_exam_study_does_not_auto_include_assignment_page_links():
    turn = [
        _assignment_details_message(),
        _open_file_message(9001, "exam-review.pdf", REVIEW),
        {
            "role": "assistant",
            "content": f"Study the review sheet. Ignore {HW4}.",
        },
    ]
    sources = app.displayed_sources_for_turn(turn, user_text="/exam what should I study for 21128")
    assert all(record["source_type"] != app.SOURCE_TYPE_ASSIGNMENT for record in sources)
    assert all(not app.is_canvas_assignment_url(record["canonical_url"]) for record in sources)
    text = app.assistant_content_with_sources(turn[-1]["content"], sources)
    assert HW4 not in text
    assert app.OPEN_ASSIGNMENT_IN_CANVAS not in text
    assert REVIEW in text


def test_course_content_path_skips_assignment_page_links():
    turn = [
        _tool("find_course_files", {"count": 1, "files": [{"file_id": 9001, "filename": "syllabus.pdf"}]}),
        _assignment_details_message(),
        _open_file_message(),
        {"role": "assistant", "content": "The late policy is in the syllabus."},
    ]
    sources = app.displayed_sources_for_turn(turn, user_text="what's the late policy?")
    assert app.infer_source_intent("what's the late policy?", ["find_course_files", "get_assignment_details", "open_file"]) == (
        app.INTENT_COURSE_CONTENT
    )
    assert all(record["source_type"] != app.SOURCE_TYPE_ASSIGNMENT for record in sources)
    text = app.assistant_content_with_sources(turn[-1]["content"], sources)
    assert HW4 not in text
    assert SYLLABUS in text


def test_final_answer_caps_source_links_at_three():
    opened = [
        _open_file_message(9001, "a.pdf", "https://cs.example.edu/a.pdf"),
        _open_file_message(9002, "b.pdf", "https://cs.example.edu/b.pdf"),
        _open_file_message(9003, "c.pdf", "https://cs.example.edu/c.pdf"),
        _open_file_message(9004, "d.pdf", "https://cs.example.edu/d.pdf"),
    ]
    turn = opened + [{"role": "assistant", "content": "Here are the readings."}]
    sources = app.displayed_sources_for_turn(turn, user_text="open the readings")
    assert len(sources) == app.MAX_DISPLAYED_SOURCE_LINKS == 3
    text = app.assistant_content_with_sources(turn[-1]["content"], sources)
    assert text.count("https://cs.example.edu/") == 3
    assert "d.pdf" not in text


def test_open_url_assignment_is_explicitly_opened_but_still_one_link():
    turn = [
        _tool(
            "open_url",
            {
                "url": HW4,
                "assignment": {"title": "Homework 4", "html_url": HW4, "instructions": "Do the work."},
                "displayed_to_student": False,
            },
        ),
        {"role": "assistant", "content": "Submit a PDF."},
    ]
    sources = app.displayed_sources_for_turn(turn, user_text="/summarize Homework 4")
    assert len(sources) == 1
    assert sources[0]["explicitly_opened"] is True
    assert sources[0]["canonical_url"] == HW4


def test_course_html_urls_from_list_my_courses_are_not_shown():
    turn = [
        _tool(
            "list_my_courses",
            {"count": 1, "courses": [{"name": "15-213", "html_url": COURSE}]},
        ),
        {"role": "assistant", "content": f"You are in 15-213 {COURSE}"},
    ]
    sources = app.displayed_sources_for_turn(turn, user_text="what courses am I in?")
    assert sources == []
    text = app.assistant_content_with_sources(turn[-1]["content"], sources)
    assert COURSE not in text
    assert "15-213" in text


def test_markdown_assignment_links_are_stripped_unless_selected():
    body = f"See [Homework 4]({HW4}) and [Homework 5]({HW5})."
    text = app.filter_unrelated_canvas_urls(body, {HW4})
    assert HW4 in text
    assert HW5 not in text
    assert "Homework 5" in text
