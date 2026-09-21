"""Headless smoke tests for the Streamlit UI, using dummy (fake) credentials.

These run the real app script through Streamlit's AppTest harness, so they catch
import errors, bad widget calls and startup exceptions without a browser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")

DUMMY_ENV = {
    "DEEPSEEK_API_KEY": "sk-dummy-key-0000000000",
    "CANVAS_API_TOKEN": "dummy-canvas-token-0000000000",
    "CANVAS_BASE_URL": "https://canvas.example.edu",
}


@pytest.fixture(autouse=True)
def clean_streamlit_caches():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def run_app(monkeypatch, tmp_path, env: dict[str, str]) -> AppTest:
    for name in ("DEEPSEEK_API_KEY", "CANVAS_API_TOKEN", "CANVAS_BASE_URL", "CANVAS_ASSISTANT_TZ"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CANVAS_ASSISTANT_DB", str(tmp_path / "history.db"))
    # Keep a stray developer .env out of the test run.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    harness = AppTest.from_file(APP_PATH, default_timeout=30)
    return harness.run()


def test_app_starts_and_shows_chat_input(monkeypatch, tmp_path):
    harness = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    assert not harness.exception
    assert harness.title[0].value == "CMU Canvas Study Assistant"
    assert len(harness.chat_input) == 1
    assert not harness.error


def test_app_explains_missing_configuration(monkeypatch, tmp_path):
    harness = run_app(monkeypatch, tmp_path, {})
    assert not harness.exception
    message = harness.error[0].value
    assert "DEEPSEEK_API_KEY" in message and "CANVAS_API_TOKEN" in message
    assert not harness.chat_input


def test_sidebar_never_renders_secret_values(monkeypatch, tmp_path):
    harness = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    rendered = " ".join(str(element.value) for element in harness.sidebar.markdown)
    rendered += " ".join(str(element.value) for element in harness.sidebar.caption)
    assert DUMMY_ENV["DEEPSEEK_API_KEY"] not in rendered
    assert DUMMY_ENV["CANVAS_API_TOKEN"] not in rendered
    assert "canvas.example.edu" in rendered


def test_inline_pdf_viewer_is_available(tmp_path):
    """st.pdf only works with the streamlit[pdf] extra installed, so keep a
    test that fails loudly if requirements.txt ever drops it."""
    from test_app import make_pdf

    pdf_path = tmp_path / "syllabus.pdf"
    pdf_path.write_bytes(make_pdf("Syllabus week 1"))
    script = (
        "import streamlit as st\n"
        f"st.pdf(open({str(pdf_path)!r}, 'rb').read(), height=200)\n"
        "st.write('rendered')\n"
    )
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception
    assert harness.markdown[0].value == "rendered"


def test_new_chat_button_creates_a_conversation(monkeypatch, tmp_path):
    harness = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    before = len(harness.sidebar.button)
    harness.sidebar.button[0].click().run()
    assert not harness.exception
    assert len(harness.sidebar.button) > before


def test_streamlit_markdown_accepts_latex_delimiters():
    """Streamlit 1.50+ renders $...$ / $$ via KaTeX; this fails if markdown chokes on them."""
    script = r"""
import streamlit as st
st.markdown(r"Inline $E=mc^2$ works.")
st.markdown("Display:\n\n$$\n\\frac{a}{b}\n$$\n")
st.write("rendered")
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception
    bodies = [element.value for element in harness.markdown]
    assert any("E=mc^2" in str(body) for body in bodies)
    assert any("frac" in str(body) for body in bodies)


def test_app_renders_seeded_math_and_assignment_instructions(monkeypatch, tmp_path):
    import json
    import sys
    from pathlib import Path as P

    sys.path.insert(0, str(P(__file__).resolve().parents[1]))
    import app

    db_path = tmp_path / "history.db"
    store = app.ConversationStore(str(db_path))
    conversation_id = store.create_conversation("Math chat")
    store.add_message(conversation_id, {"role": "user", "content": "what does the formula mean?"})
    store.add_message(
        conversation_id,
        {"role": "assistant", "content": r"That is the mass-energy relation $E=mc^2$."},
    )
    store.add_message(
        conversation_id,
        {
            "role": "tool",
            "name": "get_assignment_details",
            "tool_call_id": "call_1",
            "content": json.dumps(
                {
                    "assignment": {
                        "title": "Homework 4",
                        "instructions": r"Compute $\frac{1}{2}$ and then $$ \int_0^1 x\,dx $$.",
                    }
                }
            ),
        },
    )
    harness = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    assert not harness.exception
    rendered = " ".join(str(element.value) for element in harness.markdown)
    assert "E=mc^2" in rendered
    assert r"\frac{1}{2}" in rendered or "frac" in rendered


def test_content_preview_handles_html_tex_and_unknown_types():
    """Exercise the shared preview used by open_file and open_url, without Canvas."""
    repo = str(Path(__file__).resolve().parents[1])
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import app
import streamlit as st

app.render_content_preview(
    b"<h1>Syllabus</h1><p>Grade is \\\\(x^2\\\\).</p>",
    {{"filename": "s.html", "content_type": "text/html", "url": "https://example.com/s"}},
    caption="html from example.com",
    download_key="dl-html",
)
app.render_content_preview(
    br"\\\\frac{{1}}{{2}}",
    {{"filename": "snip.tex", "content_type": "text/x-tex"}},
    caption="tex snippet",
    download_key="dl-tex",
)
app.render_content_preview(
    b"PK binary",
    {{"filename": "slides.zip", "content_type": "application/zip"}},
    caption="zip from Canvas",
    download_key="dl-zip",
)
st.write("previewed")
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception
    rendered = " ".join(str(element.value) for element in harness.markdown)
    assert "previewed" in rendered
    assert "$x^2$" in rendered or "x^2" in rendered
    infos = [str(i.value) for i in harness.info]
    assert any("cannot be previewed" in info for info in infos)


def test_two_open_file_messages_for_the_same_file_do_not_collide():
    """Opening the same Canvas file twice used to raise StreamlitDuplicateElementKey
    because download buttons keyed only on file_id plus a 4-digit filename hash."""
    repo = str(Path(__file__).resolve().parents[1])
    script = f"""
import json
import sys
sys.path.insert(0, {repo!r})
import app

class FakeCanvas:
    def download_file(self, file_id, max_bytes=None):
        return (
            b"Homework 4 notes",
            {{
                "file_id": file_id,
                "filename": "notes.txt",
                "content_type": "text/plain",
                "size_readable": "16 B",
            }},
        )

payload = {{"file": {{"file_id": 14814596, "filename": "notes.txt"}}}}
canvas = FakeCanvas()
for call_id in ("call_hist", "call_new"):
    app.render_tool_message(
        {{
            "role": "tool",
            "name": "open_file",
            "tool_call_id": call_id,
            "content": json.dumps(payload),
        }},
        canvas,
    )
# Missing tool_call_id must still uniquify (counter, not filename hash).
app.render_file_preview(canvas, payload)
app.render_file_preview(canvas, payload)
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    keys = [button.key for button in harness.download_button]
    assert len(keys) == 4
    assert len(set(keys)) == 4
    assert all(key and key.startswith("download-") for key in keys)


def test_two_pdf_previews_for_the_same_file_do_not_collide(tmp_path):
    from test_app import make_pdf

    pdf_path = tmp_path / "syllabus.pdf"
    pdf_path.write_bytes(make_pdf("Syllabus week 1"))
    repo = str(Path(__file__).resolve().parents[1])
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import app

class FakeCanvas:
    def download_file(self, file_id, max_bytes=None):
        return (
            open({str(pdf_path)!r}, "rb").read(),
            {{
                "file_id": file_id,
                "filename": "syllabus.pdf",
                "content_type": "application/pdf",
                "size_readable": "1 KB",
            }},
        )

payload = {{"file": {{"file_id": 14814596, "filename": "syllabus.pdf"}}}}
canvas = FakeCanvas()
app.render_file_preview(canvas, payload, tool_call_id="call_a")
app.render_file_preview(canvas, payload, tool_call_id="call_b")
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    keys = [button.key for button in harness.download_button]
    assert len(keys) == 2
    assert len(set(keys)) == 2


def test_two_submission_attachments_do_not_collide_keys():
    """Two files on one get_my_submission must not share a download-button key
    (including the same file_id appearing twice)."""
    repo = str(Path(__file__).resolve().parents[1])
    script = f"""
import json
import sys
sys.path.insert(0, {repo!r})
import app

class FakeCanvas:
    def download_file(self, file_id, max_bytes=None):
        return (
            b"submission-bytes",
            {{
                "file_id": file_id,
                "filename": f"hw-{{file_id}}.pdf",
                "content_type": "text/plain",
                "size_readable": "16 B",
            }},
        )

payload = {{
    "submission": {{
        "title": "Homework 4",
        "workflow_state": "submitted",
        "attachments": [
            {{"file_id": 8801, "filename": "hw-a.pdf"}},
            {{"file_id": 8801, "filename": "hw-a.pdf"}},
        ],
    }}
}}
app.render_tool_message(
    {{
        "role": "tool",
        "name": "get_my_submission",
        "tool_call_id": "call_sub",
        "content": json.dumps(payload),
    }},
    FakeCanvas(),
)
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    keys = [button.key for button in harness.download_button]
    assert len(keys) == 2
    assert len(set(keys)) == 2
    assert all(key and key.startswith("download-") for key in keys)


def test_tool_message_render_error_leaves_the_rest_of_the_page_intact():
    """A StreamlitDuplicateElementKey (or any Exception) on one tool message
    must become an inline error, not a dead page — later messages still render."""
    repo = str(Path(__file__).resolve().parents[1])
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import app
import streamlit as st
from streamlit.errors import StreamlitDuplicateElementKey

def boom(message, canvas=None):
    raise StreamlitDuplicateElementKey("download-14814596-2110")

_orig = app.render_tool_message
app.render_tool_message = boom
try:
    app.render_message({{"role": "user", "content": "open notes.txt"}})
    app.render_message(
        {{
            "role": "tool",
            "name": "open_file",
            "tool_call_id": "call_1",
            "content": "{{}}",
        }}
    )
    app.render_message({{"role": "assistant", "content": "The notes say Friday."}})
    st.write("page survived")
finally:
    app.render_tool_message = _orig
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    rendered = " ".join(str(element.value) for element in harness.markdown)
    assert "open notes.txt" in rendered
    assert "The notes say Friday." in rendered
    assert "page survived" in rendered
    errors = " ".join(str(element.value) for element in harness.error)
    assert "download-14814596-2110" in errors


def test_sidebar_timezone_defaults_to_pittsburgh(monkeypatch, tmp_path):
    harness = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    assert not harness.exception
    assert len(harness.sidebar.selectbox) == 1
    box = harness.sidebar.selectbox[0]
    assert box.value == "America/New_York"
    joined = " ".join(str(opt) for opt in box.options)
    assert "America/New_York" in joined
    assert "Asia/Shanghai" in joined
    assert "UTC" in joined
    captions = " ".join(str(element.value) for element in harness.sidebar.caption)
    assert "America/New_York" in captions


def test_sidebar_timezone_honors_env_override(monkeypatch, tmp_path):
    harness = run_app(
        monkeypatch, tmp_path, {**DUMMY_ENV, "CANVAS_ASSISTANT_TZ": "America/Los_Angeles"}
    )
    assert not harness.exception
    assert harness.sidebar.selectbox[0].value == "America/Los_Angeles"


def _expander_element_ids(harness) -> set[int]:
    """AppTest flattens expander children into the top-level element lists."""
    ids: set[int] = set()
    for expander in getattr(harness, "expander", []) or []:
        for name in ("markdown", "caption", "title", "text", "code", "info", "warning", "error", "json"):
            for element in getattr(expander, name, []) or []:
                ids.add(id(element))
    return ids


def _default_visible_text(harness) -> str:
    """Markdown, captions, and chat text a student sees without expanding widgets.

    AppTest already flattens chat_message children into harness.markdown / caption.
    Skip expander children — those are collapsed until clicked.
    """
    skip = _expander_element_ids(harness)
    parts: list[str] = []
    for name in ("markdown", "caption", "title", "text", "code", "info", "warning", "error"):
        for element in getattr(harness, name, []) or []:
            if id(element) in skip:
                continue
            parts.append(str(getattr(element, "value", element)))
    return "\n".join(parts)


def _expander_labels(harness) -> list[str]:
    return [str(expander.label) for expander in getattr(harness, "expander", []) or []]


def _json_outside_expanders(harness) -> list:
    skip = _expander_element_ids(harness)
    return [element for element in (getattr(harness, "json", []) or []) if id(element) not in skip]


def _details_expanders(harness):
    import app

    return [exp for exp in (getattr(harness, "expander", []) or []) if exp.label == app.TURN_TRACE_EXPANDER_LABEL]


def _expander_body_text(expander) -> str:
    parts: list[str] = []
    for name in ("markdown", "caption", "title", "text", "code"):
        for element in getattr(expander, name, []) or []:
            parts.append(str(getattr(element, "value", element)))
    for element in getattr(expander, "json", []) or []:
        parts.append(str(getattr(element, "value", element)))
    return "\n".join(parts)


def test_replay_hides_tool_traces_and_shows_final_answer(monkeypatch, tmp_path):
    """On replay, SQLite still has the tool loop but the student only sees the answer."""
    import json
    import sys
    from pathlib import Path as P

    sys.path.insert(0, str(P(__file__).resolve().parents[1]))
    import app

    db_path = tmp_path / "history.db"
    store = app.ConversationStore(str(db_path))
    conversation_id = store.create_conversation("Due dates")
    store.add_message(conversation_id, {"role": "user", "content": "when is homework 4 due?"})
    store.add_message(
        conversation_id,
        {
            "role": "assistant",
            "content": "Let me search Canvas for that.",
            "reasoning_content": "I should call find_due_dates next.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "find_due_dates",
                        "arguments": '{"query": "ZZZ_TOOL_ARG", "course_id": 4242}',
                    },
                }
            ],
        },
    )
    store.add_message(
        conversation_id,
        {
            "role": "tool",
            "name": "find_due_dates",
            "tool_call_id": "call_1",
            "content": json.dumps(
                {
                    "count": 1,
                    "secret_dump": "ZZZ_TOOL_JSON",
                    "assignments": [{"title": "Homework 4", "due_at": "Friday"}],
                }
            ),
        },
    )
    store.add_message(
        conversation_id,
        {"role": "assistant", "content": "Homework 4 is due Friday."},
    )

    harness = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    assert not harness.exception, harness.exception
    visible = _default_visible_text(harness)
    assert "Homework 4 is due Friday." in visible
    assert "when is homework 4 due?" in visible
    assert "Calling" not in visible
    assert "Canvas lookup" not in visible
    assert "ZZZ_TOOL_ARG" not in visible
    assert "ZZZ_TOOL_JSON" not in visible
    assert "Let me search Canvas for that." not in visible
    assert "I should call find_due_dates" not in visible
    assert "find_due_dates" not in visible
    assert '{"query"' not in visible
    assert not any("Canvas lookup" in label for label in _expander_labels(harness))
    assert _json_outside_expanders(harness) == []
    details = _details_expanders(harness)
    assert len(details) == 1
    assert details[0].proto.expanded is False
    body = _expander_body_text(details[0])
    assert "find_due_dates" in body
    assert "ZZZ_TOOL_ARG" in body
    assert "ZZZ_TOOL_JSON" in body
    assert "Let me search Canvas for that." in body

    replayed = app.ConversationStore(str(db_path)).get_messages(conversation_id)
    assert [m["role"] for m in replayed] == ["user", "assistant", "tool", "assistant"]
    assert replayed[1]["tool_calls"][0]["function"]["name"] == "find_due_dates"
    assert "ZZZ_TOOL_JSON" in replayed[2]["content"]


def test_scripted_turn_hides_tool_traces_in_app_test():
    """AppTest after run_agent_turn + render: final answer only, no Calling/lookup/JSON."""
    repo = str(Path(__file__).resolve().parents[1])
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import app

class ScriptedDeepSeek:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, messages, tools=None, temperature=0.2):
        return self.replies.pop(0)

class FakeCanvas:
    tz = None

    def list_courses(self, include_concluded=False):
        return [{{"id": 1, "name": "Course A"}}]

deepseek = ScriptedDeepSeek(
    [
        {{
            "role": "assistant",
            "content": "Let me look that up.",
            "reasoning_content": "Call list_my_courses with ZZZ_TOOL_ARG.",
            "tool_calls": [
                {{
                    "id": "call_1",
                    "type": "function",
                    "function": {{
                        "name": "list_my_courses",
                        "arguments": '{{"ZZZ_TOOL_ARG": 4242}}',
                    }},
                }}
            ],
        }},
        {{"role": "assistant", "content": "You are enrolled in Course A."}},
    ]
)
history = [{{"role": "user", "content": "what courses am I in?"}}]
produced = app.run_agent_turn(deepseek, FakeCanvas(), history)
assert [m["role"] for m in produced] == ["assistant", "tool", "assistant"]
assert "list_my_courses" in (produced[0].get("tool_calls") or [{{}}])[0].get("function", {{}}).get("name", "")
app.render_conversation(history + produced, FakeCanvas())
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    visible = _default_visible_text(harness)
    assert "You are enrolled in Course A." in visible
    assert "what courses am I in?" in visible
    assert "Calling" not in visible
    assert "Canvas lookup" not in visible
    assert "ZZZ_TOOL_ARG" not in visible
    assert "Let me look that up." not in visible
    assert "Call list_my_courses" not in visible
    assert "list_my_courses" not in visible
    assert '{"ZZZ_TOOL_ARG"' not in visible
    assert not any("Canvas lookup" in label for label in _expander_labels(harness))
    assert _json_outside_expanders(harness) == []
    details = _details_expanders(harness)
    assert len(details) == 1
    assert details[0].proto.expanded is False
    body = _expander_body_text(details[0])
    assert "list_my_courses" in body
    assert "ZZZ_TOOL_ARG" in body
    assert "Let me look that up." in body


def test_open_file_preview_still_renders_without_lookup_expander():
    """open_file must still show the file; the Canvas lookup JSON expander must not."""
    repo = str(Path(__file__).resolve().parents[1])
    script = f"""
import json
import sys
sys.path.insert(0, {repo!r})
import app

class FakeCanvas:
    def download_file(self, file_id, max_bytes=None):
        return (
            b"Office hours are Friday.",
            {{
                "file_id": file_id,
                "filename": "notes.txt",
                "content_type": "text/plain",
                "size_readable": "24 B",
            }},
        )

turn = [
    {{
        "role": "assistant",
        "content": "Searching.",
        "tool_calls": [
            {{
                "id": "call_file",
                "type": "function",
                "function": {{
                    "name": "open_file",
                    "arguments": '{{"file_id": 14814596}}',
                }},
            }}
        ],
    }},
    {{
        "role": "tool",
        "name": "open_file",
        "tool_call_id": "call_file",
        "content": json.dumps({{"file": {{"file_id": 14814596, "filename": "notes.txt"}}}}),
    }},
    {{"role": "assistant", "content": "The notes say Friday."}},
]
app.render_conversation(
    [{{"role": "user", "content": "open notes.txt"}}] + turn,
    FakeCanvas(),
)
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    visible = _default_visible_text(harness)
    assert "The notes say Friday." in visible
    assert "open notes.txt" in visible
    assert "Calling" not in visible
    assert "Canvas lookup" not in visible
    assert "Searching." not in visible
    assert "14814596" not in visible
    assert '{"file_id"' not in visible
    keys = [button.key for button in harness.download_button]
    assert len(keys) == 1
    assert keys[0].startswith("download-")
    assert "call_file" in keys[0]
    assert not any("Canvas lookup" in label for label in _expander_labels(harness))
    assert _json_outside_expanders(harness) == []
    details = _details_expanders(harness)
    assert len(details) == 1
    assert details[0].proto.expanded is False
    assert len(details[0].download_button) == 0
    body = _expander_body_text(details[0])
    assert "open_file" in body
    assert "14814596" in body
    # Preview body still includes the file text (code/markdown), not the tool JSON.
    assert "Office hours are Friday." in visible or "notes.txt" in visible


def test_sidebar_timezone_select_persists_across_reruns(monkeypatch, tmp_path):
    harness = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    harness.sidebar.selectbox[0].select("Asia/Shanghai").run()
    assert not harness.exception
    assert harness.sidebar.selectbox[0].value == "Asia/Shanghai"

    restarted = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    assert not restarted.exception
    assert restarted.sidebar.selectbox[0].value == "Asia/Shanghai"


def test_file_panel_empty_state_on_new_chat(monkeypatch, tmp_path):
    harness = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    assert not harness.exception, harness.exception
    captions = " ".join(str(element.value) for element in harness.caption)
    markdown = " ".join(str(element.value) for element in harness.markdown)
    visible = captions + " " + markdown
    assert "No files opened in this chat yet." in visible
    assert len(harness.chat_input) == 1
    assert not any("files-sidebar-choice" in (box.key or "") for box in harness.selectbox)


def test_file_panel_keys_differ_from_chat_and_answer_still_shows():
    """In-chat preview keys stay unique from the right-hand panel of the same file."""
    repo = str(Path(__file__).resolve().parents[1])
    script = f"""
import json
import sys
sys.path.insert(0, {repo!r})
import app
import streamlit as st

class FakeCanvas:
    def download_file(self, file_id, max_bytes=None):
        return (
            b"Office hours are Friday.",
            {{
                "file_id": file_id,
                "filename": "notes.txt",
                "content_type": "text/plain",
                "size_readable": "24 B",
            }},
        )

payload = {{"file": {{"file_id": 14814596, "filename": "notes.txt"}}}}
canvas = FakeCanvas()
chat_col, files_col = st.columns([2, 1])
with chat_col:
    app.render_conversation(
        [
            {{"role": "user", "content": "open notes.txt"}},
            {{
                "role": "assistant",
                "content": "Searching.",
                "tool_calls": [
                    {{
                        "id": "call_file",
                        "type": "function",
                        "function": {{"name": "open_file", "arguments": '{{"file_id": 14814596}}'}},
                    }}
                ],
            }},
            {{
                "role": "tool",
                "name": "open_file",
                "tool_call_id": "call_file",
                "content": json.dumps(payload),
            }},
            {{"role": "assistant", "content": "The notes say Friday."}},
        ],
        canvas,
    )
with files_col:
    app.render_file_preview(canvas, payload, tool_call_id="call_file", key_suffix=app.FILE_PANEL_KEY_SUFFIX)
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    visible = _default_visible_text(harness)
    assert "The notes say Friday." in visible
    assert "open notes.txt" in visible
    keys = [button.key for button in harness.download_button]
    assert len(keys) == 2
    assert len(set(keys)) == 2
    panel_keys = [key for key in keys if key.endswith("-panel") or "-panel" in key]
    chat_keys = [key for key in keys if key not in panel_keys]
    assert len(panel_keys) == 1
    assert len(chat_keys) == 1
    assert panel_keys[0] != chat_keys[0]


def test_file_panel_click_switches_preview(tmp_path):
    repo = str(Path(__file__).resolve().parents[1])
    db_path = tmp_path / "panel.db"
    script = f"""
import json
import sys
sys.path.insert(0, {repo!r})
import app
import streamlit as st

class FakeCanvas:
    def download_file(self, file_id, max_bytes=None):
        names = {{9001: "notes.txt", 9002: "syllabus.pdf"}}
        return (
            f"body-{{file_id}}".encode(),
            {{
                "file_id": file_id,
                "filename": names[file_id],
                "content_type": "text/plain",
                "size_readable": "8 B",
            }},
        )

store = app.ConversationStore({str(db_path)!r})
existing = store.list_conversations()
if existing:
    cid = existing[0]["id"]
else:
    cid = store.create_conversation("files")
    app.remember_opened_files(
        store,
        cid,
        {{
            "role": "tool",
            "name": "open_file",
            "tool_call_id": "call_a",
            "content": json.dumps({{"file": {{"file_id": 9001, "filename": "notes.txt"}}}}),
        }},
    )
    app.remember_opened_files(
        store,
        cid,
        {{
            "role": "tool",
            "name": "open_file",
            "tool_call_id": "call_b",
            "content": json.dumps({{"file": {{"file_id": 9002, "filename": "syllabus.pdf"}}}}),
        }},
    )
app.render_file_panel(store, cid, FakeCanvas())
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    pickers = [box for box in harness.selectbox if (box.key or "").startswith("files-sidebar-choice-")]
    assert len(pickers) == 1
    picker = pickers[0]
    assert picker.options == ["notes.txt", "syllabus.pdf"]
    assert picker.value in {"file:9002", "syllabus.pdf"}
    keys = [button.key for button in harness.download_button]
    assert len(keys) == 1
    assert "-panel" in keys[0]
    rendered = " ".join(str(element.value) for element in harness.markdown)
    rendered += " ".join(str(element.value) for element in harness.caption)
    rendered += " ".join(str(element.value) for element in harness.code)
    assert "syllabus.pdf" in rendered or "body-9002" in rendered

    picker.select_index(0).run()
    assert not harness.exception, harness.exception
    pickers = [box for box in harness.selectbox if (box.key or "").startswith("files-sidebar-choice-")]
    assert pickers[0].value in {"file:9001", "notes.txt"}
    keys = [button.key for button in harness.download_button]
    assert len(keys) == 1
    assert "-panel" in keys[0]
    rendered = " ".join(str(element.value) for element in harness.markdown)
    rendered += " ".join(str(element.value) for element in harness.caption)
    rendered += " ".join(str(element.value) for element in harness.code)
    assert "notes.txt" in rendered or "body-9001" in rendered


def test_files_sidebar_collapse_hides_dropdown(tmp_path):
    repo = str(Path(__file__).resolve().parents[1])
    db_path = tmp_path / "panel.db"
    script = f"""
import json
import sys
sys.path.insert(0, {repo!r})
import app

class FakeCanvas:
    def download_file(self, file_id, max_bytes=None):
        return b"x", {{"file_id": file_id, "filename": "notes.txt", "content_type": "text/plain"}}

store = app.ConversationStore({str(db_path)!r})
cid = store.create_conversation("files")
app.remember_opened_files(
    store,
    cid,
    {{
        "role": "tool",
        "name": "open_file",
        "tool_call_id": "call_a",
        "content": json.dumps({{"file": {{"file_id": 9001, "filename": "notes.txt"}}}}),
    }},
)
app.render_file_panel(store, cid, FakeCanvas())
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    assert any((box.key or "").startswith("files-sidebar-choice-") for box in harness.selectbox)
    collapse = next(button for button in harness.button if button.key == "files-sidebar-collapse")
    collapse.click().run()
    assert not harness.exception, harness.exception
    assert not any((box.key or "").startswith("files-sidebar-choice-") for box in harness.selectbox)
    expand = next(button for button in harness.button if button.key == "files-sidebar-expand")
    expand.click().run()
    assert not harness.exception, harness.exception
    assert any((box.key or "").startswith("files-sidebar-choice-") for box in harness.selectbox)


def test_file_panel_isolated_when_switching_conversations(monkeypatch, tmp_path):
    import json
    import sys
    from pathlib import Path as P

    sys.path.insert(0, str(P(__file__).resolve().parents[1]))
    import app

    db_path = tmp_path / "history.db"
    store = app.ConversationStore(str(db_path))
    with_files = store.create_conversation("Has files")
    store.add_message(with_files, {"role": "user", "content": "open notes.txt"})
    store.add_message(
        with_files,
        {
            "role": "tool",
            "name": "open_file",
            "tool_call_id": "call_file",
            "content": json.dumps({"file": {"file_id": 9001, "filename": "notes.txt"}}),
        },
    )
    store.add_message(with_files, {"role": "assistant", "content": "The notes say Friday."})
    empty = store.create_conversation("Empty chat")
    store.add_message(empty, {"role": "user", "content": "hello"})
    store.add_message(empty, {"role": "assistant", "content": "Hi — what should we look up?"})

    harness = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    assert not harness.exception, harness.exception
    # Newest chat is Empty chat; panel must not list notes.txt.
    visible = _default_visible_text(harness)
    assert "Hi — what should we look up?" in visible
    captions = " ".join(str(element.value) for element in harness.caption)
    assert "No files opened in this chat yet." in captions + visible
    assert not any((box.key or "").startswith("files-sidebar-choice-") for box in harness.selectbox)

    opened = None
    for button in harness.sidebar.button:
        if "Has files" in str(button.label):
            opened = button
            break
    assert opened is not None
    opened.click().run()
    assert not harness.exception, harness.exception
    visible = _default_visible_text(harness)
    assert "The notes say Friday." in visible
    pickers = [box for box in harness.selectbox if (box.key or "").startswith("files-sidebar-choice-")]
    assert len(pickers) == 1
    assert pickers[0].value in {"file:9001", "notes.txt"}
    captions = " ".join(str(element.value) for element in harness.caption)
    assert "No files opened in this chat yet." not in captions

