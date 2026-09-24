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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402

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
    for name in (
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "CANVAS_API_TOKEN",
        "CANVAS_BASE_URL",
        "CANVAS_ASSISTANT_TZ",
    ):
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
    assert not harness.chat_input
    assert not harness.error


def test_app_explains_missing_configuration(monkeypatch, tmp_path):
    harness = run_app(monkeypatch, tmp_path, {})
    assert not harness.exception
    message = harness.error[0].value
    assert "DEEPSEEK_API_KEY" in message and "CANVAS_API_TOKEN" in message
    assert not harness.chat_input


def test_app_explains_missing_openai_key_without_asking_for_deepseek(monkeypatch, tmp_path):
    harness = run_app(
        monkeypatch,
        tmp_path,
        {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-4o-mini",
            "CANVAS_API_TOKEN": DUMMY_ENV["CANVAS_API_TOKEN"],
        },
    )
    assert not harness.exception
    message = harness.error[0].value
    assert "OPENAI_API_KEY" in message
    assert "DEEPSEEK_API_KEY" not in message
    assert DUMMY_ENV["CANVAS_API_TOKEN"] not in message


def test_app_explains_unknown_provider_without_falling_back(monkeypatch, tmp_path):
    harness = run_app(
        monkeypatch,
        tmp_path,
        {
            **DUMMY_ENV,
            "LLM_PROVIDER": "grok",
            "LLM_MODEL": "anything",
        },
    )
    assert not harness.exception
    message = harness.error[0].value
    assert "Unknown LLM_PROVIDER" in message
    assert "grok" in message
    assert "does not switch" in message.lower() or "not switch" in message.lower()
    assert DUMMY_ENV["DEEPSEEK_API_KEY"] not in message


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
        {
            "role": "assistant",
            "content": r"That is the mass-energy relation $E=mc^2$ and half is $\frac{1}{2}$.",
        },
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
                        "instructions": r"Compute $\int_0^1 x\,dx$ from a leftover assignment object.",
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
    # An assignment object in a tool payload must not become a full page preview.
    assert "leftover assignment object" not in rendered


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
    assert not harness.chat_input
    assert list(harness.session_state.get("opened_files") or []) == []
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
import files_rail
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
files = store.list_opened_files(cid)
chosen = app.sync_files_sidebar_selection(cid, files)
assert chosen == "file:9002"
preview = app.build_panel_preview(FakeCanvas(), app.panel_entry_for_choice(chosen, files))
assert preview["kind"] == "text"
assert "body-9002" in preview["text"]
app.apply_files_rail_value(
    {{"collapsed": False, "selected": "file:9001"}},
    conversation_id=cid,
    files=files,
)
assert st.session_state.panel_selected == "file:9001"
files_rail.mount = lambda **kwargs: {{"collapsed": False, "selected": kwargs["selected"]}}
app.render_file_panel(store, cid, FakeCanvas())
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    assert harness.session_state.panel_selected == "file:9001"


def test_files_sidebar_collapse_hides_dropdown(tmp_path):
    repo = str(Path(__file__).resolve().parents[1])
    db_path = tmp_path / "panel.db"
    script = f"""
import json
import sys
sys.path.insert(0, {repo!r})
import app
import files_rail
import streamlit as st

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
files = store.list_opened_files(cid)
files_rail.mount = lambda **kwargs: {{"collapsed": True, "selected": kwargs.get("selected")}}
app.render_file_panel(store, cid, FakeCanvas())
assert st.session_state.files_sidebar_collapsed is True
app.apply_files_rail_value(
    {{"collapsed": False, "selected": files[0]["identity"]}},
    conversation_id=cid,
    files=files,
)
assert st.session_state.files_sidebar_collapsed is False
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    assert harness.session_state.files_sidebar_collapsed is False
    assert not any((box.key or "").startswith("files-sidebar-choice-") for box in harness.selectbox)


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
    assert list(harness.session_state.get("opened_files") or []) == []
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
    opened_files = list(harness.session_state.get("opened_files") or [])
    assert [entry.get("filename") for entry in opened_files] == ["notes.txt"]
    assert not any((box.key or "").startswith("files-sidebar-choice-") for box in harness.selectbox)


def test_chat_hides_unrelated_assignment_urls_and_caps_summary_link():
    """List html_urls stay off-screen; assignment-summary shows one Canvas link."""
    repo = str(Path(__file__).resolve().parents[1])
    script = f"""
import json
import sys
sys.path.insert(0, {repo!r})
import app

hw4 = "https://canvas.cmu.edu/courses/1/assignments/3100"
hw5 = "https://canvas.cmu.edu/courses/1/assignments/3101"
app.render_conversation(
    [
        {{"role": "user", "content": "what's due?"}},
        {{
            "role": "assistant",
            "content": "Searching.",
            "tool_calls": [
                {{
                    "id": "call_list",
                    "type": "function",
                    "function": {{"name": "list_upcoming_assignments", "arguments": "{{}}"}},
                }}
            ],
        }},
        {{
            "role": "tool",
            "name": "list_upcoming_assignments",
            "tool_call_id": "call_list",
            "content": json.dumps(
                {{
                    "count": 2,
                    "assignments": [
                        {{"title": "Homework 4", "html_url": hw4}},
                        {{"title": "Homework 5", "html_url": hw5}},
                    ],
                }}
            ),
        }},
        {{
            "role": "assistant",
            "content": f"Homework 4 and Homework 5 are due. {{hw4}} {{hw5}}",
        }},
        {{"role": "user", "content": "/summarize what I need to do for Homework 4"}},
        {{
            "role": "assistant",
            "content": "Looking up the prompt.",
            "tool_calls": [
                {{
                    "id": "call_details",
                    "type": "function",
                    "function": {{"name": "get_assignment_details", "arguments": '{{"assignment_id": 3100}}'}},
                }}
            ],
        }},
        {{
            "role": "tool",
            "name": "get_assignment_details",
            "tool_call_id": "call_details",
            "content": json.dumps(
                {{
                    "assignment": {{
                        "title": "Homework 4",
                        "html_url": hw4,
                        "instructions": "Write 1200 words from a leftover assignment object.",
                    }}
                }}
            ),
        }},
        {{"role": "assistant", "content": "Write a 1200-word comparative analysis."}},
    ]
)
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    visible = _default_visible_text(harness)
    assert "Homework 4 and Homework 5 are due." in visible
    assert "Write a 1200-word comparative analysis." in visible
    assert "leftover assignment object" not in visible
    assert visible.count("https://canvas.cmu.edu/courses/1/assignments/3100") == 1
    assert "https://canvas.cmu.edu/courses/1/assignments/3101" not in visible
    assert visible.count("Open assignment in Canvas") == 1


def test_thinking_indicator_visible_until_cleared():
    """Leave the assistant-slot status open: spinner + sequential label + aria text."""
    repo = str(Path(__file__).resolve().parents[1])
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import app
import streamlit as st

with st.chat_message("user"):
    st.markdown("when is homework 4 due?")
with st.chat_message("assistant"):
    slot = st.empty()
    thinking = app.AssistantThinking(slot, conversation_id=7, turn_seq=1)
    thinking.show()
    thinking.advance(app.THINKING_STAGE_CANVAS)
    thinking.show()
    thinking.advance(app.THINKING_STAGE_ANSWER)
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    assert len(harness.status) == 1
    assert harness.status[0].icon == "spinner"
    assert harness.status[0].label == app.thinking_label(app.THINKING_STAGE_ANSWER)
    rendered = " ".join(str(element.value) for element in harness.markdown)
    assert app.THINKING_ARIA_LABEL in rendered
    assert len(harness.chat_message) == 2
    assert harness.chat_message[1].name == "assistant"
    state = harness.session_state[app.THINKING_STATE_KEY]
    assert state["active"] is True
    assert state["key"] == app.thinking_placeholder_key(7, 1)


def test_thinking_indicator_appears_during_turn_and_clears_after_success(tmp_path):
    repo = str(Path(__file__).resolve().parents[1])
    db_path = str(tmp_path / "history.db")
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import app
import streamlit as st

class SlowDeepSeek:
    def complete(self, messages, tools=None, temperature=0.2):
        st.session_state["_during"] = dict(app.get_thinking_state())
        st.session_state["_during_busy"] = bool(st.session_state.get("composer_busy"))
        st.session_state["_during_status_label"] = app.get_thinking_state().get("label")
        return {{"role": "assistant", "content": "Homework 4 is due Friday."}}

class FakeCanvas:
    tz = None

store = app.ConversationStore({db_path!r})
cid = store.create_conversation("Thinking chat")
history = store.get_messages(cid)
app.handle_user_prompt(
    "when is homework 4 due?",
    store=store,
    conversation_id=cid,
    history=history,
    deepseek=SlowDeepSeek(),
    canvas=FakeCanvas(),
    rerun=False,
)
st.session_state["_after"] = dict(app.get_thinking_state())
st.session_state["_after_busy"] = bool(st.session_state.get("composer_busy"))
st.session_state["_stored"] = store.get_messages(cid)
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    during = harness.session_state["_during"]
    assert during["active"] is True
    assert during["label"] in app.THINKING_LABELS
    assert harness.session_state["_during_busy"] is True
    assert harness.status == []
    visible = _default_visible_text(harness)
    assert "Homework 4 is due Friday." in visible
    assert "when is homework 4 due?" in visible
    assert "Calling" not in visible
    after = harness.session_state["_after"]
    assert after["active"] is False
    assert after["label"] is None
    assert harness.session_state["_after_busy"] is False
    stored = harness.session_state["_stored"]
    contents = [message.get("content") or "" for message in stored]
    assert "Homework 4 is due Friday." in contents
    for label in app.THINKING_LABELS:
        assert label not in contents
    assert app.THINKING_ARIA_LABEL not in contents
    assert [message["role"] for message in stored] == ["user", "assistant"]


def test_thinking_indicator_clears_after_failure_and_shows_error(tmp_path):
    repo = str(Path(__file__).resolve().parents[1])
    db_path = str(tmp_path / "history.db")
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import app
import streamlit as st

class BoomDeepSeek:
    def complete(self, messages, tools=None, temperature=0.2):
        st.session_state["_during"] = dict(app.get_thinking_state())
        raise RuntimeError("model unavailable")

class FakeCanvas:
    tz = None

store = app.ConversationStore({db_path!r})
cid = store.create_conversation("Error chat")
history = store.get_messages(cid)
app.handle_user_prompt(
    "when is homework 4 due?",
    store=store,
    conversation_id=cid,
    history=history,
    deepseek=BoomDeepSeek(),
    canvas=FakeCanvas(),
    rerun=False,
)
st.session_state["_after"] = dict(app.get_thinking_state())
st.session_state["_stored"] = store.get_messages(cid)
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    assert harness.session_state["_during"]["active"] is True
    assert harness.status == []
    assert harness.error
    assert "model unavailable" in str(harness.error[0].value)
    stored = harness.session_state["_stored"]
    contents = [message.get("content") or "" for message in stored]
    for label in app.THINKING_LABELS:
        assert label not in contents
    assert app.THINKING_ARIA_LABEL not in contents
    assert stored[0]["role"] == "user"


def test_thinking_indicator_does_not_duplicate_on_rerun(tmp_path):
    repo = str(Path(__file__).resolve().parents[1])
    db_path = str(tmp_path / "history.db")
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import app
import streamlit as st

class InstantDeepSeek:
    def complete(self, messages, tools=None, temperature=0.2):
        return {{"role": "assistant", "content": "You are enrolled in Course A."}}

class FakeCanvas:
    tz = None

store = app.ConversationStore({db_path!r})
cid = store.create_conversation("Replay chat")
history = store.get_messages(cid)
app.handle_user_prompt(
    "what courses am I in?",
    store=store,
    conversation_id=cid,
    history=history,
    deepseek=InstantDeepSeek(),
    canvas=FakeCanvas(),
    rerun=False,
)
# Same-run replay path: history render must not stack another thinking row.
app.render_conversation(store.get_messages(cid), FakeCanvas())
with st.chat_message("assistant"):
    slot = st.empty()
    thinking = app.AssistantThinking(slot, conversation_id=cid, turn_seq=1)
    thinking.show()
    thinking.show()
    thinking.show(app.THINKING_STAGE_CANVAS)
st.session_state["_replayed"] = store.get_messages(cid)
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    # One leftover status from the explicit second show(); the completed turn added none.
    assert len(harness.status) == 1
    assert harness.status[0].icon == "spinner"
    visible = _default_visible_text(harness)
    assert visible.count("You are enrolled in Course A.") >= 1
    stored = harness.session_state["_replayed"]
    contents = [message.get("content") or "" for message in stored]
    for label in app.THINKING_LABELS:
        assert label not in contents


def test_thinking_indicator_advances_through_tool_workflow(tmp_path):
    repo = str(Path(__file__).resolve().parents[1])
    db_path = str(tmp_path / "history.db")
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import app
import streamlit as st

class TwoStepDeepSeek:
    def __init__(self):
        self.n = 0

    def complete(self, messages, tools=None, temperature=0.2):
        snaps = st.session_state.setdefault("_snaps", [])
        snaps.append(dict(app.get_thinking_state()))
        self.n += 1
        if self.n == 1:
            return {{
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
            }}
        return {{"role": "assistant", "content": "You are enrolled in Course A."}}

class FakeCanvas:
    tz = None

    def list_courses(self, include_concluded=False):
        return [{{"id": 1, "name": "Course A"}}]

store = app.ConversationStore({db_path!r})
cid = store.create_conversation("Tools chat")
history = store.get_messages(cid)
app.handle_user_prompt(
    "what courses am I in?",
    store=store,
    conversation_id=cid,
    history=history,
    deepseek=TwoStepDeepSeek(),
    canvas=FakeCanvas(),
    rerun=False,
)
st.session_state["_stored"] = store.get_messages(cid)
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    snaps = list(harness.session_state["_snaps"])
    assert snaps[0]["active"] is True
    assert snaps[0]["label"] == app.thinking_label(app.THINKING_STAGE_THINKING)
    assert snaps[-1]["label"] == app.thinking_label(app.THINKING_STAGE_ANSWER)
    assert harness.status == []
    visible = _default_visible_text(harness)
    assert "You are enrolled in Course A." in visible
    assert "Let me look that up." not in visible
    assert "ZZZ_TOOL_ARG" not in visible
    assert "list_my_courses" not in visible
    stored = harness.session_state["_stored"]
    contents = [message.get("content") or "" for message in stored]
    for label in app.THINKING_LABELS:
        assert label not in contents
    assert stored[-1]["content"] == "You are enrolled in Course A."


def test_thinking_indicator_survives_rerun_mid_turn(tmp_path):
    """Files-rail-style st.rerun() aborts the script; session_state restores the spinner."""
    repo = str(Path(__file__).resolve().parents[1])
    db_path = str(tmp_path / "history.db")
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import app
import files_rail
import streamlit as st

class InterruptDeepSeek:
    def complete(self, messages, tools=None, temperature=0.2):
        n = int(st.session_state.get("_calls") or 0) + 1
        st.session_state["_calls"] = n
        st.session_state[f"_think_{{n}}"] = dict(app.get_thinking_state())
        st.session_state[f"_busy_{{n}}"] = bool(st.session_state.get("composer_busy"))
        st.session_state[f"_turn_{{n}}"] = dict(app.get_turn_in_progress())
        st.session_state[f"_stored_at_call_{{n}}"] = store.get_messages(cid)
        if n == 1:
            st.session_state[files_rail.COMPONENT_KEY] = {{
                "collapsed": True,
                "selected": None,
                "width_px": 320,
            }}
            st.rerun()
        return {{"role": "assistant", "content": "Homework 4 is due Friday."}}

class FakeCanvas:
    tz = None

store = app.ConversationStore({db_path!r})
if "cid" not in st.session_state:
    st.session_state.cid = store.create_conversation("Rerun chat")
cid = st.session_state.cid
history = store.get_messages(cid)
app.handle_user_prompt(
    "when is homework 4 due?",
    store=store,
    conversation_id=cid,
    history=history,
    deepseek=InterruptDeepSeek(),
    canvas=FakeCanvas(),
    rerun=False,
)
st.session_state["_after"] = dict(app.get_thinking_state())
st.session_state["_after_turn"] = dict(app.get_turn_in_progress())
st.session_state["_after_busy"] = bool(st.session_state.get("composer_busy"))
st.session_state["_stored"] = store.get_messages(cid)
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    assert harness.session_state["_calls"] == 2
    stored_at_first = harness.session_state["_stored_at_call_1"]
    stored_at_resume = harness.session_state["_stored_at_call_2"]
    assert [message["role"] for message in stored_at_first] == ["user"]
    assert [message["role"] for message in stored_at_resume] == ["user"]
    assert stored_at_resume[0]["content"] == "when is homework 4 due?"
    first = harness.session_state["_think_1"]
    second = harness.session_state["_think_2"]
    assert first["active"] is True
    assert second["active"] is True
    assert first["label"] in app.THINKING_LABELS
    assert second["label"] in app.THINKING_LABELS
    assert first["key"] == second["key"]
    assert harness.session_state["_busy_1"] is True
    assert harness.session_state["_busy_2"] is True
    assert harness.session_state["_turn_1"]["active"] is True
    assert harness.session_state["_turn_2"]["active"] is True
    assert harness.session_state["_turn_2"]["prompt"] == "when is homework 4 due?"
    after = harness.session_state["_after"]
    assert after["active"] is False
    assert harness.session_state["_after_turn"]["active"] is False
    assert harness.session_state["_after_busy"] is False
    assert harness.status == []
    visible = _default_visible_text(harness)
    assert "Homework 4 is due Friday." in visible
    stored = harness.session_state["_stored"]
    users = [message["content"] for message in stored if message.get("role") == "user"]
    assistants = [message["content"] for message in stored if message.get("role") == "assistant"]
    assert users == ["when is homework 4 due?"]
    assert assistants == ["Homework 4 is due Friday."]
    contents = [message.get("content") or "" for message in stored]
    for label in app.THINKING_LABELS:
        assert label not in contents
    assert app.THINKING_ARIA_LABEL not in contents


def test_handle_prompt_none_restarts_complete_when_no_answer(tmp_path):
    """B: handle_prompt(None) after an aborted turn with no assistant row calls complete()."""
    repo = str(Path(__file__).resolve().parents[1])
    db_path = str(tmp_path / "history.db")
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import app
import streamlit as st

class ResumeDeepSeek:
    def complete(self, messages, tools=None, temperature=0.2):
        st.session_state["_resume_complete"] = True
        st.session_state["_resume_stored_before"] = store.get_messages(cid)
        return {{"role": "assistant", "content": "Homework 4 is due Friday."}}

class FakeCanvas:
    tz = None

store = app.ConversationStore({db_path!r})
cid = store.create_conversation("Resume via handle_prompt")
prompt = "when is homework 4 due?"
store.add_message(cid, {{"role": "user", "content": prompt}})
app.mark_turn_in_progress(conversation_id=cid, prompt=prompt, turn_seq=1)
st.session_state.command_picker_submit_seq = 9
history = store.get_messages(cid)
deepseek = ResumeDeepSeek()
canvas = FakeCanvas()

def handle_prompt(prompt_text):
    if not prompt_text:
        turn = app.get_turn_in_progress()
        if not app.turn_is_active(cid):
            return
        prompt_text = turn.get("prompt")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            return
    app.handle_user_prompt(
        prompt_text,
        store=store,
        conversation_id=cid,
        history=history,
        deepseek=deepseek,
        canvas=canvas,
        rerun=False,
    )

handle_prompt(None)
st.session_state["_stored"] = store.get_messages(cid)
st.session_state["_after"] = dict(app.get_thinking_state())
st.session_state["_turn"] = dict(app.get_turn_in_progress())
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    assert harness.session_state["_resume_complete"] is True
    before = harness.session_state["_resume_stored_before"]
    assert [message["role"] for message in before] == ["user"]
    assert before[0]["content"] == "when is homework 4 due?"
    stored = harness.session_state["_stored"]
    users = [message["content"] for message in stored if message.get("role") == "user"]
    assistants = [message["content"] for message in stored if message.get("role") == "assistant"]
    assert users == ["when is homework 4 due?"]
    assert assistants == ["Homework 4 is due Friday."]
    assert harness.session_state["_after"]["active"] is False
    assert harness.session_state["_turn"]["active"] is False
    visible = _default_visible_text(harness)
    assert "Homework 4 is due Friday." in visible
    assert harness.status == []
    contents = [message.get("content") or "" for message in stored]
    for label in app.THINKING_LABELS:
        assert label not in contents


def test_thinking_indicator_restored_on_files_rail_value_change(tmp_path):
    """Aborted turn + files-rail widget value: spinner comes back, still not in SQLite."""
    repo = str(Path(__file__).resolve().parents[1])
    db_path = str(tmp_path / "history.db")
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import app
import files_rail
import streamlit as st

store = app.ConversationStore({db_path!r})
if "cid" not in st.session_state:
    st.session_state.cid = store.create_conversation("Rail chat")
cid = st.session_state.cid
prompt = "when is homework 4 due?"
if "seeded" not in st.session_state:
    store.add_message(cid, {{"role": "user", "content": prompt}})
    app.mark_turn_in_progress(conversation_id=cid, prompt=prompt, turn_seq=1)
    st.session_state.composer_busy = True
    st.session_state.command_picker_submit_seq = 3
    st.session_state.seeded = True
    st.session_state[files_rail.COMPONENT_KEY] = {{
        "collapsed": False,
        "selected": None,
        "width_px": 320,
    }}
    st.rerun()

st.session_state[files_rail.COMPONENT_KEY] = {{
    "collapsed": True,
    "selected": None,
    "width_px": 360,
}}
history = store.get_messages(cid)
app.apply_files_rail_value(
    st.session_state.get(files_rail.COMPONENT_KEY),
    conversation_id=cid,
    files=[],
)
app.render_conversation(history)
thinking = app.restore_thinking_for_active_turn(cid, history)
st.session_state["_thinking"] = dict(app.get_thinking_state())
st.session_state["_turn"] = dict(app.get_turn_in_progress())
st.session_state["_busy"] = bool(st.session_state.get("composer_busy"))
st.session_state["_stored"] = store.get_messages(cid)
st.session_state["_has_thinking"] = thinking is not None
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    assert harness.session_state["_has_thinking"] is True
    assert harness.session_state["_thinking"]["active"] is True
    assert harness.session_state["_thinking"]["label"] in app.THINKING_LABELS
    assert harness.session_state["_thinking"]["key"] == app.thinking_placeholder_key(
        harness.session_state["_turn"]["conversation_id"], 1
    )
    assert harness.session_state["_turn"]["active"] is True
    assert harness.session_state["_busy"] is True
    assert len(harness.status) == 1
    assert harness.status[0].icon == "spinner"
    rendered = " ".join(str(element.value) for element in harness.markdown)
    assert app.THINKING_ARIA_LABEL in rendered
    stored = harness.session_state["_stored"]
    assert [message["role"] for message in stored] == ["user"]
    contents = [message.get("content") or "" for message in stored]
    for label in app.THINKING_LABELS:
        assert label not in contents
    assert app.THINKING_ARIA_LABEL not in contents
    assert harness.session_state.files_sidebar_collapsed is True


def test_thinking_indicator_hidden_after_resume_completes(tmp_path):
    """After a restored turn finishes (success), the spinner is gone and a second prompt works."""
    repo = str(Path(__file__).resolve().parents[1])
    db_path = str(tmp_path / "history.db")
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import app
import streamlit as st

class InstantDeepSeek:
    def complete(self, messages, tools=None, temperature=0.2):
        n = int(st.session_state.get("_turns") or 0) + 1
        st.session_state["_turns"] = n
        return {{"role": "assistant", "content": f"Answer {{n}}."}}

class FakeCanvas:
    tz = None

store = app.ConversationStore({db_path!r})
cid = store.create_conversation("Resume finish")
prompt = "when is homework 4 due?"
store.add_message(cid, {{"role": "user", "content": prompt}})
app.mark_turn_in_progress(conversation_id=cid, prompt=prompt, turn_seq=1)
st.session_state.command_picker_submit_seq = 1
history = store.get_messages(cid)
app.handle_user_prompt(
    prompt,
    store=store,
    conversation_id=cid,
    history=history,
    deepseek=InstantDeepSeek(),
    canvas=FakeCanvas(),
    rerun=False,
)
st.session_state["_after_first"] = dict(app.get_thinking_state())
st.session_state["_after_first_busy"] = bool(st.session_state.get("composer_busy"))
st.session_state["_after_first_turn"] = dict(app.get_turn_in_progress())
history = store.get_messages(cid)
app.handle_user_prompt(
    "and homework 5?",
    store=store,
    conversation_id=cid,
    history=history,
    deepseek=InstantDeepSeek(),
    canvas=FakeCanvas(),
    rerun=False,
)
st.session_state["_after"] = dict(app.get_thinking_state())
st.session_state["_stored"] = store.get_messages(cid)
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    assert harness.session_state["_after_first"]["active"] is False
    assert harness.session_state["_after_first_busy"] is False
    assert harness.session_state["_after_first_turn"]["active"] is False
    assert harness.session_state["_after"]["active"] is False
    assert harness.status == []
    stored = harness.session_state["_stored"]
    users = [message["content"] for message in stored if message.get("role") == "user"]
    assistants = [message["content"] for message in stored if message.get("role") == "assistant"]
    assert users == ["when is homework 4 due?", "and homework 5?"]
    assert assistants == ["Answer 1.", "Answer 2."]
    contents = [message.get("content") or "" for message in stored]
    for label in app.THINKING_LABELS:
        assert label not in contents


def test_resume_does_not_call_deepseek_when_answer_already_stored(tmp_path):
    repo = str(Path(__file__).resolve().parents[1])
    db_path = str(tmp_path / "history.db")
    script = f"""
import sys
sys.path.insert(0, {repo!r})
import app
import streamlit as st

class BoomIfCalled:
    def complete(self, messages, tools=None, temperature=0.2):
        st.session_state["_called"] = True
        raise RuntimeError("should not call DeepSeek again")

class FakeCanvas:
    tz = None

store = app.ConversationStore({db_path!r})
cid = store.create_conversation("Already done")
prompt = "when is homework 4 due?"
store.add_message(cid, {{"role": "user", "content": prompt}})
store.add_message(cid, {{"role": "assistant", "content": "Friday."}})
app.mark_turn_in_progress(conversation_id=cid, prompt=prompt, turn_seq=2)
st.session_state.command_picker_submit_seq = 4
history = store.get_messages(cid)
app.handle_user_prompt(
    prompt,
    store=store,
    conversation_id=cid,
    history=history,
    deepseek=BoomIfCalled(),
    canvas=FakeCanvas(),
    rerun=False,
)
st.session_state["_called"] = bool(st.session_state.get("_called"))
st.session_state["_after"] = dict(app.get_thinking_state())
st.session_state["_stored"] = store.get_messages(cid)
st.session_state["_turn"] = dict(app.get_turn_in_progress())
"""
    harness = AppTest.from_string(script, default_timeout=30).run()
    assert not harness.exception, harness.exception
    assert harness.session_state["_called"] is False
    assert harness.session_state["_after"]["active"] is False
    assert harness.session_state["_turn"]["active"] is False
    stored = harness.session_state["_stored"]
    users = [message["content"] for message in stored if message.get("role") == "user"]
    assistants = [message["content"] for message in stored if message.get("role") == "assistant"]
    assert users == ["when is homework 4 due?"]
    assert assistants == ["Friday."]
    assert harness.status == []

