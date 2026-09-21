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
    for name in ("DEEPSEEK_API_KEY", "CANVAS_API_TOKEN", "CANVAS_BASE_URL"):
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
