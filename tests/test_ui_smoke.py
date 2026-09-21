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


def test_sidebar_timezone_select_persists_across_reruns(monkeypatch, tmp_path):
    harness = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    harness.sidebar.selectbox[0].select("Asia/Shanghai").run()
    assert not harness.exception
    assert harness.sidebar.selectbox[0].value == "Asia/Shanghai"

    restarted = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    assert not restarted.exception
    assert restarted.sidebar.selectbox[0].value == "Asia/Shanghai"
