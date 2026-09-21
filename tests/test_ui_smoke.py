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
