"""Files rail custom component: payload helpers and iframe contract."""

from __future__ import annotations

from pathlib import Path

import app
import files_rail


class FakeCanvas:
    def download_file(self, file_id, max_bytes=None):
        return (
            b"Office hours are Friday.",
            {
                "file_id": file_id,
                "filename": "notes.txt",
                "content_type": "text/plain",
                "size_readable": "24 B",
            },
        )

    def fetch_url(self, url, max_bytes=None):
        return (
            b"%PDF-1.4",
            {
                "url": url,
                "filename": "syllabus.pdf",
                "content_type": "application/pdf",
                "size_readable": "8 B",
            },
        )


def test_frontend_is_a_declare_component_iframe():
    html = (files_rail.FRONTEND_DIR / "index.html").read_text()
    assert files_rail.FRONTEND_DIR.is_dir()
    assert (files_rail.FRONTEND_DIR / "index.html").is_file()
    assert "streamlit:componentReady" in html
    assert "apiVersion" in html
    assert "streamlit:setComponentValue" in html
    assert files_rail.HOST_ID in html
    assert files_rail.STYLE_ID in html
    assert "position: fixed" in html
    assert "stAppScrollToBottomContainer" in html
    assert '[data-testid="stMain"]' in html
    assert "padding-right: var(--canvas-files-rail-width)" in html
    assert '[data-testid="stChatMessage"]' in html
    assert "doc.body.appendChild(host)" in html
    assert "files-rail-toggle" in html
    assert "files-rail-chevron" in html
    assert "files-rail-resize" in html
    assert "col-resize" in html
    assert "cursor: col-resize" in html
    assert "ew-resize" not in html
    assert "border-left: 1px solid" in html
    assert "rgba(49, 51, 63, 0.2)" in html
    resize_chunk = html[
        html.find("#canvas-files-rail-host .files-rail-resize") : html.find(
            "#canvas-files-rail-host .files-rail-toggle"
        )
    ]
    assert "cursor: col-resize" in resize_chunk
    assert "linear-gradient" not in resize_chunk
    assert "ff4b4b" not in resize_chunk
    assert "primary" not in resize_chunk
    assert "button.files-rail-download" in html
    assert "createElement(\"button\")" in html
    assert "a.files-rail-download" not in html
    assert "files-rail-preview" in html
    assert "files-rail-preview-pdf" in html
    pdf_chunk = html[
        html.find("#canvas-files-rail-host iframe.files-rail-pdf") : html.find(
            "#canvas-files-rail-host button.files-rail-download"
        )
    ]
    assert "height: 100%" in pdf_chunk
    assert "flex: 1 1 auto" in pdf_chunk
    assert "min-height: 0" in pdf_chunk
    assert "min(55vh" not in html
    assert "max-height: 55vh" not in html
    assert "bottom: 0" in html
    assert "height: 100vh" in html
    assert "height: 100dvh" in html
    assert "localStorage" in html
    assert "st-key-files-sidebar" not in html
    assert "position: sticky" not in html
    assert '[data-testid="stColumn"]' not in html
    assert "chevron_right" not in html
    assert "chevron_left" not in html
    assert "AppViewContainer" not in html
    assert Path(files_rail._component.path) == files_rail.FRONTEND_DIR


def test_frontend_insets_stmain_by_live_rail_width():
    html = (files_rail.FRONTEND_DIR / "index.html").read_text()
    main_rule = html[html.find('[data-testid="stMain"]') : html.find('[data-testid="stChatMessage"]')]
    assert "padding-right: var(--canvas-files-rail-width)" in main_rule
    assert "box-sizing: border-box" in main_rule
    assert '[data-testid="stColumn"]' not in html
    assert "position: sticky" not in html
    chrome = html[html.find("function chromeCss") : html.find("function applyChrome")]
    assert "collapsed ? \"0px\"" in chrome or "collapsed ? \"0px\"" in html


def test_clamp_width_matches_sidebar_band():
    assert files_rail.clamp_width(320) == 320
    assert files_rail.clamp_width(80) == files_rail.RAIL_MIN_WIDTH_PX
    assert files_rail.clamp_width(900) == files_rail.RAIL_MAX_WIDTH_PX
    assert files_rail.clamp_width("nope") == files_rail.RAIL_WIDTH_PX


def test_build_panel_preview_text_and_pdf_url():
    text = app.build_panel_preview(
        FakeCanvas(),
        {"kind": "file", "identity": "file:1", "file_id": 1, "filename": "notes.txt"},
    )
    assert text["kind"] == "text"
    assert "Office hours are Friday." in text["text"]
    assert text["data_url"].startswith("data:text/plain;base64,")

    pdf = app.build_panel_preview(
        FakeCanvas(),
        {
            "kind": "url",
            "identity": "url:https://example.edu/s.pdf",
            "url": "https://example.edu/s.pdf",
            "filename": "syllabus.pdf",
        },
    )
    assert pdf["kind"] == "pdf"
    assert pdf["data_url"].startswith("data:application/pdf;base64,")


def test_build_panel_preview_empty_and_missing_canvas():
    assert app.build_panel_preview(FakeCanvas(), None) == {"kind": "empty"}
    missing = app.build_panel_preview(
        None, {"identity": "file:1", "filename": "notes.txt"}
    )
    assert missing["kind"] == "error"
    assert "Canvas" in missing["message"]


def test_apply_files_rail_value_stores_width(monkeypatch):
    class State(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

        def __setattr__(self, name, value):
            self[name] = value

    state = State()
    monkeypatch.setattr(app.st, "session_state", state, raising=False)
    files = [{"identity": "file:1", "filename": "notes.txt"}]
    app.apply_files_rail_value(
        {"collapsed": False, "selected": "file:1", "width_px": 420},
        conversation_id=1,
        files=files,
    )
    assert state["files_sidebar_collapsed"] is False
    assert state["panel_selected"] == "file:1"
    assert state["files_rail_width_px"] == 420
