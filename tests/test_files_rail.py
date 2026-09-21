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
    assert "doc.body.appendChild(host)" in html
    assert "files-rail-toggle" in html
    assert "files-rail-chevron" in html
    assert "st-key-files-sidebar" not in html
    assert "position: sticky" not in html
    assert '[data-testid="stColumn"]' not in html
    assert "chevron_right" not in html
    assert "chevron_left" not in html
    assert "AppViewContainer" not in html
    assert Path(files_rail._component.path) == files_rail.FRONTEND_DIR


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
