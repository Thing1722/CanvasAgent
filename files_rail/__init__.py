"""Pinned files rail as a Streamlit custom component.

The visible rail is HTML we own. A ``declare_component`` iframe with a stable
key stays mounted across Streamlit 1.64 reruns and paints one
``#canvas-files-rail-host`` on ``document.body``. Streamlit widgets are never
moved; collapse / selection return through ``setComponentValue``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
HOST_ID = "canvas-files-rail-host"
STYLE_ID = "canvas-files-rail-style"
COMPONENT_KEY = "files-rail"
RAIL_WIDTH_PX = 320

_component = components.declare_component("files_rail", path=str(FRONTEND_DIR))


def mount(
    *,
    files: list[dict[str, Any]],
    selected: str | None,
    collapsed: bool,
    preview: dict[str, Any] | None,
    empty_copy: str,
    key: str = COMPONENT_KEY,
) -> Any:
    """Render the rail and return ``{collapsed, selected}`` from the iframe."""
    slim = [
        {
            "identity": entry.get("identity"),
            "label": entry.get("label") or entry.get("filename") or entry.get("identity"),
            "filename": entry.get("filename"),
        }
        for entry in files
        if entry.get("identity")
    ]
    return _component(
        files=slim,
        selected=selected,
        collapsed=bool(collapsed),
        preview=preview or {},
        empty_copy=empty_copy,
        host_id=HOST_ID,
        style_id=STYLE_ID,
        width_px=RAIL_WIDTH_PX,
        key=key,
        default={"collapsed": bool(collapsed), "selected": selected},
    )
