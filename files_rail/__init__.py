"""Pinned files rail as a Streamlit custom component.

The visible rail is HTML we own. A ``declare_component`` iframe with a stable
key stays mounted across Streamlit 1.64 reruns and paints one
``#canvas-files-rail-host`` on ``document.body``. Streamlit widgets are never
moved; collapse / selection / width return through ``setComponentValue``.
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
RAIL_MIN_WIDTH_PX = 200
RAIL_MAX_WIDTH_PX = 600
WIDTH_STORAGE_KEY = "canvas-files-rail-width"

_component = components.declare_component("files_rail", path=str(FRONTEND_DIR))


def clamp_width(width_px: Any, default: int = RAIL_WIDTH_PX) -> int:
    """Keep a dragged rail width inside the same band as Streamlit's sidebar."""
    try:
        value = int(width_px)
    except (TypeError, ValueError):
        return default
    return max(RAIL_MIN_WIDTH_PX, min(RAIL_MAX_WIDTH_PX, value))


def mount(
    *,
    files: list[dict[str, Any]],
    selected: str | None,
    collapsed: bool,
    preview: dict[str, Any] | None,
    empty_copy: str,
    width_px: int = RAIL_WIDTH_PX,
    key: str = COMPONENT_KEY,
) -> Any:
    """Render the rail and return ``{collapsed, selected, width_px}`` from the iframe."""
    slim = [
        {
            "identity": entry.get("identity"),
            "label": entry.get("label") or entry.get("filename") or entry.get("identity"),
            "filename": entry.get("filename"),
        }
        for entry in files
        if entry.get("identity")
    ]
    width = clamp_width(width_px)
    return _component(
        files=slim,
        selected=selected,
        collapsed=bool(collapsed),
        preview=preview or {},
        empty_copy=empty_copy,
        host_id=HOST_ID,
        style_id=STYLE_ID,
        width_px=width,
        min_width_px=RAIL_MIN_WIDTH_PX,
        max_width_px=RAIL_MAX_WIDTH_PX,
        width_storage_key=WIDTH_STORAGE_KEY,
        key=key,
        default={
            "collapsed": bool(collapsed),
            "selected": selected,
            "width_px": width,
        },
    )
