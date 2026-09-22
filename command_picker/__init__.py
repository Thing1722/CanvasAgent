"""Custom chat composer with live slash-command filtering.

A ``declare_component`` iframe holds a real auto-resizing ``<textarea>``.
Typing a leading ``/`` filters the shared command list in-place. Picking a
row writes ``/token `` and does not submit. Enter sends ``{submit, seq}``;
Shift+Enter inserts a newline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit.components.v1 as components

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
HOST_ID = "canvas-command-picker-host"
MENU_ID = "canvas-command-picker-menu"
STYLE_ID = "canvas-command-picker-style"
COMPONENT_KEY = "command-picker"
DEFAULT_PLACEHOLDER = "What's due this week?"

_component = components.declare_component("command_picker", path=str(FRONTEND_DIR))


def commands_payload(commands: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Strip anything that is not a user-facing token + description."""
    rows: list[dict[str, str]] = []
    for entry in commands:
        token = str(entry.get("token") or "").strip().lstrip("/").lower()
        description = str(entry.get("description") or "").strip()
        if not token:
            continue
        rows.append({"token": token, "description": description})
    return rows


def mount(
    *,
    commands: list[dict[str, Any]],
    placeholder: str = DEFAULT_PLACEHOLDER,
    busy: bool = False,
    key: str = COMPONENT_KEY,
) -> Any:
    """Render the composer. Returns ``{submit, seq}`` only when the user sends."""
    return _component(
        commands=commands_payload(commands),
        placeholder=placeholder or DEFAULT_PLACEHOLDER,
        busy=bool(busy),
        host_id=HOST_ID,
        menu_id=MENU_ID,
        style_id=STYLE_ID,
        key=key,
        default=None,
    )
