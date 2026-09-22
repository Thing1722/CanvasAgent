"""Command picker icon + menu as a Streamlit custom component.

The visible sparkle sits beside the pinned chat input. A ``declare_component``
iframe with a stable key stays mounted across Streamlit 1.64 reruns and paints
one ``#canvas-command-picker-host`` on ``document.body``. Picking a command
returns ``{insert, seq}`` through ``setComponentValue``; Python prefills
``st.chat_input`` via session state and does not submit.
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


def mount(*, commands: list[dict[str, Any]], key: str = COMPONENT_KEY) -> Any:
    """Render the picker and return ``{insert, seq}`` when a command is chosen."""
    return _component(
        commands=commands_payload(commands),
        host_id=HOST_ID,
        menu_id=MENU_ID,
        style_id=STYLE_ID,
        key=key,
        default=None,
    )
