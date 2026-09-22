"""Command picker: shared defs, insert-without-submit, filtering, /files routing."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402
import command_picker  # noqa: E402

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")

DUMMY_ENV = {
    "DEEPSEEK_API_KEY": "sk-dummy-key-0000000000",
    "CANVAS_API_TOKEN": "dummy-canvas-token-0000000000",
    "CANVAS_BASE_URL": "https://canvas.example.edu",
}

PICKER_SCRIPT = """
import streamlit as st
import app

if "command_picker_open" not in st.session_state:
    st.session_state.command_picker_open = False

if st.button("Commands", key="command-picker-toggle", help="Commands"):
    st.session_state.command_picker_open = not st.session_state.command_picker_open

if st.session_state.command_picker_open:
    for command in app.COMMANDS:
        if st.button(
            f"/{command.token}",
            key=f"command-pick-{command.token}",
            help=command.description,
        ):
            app.insert_command_into_chat(command.token)
            st.session_state.command_picker_open = False

prompt = st.chat_input("What's due this week?", key=app.CHAT_INPUT_KEY)
if prompt:
    st.session_state["submitted_prompt"] = prompt
    st.write("SUBMITTED:" + prompt)
"""


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
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    return AppTest.from_file(APP_PATH, default_timeout=30).run()


def test_slash_maps_are_derived_from_commands():
    assert app.SLASH_COMMANDS == {command.token: command.skill for command in app.COMMANDS}
    assert app.SLASH_COMMAND_DEFAULTS == {
        command.skill: command.default_request for command in app.COMMANDS
    }
    tokens = [command.token for command in app.COMMANDS]
    assert tokens == ["schedule", "summarize", "exam", "deadlines", "files"]


def test_parser_uses_the_same_command_table():
    source = inspect.getsource(app.parse_user_message)
    assert "SLASH_COMMANDS" in source
    assert "SLASH_COMMAND_DEFAULTS" in source
    assert app.parse_user_message("/files") == (
        "files",
        "Search my course files, modules, and linked materials.",
    )
    assert app.parse_user_message("What's due this week?") == (
        None,
        "What's due this week?",
    )


def test_picker_payload_hides_skills_and_filenames():
    rows = app.commands_for_picker()
    assert [row["token"] for row in rows] == [command.token for command in app.COMMANDS]
    dumped = str(rows)
    for command in app.COMMANDS:
        assert command.skill not in dumped or command.skill == command.token
        assert f"{command.skill}.md" not in dumped
    assert "scheduling.md" not in dumped
    assert "files.md" not in dumped
    assert "assignment_summary" not in dumped
    assert command_picker.commands_payload(rows) == rows


def test_filter_commands_hides_menu_for_ordinary_text():
    assert app.filter_commands("What's due this week?") == ()
    assert app.filter_commands("please /schedule my week") == ()
    assert app.filter_commands("") == ()
    assert app.filter_commands(None) == ()
    assert app.filter_commands("/") == app.COMMANDS
    tokens = [command.token for command in app.filter_commands("/sch")]
    assert tokens == ["schedule"]
    assert [command.token for command in app.filter_commands("/s")] == [
        "schedule",
        "summarize",
    ]


def test_insert_command_text_and_unknown_token():
    assert app.command_insert_text("schedule") == "/schedule "
    assert app.command_insert_text("/EXAM") == "/exam "
    with pytest.raises(ValueError):
        app.command_insert_text("nope")
    assert app.known_command_token("files") == "files"
    assert app.known_command_token("/foo") is None


def test_apply_command_picker_value_prefills_without_duplicate(monkeypatch):
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
    assert app.apply_command_picker_value({"insert": "schedule", "seq": 1}) == "/schedule "
    assert state[app.CHAT_INPUT_KEY] == "/schedule "
    assert app.apply_command_picker_value({"insert": "schedule", "seq": 1}) is None
    assert app.apply_command_picker_value({"insert": "files", "seq": 2}) == "/files "
    assert state[app.CHAT_INPUT_KEY] == "/files "
    assert app.apply_command_picker_value("nope") is None
    assert app.apply_command_picker_value({"insert": "unknown", "seq": 3}) is None


def test_opening_picker_lists_commands():
    harness = AppTest.from_string(PICKER_SCRIPT, default_timeout=30).run()
    assert not harness.exception
    assert len(harness.chat_input) == 1
    labels_before = [button.label for button in harness.button]
    assert "Commands" in labels_before
    assert "/schedule" not in labels_before
    harness.button(key="command-picker-toggle").click().run()
    assert not harness.exception
    labels = [button.label for button in harness.button]
    for command in app.COMMANDS:
        assert f"/{command.token}" in labels
        assert command.skill == command.token or command.skill not in labels
        assert f"{command.skill}.md" not in labels
    assert "scheduling.md" not in labels
    assert "files.md" not in labels


def test_selecting_command_inserts_text_without_submitting():
    harness = AppTest.from_string(PICKER_SCRIPT, default_timeout=30).run()
    harness.button(key="command-picker-toggle").click().run()
    harness.button(key="command-pick-schedule").click().run()
    assert not harness.exception
    bodies = [str(element.value) for element in harness.markdown]
    assert not any(body.startswith("SUBMITTED:") for body in bodies)
    assert harness.session_state.get("submitted_prompt") is None
    chat = harness.chat_input(key=app.CHAT_INPUT_KEY)
    assert chat.proto.set_value
    assert chat.proto.value == "/schedule "
    # After the one-shot prefill, the widget has not been submitted.
    assert harness.session_state.get(app.CHAT_INPUT_KEY) in (None, "/schedule ")


def test_inserted_command_can_be_edited_then_submitted():
    harness = AppTest.from_string(PICKER_SCRIPT, default_timeout=30).run()
    harness.button(key="command-picker-toggle").click().run()
    harness.button(key="command-pick-schedule").click().run()
    harness.chat_input(key=app.CHAT_INPUT_KEY).set_value("/schedule plan next week").run()
    assert not harness.exception
    assert harness.session_state["submitted_prompt"] == "/schedule plan next week"
    bodies = [str(element.value) for element in harness.markdown]
    assert any(body == "SUBMITTED:/schedule plan next week" for body in bodies)


def test_app_mounts_picker_and_keeps_chat_input(monkeypatch, tmp_path):
    seen: list[list[dict[str, str]]] = []

    def fake_mount(*, commands, key=command_picker.COMPONENT_KEY):
        seen.append(list(commands))
        return None

    monkeypatch.setattr(command_picker, "mount", fake_mount)
    harness = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    assert not harness.exception
    assert len(harness.chat_input) == 1
    assert harness.chat_input[0].key == app.CHAT_INPUT_KEY
    assert seen
    assert [row["token"] for row in seen[0]] == [command.token for command in app.COMMANDS]
    dumped = str(seen[0])
    assert "scheduling.md" not in dumped
    assert "assignment_summary" not in dumped


def test_app_picker_insert_does_not_submit(monkeypatch, tmp_path):
    def fake_mount(*, commands, key=command_picker.COMPONENT_KEY):
        return {"insert": "schedule", "seq": 99}

    monkeypatch.setattr(command_picker, "mount", fake_mount)
    harness = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    assert not harness.exception
    assert len(harness.chat_input) == 1
    chat = harness.chat_input(key=app.CHAT_INPUT_KEY)
    assert chat.proto.set_value
    assert chat.proto.value == "/schedule "
    assert not harness.chat_message
    store = app.ConversationStore(str(tmp_path / "history.db"))
    conversation_id = harness.session_state["conversation_id"]
    assert store.get_messages(conversation_id) == []


def test_ordinary_messages_do_not_open_typeahead():
    assert app.filter_commands("What's due this week?") == ()
    assert app.parse_user_message("What's due this week?") == (
        None,
        "What's due this week?",
    )


def test_frontend_is_a_declare_component_iframe():
    html = (command_picker.FRONTEND_DIR / "index.html").read_text()
    assert command_picker.FRONTEND_DIR.is_dir()
    assert (command_picker.FRONTEND_DIR / "index.html").is_file()
    assert "streamlit:componentReady" in html
    assert "streamlit:setComponentValue" in html
    assert command_picker.HOST_ID in html
    assert command_picker.MENU_ID in html
    assert command_picker.STYLE_ID in html
    assert "command-picker-icon" in html
    assert 'aria-label", "Commands"' in html or "aria-label" in html
    assert "title" in html and "Commands" in html
    assert "ArrowDown" in html
    assert "ArrowUp" in html
    assert "Escape" in html
    assert "Enter" in html
    assert "insert" in html
    assert "scheduling.md" not in html
    assert "files.md" not in html
    assert "doc.body.appendChild(host)" in html
    assert Path(command_picker._component.path) == command_picker.FRONTEND_DIR


def test_main_keeps_chat_input_and_files_rail():
    main_src = inspect.getsource(app.main)
    assert "render_command_picker" in main_src
    assert "st.chat_input" in main_src
    assert "CHAT_INPUT_KEY" in main_src
    assert "render_file_panel" in main_src
    assert "st.columns([2, 1]" not in main_src
    picker_src = inspect.getsource(app.render_command_picker)
    assert "command_picker.mount" in picker_src
    assert "files.md" not in picker_src
    assert "scheduling.md" not in picker_src
