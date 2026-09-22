"""Custom composer: live ``/`` filter, insert-without-submit, submit via component."""

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

COMPOSER_SCRIPT = """
import streamlit as st
import app

pending = st.session_state.pop("pending_insert", None)
if pending is not None:
    st.session_state["draft-box"] = pending

draft = st.text_input("composer", key="draft-box") or ""

for command in app.filter_commands(draft):
    if st.button(f"/{command.token}", key=f"command-pick-{command.token}"):
        st.session_state["pending_insert"] = app.command_insert_text(command.token)

if st.button("Send", key="composer-send"):
    action = app.picker_enter_action(st.session_state.get("draft-box") or "", -1)
    if action["kind"] == "submit":
        sent = app.apply_command_picker_value({"submit": action["text"], "seq": 1})
        if sent:
            st.session_state["submitted_prompt"] = sent
            st.write("SUBMITTED:" + sent)
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


def test_live_filter_slash_sch_and_sum():
    assert [command.token for command in app.filter_commands("/")] == [
        command.token for command in app.COMMANDS
    ]
    assert [command.token for command in app.filter_commands("/sch")] == ["schedule"]
    assert [command.token for command in app.filter_commands("/sum")] == ["summarize"]
    assert [command.token for command in app.filter_commands("/s")] == [
        "schedule",
        "summarize",
    ]


def test_filter_commands_hides_menu_for_ordinary_text():
    assert app.filter_commands("What's due this week?") == ()
    assert app.filter_commands("please /schedule my week") == ()
    assert app.filter_commands("/schedule plan next week") == ()
    assert app.filter_commands("/schedule ") == ()
    assert app.filter_commands("") == ()
    assert app.filter_commands(None) == ()


def test_icon_open_lists_every_command():
    assert app.commands_matching("What's due this week?", icon_open=True) == app.COMMANDS
    assert app.commands_matching("/", icon_open=False) == app.COMMANDS


def test_enter_inserts_highlight_or_submits():
    assert app.picker_enter_action("/sch", 0) == {"kind": "insert", "text": "/schedule "}
    assert app.picker_enter_action("/sum", 0) == {"kind": "insert", "text": "/summarize "}
    assert app.picker_enter_action("/schedule plan next week", -1) == {
        "kind": "submit",
        "text": "/schedule plan next week",
    }
    assert app.picker_enter_action("What's due this week?", -1) == {
        "kind": "submit",
        "text": "What's due this week?",
    }
    assert app.picker_enter_action("", -1) == {"kind": "noop"}
    assert app.picker_enter_action("/", 0)["kind"] == "insert"
    assert app.picker_enter_action("/", -1) == {"kind": "insert", "text": "/schedule "}


def test_enter_in_open_menu_does_not_submit():
    action = app.picker_key_action("Enter", "/sch", -1)
    assert action == {"kind": "insert", "text": "/schedule "}
    assert action["kind"] != "submit"


def test_shift_enter_inserts_newline():
    assert app.picker_key_action("Enter", "/schedule plan", -1, shift=True) == {
        "kind": "newline",
        "text": "/schedule plan\n",
    }
    assert app.picker_enter_action("line one", shift=True) == {
        "kind": "newline",
        "text": "line one\n",
    }


def test_escape_closes_menu_keeps_text():
    assert app.picker_key_action("Escape", "/sch plan") == {
        "kind": "close_menu",
        "text": "/sch plan",
    }
    assert app.picker_key_action("Escape", "What's due?")["text"] == "What's due?"
    html = (command_picker.FRONTEND_DIR / "index.html").read_text()
    assert "dismissed = true" in html
    assert "if (dismissed) return []" in html


def test_mid_sentence_slash_does_not_open_menu():
    assert app.filter_commands("see /files tomorrow") == ()
    assert app.filter_commands("hello /sch") == ()
    assert app.filter_commands("please /schedule my week") == ()
    assert app.filter_commands("line one\n/schedule") == ()


def test_apply_submit_is_idempotent_per_seq(monkeypatch):
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
    assert app.apply_command_picker_value({"submit": "What's due?", "seq": 1}) == "What's due?"
    assert app.apply_command_picker_value({"submit": "What's due?", "seq": 1}) is None
    assert app.apply_command_picker_value({"submit": "again", "seq": 2}) == "again"
    assert app.apply_command_picker_value({"insert": "schedule", "seq": 3}) is None
    assert app.apply_command_picker_value("nope") is None
    assert app.apply_command_picker_value({"submit": "   ", "seq": 4}) is None


def test_typing_slash_lists_commands_in_harness():
    harness = AppTest.from_string(COMPOSER_SCRIPT, default_timeout=30).run()
    assert not harness.exception
    assert not any(button.label.startswith("/") for button in harness.button)
    harness.text_input(key="draft-box").set_value("/").run()
    labels = [button.label for button in harness.button]
    for command in app.COMMANDS:
        assert f"/{command.token}" in labels
        assert f"{command.skill}.md" not in labels
    assert "scheduling.md" not in labels


def test_filter_sch_and_sum_in_harness():
    harness = AppTest.from_string(COMPOSER_SCRIPT, default_timeout=30).run()
    harness.text_input(key="draft-box").set_value("/sch").run()
    labels = [button.label for button in harness.button if button.label.startswith("/")]
    assert labels == ["/schedule"]
    harness.text_input(key="draft-box").set_value("/sum").run()
    labels = [button.label for button in harness.button if button.label.startswith("/")]
    assert labels == ["/summarize"]
    harness.text_input(key="draft-box").set_value("What's due this week?").run()
    assert [button.label for button in harness.button if button.label.startswith("/")] == []


def test_selecting_command_inserts_text_without_submitting():
    harness = AppTest.from_string(COMPOSER_SCRIPT, default_timeout=30).run()
    harness.text_input(key="draft-box").set_value("/").run()
    harness.button(key="command-pick-schedule").click().run()
    harness.run()
    assert not harness.exception
    bodies = [str(element.value) for element in harness.markdown]
    assert not any(body.startswith("SUBMITTED:") for body in bodies)
    assert harness.session_state.get("submitted_prompt") is None
    assert harness.session_state.get("draft-box") == "/schedule "
    assert harness.text_input(key="draft-box").value == "/schedule "


def test_submit_still_works_after_command_plus_text():
    harness = AppTest.from_string(COMPOSER_SCRIPT, default_timeout=30).run()
    harness.text_input(key="draft-box").set_value("/").run()
    harness.button(key="command-pick-schedule").click().run()
    harness.run()
    harness.text_input(key="draft-box").set_value("/schedule plan next week").run()
    harness.button(key="composer-send").click().run()
    assert not harness.exception
    assert harness.session_state["submitted_prompt"] == "/schedule plan next week"
    bodies = [str(element.value) for element in harness.markdown]
    assert any(body == "SUBMITTED:/schedule plan next week" for body in bodies)


def test_ordinary_submit_without_slash():
    harness = AppTest.from_string(COMPOSER_SCRIPT, default_timeout=30).run()
    harness.text_input(key="draft-box").set_value("What's due this week?").run()
    harness.button(key="composer-send").click().run()
    assert harness.session_state["submitted_prompt"] == "What's due this week?"


def test_app_uses_custom_composer_not_st_chat_input(monkeypatch, tmp_path):
    seen: list[list[dict[str, str]]] = []

    def fake_mount(*, commands, placeholder=None, busy=False, key=command_picker.COMPONENT_KEY):
        seen.append(list(commands))
        return None

    monkeypatch.setattr(command_picker, "mount", fake_mount)
    harness = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    assert not harness.exception
    assert not harness.chat_input
    assert seen
    assert [row["token"] for row in seen[0]] == [command.token for command in app.COMMANDS]
    dumped = str(seen[0])
    assert "scheduling.md" not in dumped
    assert "assignment_summary" not in dumped


def test_app_component_submit_is_not_an_insert(monkeypatch, tmp_path):
    def fake_mount(*, commands, placeholder=None, busy=False, key=command_picker.COMPONENT_KEY):
        return {"insert": "schedule", "seq": 99}

    monkeypatch.setattr(command_picker, "mount", fake_mount)
    harness = run_app(monkeypatch, tmp_path, DUMMY_ENV)
    assert not harness.exception
    store = app.ConversationStore(str(tmp_path / "history.db"))
    conversation_id = harness.session_state["conversation_id"]
    assert store.get_messages(conversation_id) == []


def test_frontend_owns_a_real_textarea_and_parity_keys():
    html = (command_picker.FRONTEND_DIR / "index.html").read_text()
    assert command_picker.FRONTEND_DIR.is_dir()
    assert "<textarea" in html
    assert "<input" not in html or 'type="text"' not in html
    assert 'class="command-picker-input"' in html
    assert "function autosize" in html
    assert "insertNewline" in html
    assert "shiftKey" in html
    assert "focusComposer" in html
    assert "setBusy" in html
    assert "aria-busy" in html
    assert "input.removeAttribute(\"disabled\")" in html
    assert "function draftBucket" in html
    assert "function saveDraft" in html
    assert "function restoreDraft" in html
    assert "function clearDraft" in html
    assert "__canvasCommandPickerDraft" in html
    assert "setInterval(onPlace" in html
    assert "filterCommands" in html
    assert "ArrowDown" in html
    assert "ArrowUp" in html
    assert "Escape" in html
    assert "Enter" in html
    assert "command-picker-icon" in html
    assert "scheduling.md" not in html
    assert "files.md" not in html
    assert "st.chat_input" not in html
    assert Path(command_picker._component.path) == command_picker.FRONTEND_DIR


def test_main_replaces_st_chat_input_and_keeps_files_rail():
    main_src = inspect.getsource(app.main)
    assert "render_command_picker" in main_src
    assert "st.chat_input" not in main_src
    assert "handle_prompt(render_command_picker())" in main_src
    assert "render_file_panel" in main_src
    assert "st.columns([2, 1]" not in main_src
    picker_src = inspect.getsource(app.render_command_picker)
    assert "command_picker.mount" in picker_src
    assert "files.md" not in picker_src
    assert "scheduling.md" not in picker_src
    assert not hasattr(app, "insert_command_into_chat")
    assert not hasattr(app, "CHAT_INPUT_KEY")
    # After a reply Streamlit remounts the iframe so focusComposer can run.
    assert "composer_busy = True" in main_src
    assert "composer_busy = False" in main_src
    assert "focus returns" in main_src
    assert main_src.count("st.rerun()") >= 1
