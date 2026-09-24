"""Multi-provider LLM selection and adapters, with mocked HTTP only."""

from __future__ import annotations

import inspect
import io
import json
import sqlite3
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402
from llm.anthropic import AnthropicClient, openai_tools_to_anthropic, to_anthropic_messages
from llm.client import (
    LLMConfigError,
    LLMError,
    LLMResponse,
    LLMToolCall,
    SUPPORTED_PROVIDERS,
    build_llm_client,
    coerce_llm_response,
)
from llm.deepseek import DeepSeekClient
from llm.openai import OpenAIClient
from test_app import ScriptedDeepSeek  # noqa: E402

SECRET_DEEPSEEK = "sk-deepseek-secret-KEY-11111111"
SECRET_OPENAI = "sk-openai-secret-KEY-22222222"
SECRET_ANTHROPIC = "sk-ant-secret-KEY-33333333"
CANVAS_TOKEN = "canvas-token-abcdef123456"

HISTORY = [
    {"role": "system", "content": "You are a study assistant. Canvas is read-only."},
    {"role": "user", "content": "what courses am I in?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "list_my_courses", "arguments": "{}"},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "list_my_courses",
        "content": '{"count": 1, "courses": [{"name": "Course A"}]}',
    },
]


def _json_response(payload: dict, status: int = 200) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.raw = io.BytesIO(json.dumps(payload).encode())
    response.headers["Content-Type"] = "application/json"
    return response


class RecordingSession:
    def __init__(self, response: requests.Response):
        self.response = response
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": dict(headers or {}), "json": json, "timeout": timeout})
        return self.response


def openai_tool_payload(name: str = "list_my_courses", arguments: str = '{"course": "15-213"}') -> dict:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_9",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            }
        ]
    }


def openai_text_payload(text: str = "You are enrolled in Course A.") -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


def anthropic_tool_payload(name: str = "list_my_courses", arguments: dict | None = None) -> dict:
    return {
        "content": [
            {"type": "text", "text": "Looking that up."},
            {
                "type": "tool_use",
                "id": "toolu_9",
                "name": name,
                "input": arguments or {"course": "15-213"},
            },
        ]
    }


def anthropic_text_payload(text: str = "You are enrolled in Course A.") -> dict:
    return {"content": [{"type": "text", "text": text}]}


# --- selection / configuration ---------------------------------------------


def test_default_provider_is_deepseek_flash():
    settings = app.load_settings({"CANVAS_API_TOKEN": CANVAS_TOKEN, "DEEPSEEK_API_KEY": SECRET_DEEPSEEK})
    assert settings.llm_provider == "deepseek"
    assert settings.model == "deepseek-flash"
    assert settings.missing == []
    assert settings.config_error is None
    client = build_llm_client(settings)
    assert isinstance(client, DeepSeekClient)
    assert client.model == "deepseek-flash"


@pytest.mark.parametrize(
    ("provider", "key_name", "key", "model", "client_type"),
    [
        ("openai", "OPENAI_API_KEY", SECRET_OPENAI, "gpt-4o-mini", OpenAIClient),
        ("anthropic", "ANTHROPIC_API_KEY", SECRET_ANTHROPIC, "claude-sonnet-4-5", AnthropicClient),
        ("deepseek", "DEEPSEEK_API_KEY", SECRET_DEEPSEEK, "deepseek-chat", DeepSeekClient),
    ],
)
def test_provider_selection_from_env(provider, key_name, key, model, client_type):
    settings = app.load_settings(
        {
            "LLM_PROVIDER": provider,
            "LLM_MODEL": model,
            key_name: key,
            "CANVAS_API_TOKEN": CANVAS_TOKEN,
        }
    )
    assert settings.llm_provider == provider
    assert settings.model == model
    assert settings.missing == []
    client = build_llm_client(settings)
    assert isinstance(client, client_type)
    assert client.model == model


def test_missing_key_for_selected_provider_fails_without_secret():
    settings = app.load_settings(
        {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-4o-mini",
            "CANVAS_API_TOKEN": CANVAS_TOKEN,
            "DEEPSEEK_API_KEY": SECRET_DEEPSEEK,
        }
    )
    assert settings.missing == ["OPENAI_API_KEY"]
    assert "DEEPSEEK_API_KEY" not in settings.missing
    with pytest.raises(LLMConfigError) as excinfo:
        build_llm_client(settings)
    message = str(excinfo.value)
    assert "OPENAI_API_KEY" in message
    assert SECRET_DEEPSEEK not in message
    assert CANVAS_TOKEN not in message
    assert "***" not in message or SECRET_DEEPSEEK not in message


def test_unknown_provider_fails_clearly_without_fallback():
    settings = app.load_settings(
        {
            "LLM_PROVIDER": "grok",
            "LLM_MODEL": "anything",
            "DEEPSEEK_API_KEY": SECRET_DEEPSEEK,
            "OPENAI_API_KEY": SECRET_OPENAI,
            "CANVAS_API_TOKEN": CANVAS_TOKEN,
        }
    )
    assert settings.config_error
    assert "Unknown LLM_PROVIDER" in settings.config_error
    assert "grok" in settings.config_error
    assert "deepseek, openai, anthropic" in settings.config_error
    with pytest.raises(LLMConfigError) as excinfo:
        build_llm_client(settings)
    message = str(excinfo.value)
    assert "grok" in message
    assert SECRET_DEEPSEEK not in message
    assert SECRET_OPENAI not in message


def test_openai_without_model_fails_and_does_not_use_deepseek_default():
    settings = app.load_settings(
        {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": SECRET_OPENAI,
            "CANVAS_API_TOKEN": CANVAS_TOKEN,
            "DEEPSEEK_MODEL": "deepseek-flash",
        }
    )
    assert settings.config_error
    assert "LLM_MODEL" in settings.config_error
    assert settings.model == ""
    with pytest.raises(LLMConfigError, match="LLM_MODEL"):
        build_llm_client(settings)


def test_deepseek_model_alias_only_applies_to_deepseek():
    settings = app.load_settings(
        {
            "LLM_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": SECRET_DEEPSEEK,
            "CANVAS_API_TOKEN": CANVAS_TOKEN,
            "DEEPSEEK_MODEL": "deepseek-chat",
        }
    )
    assert settings.model == "deepseek-chat"
    assert settings.config_error is None


def test_build_llm_client_does_not_import_unused_adapters():
    sys.modules.pop("llm.openai", None)
    sys.modules.pop("llm.anthropic", None)
    settings = app.load_settings(
        {"DEEPSEEK_API_KEY": SECRET_DEEPSEEK, "CANVAS_API_TOKEN": CANVAS_TOKEN}
    )
    client = build_llm_client(settings)
    assert isinstance(client, DeepSeekClient)
    assert "llm.openai" not in sys.modules
    assert "llm.anthropic" not in sys.modules


def test_401_error_mentions_env_name_not_the_key():
    app.register_secret(SECRET_OPENAI)
    client = OpenAIClient(SECRET_OPENAI, "https://api.openai.com/v1", "gpt-4o-mini")
    client.session = RecordingSession(_json_response({"error": SECRET_OPENAI}, status=401))
    with pytest.raises(LLMError) as excinfo:
        client.complete([{"role": "user", "content": "hi"}])
    message = str(excinfo.value)
    assert "401" in message
    assert "OPENAI_API_KEY" in message
    assert SECRET_OPENAI not in message


def test_http_error_body_is_redacted():
    app.register_secret(SECRET_DEEPSEEK)
    client = DeepSeekClient(SECRET_DEEPSEEK, "https://api.deepseek.com", "deepseek-flash")
    client.session = RecordingSession(
        _json_response({"error": f"bad key {SECRET_DEEPSEEK}"}, status=500)
    )
    with pytest.raises(LLMError) as excinfo:
        client.complete([{"role": "user", "content": "hi"}])
    assert SECRET_DEEPSEEK not in str(excinfo.value)
    assert "500" in str(excinfo.value)


# --- adapters --------------------------------------------------------------


def _assert_openai_compat_roundtrip(client, recorded: RecordingSession, expect_host: str):
    first = client.complete(HISTORY, tools=app.TOOL_SCHEMAS)
    assert first.tool_call is True
    assert first.tool_name == "list_my_courses"
    assert first.tool_arguments == {"course": "15-213"}
    assert first.text == ""
    request = recorded.calls[0]
    assert request["url"].startswith(expect_host)
    assert request["url"].endswith("/chat/completions")
    assert request["json"]["messages"][0]["role"] == "system"
    assert "study assistant" in request["json"]["messages"][0]["content"]
    assert request["json"]["messages"][1]["role"] == "user"
    assert request["json"]["messages"][2]["tool_calls"][0]["function"]["name"] == "list_my_courses"
    assert request["json"]["messages"][3]["role"] == "tool"
    assert request["json"]["messages"][3]["tool_call_id"] == "call_1"
    assert request["json"]["tools"] == app.TOOL_SCHEMAS
    assert "Authorization" in request["headers"]
    secret = client._api_key
    dumped = json.dumps(request["json"])
    assert secret not in dumped

    recorded.response = _json_response(openai_text_payload())
    second = client.complete(HISTORY + [first.as_assistant_message()])
    assert second.tool_call is False
    assert second.text == "You are enrolled in Course A."
    assert second.tool_name == ""
    assert second.tool_arguments == {}


def test_deepseek_adapter_system_history_tool_call_result_and_final_text():
    client = DeepSeekClient(SECRET_DEEPSEEK, "https://api.deepseek.com", "deepseek-flash")
    recorded = RecordingSession(_json_response(openai_tool_payload()))
    client.session = recorded
    _assert_openai_compat_roundtrip(client, recorded, "https://api.deepseek.com")


def test_openai_adapter_system_history_tool_call_result_and_final_text():
    client = OpenAIClient(SECRET_OPENAI, "https://api.openai.com/v1", "gpt-4o-mini")
    recorded = RecordingSession(_json_response(openai_tool_payload()))
    client.session = recorded
    _assert_openai_compat_roundtrip(client, recorded, "https://api.openai.com/v1")


def test_anthropic_adapter_system_history_tool_call_result_and_final_text():
    client = AnthropicClient(SECRET_ANTHROPIC, "https://api.anthropic.com", "claude-sonnet-4-5")
    recorded = RecordingSession(_json_response(anthropic_tool_payload()))
    client.session = recorded

    first = client.complete(HISTORY, tools=app.TOOL_SCHEMAS)
    assert first.tool_call is True
    assert first.tool_name == "list_my_courses"
    assert first.tool_arguments == {"course": "15-213"}
    assert first.text == "Looking that up."

    request = recorded.calls[0]
    assert request["url"] == "https://api.anthropic.com/v1/messages"
    assert request["json"]["system"].startswith("You are a study assistant")
    assert request["json"]["messages"][0]["role"] == "user"
    assert request["json"]["messages"][1]["role"] == "assistant"
    assert request["json"]["messages"][1]["content"][0]["type"] == "tool_use"
    tool_user = request["json"]["messages"][2]
    assert tool_user["role"] == "user"
    assert tool_user["content"][0]["type"] == "tool_result"
    assert tool_user["content"][0]["tool_use_id"] == "call_1"
    converted_tools = request["json"]["tools"]
    assert {item["name"] for item in converted_tools} == set(app.TOOL_NAMES)
    assert all("input_schema" in item for item in converted_tools)
    assert "function" not in converted_tools[0]
    assert request["headers"]["anthropic-version"] == "2023-06-01"
    assert "Authorization" not in request["headers"]
    assert SECRET_ANTHROPIC not in json.dumps(request["json"])

    recorded.response = _json_response(anthropic_text_payload())
    second = client.complete(HISTORY + [first.as_assistant_message()])
    assert second.tool_call is False
    assert second.text == "You are enrolled in Course A."


def test_anthropic_tool_schema_conversion_matches_app_tools():
    converted = openai_tools_to_anthropic(app.TOOL_SCHEMAS)
    assert [item["name"] for item in converted] == list(app.TOOL_NAMES)
    due = next(item for item in converted if item["name"] == "find_due_dates")
    assert due["input_schema"]["properties"]["query"]["type"] == "string"


def test_anthropic_history_conversion_keeps_system_out_of_messages():
    system, messages = to_anthropic_messages(HISTORY)
    assert "read-only" in system
    assert all(item["role"] != "system" for item in messages)
    assert messages[-1]["content"][0]["type"] == "tool_result"


# --- canvas dispatch stays provider-independent ----------------------------


def test_dispatch_tool_source_does_not_branch_on_provider():
    source = inspect.getsource(app.dispatch_tool)
    lowered = source.lower()
    assert "llm_provider" not in lowered
    assert "openai" not in lowered
    assert "anthropic" not in lowered
    assert "deepseek" not in lowered


def test_run_agent_turn_uses_normalized_fields_not_provider_names():
    source = inspect.getsource(app.run_agent_turn)
    body = source.split('"""', 2)[-1]
    assert "reply.tool_call" in body
    assert "tool_name" in body
    assert "tool_arguments" in body
    assert "LLM_PROVIDER" not in body
    assert "from_anthropic" not in body
    assert "/chat/completions" not in body


def test_app_py_has_no_provider_http():
    source = Path(app.__file__).read_text()
    assert "/chat/completions" not in source
    assert "anthropic-version" not in source
    assert "class DeepSeekClient" not in source
    assert "x-api-key" not in source


def test_canvas_dispatch_unchanged_for_scripted_clients():
    class FakeCanvas:
        tz = None

        def list_courses(self, include_concluded=False):
            return [{"course_id": 1, "name": "Course A"}]

    llm = ScriptedDeepSeek(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "list_my_courses", "arguments": "{}"},
                    }
                ],
            },
            {"role": "assistant", "content": "You are enrolled in Course A."},
        ]
    )
    produced = app.run_agent_turn(llm, FakeCanvas(), [{"role": "user", "content": "my courses?"}])
    assert [message["role"] for message in produced] == ["assistant", "tool", "assistant"]
    payload = json.loads(produced[1]["content"])
    assert payload["count"] == 1
    assert produced[-1]["content"] == "You are enrolled in Course A."


def test_normalized_response_drives_dispatch_the_same_way():
    class RecordingCanvas:
        tz = None
        calls: list[tuple[str, dict]]

        def __init__(self):
            self.calls = []

        def list_courses(self, include_concluded=False):
            self.calls.append(("list_my_courses", {"include_concluded": include_concluded}))
            return [{"course_id": 1, "name": "Course A"}]

    class NormalizedClient:
        def complete(self, messages, tools=None, temperature=0.2):
            if any(item.get("role") == "tool" for item in messages):
                return LLMResponse(text="You are enrolled in Course A.")
            return LLMResponse(
                text="",
                tool_calls=(
                    LLMToolCall(id="call_n", name="list_my_courses", arguments="{}"),
                ),
            )

    canvas = RecordingCanvas()
    produced = app.run_agent_turn(
        NormalizedClient(), canvas, [{"role": "user", "content": "my courses?"}]
    )
    assert canvas.calls[0][0] == "list_my_courses"
    assert produced[0]["tool_calls"][0]["function"]["name"] == "list_my_courses"
    assert json.loads(produced[1]["content"])["count"] == 1


def test_coerce_accepts_openai_shaped_test_doubles():
    reply = coerce_llm_response(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "open_file", "arguments": '{"file_id": 9}'}}
            ],
        }
    )
    assert reply.tool_call is True
    assert reply.tool_name == "open_file"
    assert reply.tool_arguments == {"file_id": 9}


# --- secrets stay out of sqlite --------------------------------------------


def test_keys_never_written_to_sqlite(tmp_path):
    app.register_secret(SECRET_DEEPSEEK)
    app.register_secret(SECRET_OPENAI)
    app.register_secret(SECRET_ANTHROPIC)
    settings = app.load_settings(
        {
            "DEEPSEEK_API_KEY": SECRET_DEEPSEEK,
            "OPENAI_API_KEY": SECRET_OPENAI,
            "ANTHROPIC_API_KEY": SECRET_ANTHROPIC,
            "CANVAS_API_TOKEN": CANVAS_TOKEN,
        }
    )
    db_path = tmp_path / "history.db"
    store = app.ConversationStore(str(db_path))
    conversation_id = store.create_conversation("secret check")
    store.add_message(conversation_id, {"role": "user", "content": "what is due?"})
    store.add_message(conversation_id, {"role": "assistant", "content": "Friday."})
    store.set_setting("timezone", "America/New_York")

    raw = db_path.read_bytes()
    for secret in (SECRET_DEEPSEEK, SECRET_OPENAI, SECRET_ANTHROPIC, CANVAS_TOKEN):
        assert secret.encode() not in raw

    with sqlite3.connect(db_path) as conn:
        setting_values = [row[0] for row in conn.execute("SELECT value FROM settings")]
        contents = [row[0] or "" for row in conn.execute("SELECT content FROM messages")]
    assert settings.deepseek_api_key not in setting_values
    assert all(SECRET_DEEPSEEK not in text for text in contents)
    assert "America/New_York" in setting_values


def test_supported_providers_are_exactly_the_three():
    assert SUPPORTED_PROVIDERS == ("deepseek", "openai", "anthropic")
