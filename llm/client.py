"""Provider-agnostic LLM types, errors, and factory.

Adapters normalize every vendor response into :class:`LLMResponse` so the
agent loop can dispatch Canvas tools without branching on ``LLM_PROVIDER``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

DEFAULT_PROVIDER = "deepseek"
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"

SUPPORTED_PROVIDERS = ("deepseek", "openai", "anthropic")

PROVIDER_API_KEY_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

PROVIDER_DEFAULT_MODEL = {
    "deepseek": DEFAULT_DEEPSEEK_MODEL,
}

UNKNOWN_PROVIDER_TEMPLATE = (
    "Unknown LLM_PROVIDER {raw!r}. Supported providers: {supported}. "
    "The app does not switch providers automatically."
)
MISSING_KEY_TEMPLATE = (
    "{env_name} is missing. Set it in .env or Streamlit secrets for LLM_PROVIDER={provider}."
)
MISSING_MODEL_TEMPLATE = (
    "LLM_MODEL is missing. Set LLM_MODEL to the model name for LLM_PROVIDER={provider}."
)


class LLMError(RuntimeError):
    """The selected provider rejected or failed a completion call."""


class LLMConfigError(LLMError):
    """Provider, model, or API key configuration is invalid."""


@dataclass(frozen=True)
class LLMToolCall:
    """One tool/function call in the internal format."""

    id: str
    name: str
    arguments: str = "{}"

    def parsed_arguments(self) -> dict[str, Any]:
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @property
    def tool_name(self) -> str:
        return self.name

    @property
    def tool_arguments(self) -> dict[str, Any]:
        return self.parsed_arguments()


@dataclass(frozen=True)
class LLMResponse:
    """Normalized completion: final text and/or tool calls.

    Fields used by the agent loop:
    ``text``, ``tool_call``, ``tool_name``, ``tool_arguments``.
    """

    text: str = ""
    tool_calls: tuple[LLMToolCall, ...] = field(default_factory=tuple)

    @property
    def tool_call(self) -> bool:
        return bool(self.tool_calls)

    @property
    def tool_name(self) -> str:
        return self.tool_calls[0].name if self.tool_calls else ""

    @property
    def tool_arguments(self) -> dict[str, Any]:
        return self.tool_calls[0].parsed_arguments() if self.tool_calls else {}

    def as_assistant_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self.text or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        return message

    @classmethod
    def from_openai_message(cls, message: dict[str, Any] | None) -> "LLMResponse":
        payload = message or {}
        calls: list[LLMToolCall] = []
        for raw in payload.get("tool_calls") or []:
            if not isinstance(raw, dict):
                continue
            function = raw.get("function") or {}
            calls.append(
                LLMToolCall(
                    id=str(raw.get("id") or ""),
                    name=str(function.get("name") or ""),
                    arguments=_arguments_json(function.get("arguments")),
                )
            )
        return cls(text=_text_from_content(payload.get("content")), tool_calls=tuple(calls))

    @classmethod
    def from_anthropic_message(cls, payload: dict[str, Any] | None) -> "LLMResponse":
        data = payload or {}
        blocks = data.get("content")
        if isinstance(data.get("message"), dict) and blocks is None:
            blocks = data["message"].get("content")
        text_parts: list[str] = []
        calls: list[LLMToolCall] = []
        for block in blocks or []:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                text_parts.append(str(block.get("text") or ""))
            elif kind == "tool_use":
                calls.append(
                    LLMToolCall(
                        id=str(block.get("id") or ""),
                        name=str(block.get("name") or ""),
                        arguments=_arguments_json(block.get("input")),
                    )
                )
        return cls(text="".join(text_parts), tool_calls=tuple(calls))


class LLMClient(Protocol):
    """Anything the agent loop can call. Test doubles may return a dict."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse | dict[str, Any]:
        ...


def coerce_llm_response(reply: Any) -> LLMResponse:
    """Accept an adapter result or an OpenAI-shaped test double."""
    if isinstance(reply, LLMResponse):
        return reply
    if isinstance(reply, dict):
        return LLMResponse.from_openai_message(reply)
    raise LLMError("LLM client returned an unexpected response type.")


def normalize_provider(name: str | None) -> str:
    return (name or "").strip().lower()


def unknown_provider_message(raw: str) -> str:
    return UNKNOWN_PROVIDER_TEMPLATE.format(
        raw=raw.strip() or raw,
        supported=", ".join(SUPPORTED_PROVIDERS),
    )


def missing_key_message(provider: str) -> str:
    env_name = PROVIDER_API_KEY_ENV.get(provider, "API_KEY")
    return MISSING_KEY_TEMPLATE.format(env_name=env_name, provider=provider)


def missing_model_message(provider: str) -> str:
    return MISSING_MODEL_TEMPLATE.format(provider=provider)


def resolve_provider(raw: str | None) -> tuple[str, str | None]:
    """Return ``(provider, error)``. Error is set for unknown names; no fallback."""
    text = (raw or "").strip()
    if not text:
        return DEFAULT_PROVIDER, None
    provider = normalize_provider(text)
    if provider not in SUPPORTED_PROVIDERS:
        return provider, unknown_provider_message(text)
    return provider, None


def resolve_model(provider: str, llm_model: str | None, deepseek_model: str | None = None) -> tuple[str, str | None]:
    """Pick the model for ``provider``. Empty model is an error except DeepSeek's default."""
    model = (llm_model or "").strip()
    if not model and provider == "deepseek":
        model = (deepseek_model or "").strip()
    if model:
        return model, None
    default = PROVIDER_DEFAULT_MODEL.get(provider)
    if default:
        return default, None
    return "", missing_model_message(provider)


def _arguments_json(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value if value.strip() else "{}"
    try:
        return json.dumps(value)
    except TypeError:
        return "{}"


def _text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in (None, "text"):
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return str(content)


def _redact(text: Any) -> str:
    try:
        from app import redact

        return redact(text)
    except Exception:
        return text if isinstance(text, str) else str(text)


def safe_provider_error(text: Any) -> str:
    """Scrub secrets from provider error text. Never include API keys."""
    return _redact(text)


def provider_key_value(settings: Any, provider: str) -> str:
    mapping = {
        "deepseek": getattr(settings, "deepseek_api_key", "") or "",
        "openai": getattr(settings, "openai_api_key", "") or "",
        "anthropic": getattr(settings, "anthropic_api_key", "") or "",
    }
    return str(mapping.get(provider, "")).strip()


def build_llm_client(settings: Any) -> LLMClient:
    """Construct the adapter for ``settings.llm_provider``. No fallback chain."""
    error = getattr(settings, "config_error", None)
    if error:
        raise LLMConfigError(str(error))

    provider = normalize_provider(getattr(settings, "llm_provider", "") or DEFAULT_PROVIDER)
    if provider not in SUPPORTED_PROVIDERS:
        raw = getattr(settings, "llm_provider", provider)
        raise LLMConfigError(unknown_provider_message(str(raw)))

    model = str(getattr(settings, "model", "") or "").strip()
    if not model:
        raise LLMConfigError(missing_model_message(provider))

    api_key = provider_key_value(settings, provider)
    if not api_key:
        raise LLMConfigError(missing_key_message(provider))

    if provider == "deepseek":
        from llm.deepseek import DeepSeekClient

        return DeepSeekClient(
            api_key,
            getattr(settings, "deepseek_base_url", DEFAULT_DEEPSEEK_BASE_URL),
            model,
        )
    if provider == "openai":
        from llm.openai import OpenAIClient

        return OpenAIClient(
            api_key,
            getattr(settings, "openai_base_url", DEFAULT_OPENAI_BASE_URL),
            model,
        )
    if provider == "anthropic":
        from llm.anthropic import AnthropicClient

        return AnthropicClient(
            api_key,
            getattr(settings, "anthropic_base_url", DEFAULT_ANTHROPIC_BASE_URL),
            model,
        )
    raise LLMConfigError(unknown_provider_message(provider))
