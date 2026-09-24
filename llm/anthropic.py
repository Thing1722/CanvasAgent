"""Anthropic Claude adapter (Messages API + native tools)."""

from __future__ import annotations

import json
from typing import Any

import requests

from llm.client import LLMError, LLMResponse, safe_provider_error

ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_MAX_TOKENS = 4096


class AnthropicClient:
    """Claude Messages API. Uses ``requests`` so ``anthropic`` is optional."""

    provider_name = "Anthropic"
    api_key_env = "ANTHROPIC_API_KEY"

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 120.0) -> None:
        self._api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> LLMResponse:
        system, converted = to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": converted,
            "temperature": temperature,
            "max_tokens": ANTHROPIC_MAX_TOKENS,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = openai_tools_to_anthropic(tools)
        try:
            response = self.session.post(
                f"{self.base_url}/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": ANTHROPIC_API_VERSION,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError(f"Could not reach {self.provider_name}: {safe_provider_error(exc)}") from exc

        if response.status_code == 401:
            raise LLMError(
                f"{self.provider_name} rejected the API key (401). Check {self.api_key_env} in your .env."
            )
        if response.status_code >= 400:
            body = safe_provider_error(response.text)[:300]
            raise LLMError(f"{self.provider_name} returned HTTP {response.status_code}: {body}")
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"{self.provider_name} returned an unexpected response shape.") from exc
        return LLMResponse.from_anthropic_message(data)


def openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the app's OpenAI function schemas to Anthropic ``tools``."""
    converted: list[dict[str, Any]] = []
    for schema in tools:
        function = schema.get("function") if isinstance(schema, dict) else None
        spec = function if isinstance(function, dict) else schema
        if not isinstance(spec, dict):
            continue
        name = spec.get("name")
        if not name:
            continue
        parameters = spec.get("parameters") or spec.get("input_schema") or {"type": "object", "properties": {}}
        converted.append(
            {
                "name": name,
                "description": spec.get("description") or "",
                "input_schema": parameters,
            }
        )
    return converted


def to_anthropic_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Split system text and rewrite history into Anthropic message blocks."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role")
        if role == "system":
            text = message.get("content") or ""
            if text:
                system_parts.append(str(text))
            continue
        if role == "user":
            _append_user_text(out, str(message.get("content") or ""))
            continue
        if role == "assistant":
            content = _assistant_content(message)
            if content:
                out.append({"role": "assistant", "content": content})
            continue
        if role == "tool":
            _append_tool_result(
                out,
                tool_use_id=str(message.get("tool_call_id") or ""),
                content=str(message.get("content") or ""),
            )
    return "\n\n".join(system_parts), out


def _assistant_content(message: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    text = message.get("content") or ""
    if text:
        content.append({"type": "text", "text": str(text)})
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        raw_args = function.get("arguments") or "{}"
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            parsed = {}
        if not isinstance(parsed, dict):
            parsed = {}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or "",
                "name": function.get("name") or "",
                "input": parsed,
            }
        )
    return content


def _append_user_text(out: list[dict[str, Any]], text: str) -> None:
    if out and out[-1].get("role") == "user":
        previous = out[-1]["content"]
        if isinstance(previous, str):
            out[-1]["content"] = previous + ("\n\n" + text if text else "")
            return
        if isinstance(previous, list) and text:
            previous.append({"type": "text", "text": text})
            return
    out.append({"role": "user", "content": text})


def _append_tool_result(out: list[dict[str, Any]], tool_use_id: str, content: str) -> None:
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if out and out[-1].get("role") == "user" and isinstance(out[-1].get("content"), list):
        out[-1]["content"].append(block)
        return
    out.append({"role": "user", "content": [block]})
