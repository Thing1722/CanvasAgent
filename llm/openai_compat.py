"""Shared OpenAI-compatible chat-completions HTTP client (DeepSeek and OpenAI)."""

from __future__ import annotations

from typing import Any

import requests

from llm.client import LLMError, LLMResponse, safe_provider_error


class OpenAICompatibleClient:
    """POST ``/chat/completions`` and normalize the first choice message."""

    provider_name = "OpenAI"
    api_key_env = "OPENAI_API_KEY"

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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
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
            return LLMResponse.from_openai_message(data["choices"][0]["message"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.provider_name} returned an unexpected response shape.") from exc
