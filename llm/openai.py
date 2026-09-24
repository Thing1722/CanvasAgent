"""OpenAI adapter (chat completions + tools)."""

from __future__ import annotations

from llm.openai_compat import OpenAICompatibleClient


class OpenAIClient(OpenAICompatibleClient):
    """OpenAI chat completions. Uses ``requests`` so ``openai`` is optional."""

    provider_name = "OpenAI"
    api_key_env = "OPENAI_API_KEY"
