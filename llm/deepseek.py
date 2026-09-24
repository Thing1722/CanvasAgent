"""DeepSeek adapter (OpenAI-compatible chat completions)."""

from __future__ import annotations

from llm.openai_compat import OpenAICompatibleClient


class DeepSeekClient(OpenAICompatibleClient):
    """DeepSeek chat completions. Uses ``requests``; no extra package."""

    provider_name = "DeepSeek"
    api_key_env = "DEEPSEEK_API_KEY"
