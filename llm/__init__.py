"""LLM provider adapters.

``app.py`` talks to :class:`LLMClient` only. Provider HTTP lives in the adapter
modules, which are imported when that provider is selected.
"""

from llm.client import (
    DEFAULT_PROVIDER,
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMResponse,
    LLMToolCall,
    SUPPORTED_PROVIDERS,
    build_llm_client,
    coerce_llm_response,
)

__all__ = [
    "DEFAULT_PROVIDER",
    "LLMClient",
    "LLMConfigError",
    "LLMError",
    "LLMResponse",
    "LLMToolCall",
    "SUPPORTED_PROVIDERS",
    "build_llm_client",
    "coerce_llm_response",
]
