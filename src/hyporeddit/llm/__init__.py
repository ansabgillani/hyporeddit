"""LLM client factory — returns the configured client (Anthropic or OpenAI-compatible)."""

from hyporeddit.config import settings
from hyporeddit.llm.base import LLMClient


def get_llm_client() -> LLMClient:
    if settings.llm_provider == "openai":
        from hyporeddit.llm.openai_compat import OpenAICompatLLMClient
        return OpenAICompatLLMClient()
    from hyporeddit.llm.anthropic import AnthropicLLMClient
    return AnthropicLLMClient()
