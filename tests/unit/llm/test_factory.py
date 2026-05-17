"""Unit tests for llm/__init__.py — get_llm_client() provider routing."""

from unittest.mock import patch

import pytest

from hyporeddit.llm import get_llm_client
from hyporeddit.llm.anthropic import AnthropicLLMClient
from hyporeddit.llm.openai_compat import OpenAICompatLLMClient


def test_get_llm_client_returns_openai_client_when_provider_is_openai(monkeypatch) -> None:
    import hyporeddit.llm as llm_module
    monkeypatch.setattr(llm_module.settings, "llm_provider", "openai")

    with patch("hyporeddit.llm.openai_compat.OpenAI"):
        client = get_llm_client()

    assert isinstance(client, OpenAICompatLLMClient)


def test_get_llm_client_returns_anthropic_client_when_provider_is_anthropic(monkeypatch) -> None:
    import hyporeddit.llm as llm_module
    monkeypatch.setattr(llm_module.settings, "llm_provider", "anthropic")

    with patch("hyporeddit.llm.anthropic.anthropic"):
        client = get_llm_client()

    assert isinstance(client, AnthropicLLMClient)


def test_get_llm_client_defaults_to_anthropic_for_unknown_provider(monkeypatch) -> None:
    import hyporeddit.llm as llm_module
    monkeypatch.setattr(llm_module.settings, "llm_provider", "unknown")

    with patch("hyporeddit.llm.anthropic.anthropic"):
        client = get_llm_client()

    assert isinstance(client, AnthropicLLMClient)
