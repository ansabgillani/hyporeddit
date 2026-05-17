"""Unit tests for llm/openai_compat.py — all OpenAI SDK calls are mocked."""

import json
from unittest.mock import MagicMock, patch

import pytest

from hyporeddit.llm.openai_compat import OpenAICompatLLMClient
from hyporeddit.llm.base import ChunkWithContext, StanceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(i: int) -> ChunkWithContext:
    return ChunkWithContext(
        chunk_id=f"c{i}",
        text_de=f"Kommentar {i}",
        text_en=f"Comment {i}",
        parent_post_title="Bauantrag",
        parent_post_body="Body",
        retrieval_score=0.8,
    )


def make_stance_json(chunks: list[ChunkWithContext]) -> str:
    return json.dumps([
        {"chunk_id": c.chunk_id, "stance": "supports", "rationale": "relevant"}
        for c in chunks
    ])


def make_mock_completion(content: str) -> MagicMock:
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = content
    return completion


# ---------------------------------------------------------------------------
# classify_stances — batching
# ---------------------------------------------------------------------------

def test_classify_stances_single_batch_makes_one_api_call() -> None:
    chunks = [make_chunk(i) for i in range(5)]
    response_json = make_stance_json(chunks)

    with patch("hyporeddit.llm.openai_compat.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_completion(response_json)

        client = OpenAICompatLLMClient()
        results = client.classify_stances("test hypothesis", chunks)

    assert mock_client.chat.completions.create.call_count == 1
    assert len(results) == 5


def test_classify_stances_30_chunks_makes_two_api_calls() -> None:
    chunks = [make_chunk(i) for i in range(30)]

    with patch("hyporeddit.llm.openai_compat.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        def make_response(model, max_tokens, messages, **kwargs):
            call_no = mock_client.chat.completions.create.call_count - 1
            start = call_no * 15
            batch_chunks = chunks[start : start + 15]
            return make_mock_completion(make_stance_json(batch_chunks))

        mock_client.chat.completions.create.side_effect = make_response

        client = OpenAICompatLLMClient()
        results = client.classify_stances("test hypothesis", chunks)

    assert mock_client.chat.completions.create.call_count == 2
    assert len(results) == 30


def test_classify_stances_returns_stance_results() -> None:
    chunks = [make_chunk(0)]
    response_json = json.dumps([
        {"chunk_id": "c0", "stance": "contradicts", "rationale": "disagrees"}
    ])

    with patch("hyporeddit.llm.openai_compat.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_completion(response_json)

        client = OpenAICompatLLMClient()
        results = client.classify_stances("hypothesis", chunks)

    assert len(results) == 1
    assert isinstance(results[0], StanceResult)
    assert results[0].chunk_id == "c0"
    assert results[0].stance == "contradicts"
    assert results[0].rationale == "disagrees"


def test_classify_stances_handles_malformed_json_with_retry() -> None:
    chunks = [make_chunk(0)]
    good_json = json.dumps([{"chunk_id": "c0", "stance": "neutral", "rationale": "ok"}])

    with patch("hyporeddit.llm.openai_compat.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = [
            make_mock_completion("not valid json {{{"),
            make_mock_completion(good_json),
        ]

        client = OpenAICompatLLMClient()
        results = client.classify_stances("hypothesis", chunks)

    assert mock_client.chat.completions.create.call_count == 2
    assert len(results) == 1
    assert results[0].stance == "neutral"


def test_classify_stances_empty_chunks_returns_empty() -> None:
    with patch("hyporeddit.llm.openai_compat.OpenAI"):
        client = OpenAICompatLLMClient()
        results = client.classify_stances("hypothesis", [])
    assert results == []


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------

def test_synthesize_makes_one_api_call() -> None:
    with patch("hyporeddit.llm.openai_compat.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_completion("Strong support found.")

        client = OpenAICompatLLMClient()
        result = client.synthesize("hypothesis", [], {})

    assert mock_client.chat.completions.create.call_count == 1
    assert result == "Strong support found."


def test_synthesize_returns_string() -> None:
    with patch("hyporeddit.llm.openai_compat.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_completion("Summary text.")

        client = OpenAICompatLLMClient()
        result = client.synthesize("any hypothesis", [], {"score": 0.7})

    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# translate — KEY: must use OpenAI, not Anthropic
# ---------------------------------------------------------------------------

def test_translate_uses_openai_chat_completions() -> None:
    """translate() must call OpenAI chat.completions, not the Anthropic translator."""
    with patch("hyporeddit.llm.openai_compat.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_completion("Hello world")

        client = OpenAICompatLLMClient()
        result = client.translate("Hallo Welt")

    mock_client.chat.completions.create.assert_called_once()
    assert result == "Hello world"


def test_translate_returns_string() -> None:
    with patch("hyporeddit.llm.openai_compat.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_completion("Translated text")

        client = OpenAICompatLLMClient()
        result = client.translate("Irgendein Text")

    assert isinstance(result, str)
    assert result == "Translated text"


def test_translate_sends_source_text_in_message() -> None:
    """The German source text must appear in the user message sent to OpenAI."""
    german = "Wir haben lange gewartet."

    with patch("hyporeddit.llm.openai_compat.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_completion("We waited a long time.")

        client = OpenAICompatLLMClient()
        client.translate(german)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs.get("messages", [])
    all_content = " ".join(m["content"] for m in messages if isinstance(m.get("content"), str))
    assert german in all_content


# ---------------------------------------------------------------------------
# translate_batch
# ---------------------------------------------------------------------------

def test_translate_batch_returns_one_translation_per_input() -> None:
    with patch("hyporeddit.llm.openai_compat.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_mock_completion(
            '["English A", "English B"]'
        )

        client = OpenAICompatLLMClient()
        results = client.translate_batch(["Text A", "Text B"])

    assert results == ["English A", "English B"]


def test_translate_batch_empty_returns_empty() -> None:
    with patch("hyporeddit.llm.openai_compat.OpenAI"):
        client = OpenAICompatLLMClient()
        results = client.translate_batch([])
    assert results == []
