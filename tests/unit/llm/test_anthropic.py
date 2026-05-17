"""Unit tests for llm/anthropic.py — all Anthropic SDK calls are mocked."""

import json
from unittest.mock import MagicMock, patch, call

import pytest

from hyporeddit.llm.anthropic import AnthropicLLMClient
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


def make_mock_message(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=content)]
    return msg


# ---------------------------------------------------------------------------
# classify_stances — batching
# ---------------------------------------------------------------------------

def test_classify_stances_single_batch_makes_one_api_call() -> None:
    chunks = [make_chunk(i) for i in range(5)]
    response_json = make_stance_json(chunks)

    with patch("hyporeddit.llm.anthropic.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = make_mock_message(response_json)

        client = AnthropicLLMClient()
        results = client.classify_stances("test hypothesis", chunks)

    assert mock_client.messages.create.call_count == 1
    assert len(results) == 5


def test_classify_stances_30_chunks_makes_two_api_calls() -> None:
    """30 chunks with batch_size=15 → 2 calls."""
    chunks = [make_chunk(i) for i in range(30)]

    with patch("hyporeddit.llm.anthropic.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        def make_response(model, max_tokens, messages, system=None):
            # Determine which batch was sent based on call count
            call_no = mock_client.messages.create.call_count - 1
            start = call_no * 15
            batch_chunks = chunks[start:start + 15]
            return make_mock_message(make_stance_json(batch_chunks))

        mock_client.messages.create.side_effect = make_response

        client = AnthropicLLMClient()
        results = client.classify_stances("test hypothesis", chunks)

    assert mock_client.messages.create.call_count == 2
    assert len(results) == 30


def test_classify_stances_returns_stance_results() -> None:
    chunks = [make_chunk(0)]
    response_json = json.dumps([
        {"chunk_id": "c0", "stance": "contradicts", "rationale": "disagrees"}
    ])

    with patch("hyporeddit.llm.anthropic.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = make_mock_message(response_json)

        client = AnthropicLLMClient()
        results = client.classify_stances("hypothesis", chunks)

    assert len(results) == 1
    assert isinstance(results[0], StanceResult)
    assert results[0].chunk_id == "c0"
    assert results[0].stance == "contradicts"
    assert results[0].rationale == "disagrees"


def test_classify_stances_handles_malformed_json_with_retry() -> None:
    """Malformed JSON triggers one retry with repair instruction."""
    chunks = [make_chunk(0)]
    good_json = json.dumps([{"chunk_id": "c0", "stance": "neutral", "rationale": "ok"}])

    with patch("hyporeddit.llm.anthropic.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        # First call returns bad JSON, second call returns good JSON
        mock_client.messages.create.side_effect = [
            make_mock_message("not valid json {{{"),
            make_mock_message(good_json),
        ]

        client = AnthropicLLMClient()
        results = client.classify_stances("hypothesis", chunks)

    assert mock_client.messages.create.call_count == 2
    assert len(results) == 1
    assert results[0].stance == "neutral"


def test_classify_stances_empty_chunks_returns_empty() -> None:
    with patch("hyporeddit.llm.anthropic.anthropic"):
        client = AnthropicLLMClient()
        results = client.classify_stances("hypothesis", [])
    assert results == []


# ---------------------------------------------------------------------------
# synthesize
# ---------------------------------------------------------------------------

def test_synthesize_makes_one_api_call() -> None:
    with patch("hyporeddit.llm.anthropic.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = make_mock_message("Strong support found.")

        client = AnthropicLLMClient()
        result = client.synthesize("hypothesis", [], {})

    assert mock_client.messages.create.call_count == 1
    assert result == "Strong support found."


def test_synthesize_returns_string() -> None:
    with patch("hyporeddit.llm.anthropic.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = make_mock_message("Summary text.")

        client = AnthropicLLMClient()
        result = client.synthesize("any hypothesis", [], {"score": 0.7})

    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# translate_batch
# ---------------------------------------------------------------------------

def test_translate_batch_delegates_to_translate_batch_de_to_en() -> None:
    """AnthropicLLMClient.translate_batch delegates to translate_batch_de_to_en."""
    with patch("hyporeddit.llm.anthropic.anthropic"):
        client = AnthropicLLMClient()

    with patch(
        "hyporeddit.llm.anthropic.translate_batch_de_to_en",
        return_value=["A", "B"],
    ) as mock_batch:
        result = client.translate_batch(["Eins", "Zwei"])

    mock_batch.assert_called_once_with(["Eins", "Zwei"])
    assert result == ["A", "B"]


def test_translate_batch_empty_returns_empty() -> None:
    with patch("hyporeddit.llm.anthropic.anthropic"):
        client = AnthropicLLMClient()

    with patch(
        "hyporeddit.llm.anthropic.translate_batch_de_to_en",
        return_value=[],
    ) as mock_batch:
        result = client.translate_batch([])

    mock_batch.assert_called_once_with([])
    assert result == []


# ---------------------------------------------------------------------------
# _parse_stance_json — direct tests
# ---------------------------------------------------------------------------

class TestParseStanceJson:
    def test_valid_json_array_is_parsed(self) -> None:
        from hyporeddit.llm.anthropic import _parse_stance_json

        raw = '[{"chunk_id": "c1", "stance": "supports", "rationale": "ok"}]'
        result = _parse_stance_json(raw)

        assert result is not None
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c1"
        assert result[0]["stance"] == "supports"

    def test_strips_markdown_code_fences(self) -> None:
        from hyporeddit.llm.anthropic import _parse_stance_json

        raw = '```json\n[{"chunk_id": "c1", "stance": "neutral", "rationale": ""}]\n```'
        result = _parse_stance_json(raw)

        assert result is not None
        assert result[0]["stance"] == "neutral"

    def test_strips_generic_code_fences(self) -> None:
        from hyporeddit.llm.anthropic import _parse_stance_json

        raw = '```\n[{"chunk_id": "c2", "stance": "contradicts", "rationale": ""}]\n```'
        result = _parse_stance_json(raw)

        assert result is not None
        assert result[0]["chunk_id"] == "c2"

    def test_invalid_json_returns_none(self) -> None:
        from hyporeddit.llm.anthropic import _parse_stance_json

        assert _parse_stance_json("not json {{{") is None

    def test_json_object_not_array_returns_none(self) -> None:
        from hyporeddit.llm.anthropic import _parse_stance_json

        assert _parse_stance_json('{"chunk_id": "c1", "stance": "supports"}') is None

    def test_empty_array_is_valid(self) -> None:
        from hyporeddit.llm.anthropic import _parse_stance_json

        result = _parse_stance_json("[]")

        assert result == []

    def test_multiple_items(self) -> None:
        from hyporeddit.llm.anthropic import _parse_stance_json

        raw = '[{"chunk_id": "c1", "stance": "supports", "rationale": "a"}, {"chunk_id": "c2", "stance": "irrelevant", "rationale": "b"}]'
        result = _parse_stance_json(raw)

        assert result is not None
        assert len(result) == 2

    def test_leading_trailing_whitespace_is_stripped(self) -> None:
        from hyporeddit.llm.anthropic import _parse_stance_json

        raw = '  \n[{"chunk_id": "c1", "stance": "neutral", "rationale": ""}]\n  '
        result = _parse_stance_json(raw)

        assert result is not None
