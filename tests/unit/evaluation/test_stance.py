"""Unit tests for evaluation/stance.py."""

import time
from unittest.mock import MagicMock

import pytest

from hyporeddit.evaluation.aggregator import ClassifiedEvidence
from hyporeddit.evaluation.stance import classify
from hyporeddit.llm.base import ChunkWithContext, StanceResult
from hyporeddit.models.ingestion import Chunk, ChunkMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_metadata() -> ChunkMetadata:
    return ChunkMetadata(
        score=10,
        upvote_ratio=0.9,
        created_utc=int(time.time()) - 30 * 86400,
        depth=0,
        author_karma=1000,
        num_comments=5,
        awards_count=0,
    )


def make_chunk(chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_type="comment",
        source_id="s1",
        text_de="Kommentar",
        text_en="Comment",
        parent_post_id="p1",
        parent_post_title="Title",
        parent_post_body="Body",
        char_offset=0,
        metadata=make_metadata(),
    )


def make_retrieved(chunk_id: str = "c1", cosine_score: float = 0.8):
    from hyporeddit.evaluation.retriever import RetrievedChunk
    return RetrievedChunk(chunk=make_chunk(chunk_id), cosine_score=cosine_score)


def make_llm_client(stance_results: list[StanceResult]) -> MagicMock:
    client = MagicMock()
    client.classify_stances.return_value = stance_results
    return client


# ---------------------------------------------------------------------------
# classify() — core behaviour
# ---------------------------------------------------------------------------

def test_classify_calls_llm_client() -> None:
    retrieved = [make_retrieved("c1")]
    stance_results = [StanceResult(chunk_id="c1", stance="supports", rationale="relevant")]
    llm = make_llm_client(stance_results)

    classify("hypothesis", retrieved, llm)

    llm.classify_stances.assert_called_once()


def test_classify_returns_classified_evidence() -> None:
    retrieved = [make_retrieved("c1")]
    stance_results = [StanceResult(chunk_id="c1", stance="supports", rationale="ok")]
    llm = make_llm_client(stance_results)

    results = classify("hypothesis", retrieved, llm)

    assert len(results) == 1
    assert isinstance(results[0], ClassifiedEvidence)
    assert results[0].stance == "supports"
    assert results[0].rationale == "ok"


def test_classify_preserves_retrieval_score() -> None:
    retrieved = [make_retrieved("c1", cosine_score=0.92)]
    stance_results = [StanceResult(chunk_id="c1", stance="neutral", rationale="")]
    llm = make_llm_client(stance_results)

    results = classify("hypothesis", retrieved, llm)

    assert abs(results[0].retrieval_score - 0.92) < 1e-6


def test_classify_computes_weight() -> None:
    retrieved = [make_retrieved("c1")]
    stance_results = [StanceResult(chunk_id="c1", stance="supports", rationale="")]
    llm = make_llm_client(stance_results)

    results = classify("hypothesis", retrieved, llm)

    assert results[0].weight > 0.0


def test_classify_handles_multiple_chunks() -> None:
    retrieved = [make_retrieved(f"c{i}") for i in range(5)]
    stance_results = [
        StanceResult(chunk_id=f"c{i}", stance="supports", rationale="")
        for i in range(5)
    ]
    llm = make_llm_client(stance_results)

    results = classify("hypothesis", retrieved, llm)

    assert len(results) == 5


def test_classify_drops_llm_results_with_unknown_chunk_id() -> None:
    """If LLM returns a chunk_id not in the retrieved set, it is silently dropped."""
    retrieved = [make_retrieved("c1")]
    stance_results = [
        StanceResult(chunk_id="c1", stance="supports", rationale=""),
        StanceResult(chunk_id="UNKNOWN", stance="supports", rationale=""),
    ]
    llm = make_llm_client(stance_results)

    results = classify("hypothesis", retrieved, llm)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "c1"


def test_classify_passes_chunks_as_chunk_with_context() -> None:
    """The LLM client must receive ChunkWithContext objects, not raw Chunk objects."""
    retrieved = [make_retrieved("c1")]
    stance_results = [StanceResult(chunk_id="c1", stance="neutral", rationale="")]
    llm = make_llm_client(stance_results)

    classify("hypothesis", retrieved, llm)

    call_args = llm.classify_stances.call_args
    chunks_arg = call_args[0][1]  # second positional arg
    assert all(isinstance(c, ChunkWithContext) for c in chunks_arg)
