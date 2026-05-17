"""Unit tests for evaluation/aggregator.py.

Pure function tests — no env vars, no network, no mocks.
These are the most valuable tests in the system.
"""

import math
import time

import pytest

from hyporeddit.evaluation.aggregator import (
    AggregationResult,
    ClassifiedEvidence,
    aggregate,
    compute_weight,
)
from hyporeddit.models.ingestion import Chunk, ChunkMetadata


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_metadata(
    score: int = 10,
    created_utc: int | None = None,
    depth: int = 0,
    author_karma: int = 1000,
    upvote_ratio: float = 0.9,
    num_comments: int = 5,
) -> ChunkMetadata:
    if created_utc is None:
        # recent post — within last 30 days
        created_utc = int(time.time()) - 30 * 86400
    return ChunkMetadata(
        score=score,
        upvote_ratio=upvote_ratio,
        created_utc=created_utc,
        depth=depth,
        author_karma=author_karma,
        num_comments=num_comments,
        awards_count=0,
    )


def make_chunk(
    chunk_id: str = "c1",
    source_id: str = "src1",
    metadata: ChunkMetadata | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_type="comment",
        source_id=source_id,
        text_de="Test Kommentar",
        parent_post_id="p1",
        parent_post_title="Post Titel",
        parent_post_body="Post Body",
        char_offset=0,
        metadata=metadata or make_metadata(),
    )


def make_evidence(
    stance: str = "supports",
    weight: float = 1.0,
    retrieval_score: float = 0.8,
    chunk: Chunk | None = None,
) -> ClassifiedEvidence:
    return ClassifiedEvidence(
        chunk=chunk or make_chunk(),
        stance=stance,
        rationale="Test rationale",
        retrieval_score=retrieval_score,
        weight=weight,
    )


# ---------------------------------------------------------------------------
# compute_weight tests
# ---------------------------------------------------------------------------

def test_weight_is_positive_for_valid_inputs() -> None:
    chunk = make_chunk(metadata=make_metadata(score=10, depth=0, author_karma=1000))
    ev = make_evidence(chunk=chunk, retrieval_score=0.8)
    w = compute_weight(ev)
    assert w > 0.0


def test_weight_decreases_with_depth() -> None:
    shallow = make_evidence(chunk=make_chunk(metadata=make_metadata(depth=0)), retrieval_score=0.8)
    deep = make_evidence(chunk=make_chunk(metadata=make_metadata(depth=5)), retrieval_score=0.8)
    assert compute_weight(shallow) > compute_weight(deep)


def test_weight_decreases_with_age() -> None:
    recent_utc = int(time.time()) - 10 * 86400   # 10 days ago
    old_utc = int(time.time()) - 400 * 86400     # 400 days ago
    recent = make_evidence(chunk=make_chunk(metadata=make_metadata(created_utc=recent_utc)), retrieval_score=0.8)
    old = make_evidence(chunk=make_chunk(metadata=make_metadata(created_utc=old_utc)), retrieval_score=0.8)
    assert compute_weight(recent) > compute_weight(old)


def test_weight_increases_with_score() -> None:
    low = make_evidence(chunk=make_chunk(metadata=make_metadata(score=1)), retrieval_score=0.8)
    high = make_evidence(chunk=make_chunk(metadata=make_metadata(score=500)), retrieval_score=0.8)
    assert compute_weight(low) < compute_weight(high)


def test_weight_increases_with_retrieval_score() -> None:
    low_rel = make_evidence(retrieval_score=0.3)
    high_rel = make_evidence(retrieval_score=0.9)
    assert compute_weight(low_rel) < compute_weight(high_rel)


def test_weight_never_negative() -> None:
    edge = make_evidence(
        chunk=make_chunk(metadata=make_metadata(score=0, depth=10, author_karma=0,
                                                 created_utc=int(time.time()) - 1000 * 86400)),
        retrieval_score=0.01,
    )
    assert compute_weight(edge) >= 0.0


# ---------------------------------------------------------------------------
# aggregate tests — score computation
# ---------------------------------------------------------------------------

def test_all_supports_gives_score_one() -> None:
    evidence = [make_evidence(stance="supports", weight=1.0) for _ in range(5)]
    result = aggregate(evidence)
    assert abs(result.score - 1.0) < 1e-9


def test_all_contradicts_gives_score_zero() -> None:
    evidence = [make_evidence(stance="contradicts", weight=1.0) for _ in range(5)]
    result = aggregate(evidence)
    assert abs(result.score - 0.0) < 1e-9


def test_all_neutral_gives_score_half() -> None:
    evidence = [make_evidence(stance="neutral", weight=1.0) for _ in range(5)]
    result = aggregate(evidence)
    assert abs(result.score - 0.5) < 1e-9


def test_irrelevant_excluded_from_score() -> None:
    evidence = [
        make_evidence(stance="supports", weight=1.0),
        make_evidence(stance="irrelevant", weight=999.0),  # huge weight, but excluded
    ]
    result = aggregate(evidence)
    assert abs(result.score - 1.0) < 1e-9


def test_all_irrelevant_returns_midpoint_zero_confidence() -> None:
    evidence = [make_evidence(stance="irrelevant") for _ in range(5)]
    result = aggregate(evidence)
    assert abs(result.score - 0.5) < 1e-9
    assert abs(result.confidence - 0.0) < 1e-9


def test_empty_evidence_returns_midpoint_zero_confidence() -> None:
    result = aggregate([])
    assert abs(result.score - 0.5) < 1e-9
    assert abs(result.confidence - 0.0) < 1e-9


def test_mixed_evidence_score_between_zero_and_one() -> None:
    evidence = [
        make_evidence(stance="supports", weight=1.0),
        make_evidence(stance="contradicts", weight=1.0),
    ]
    result = aggregate(evidence)
    assert 0.0 < result.score < 1.0


def test_weighted_score_respects_weights() -> None:
    evidence = [
        make_evidence(stance="supports", weight=3.0),
        make_evidence(stance="contradicts", weight=1.0),
    ]
    result = aggregate(evidence)
    # Weighted: (3*1.0 + 1*0.0) / (3+1) = 0.75
    assert abs(result.score - 0.75) < 1e-9


# ---------------------------------------------------------------------------
# aggregate tests — confidence computation
# ---------------------------------------------------------------------------

def test_confidence_zero_for_no_relevant_evidence() -> None:
    result = aggregate([make_evidence(stance="irrelevant")])
    assert result.confidence == 0.0


def test_confidence_increases_with_sample_size() -> None:
    small = aggregate([make_evidence(stance="supports") for _ in range(5)])
    large = aggregate([make_evidence(stance="supports") for _ in range(50)])
    assert small.confidence < large.confidence


def test_confidence_saturates_at_n_50() -> None:
    exact_50 = aggregate([make_evidence(stance="supports") for _ in range(50)])
    more_than_50 = aggregate([make_evidence(stance="supports") for _ in range(100)])
    assert abs(exact_50.confidence - more_than_50.confidence) < 1e-9


def test_confidence_one_sided_evidence_higher() -> None:
    one_sided = aggregate([make_evidence(stance="supports") for _ in range(20)])
    split = aggregate(
        [make_evidence(stance="supports") for _ in range(10)]
        + [make_evidence(stance="contradicts") for _ in range(10)]
    )
    assert one_sided.confidence > split.confidence


# ---------------------------------------------------------------------------
# AggregationResult structure
# ---------------------------------------------------------------------------

def test_aggregate_returns_stance_distribution() -> None:
    evidence = [
        make_evidence(stance="supports"),
        make_evidence(stance="supports"),
        make_evidence(stance="contradicts"),
        make_evidence(stance="neutral"),
        make_evidence(stance="irrelevant"),
    ]
    result = aggregate(evidence)
    dist = result.stance_distribution
    assert dist["supports"] == 2
    assert dist["contradicts"] == 1
    assert dist["neutral"] == 1
    assert dist["irrelevant"] == 1
