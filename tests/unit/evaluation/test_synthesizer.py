"""Unit tests for evaluation/synthesizer.py."""

import time
from unittest.mock import MagicMock

from hyporeddit.evaluation.aggregator import AggregationResult, ClassifiedEvidence
from hyporeddit.evaluation.synthesizer import synthesize
from hyporeddit.models.ingestion import Chunk, ChunkMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_metadata() -> ChunkMetadata:
    return ChunkMetadata(
        score=10, upvote_ratio=0.9, created_utc=int(time.time()) - 30 * 86400,
        depth=0, author_karma=1000, num_comments=5, awards_count=0,
    )


def make_chunk(chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, source_type="comment", source_id="s1",
        text_de="Kommentar", text_en="Comment",
        parent_post_id="p1", parent_post_title="Title", parent_post_body="Body",
        char_offset=0, metadata=make_metadata(),
    )


def make_evidence(stance: str, chunk_id: str = "c1", weight: float = 1.0) -> ClassifiedEvidence:
    ev = ClassifiedEvidence(
        chunk=make_chunk(chunk_id),
        stance=stance,
        rationale="test",
        retrieval_score=0.8,
        weight=weight,
    )
    return ev


def make_agg_result() -> AggregationResult:
    return AggregationResult(
        score=0.73,
        confidence=0.81,
        stance_distribution={"supports": 10, "contradicts": 3, "neutral": 5, "irrelevant": 2},
        sample_size=18,
    )


def make_llm_client(synthesis_text: str = "Synthesis result.") -> MagicMock:
    client = MagicMock()
    client.synthesize.return_value = synthesis_text
    return client


# ---------------------------------------------------------------------------
# synthesize() — core behaviour
# ---------------------------------------------------------------------------

def test_synthesize_calls_llm_client() -> None:
    evidence = [make_evidence("supports")]
    llm = make_llm_client()

    synthesize("hypothesis", evidence, make_agg_result(), llm)

    llm.synthesize.assert_called_once()


def test_synthesize_returns_string() -> None:
    evidence = [make_evidence("supports")]
    llm = make_llm_client("Strong support found.")

    result = synthesize("hypothesis", evidence, make_agg_result(), llm)

    assert result == "Strong support found."


def test_synthesize_selects_top_3_supporting() -> None:
    """Only top 3 supporting items (by weight) are passed to the LLM."""
    supporting = [make_evidence("supports", chunk_id=f"c{i}", weight=float(i)) for i in range(5)]
    evidence = supporting
    llm = make_llm_client()

    synthesize("hypothesis", evidence, make_agg_result(), llm)

    call_kwargs = llm.synthesize.call_args
    # The second positional arg is the evidence list passed to synthesize
    passed_evidence = call_kwargs[0][1]
    support_items = [e for e in passed_evidence if e.stance == "supports"]
    assert len(support_items) <= 3


def test_synthesize_selects_top_3_contradicting() -> None:
    contradicting = [
        make_evidence("contradicts", chunk_id=f"c{i}", weight=float(i)) for i in range(5)
    ]
    llm = make_llm_client()

    synthesize("hypothesis", contradicting, make_agg_result(), llm)

    passed_evidence = llm.synthesize.call_args[0][1]
    contra_items = [e for e in passed_evidence if e.stance == "contradicts"]
    assert len(contra_items) <= 3


def test_synthesize_passes_stats_to_llm() -> None:
    evidence = [make_evidence("supports")]
    agg = make_agg_result()
    llm = make_llm_client()

    synthesize("hypothesis", evidence, agg, llm)

    stats_arg = llm.synthesize.call_args[0][2]
    assert stats_arg["score"] == agg.score
    assert stats_arg["confidence"] == agg.confidence
    assert stats_arg["sample_size"] == agg.sample_size


def test_synthesize_empty_evidence_still_calls_llm() -> None:
    llm = make_llm_client("No evidence found.")

    result = synthesize("hypothesis", [], make_agg_result(), llm)

    llm.synthesize.assert_called_once()
    assert result == "No evidence found."
