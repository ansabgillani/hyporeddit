"""Unit tests for evaluation/pipeline.py.

All external dependencies (embedder, retriever, LLM, SQLite) are mocked.
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hyporeddit.evaluation.aggregator import AggregationResult, ClassifiedEvidence
from hyporeddit.evaluation.pipeline import evaluate_hypothesis
from hyporeddit.evaluation.retriever import RetrievedChunk
from hyporeddit.models.evaluation import EvaluationResult
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


def make_retrieved(chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(chunk=make_chunk(chunk_id), cosine_score=0.85)


def make_classified(chunk_id: str = "c1", stance: str = "supports") -> ClassifiedEvidence:
    ev = ClassifiedEvidence(
        chunk=make_chunk(chunk_id),
        stance=stance,
        rationale="relevant",
        retrieval_score=0.85,
        weight=0.5,
    )
    return ev


def make_query_vector() -> np.ndarray:
    rng = np.random.default_rng(0)
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


def _patch_pipeline(
    retrieved: list[RetrievedChunk] | None = None,
    classified: list[ClassifiedEvidence] | None = None,
    synthesis_text: str = "Summary.",
    existing_run=None,
):
    """Context manager factory — returns a dict of mocks."""
    if retrieved is None:
        retrieved = [make_retrieved()]
    if classified is None:
        classified = [make_classified()]

    agg_result = AggregationResult(
        score=0.75, confidence=0.80,
        stance_distribution={"supports": 1, "contradicts": 0, "neutral": 0, "irrelevant": 0},
        sample_size=1,
    )

    patches = {
        "encoder": patch("hyporeddit.evaluation.pipeline.BGE_M3_Encoder"),
        "retriever": patch("hyporeddit.evaluation.pipeline.retrieve", return_value=retrieved),
        "stance": patch("hyporeddit.evaluation.pipeline.classify", return_value=classified),
        "aggregate": patch("hyporeddit.evaluation.pipeline.aggregate", return_value=agg_result),
        "synthesize": patch("hyporeddit.evaluation.pipeline.synthesize", return_value=synthesis_text),
        "llm_client": patch("hyporeddit.evaluation.pipeline.get_llm_client"),
        "get_db": patch("hyporeddit.evaluation.pipeline.get_db"),
        "insert_hypothesis": patch("hyporeddit.evaluation.pipeline.insert_hypothesis"),
        "get_hypothesis_by_text": patch(
            "hyporeddit.evaluation.pipeline.get_hypothesis_by_text",
            return_value=None,
        ),
        "get_latest_run": patch(
            "hyporeddit.evaluation.pipeline.get_latest_run_for_hypothesis",
            return_value=existing_run,
        ),
        "insert_run": patch("hyporeddit.evaluation.pipeline.insert_evaluation_run"),
        "insert_evidence": patch("hyporeddit.evaluation.pipeline.insert_evidence_classifications"),
    }
    return patches


# ---------------------------------------------------------------------------
# Pipeline returns correct type
# ---------------------------------------------------------------------------

def test_evaluate_hypothesis_returns_evaluation_result() -> None:
    p = _patch_pipeline()
    with p["encoder"] as enc_cls, p["retriever"], p["stance"], p["aggregate"], \
         p["synthesize"], p["llm_client"], p["get_db"], p["insert_hypothesis"], \
         p["get_hypothesis_by_text"], p["get_latest_run"], p["insert_run"], p["insert_evidence"]:

        enc_cls.return_value.encode_query.return_value = make_query_vector()
        result = evaluate_hypothesis("test hypothesis")

    assert isinstance(result, EvaluationResult)


def test_evaluate_hypothesis_result_has_score() -> None:
    p = _patch_pipeline()
    with p["encoder"] as enc_cls, p["retriever"], p["stance"], p["aggregate"], \
         p["synthesize"], p["llm_client"], p["get_db"], p["insert_hypothesis"], \
         p["get_hypothesis_by_text"], p["get_latest_run"], p["insert_run"], p["insert_evidence"]:

        enc_cls.return_value.encode_query.return_value = make_query_vector()
        result = evaluate_hypothesis("test hypothesis")

    assert 0.0 <= result.score <= 1.0


def test_evaluate_hypothesis_result_has_synthesis() -> None:
    p = _patch_pipeline(synthesis_text="Custom synthesis.")
    with p["encoder"] as enc_cls, p["retriever"], p["stance"], p["aggregate"], \
         p["synthesize"], p["llm_client"], p["get_db"], p["insert_hypothesis"], \
         p["get_hypothesis_by_text"], p["get_latest_run"], p["insert_run"], p["insert_evidence"]:

        enc_cls.return_value.encode_query.return_value = make_query_vector()
        result = evaluate_hypothesis("test hypothesis")

    assert result.synthesis == "Custom synthesis."


def test_evaluate_hypothesis_result_has_evidence() -> None:
    classified = [make_classified("c1", "supports"), make_classified("c2", "contradicts")]
    p = _patch_pipeline(classified=classified)
    with p["encoder"] as enc_cls, p["retriever"], p["stance"], p["aggregate"], \
         p["synthesize"], p["llm_client"], p["get_db"], p["insert_hypothesis"], \
         p["get_hypothesis_by_text"], p["get_latest_run"], p["insert_run"], p["insert_evidence"]:

        enc_cls.return_value.encode_query.return_value = make_query_vector()
        result = evaluate_hypothesis("test hypothesis")

    assert len(result.evidence) == 2


# ---------------------------------------------------------------------------
# Caching behaviour
# ---------------------------------------------------------------------------

def test_evaluate_returns_cached_result_when_available() -> None:
    """If force_rerun=False and a recent run exists, return it without re-running."""
    cached_run = MagicMock()
    cached_run.__getitem__ = lambda self, key: {
        "id": "run-cached",
        "hypothesis_id": "hyp-001",
        "score": 0.6,
        "confidence": 0.7,
        "sample_size": 10,
        "stance_supports": 7, "stance_contradicts": 2, "stance_neutral": 1, "stance_irrelevant": 0,
        "synthesis": "Cached synthesis.",
        "model_classification": "haiku",
        "model_synthesis": "sonnet",
    }[key]

    p = _patch_pipeline(existing_run=cached_run)
    with p["encoder"] as enc_cls, p["retriever"] as mock_retrieve, p["stance"], \
         p["aggregate"], p["synthesize"], p["llm_client"], p["get_db"], \
         p["insert_hypothesis"], p["get_hypothesis_by_text"] as mock_get_hyp, \
         p["get_latest_run"], p["insert_run"], p["insert_evidence"]:

        mock_get_hyp.return_value = MagicMock()  # hypothesis exists
        enc_cls.return_value.encode_query.return_value = make_query_vector()
        result = evaluate_hypothesis("test hypothesis", force_rerun=False)

    # retriever should NOT have been called when cache hit
    mock_retrieve.assert_not_called()
    assert result.run_id == "run-cached"


def test_force_rerun_bypasses_cache() -> None:
    cached_run = MagicMock()
    cached_run.__getitem__ = lambda self, key: {
        "id": "run-old", "hypothesis_id": "hyp-001", "score": 0.5,
        "confidence": 0.5, "sample_size": 5,
        "stance_supports": 5, "stance_contradicts": 0, "stance_neutral": 0, "stance_irrelevant": 0,
        "synthesis": "Old.", "model_classification": "haiku", "model_synthesis": "sonnet",
    }[key]

    p = _patch_pipeline(existing_run=cached_run)
    with p["encoder"] as enc_cls, p["retriever"] as mock_retrieve, p["stance"], \
         p["aggregate"], p["synthesize"], p["llm_client"], p["get_db"], \
         p["insert_hypothesis"], p["get_hypothesis_by_text"], \
         p["get_latest_run"], p["insert_run"], p["insert_evidence"]:

        enc_cls.return_value.encode_query.return_value = make_query_vector()
        evaluate_hypothesis("test hypothesis", force_rerun=True)

    # retriever must be called when force_rerun=True
    mock_retrieve.assert_called_once()


# ---------------------------------------------------------------------------
# Pipeline calls components in order
# ---------------------------------------------------------------------------

def test_evaluate_calls_classify_after_retrieve() -> None:
    call_order = []

    p = _patch_pipeline()
    with p["encoder"] as enc_cls, \
         patch("hyporeddit.evaluation.pipeline.retrieve",
               side_effect=lambda *a, **kw: (call_order.append("retrieve") or [make_retrieved()])) as _, \
         patch("hyporeddit.evaluation.pipeline.classify",
               side_effect=lambda *a, **kw: (call_order.append("classify") or [make_classified()])) as _, \
         p["aggregate"], p["synthesize"], p["llm_client"], p["get_db"], \
         p["insert_hypothesis"], p["get_hypothesis_by_text"], \
         p["get_latest_run"], p["insert_run"], p["insert_evidence"]:

        enc_cls.return_value.encode_query.return_value = make_query_vector()
        evaluate_hypothesis("hypothesis")

    assert call_order.index("retrieve") < call_order.index("classify")
