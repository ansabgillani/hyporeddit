"""Unit tests for evaluation/display.py.

We test that display functions run without error and produce some output,
without asserting on exact formatting (which is brittle and UI-specific).
"""

from io import StringIO
from unittest.mock import patch

import pytest

from hyporeddit.evaluation.display import display_history, display_hypothesis_list, display_result
from hyporeddit.models.evaluation import EvaluationResult, EvidenceItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_evidence_item(stance: str = "supports") -> EvidenceItem:
    return EvidenceItem(
        chunk_id="c1", stance=stance, rationale="relevant",
        text_de="Kommentar", text_en="Comment",
        parent_post_title="Title",
        source_url="https://reddit.com/r/hausbau/comments/abc/",
        weight=0.75, retrieval_score=0.85,
    )


def make_result(**kwargs) -> EvaluationResult:
    defaults = dict(
        run_id="run-001",
        hypothesis_id="hyp-001",
        hypothesis_text="Homebuilders find planning too slow",
        score=0.73,
        confidence=0.81,
        sample_size=45,
        stance_distribution={"supports": 30, "contradicts": 5, "neutral": 8, "irrelevant": 2},
        evidence=[make_evidence_item("supports"), make_evidence_item("contradicts")],
        synthesis="Strong evidence supports this hypothesis.",
        model_classification="claude-haiku-4-5-20251001",
        model_synthesis="claude-sonnet-4-6",
    )
    defaults.update(kwargs)
    return EvaluationResult(**defaults)


# ---------------------------------------------------------------------------
# display_result
# ---------------------------------------------------------------------------

def test_display_result_runs_without_error() -> None:
    result = make_result()
    # Should not raise
    display_result(result)


def test_display_result_with_empty_evidence_runs_without_error() -> None:
    result = make_result(evidence=[])
    display_result(result)


def test_display_result_with_zero_confidence_runs_without_error() -> None:
    result = make_result(score=0.5, confidence=0.0, sample_size=0)
    display_result(result)


# ---------------------------------------------------------------------------
# display_history
# ---------------------------------------------------------------------------

def test_display_history_empty_list_runs_without_error() -> None:
    display_history([])


def test_display_history_with_runs_runs_without_error() -> None:
    from unittest.mock import MagicMock
    run = MagicMock()
    run.__getitem__ = lambda self, key: {
        "id": "r1", "run_at": "2026-05-16T12:00:00",
        "score": 0.7, "confidence": 0.8, "sample_size": 20,
    }[key]
    display_history([run])


# ---------------------------------------------------------------------------
# display_hypothesis_list
# ---------------------------------------------------------------------------

def test_display_hypothesis_list_empty_runs_without_error() -> None:
    display_hypothesis_list([])


def test_display_hypothesis_list_with_items_runs_without_error() -> None:
    from unittest.mock import MagicMock
    hyp = MagicMock()
    hyp.__getitem__ = lambda self, key: {
        "id": "h1", "text": "Test hypothesis",
        "score": 0.65, "confidence": 0.75, "run_at": "2026-05-16T12:00:00",
    }.get(key)
    display_hypothesis_list([hyp])
