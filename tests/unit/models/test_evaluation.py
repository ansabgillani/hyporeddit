"""Unit tests for models/evaluation.py."""

import pytest
from hyporeddit.models.evaluation import EvaluationResult, EvidenceItem


def make_evidence_item(**kwargs) -> EvidenceItem:
    defaults = dict(
        chunk_id="c1",
        stance="supports",
        rationale="supports the hypothesis",
        text_de="Wir haben 14 Monate gewartet",
        text_en="We waited 14 months",
        parent_post_title="Bauantrag dauert ewig",
        source_url="https://reddit.com/r/hausbau/comments/abc/",
        weight=0.75,
        retrieval_score=0.85,
    )
    defaults.update(kwargs)
    return EvidenceItem(**defaults)


def make_result(**kwargs) -> EvaluationResult:
    defaults = dict(
        run_id="run-001",
        hypothesis_id="hyp-001",
        hypothesis_text="Homebuilders find planning too slow",
        score=0.73,
        confidence=0.81,
        sample_size=45,
        stance_distribution={"supports": 30, "contradicts": 5, "neutral": 8, "irrelevant": 2},
        evidence=[make_evidence_item()],
        synthesis="Strong evidence supports this hypothesis.",
        model_classification="claude-haiku-4-5-20251001",
        model_synthesis="claude-sonnet-4-6",
    )
    defaults.update(kwargs)
    return EvaluationResult(**defaults)


# ---------------------------------------------------------------------------
# EvidenceItem
# ---------------------------------------------------------------------------

def test_evidence_item_fields() -> None:
    item = make_evidence_item()
    assert item.chunk_id == "c1"
    assert item.stance == "supports"
    assert item.text_en == "We waited 14 months"
    assert item.weight == 0.75
    assert item.retrieval_score == 0.85


def test_evidence_item_stance_values() -> None:
    for stance in ("supports", "contradicts", "neutral", "irrelevant"):
        item = make_evidence_item(stance=stance)
        assert item.stance == stance


def test_evidence_item_text_en_optional() -> None:
    item = make_evidence_item(text_en=None)
    assert item.text_en is None


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------

def test_evaluation_result_fields() -> None:
    result = make_result()
    assert result.run_id == "run-001"
    assert result.score == 0.73
    assert result.confidence == 0.81
    assert result.sample_size == 45
    assert result.synthesis == "Strong evidence supports this hypothesis."
    assert len(result.evidence) == 1


def test_evaluation_result_has_model_dump() -> None:
    result = make_result()
    d = result.model_dump()
    assert isinstance(d, dict)
    assert "run_id" in d
    assert "score" in d
    assert "evidence" in d
    assert isinstance(d["evidence"], list)


def test_evaluation_result_stance_distribution() -> None:
    result = make_result()
    dist = result.stance_distribution
    assert dist["supports"] == 30
    assert dist["contradicts"] == 5


def test_evaluation_result_score_range() -> None:
    result = make_result(score=0.0)
    assert result.score == 0.0
    result2 = make_result(score=1.0)
    assert result2.score == 1.0


def test_evaluation_result_empty_evidence() -> None:
    result = make_result(evidence=[])
    assert result.evidence == []
    d = result.model_dump()
    assert d["evidence"] == []
