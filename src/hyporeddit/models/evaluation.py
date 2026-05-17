"""Pydantic models for evaluation results."""

from pydantic import BaseModel


class EvidenceItem(BaseModel):
    chunk_id: str
    stance: str                  # 'supports' | 'contradicts' | 'neutral' | 'irrelevant'
    rationale: str
    text_de: str
    text_en: str | None
    parent_post_title: str
    source_url: str
    weight: float
    retrieval_score: float


class EvaluationResult(BaseModel):
    run_id: str
    hypothesis_id: str
    hypothesis_text: str
    score: float
    confidence: float
    sample_size: int
    stance_distribution: dict[str, int]
    evidence: list[EvidenceItem]
    synthesis: str
    model_classification: str
    model_synthesis: str
