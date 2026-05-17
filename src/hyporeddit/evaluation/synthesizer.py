"""Prose synthesis of evaluation evidence via the LLM client."""

from hyporeddit.evaluation.aggregator import AggregationResult, ClassifiedEvidence
from hyporeddit.llm.base import LLMClient


def synthesize(
    hypothesis: str,
    evidence: list[ClassifiedEvidence],
    agg_result: AggregationResult,
    llm_client: LLMClient,
) -> str:
    """Generate a prose synthesis summarising the evidence for a hypothesis.

    Selects top 3 supporting + top 3 contradicting items by weight and
    delegates prose generation to the LLM client.
    """
    supporting = sorted(
        (e for e in evidence if e.stance == "supports"),
        key=lambda e: e.weight,
        reverse=True,
    )[:3]
    contradicting = sorted(
        (e for e in evidence if e.stance == "contradicts"),
        key=lambda e: e.weight,
        reverse=True,
    )[:3]

    top_evidence = supporting + contradicting

    stats = {
        "score": agg_result.score,
        "confidence": agg_result.confidence,
        "sample_size": agg_result.sample_size,
        "supports": agg_result.stance_distribution.get("supports", 0),
        "contradicts": agg_result.stance_distribution.get("contradicts", 0),
        "neutral": agg_result.stance_distribution.get("neutral", 0),
        "irrelevant": agg_result.stance_distribution.get("irrelevant", 0),
    }

    return llm_client.synthesize(hypothesis, top_evidence, stats)
