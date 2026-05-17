"""Confidence-weighted stance aggregation.

Implements the formula from ARCHITECTURE.md:
  weight = relevance × recency × engagement × depth_penalty × author_signal

Score = Σ(weight_i × stance_value_i) / Σ(weight_i)
  where stance_values = {supports: 1.0, contradicts: 0.0, neutral: 0.5, irrelevant: excluded}

Confidence = agreement_rate × sample_factor
  agreement_rate = max(supports, relevant-supports) / relevant_count
  sample_factor  = min(relevant_count / 50, 1.0)
"""

import math
import time
from dataclasses import dataclass

from hyporeddit.config import settings
from hyporeddit.models.ingestion import Chunk

_STANCE_VALUES: dict[str, float | None] = {
    "supports": 1.0,
    "contradicts": 0.0,
    "neutral": 0.5,
    "irrelevant": None,  # excluded from score computation
}


@dataclass
class ClassifiedEvidence:
    chunk: Chunk
    stance: str           # 'supports' | 'contradicts' | 'neutral' | 'irrelevant'
    rationale: str
    retrieval_score: float
    weight: float = 0.0   # populated by compute_weight


@dataclass
class AggregationResult:
    score: float                         # 0–1
    confidence: float                    # 0–1
    stance_distribution: dict[str, int]  # counts per stance
    sample_size: int                     # total evidence chunks classified


def compute_weight(evidence: ClassifiedEvidence) -> float:
    """Compute the confidence weight for one piece of evidence.

    Formula (all factors multiplied):
      relevance    = cosine similarity from vector search (0–1)
      recency      = exp(-0.693 * age_days / half_life)  [half-life = 180 days]
      engagement   = log1p(max(score, 1)) / log1p(1000)
      depth_penalty = 1 / (1 + depth * factor)
      author_signal = 0.8 + 0.2 * log1p(max(karma, 1)) / log1p(100_000)
    """
    chunk = evidence.chunk
    meta = chunk.metadata

    relevance = max(0.0, min(1.0, evidence.retrieval_score))

    age_days = (time.time() - meta.created_utc) / 86400.0
    half_life = settings.recency_half_life_days
    recency = math.exp(-0.693 * age_days / half_life)

    engagement = math.log1p(max(meta.score, 1)) / math.log1p(1000)

    depth_penalty = 1.0 / (1.0 + meta.depth * settings.depth_penalty_factor)

    karma = math.log1p(max(meta.author_karma, 1)) / math.log1p(100_000)
    author_signal = 0.8 + 0.2 * karma

    return relevance * recency * engagement * depth_penalty * author_signal


def aggregate(classified: list[ClassifiedEvidence]) -> AggregationResult:
    """Aggregate classified evidence into a score and confidence.

    Returns midpoint score (0.5) with zero confidence when there is no
    relevant evidence.
    """
    stance_distribution = {"supports": 0, "contradicts": 0, "neutral": 0, "irrelevant": 0}
    for ev in classified:
        if ev.stance in stance_distribution:
            stance_distribution[ev.stance] += 1

    relevant = [ev for ev in classified if ev.stance != "irrelevant"]

    if not relevant:
        return AggregationResult(
            score=0.5,
            confidence=0.0,
            stance_distribution=stance_distribution,
            sample_size=len(classified),
        )

    weighted_sum = sum(
        ev.weight * _STANCE_VALUES[ev.stance]  # type: ignore[operator]
        for ev in relevant
    )
    total_weight = sum(ev.weight for ev in relevant)
    score = weighted_sum / total_weight if total_weight > 0 else 0.5

    n = len(relevant)
    supports_count = stance_distribution["supports"]
    agreement = max(supports_count, n - supports_count) / n
    sample_factor = min(n / settings.confidence_saturation_n, 1.0)
    confidence = agreement * sample_factor

    return AggregationResult(
        score=score,
        confidence=confidence,
        stance_distribution=stance_distribution,
        sample_size=len(classified),
    )


if __name__ == "__main__":
    # Run: python -m hyporeddit.evaluation.aggregator
    # Env: none required
    from loguru import logger

    recent_utc = int(time.time()) - 30 * 86400
    from hyporeddit.models.ingestion import ChunkMetadata

    def _ev(stance: str, score: int = 10, depth: int = 0) -> ClassifiedEvidence:
        from hyporeddit.models.ingestion import Chunk
        meta = ChunkMetadata(score=score, upvote_ratio=0.9, created_utc=recent_utc,
                              depth=depth, author_karma=1000, num_comments=5, awards_count=0)
        chunk = Chunk("c", "comment", "s", "Text.", "p", "T", "B", 0, meta)
        ev = ClassifiedEvidence(chunk=chunk, stance=stance, rationale="test", retrieval_score=0.8)
        ev.weight = compute_weight(ev)
        return ev

    sample = [_ev("supports")] * 30 + [_ev("contradicts")] * 5 + [_ev("neutral")] * 10
    result = aggregate(sample)
    logger.info("Score: {:.2f} | Confidence: {:.2f} | Distribution: {}",
                result.score, result.confidence, result.stance_distribution)
