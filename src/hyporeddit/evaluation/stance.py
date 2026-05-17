"""Batched stance classification of retrieved chunks."""

from loguru import logger

from hyporeddit.evaluation.aggregator import ClassifiedEvidence, compute_weight
from hyporeddit.evaluation.retriever import RetrievedChunk
from hyporeddit.llm.base import ChunkWithContext, LLMClient


def classify(
    hypothesis: str,
    retrieved: list[RetrievedChunk],
    llm_client: LLMClient,
) -> list[ClassifiedEvidence]:
    """Classify each retrieved chunk's stance toward the hypothesis.

    Returns ClassifiedEvidence list with weight pre-computed.
    """
    if not retrieved:
        return []

    chunk_map = {r.chunk.chunk_id: r for r in retrieved}

    chunks_with_context = [
        ChunkWithContext(
            chunk_id=r.chunk.chunk_id,
            text_de=r.chunk.text_de,
            text_en=r.chunk.text_en,
            parent_post_title=r.chunk.parent_post_title,
            parent_post_body=r.chunk.parent_post_body,
            retrieval_score=r.cosine_score,
        )
        for r in retrieved
    ]

    stance_results = llm_client.classify_stances(hypothesis, chunks_with_context)

    classified: list[ClassifiedEvidence] = []
    for sr in stance_results:
        if sr.chunk_id not in chunk_map:
            logger.debug("LLM returned unknown chunk_id {} — skipping", sr.chunk_id)
            continue
        retrieved_chunk = chunk_map[sr.chunk_id]
        ev = ClassifiedEvidence(
            chunk=retrieved_chunk.chunk,
            stance=sr.stance,
            rationale=sr.rationale,
            retrieval_score=retrieved_chunk.cosine_score,
        )
        ev.weight = compute_weight(ev)
        classified.append(ev)

    dist = {"supports": 0, "contradicts": 0, "neutral": 0, "irrelevant": 0}
    for ev in classified:
        if ev.stance in dist:
            dist[ev.stance] += 1
    logger.info(
        "Stance distribution: {} supports, {} contradicts, {} neutral, {} irrelevant",
        dist["supports"], dist["contradicts"], dist["neutral"], dist["irrelevant"],
    )

    return classified
