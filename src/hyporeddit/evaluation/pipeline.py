"""Evaluation pipeline orchestration."""

import time
import uuid
from datetime import datetime, timezone

from loguru import logger

from hyporeddit.config import settings
from hyporeddit.evaluation.aggregator import aggregate
from hyporeddit.evaluation.retriever import retrieve
from hyporeddit.evaluation.stance import classify
from hyporeddit.evaluation.synthesizer import synthesize
from hyporeddit.llm import get_llm_client
from hyporeddit.models.evaluation import EvaluationResult, EvidenceItem
from hyporeddit.storage.sqlite import (
    get_db,
    get_hypothesis_by_text,
    insert_evaluation_run,
    insert_evidence_classifications,
    insert_hypothesis,
)

try:
    from hyporeddit.embedding.bge_m3 import BGE_M3_Encoder
except ImportError:
    BGE_M3_Encoder = None  # type: ignore[assignment,misc]


def get_latest_run_for_hypothesis(db, hypothesis_id: str):
    """Return the most recent evaluation run for this hypothesis, or None."""
    rows = db.execute(
        "SELECT * FROM evaluation_runs WHERE hypothesis_id=? ORDER BY run_at DESC LIMIT 1",
        (hypothesis_id,),
    ).fetchall()
    return rows[0] if rows else None


def _build_result_from_run(run, hypothesis_text: str) -> EvaluationResult:
    """Re-construct EvaluationResult from a persisted run row (cache hit)."""
    return EvaluationResult(
        run_id=run["id"],
        hypothesis_id=run["hypothesis_id"],
        hypothesis_text=hypothesis_text,
        score=run["score"],
        confidence=run["confidence"],
        sample_size=run["sample_size"],
        stance_distribution={
            "supports": run["stance_supports"],
            "contradicts": run["stance_contradicts"],
            "neutral": run["stance_neutral"],
            "irrelevant": run["stance_irrelevant"],
        },
        evidence=[],  # evidence items are not re-fetched for cached results
        synthesis=run["synthesis"],
        model_classification=run["model_classification"],
        model_synthesis=run["model_synthesis"],
    )


def evaluate_hypothesis(
    text: str,
    top_k: int = 100,
    force_rerun: bool = False,
) -> EvaluationResult:
    """Run the full hypothesis evaluation pipeline.

    Steps: embed → retrieve → classify → aggregate → synthesize → persist.
    Returns cached result if an existing run is found and force_rerun=False.
    """
    t_start = time.time()
    db = get_db()

    # ── Upsert hypothesis ──────────────────────────────────────────────────
    existing_hyp = get_hypothesis_by_text(db, text)
    if existing_hyp is None:
        hypothesis_id = str(uuid.uuid4())
        insert_hypothesis(
            db,
            id=hypothesis_id,
            text=text,
            language="en",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    else:
        hypothesis_id = existing_hyp["id"]

    # ── Cache check ────────────────────────────────────────────────────────
    if not force_rerun:
        latest_run = get_latest_run_for_hypothesis(db, hypothesis_id)
        if latest_run is not None:
            logger.info("Cache hit for hypothesis {} — returning run {}", hypothesis_id, latest_run["id"])
            return _build_result_from_run(latest_run, text)

    # ── Step 1: Embed hypothesis ──────────────────────────────────────────
    t0 = time.time()
    encoder = BGE_M3_Encoder()
    query_vector = encoder.encode_query(text)
    logger.info("Embedding: {:.2f}s", time.time() - t0)

    # ── Step 2: Retrieve ───────────────────────────────────────────────────
    t0 = time.time()
    retrieved = retrieve(query_vector, top_k=top_k)
    logger.info("Retrieval: {:.2f}s | {} chunks", time.time() - t0, len(retrieved))

    # ── Step 3: Classify stances ───────────────────────────────────────────
    t0 = time.time()
    llm_client = get_llm_client()
    classified = classify(text, retrieved, llm_client)
    logger.info("Classification: {:.2f}s", time.time() - t0)

    # ── Step 4: Aggregate ─────────────────────────────────────────────────
    agg = aggregate(classified)

    # ── Step 5: Synthesize ────────────────────────────────────────────────
    t0 = time.time()
    synthesis_text = synthesize(text, classified, agg, llm_client)
    logger.info("Synthesis: {:.2f}s", time.time() - t0)

    # ── Step 6: Persist ───────────────────────────────────────────────────
    run_id = str(uuid.uuid4())
    run_at = datetime.now(timezone.utc).isoformat()

    insert_evaluation_run(
        db,
        id=run_id,
        hypothesis_id=hypothesis_id,
        run_at=run_at,
        score=agg.score,
        confidence=agg.confidence,
        sample_size=agg.sample_size,
        stance_supports=agg.stance_distribution.get("supports", 0),
        stance_contradicts=agg.stance_distribution.get("contradicts", 0),
        stance_neutral=agg.stance_distribution.get("neutral", 0),
        stance_irrelevant=agg.stance_distribution.get("irrelevant", 0),
        synthesis=synthesis_text,
        model_classification=settings.llm_classification_model,
        model_synthesis=settings.llm_synthesis_model,
    )

    evidence_rows = [
        {
            "id": str(uuid.uuid4()),
            "chunk_id": ev.chunk.chunk_id,
            "stance": ev.stance,
            "rationale": ev.rationale,
            "weight": ev.weight,
            "retrieval_score": ev.retrieval_score,
            "created_at": run_at,
        }
        for ev in classified
    ]
    insert_evidence_classifications(db, run_id, evidence_rows)

    logger.info("Total: {:.2f}s", time.time() - t_start)

    # ── Build result ───────────────────────────────────────────────────────
    evidence_items = [
        EvidenceItem(
            chunk_id=ev.chunk.chunk_id,
            stance=ev.stance,
            rationale=ev.rationale,
            text_de=ev.chunk.text_de,
            text_en=ev.chunk.text_en,
            parent_post_title=ev.chunk.parent_post_title,
            source_url=f"https://reddit.com/r/{settings.reddit_subreddit}/comments/{ev.chunk.parent_post_id}/",
            weight=ev.weight,
            retrieval_score=ev.retrieval_score,
        )
        for ev in classified
    ]

    return EvaluationResult(
        run_id=run_id,
        hypothesis_id=hypothesis_id,
        hypothesis_text=text,
        score=agg.score,
        confidence=agg.confidence,
        sample_size=agg.sample_size,
        stance_distribution=agg.stance_distribution,
        evidence=evidence_items,
        synthesis=synthesis_text,
        model_classification=settings.llm_classification_model,
        model_synthesis=settings.llm_synthesis_model,
    )
