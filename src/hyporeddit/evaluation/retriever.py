"""Vector retrieval + SQLite hydration for the evaluation pipeline."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from hyporeddit.config import settings
from hyporeddit.models.ingestion import Chunk, ChunkMetadata
from hyporeddit.storage.lance import VectorStore
from hyporeddit.storage.sqlite import get_chunks, get_db


@dataclass
class RetrievedChunk:
    chunk: Chunk
    cosine_score: float


def _row_to_chunk(row: Any) -> Chunk:
    d = dict(row)
    meta = ChunkMetadata(
        score=d.get("score", 0),
        upvote_ratio=d.get("upvote_ratio", 0.0),
        created_utc=d.get("created_utc", 0),
        depth=d.get("depth", 0),
        author_karma=d.get("author_karma", 0),
        num_comments=d.get("num_comments", 0),
        awards_count=d.get("awards_count", 0),
    )
    return Chunk(
        chunk_id=d["chunk_id"],
        source_type=d["source_type"],
        source_id=d["source_id"],
        text_de=d["text_de"],
        text_en=d.get("text_en"),
        parent_post_id=d["parent_post_id"],
        parent_post_title=d.get("parent_post_title", ""),
        parent_post_body=d.get("parent_post_body", ""),
        char_offset=d.get("char_offset", 0),
        metadata=meta,
    )


def retrieve(
    query_vector: np.ndarray,
    top_k: int = 100,
    filters: dict[str, Any] | None = None,
) -> list[RetrievedChunk]:
    """Search LanceDB and hydrate results from SQLite.

    Returns list of RetrievedChunk ordered by cosine similarity (descending).
    """
    vs = VectorStore()
    vs.initialize()

    hits = vs.search(query_vector, top_k=top_k, filters=filters)
    if not hits:
        logger.info("Retrieved 0 chunks")
        return []

    score_map = dict(hits)
    chunk_ids = list(score_map)

    db = get_db()
    rows = get_chunks(db, chunk_ids)
    row_map = {row["chunk_id"]: row for row in rows}

    results = []
    for chunk_id in chunk_ids:
        if chunk_id not in row_map:
            logger.warning("Chunk {} found in LanceDB but not in SQLite — skipping", chunk_id)
            continue
        chunk = _row_to_chunk(row_map[chunk_id])
        results.append(RetrievedChunk(chunk=chunk, cosine_score=score_map[chunk_id]))

    scores = [r.cosine_score for r in results]
    if scores:
        logger.info(
            "Retrieved {} chunks (cosine score range: {:.2f}–{:.2f})",
            len(results),
            min(scores),
            max(scores),
        )
    return results
