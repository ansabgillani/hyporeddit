"""Unified write path — all chunk writes go through here.

Direct writes to sqlite.py or lance.py for chunks are prohibited outside this module.
"""

from datetime import datetime, timezone
from typing import Any

import numpy as np
from loguru import logger

from hyporeddit.models.ingestion import Chunk
from hyporeddit.storage import sqlite as _sqlite
from hyporeddit.storage.sqlite import Database


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_default_vector_store() -> Any:
    from hyporeddit.storage.lance import VectorStore

    vs = VectorStore()
    vs.initialize()
    return vs


def store_chunk(
    chunk: Chunk,
    embedding: np.ndarray,
    *,
    db: Database | None = None,
    vector_store: Any = None,
) -> None:
    """Write a chunk to both SQLite and LanceDB.

    SQLite write is attempted first. If LanceDB fails after SQLite succeeds,
    the chunk is marked is_orphaned=1 and logged as an error. Orphans are
    retried on next startup via verify_stores(fix=True).
    """
    db = db or _sqlite.get_db()
    vs = vector_store or _get_default_vector_store()

    _sqlite.insert_chunk(
        db,
        chunk_id=chunk.chunk_id,
        source_type=chunk.source_type,
        source_id=chunk.source_id,
        parent_post_id=chunk.parent_post_id,
        parent_post_title=chunk.parent_post_title,
        parent_post_body=chunk.parent_post_body,
        text_de=chunk.text_de,
        text_en=chunk.text_en,
        char_offset=chunk.char_offset,
        created_at=_now(),
    )

    metadata = {
        "source_type": chunk.source_type,
        "parent_post_id": chunk.parent_post_id,
        "created_utc": chunk.metadata.created_utc,
        "score": chunk.metadata.score,
    }

    try:
        vs.upsert_vector(chunk.chunk_id, embedding, metadata)
    except Exception as exc:
        logger.error(
            "LanceDB write failed for chunk {} — marking orphaned: {}", chunk.chunk_id, exc
        )
        _sqlite.mark_chunk_orphaned(db, chunk.chunk_id)


def verify_stores(
    *,
    db: Database | None = None,
    vector_store: Any = None,
    fix: bool = False,
    encoder: Any = None,
) -> dict[str, Any]:
    """Compare chunk IDs in SQLite and LanceDB; report discrepancies.

    Returns a dict with keys: missing_vectors, orphaned_vectors, status.
    If fix=True, re-embeds SQLite-only orphaned chunks into LanceDB.
    """
    db = db or _sqlite.get_db()
    vs = vector_store or _get_default_vector_store()

    sqlite_ids: set[str] = _sqlite.get_all_chunk_ids(db)
    lance_ids: set[str] = set(vs.get_all_chunk_ids())

    missing = sqlite_ids - lance_ids          # in SQLite but not LanceDB
    orphaned = lance_ids - sqlite_ids         # in LanceDB but not SQLite

    logger.info(
        "verify-stores: SQLite={} LanceDB={} missing_vectors={} orphaned_vectors={}",
        len(sqlite_ids), len(lance_ids), len(missing), len(orphaned),
    )

    if fix and missing:
        _fix_missing_vectors(db, vs, missing, encoder)
        lance_ids = set(vs.get_all_chunk_ids())
        missing = sqlite_ids - lance_ids
        orphaned = lance_ids - sqlite_ids

    status = "OK" if not missing and not orphaned else "OUT OF SYNC"

    if status == "OK":
        logger.info("verify-stores: stores are IN SYNC")
    else:
        logger.warning("verify-stores: stores are OUT OF SYNC")

    return {
        "sqlite_count": len(sqlite_ids),
        "lance_count": len(lance_ids),
        "missing_vectors": len(missing),
        "orphaned_vectors": len(orphaned),
        "status": status,
    }


def _fix_missing_vectors(
    db: Database,
    vs: Any,
    missing_ids: set[str],
    encoder: Any,
) -> None:
    if encoder is None:
        from hyporeddit.embedding.bge_m3 import BGE_M3_Encoder

        encoder = BGE_M3_Encoder()

    rows = _sqlite.get_chunks(db, list(missing_ids))
    texts = [r["text_de"] for r in rows]
    if not texts:
        return

    logger.info("Re-embedding {} chunks missing from LanceDB", len(texts))
    embeddings = encoder.encode(texts)

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

    fixed = 0
    failed = 0
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Syncing to LanceDB", total=len(rows))
        for row, embedding in zip(rows, embeddings):
            metadata = {
                "source_type": row["source_type"],
                "parent_post_id": row["parent_post_id"],
                "created_utc": 0,
                "score": 0,
            }
            try:
                vs.upsert_vector(row["chunk_id"], embedding, metadata)
                db.execute("UPDATE chunks SET is_orphaned=0 WHERE chunk_id=?", (row["chunk_id"],))
                db.commit()
                fixed += 1
            except Exception as exc:
                logger.error("Failed to sync chunk {} to LanceDB: {}", row["chunk_id"], exc)
                failed += 1
            progress.advance(task)

    logger.info("LanceDB sync complete: {} written, {} failed", fixed, failed)
