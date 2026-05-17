"""Post-processing pipeline — filter → chunk → translate → embed → store.

Called via `hyporeddit process [--reprocess]` after ingestion.
Processing is deliberately separate from ingestion so backfill can run
without requiring the embedding model or API key.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from loguru import logger
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from hyporeddit.config import settings
from hyporeddit.ingestion.chunker import chunk_comment, chunk_post
from hyporeddit.ingestion.filters import apply_filters
from hyporeddit.models.ingestion import Chunk
from hyporeddit.sources.base import RawComment, RawPost
from hyporeddit.storage import sqlite as _sqlite
from hyporeddit.storage.sqlite import Database
from hyporeddit.storage.unified import store_chunk, verify_stores


def _row_to_raw_post(row: Any) -> RawPost:
    return RawPost(
        id=row["id"],
        title=row["title"] or "",
        author=row["author"],
        selftext=row["body"] or "",
        score=row["score"] or 0,
        upvote_ratio=row["upvote_ratio"] or 0.0,
        num_comments=row["num_comments"] or 0,
        created_utc=row["created_utc"] or 0,
        url=row["url"] or "",
        is_self=bool(row["is_self"]),
        flair=row["flair"],
        edited=bool(row["edited"]),
        awards_count=0,
    )


def _row_to_raw_comment(row: Any) -> RawComment:
    return RawComment(
        id=row["id"],
        post_id=row["post_id"],
        parent_id=row["parent_id"] or row["post_id"],
        author=row["author"],
        body=row["body"] or "",
        score=row["score"] or 0,
        depth=row["depth"] or 0,
        created_utc=row["created_utc"] or 0,
        is_submitter=bool(row["is_submitter"]),
        edited=bool(row["edited"]),
        awards_count=0,
    )


def _get_default_encoder() -> Any:
    from hyporeddit.embedding.adapter import AdaptiveEncoder

    return AdaptiveEncoder()


def _get_default_vector_store() -> Any:
    from hyporeddit.storage.lance import VectorStore

    vs = VectorStore()
    vs.initialize()
    return vs


def _prepare_post(post_row: Any, db_path: str) -> tuple[str, list[Chunk], int, int]:
    """Load, filter, and chunk one post. Runs in a worker thread with its own DB connection."""
    from hyporeddit.storage.sqlite import Database

    _db = Database(db_path)
    comment_rows = _sqlite.get_comments_for_post(_db, post_row["id"])
    raw_comments = [_row_to_raw_comment(r) for r in comment_rows]
    kept, _ = apply_filters(raw_comments)
    raw_post = _row_to_raw_post(post_row)
    chunks: list[Chunk] = chunk_post(raw_post)
    for comment in kept:
        chunks.extend(chunk_comment(comment, raw_post))
    return post_row["id"], chunks, len(raw_comments), len(kept)


def process_all(
    reprocess: bool = False,
    *,
    db: Database | None = None,
    encoder: Any = None,
    vector_store: Any = None,
    make_translation: bool = False,
    train_adapter: bool = False,
) -> None:
    """Process all fetched posts into chunks, embeddings, and translations.

    Phase 1 — parallel prepare: each post is loaded, filtered, and chunked
               concurrently using one thread per CPU core.
    Phase 2 — embed: one batched BGE-M3 call across all collected texts.
    Phase 2.5 — translate (opt-in): DE→EN via LLM when --make-translation is set.
    Phase 3 — store: write chunks + embeddings to SQLite + LanceDB (main thread).
    """
    db = db or _sqlite.get_db()
    vs = vector_store or _get_default_vector_store()

    posts = _sqlite.get_all_posts(db)
    if not posts:
        logger.info("process-all: no posts found in database")
        return

    # Filter out already-processed posts in the main thread before spawning workers
    if reprocess:
        unprocessed = list(posts)
    else:
        unprocessed = [
            row for row in posts
            if not _sqlite.get_chunk_ids_for_post(db, row["id"])
        ]

    if not unprocessed:
        logger.info("process-all: all posts already processed")
        sqlite_ids = _sqlite.get_all_chunk_ids(db)
        missing = sqlite_ids - set(vs.get_all_chunk_ids())
        if missing:
            logger.info(
                "process-all: {} SQLite chunks have no LanceDB vector — re-embedding",
                len(missing),
            )
            enc = encoder or _get_default_encoder()
            verify_stores(db=db, vector_store=vs, fix=True, encoder=enc)
        return

    enc = encoder or _get_default_encoder()

    n_workers = min(os.cpu_count() or 4, len(unprocessed))
    logger.info(
        "Preparing {} posts with {} parallel workers", len(unprocessed), n_workers
    )

    # Phase 1: parallel load + filter + chunk across all unprocessed posts
    post_chunks: list[tuple[str, list[Chunk]]] = []
    db_path = db._path

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        prep_task = progress.add_task("Preparing chunks", total=len(unprocessed))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_prepare_post, row, db_path): row["id"]
                for row in unprocessed
            }
            for future in as_completed(futures):
                post_id, chunks, n_raw, n_kept = future.result()
                logger.info(
                    "Post {}: {} comments → {} kept after filtering",
                    post_id, n_raw, n_kept,
                )
                if chunks:
                    post_chunks.append((post_id, chunks))
                progress.advance(prep_task)

    if not post_chunks:
        logger.info("process-all: no chunks generated")
        return

    # Phase 2: single batched embedding call across all posts
    all_texts = [c.text_de for _, chunks in post_chunks for c in chunks]
    total_texts = len(all_texts)
    logger.info("Embedding {} texts across {} posts", total_texts, len(post_chunks))

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        embed_task = progress.add_task(f"Embedding {total_texts} texts", total=total_texts)
        all_embeddings: np.ndarray = enc.encode(all_texts, progress=progress, task_id=embed_task)

    # Phase 2.5: translate DE → EN (opt-in via --make-translation)
    # translate_batch_de_to_en owns its own Progress display, so it runs outside
    # the embedding progress context to avoid nested Live conflicts.
    if make_translation:
        from hyporeddit.translation.translator import translate_batch_de_to_en

        logger.info("Translating {} chunks DE→EN …", total_texts)
        translations = translate_batch_de_to_en(all_texts)
        idx = 0
        for _, chunks in post_chunks:
            for chunk in chunks:
                if idx < len(translations):
                    chunk.text_en = translations[idx]
                idx += 1

    # Phase 3: store (main thread — SQLite + LanceDB writes must be serial)
    total_chunks = 0
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        store_task = progress.add_task("Storing chunks", total=total_texts)
        emb_offset = 0

        for post_id, chunks in post_chunks:
            n = len(chunks)
            embeddings = all_embeddings[emb_offset: emb_offset + n]
            emb_offset += n

            for chunk, embedding in zip(chunks, embeddings):
                store_chunk(chunk, embedding, db=db, vector_store=vs)
                progress.advance(store_task)

            total_chunks += n
            logger.info("Stored post {}: {} chunks", post_id, n)

    logger.info("process-all complete: {} total chunks stored", total_chunks)

    verify_stores(db=db, vector_store=vs, fix=True, encoder=enc)

    if train_adapter or total_chunks >= settings.adapter_train_threshold:
        from hyporeddit.embedding.adapter import AdaptiveEncoder
        if isinstance(enc, AdaptiveEncoder):
            logger.info(
                "adapter-train triggered: {} new chunks (threshold={}, forced={})",
                total_chunks, settings.adapter_train_threshold, train_adapter,
            )
            enc.train_adapter(db)
        else:
            logger.debug("adapter-train skipped: encoder is not AdaptiveEncoder")
