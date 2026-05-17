"""Ingestion orchestration — backfill and daily delta modes.

Two modes:
  backfill: page through /new.json up to `limit` posts, resumable on failure.
  delta:    fetch first page only, upsert posts not already in SQLite.
"""

import uuid
from datetime import datetime, timezone

from loguru import logger

from hyporeddit.sources.reddit_json import RedditJsonAdapter
from hyporeddit.storage.sqlite import (
    Database,
    ensure_default_source,
    get_all_post_ids,
    get_db,
    get_resumable_backfill_job,
    insert_comment,
    insert_ingestion_job,
    update_ingestion_job,
    upsert_post,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _post_to_kwargs(post: object, source_id: str) -> dict:
    from hyporeddit.sources.base import RawPost

    p: RawPost = post  # type: ignore[assignment]
    return {
        "id": p.id,
        "source_id": source_id,
        "title": p.title,
        "body": p.selftext,
        "author": p.author,
        "created_utc": p.created_utc,
        "score": p.score,
        "upvote_ratio": p.upvote_ratio,
        "num_comments": p.num_comments,
        "flair": p.flair,
        "url": p.url,
        "is_self": int(p.is_self),
        "edited": int(p.edited),
        "fetched_at": _now(),
    }


def _comment_to_kwargs(comment: object) -> dict:
    from hyporeddit.sources.base import RawComment

    c: RawComment = comment  # type: ignore[assignment]
    return {
        "id": c.id,
        "post_id": c.post_id,
        "parent_id": c.parent_id,
        "author": c.author,
        "body": c.body,
        "created_utc": c.created_utc,
        "score": c.score,
        "depth": c.depth,
        "is_submitter": int(c.is_submitter),
        "edited": int(c.edited),
        "fetched_at": _now(),
    }


def _ingest_post(
    adapter: RedditJsonAdapter,
    db: "Database",
    source_id: str,
    post: object,
) -> tuple[str, int]:
    """Fetch comments for one post, then write post + comments to DB."""
    from hyporeddit.sources.base import RawPost

    p: RawPost = post  # type: ignore[assignment]
    comments = adapter.fetch_comments(p.id)
    upsert_post(db, **_post_to_kwargs(post, source_id))
    for comment in comments:
        insert_comment(db, **_comment_to_kwargs(comment))
    return p.id, len(comments)


def run_backfill(limit: int = 1000) -> None:
    """Fetch up to `limit` posts and their comment trees, resumable on failure."""
    db = get_db()
    adapter = RedditJsonAdapter()
    source_id = adapter.source_name()
    ensure_default_source(db)

    existing_job = get_resumable_backfill_job(db)
    if existing_job is not None:
        job_id = existing_job["id"]
        cursor = existing_job["cursor"]
        posts_fetched = existing_job["posts_fetched"]
        logger.info("Resuming backfill job {} from cursor={}, posts_fetched={}",
                    job_id, cursor, posts_fetched)
        update_ingestion_job(db, id=job_id, status="running")
    else:
        job_id = str(uuid.uuid4())
        cursor = None
        posts_fetched = 0
        insert_ingestion_job(db, id=job_id, mode="backfill",
                             source_id=source_id, status="running",
                             started_at=_now())

    try:
        while posts_fetched < limit:
            page_limit = min(100, limit - posts_fetched)
            posts, next_cursor = adapter.fetch_posts_with_cursor(limit=page_limit, after=cursor)
            if not posts:
                break

            for post in posts:
                _ingest_post(adapter, db, source_id, post)

            posts_fetched += len(posts)
            cursor = next_cursor
            update_ingestion_job(db, id=job_id, status="running",
                                 cursor=cursor, posts_fetched=posts_fetched)
            logger.info("Backfill progress: {}/{} posts", posts_fetched, limit)

            if next_cursor is None:
                break

        update_ingestion_job(db, id=job_id, status="complete", posts_fetched=posts_fetched)
        logger.info("Backfill complete: {} posts stored", posts_fetched)

    except Exception as exc:
        update_ingestion_job(db, id=job_id, status="failed", error=str(exc))
        logger.error("Backfill failed: {}", exc)
        raise


def run_delta() -> None:
    """Fetch the first page of /new.json and store only posts not already in SQLite."""
    db = get_db()
    adapter = RedditJsonAdapter()
    source_id = adapter.source_name()
    ensure_default_source(db)

    job_id = str(uuid.uuid4())
    insert_ingestion_job(db, id=job_id, mode="delta",
                         source_id=source_id, status="running",
                         started_at=_now())

    try:
        posts, _ = adapter.fetch_posts_with_cursor(limit=100)
        existing_ids = get_all_post_ids(db)
        new_posts = [p for p in posts if p.id not in existing_ids]

        if not new_posts:
            logger.info("Delta run: 0 new posts found")
            update_ingestion_job(db, id=job_id, status="complete", posts_fetched=0)
            return

        for post in new_posts:
            _ingest_post(adapter, db, source_id, post)

        update_ingestion_job(db, id=job_id, status="complete", posts_fetched=len(new_posts))
        logger.info("Delta run: {} new posts stored", len(new_posts))

    except Exception as exc:
        update_ingestion_job(db, id=job_id, status="failed", error=str(exc))
        logger.error("Delta run failed: {}", exc)
        raise


if __name__ == "__main__":
    # Run: python -m hyporeddit.ingestion.scheduler
    # Env: SQLITE_PATH, ANTHROPIC_API_KEY (for processing), or none for raw ingest only
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "delta"
    if mode == "backfill":
        run_backfill(limit=50)
    else:
        run_delta()
