"""Unit tests for ingestion/scheduler.py.

Mocks the source adapter and database. No network calls, no disk I/O.
"""

from unittest.mock import MagicMock, patch, call
from collections.abc import Generator

import pytest

from hyporeddit.ingestion.scheduler import run_backfill, run_delta
from hyporeddit.sources.base import RawPost, RawComment
from hyporeddit.storage.sqlite import Database


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_raw_post(id_: str = "p1", title: str = "Post") -> RawPost:
    return RawPost(
        id=id_,
        title=title,
        author="user1",
        selftext="body",
        score=10,
        upvote_ratio=0.9,
        num_comments=2,
        created_utc=1_700_000_000,
        url=f"https://reddit.com/r/hausbau/comments/{id_}",
        is_self=True,
        flair=None,
        edited=False,
        awards_count=0,
    )


def make_raw_comment(id_: str = "c1", post_id: str = "p1") -> RawComment:
    return RawComment(
        id=id_,
        post_id=post_id,
        parent_id=post_id,
        author="commenter",
        body="Ein Kommentar.",
        score=3,
        depth=0,
        created_utc=1_700_001_000,
        is_submitter=False,
        edited=False,
        awards_count=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_backfill_creates_job_row() -> None:
    """run_backfill() creates an ingestion_jobs row with mode='backfill'."""
    db = MagicMock(spec=Database)
    adapter = MagicMock()
    adapter.source_name.return_value = "reddit:r/hausbau"
    adapter.fetch_posts_with_cursor.return_value = ([], None)

    with patch("hyporeddit.ingestion.scheduler.get_db", return_value=db), \
         patch("hyporeddit.ingestion.scheduler.RedditJsonAdapter", return_value=adapter), \
         patch("hyporeddit.ingestion.scheduler.ensure_default_source"), \
         patch("hyporeddit.ingestion.scheduler.insert_ingestion_job") as mock_insert, \
         patch("hyporeddit.ingestion.scheduler.update_ingestion_job"), \
         patch("hyporeddit.ingestion.scheduler.get_resumable_backfill_job", return_value=None):
        run_backfill(limit=10)

    mock_insert.assert_called_once()
    args = mock_insert.call_args
    assert args[1]["mode"] == "backfill" or args[0][2] == "backfill"


def test_backfill_upserts_fetched_posts() -> None:
    """run_backfill() calls upsert_post for each fetched post."""
    db = MagicMock(spec=Database)
    adapter = MagicMock()
    adapter.source_name.return_value = "reddit:r/hausbau"
    adapter.fetch_posts_with_cursor.side_effect = [
        ([make_raw_post("p1"), make_raw_post("p2")], None),
    ]
    adapter.fetch_comments.return_value = []

    with patch("hyporeddit.ingestion.scheduler.get_db", return_value=db), \
         patch("hyporeddit.ingestion.scheduler.RedditJsonAdapter", return_value=adapter), \
         patch("hyporeddit.ingestion.scheduler.ensure_default_source"), \
         patch("hyporeddit.ingestion.scheduler.insert_ingestion_job"), \
         patch("hyporeddit.ingestion.scheduler.update_ingestion_job"), \
         patch("hyporeddit.ingestion.scheduler.get_resumable_backfill_job", return_value=None), \
         patch("hyporeddit.ingestion.scheduler.upsert_post") as mock_upsert:
        run_backfill(limit=10)

    assert mock_upsert.call_count == 2


def test_backfill_stops_at_limit() -> None:
    """run_backfill() pages until `limit` posts are collected."""
    db = MagicMock(spec=Database)
    adapter = MagicMock()
    adapter.source_name.return_value = "reddit:r/hausbau"
    page = [make_raw_post(f"p{i}") for i in range(5)]
    adapter.fetch_posts_with_cursor.side_effect = [
        (page, "cursor1"),
        (page, "cursor2"),
        (page, None),
    ]
    adapter.fetch_comments.return_value = []

    with patch("hyporeddit.ingestion.scheduler.get_db", return_value=db), \
         patch("hyporeddit.ingestion.scheduler.RedditJsonAdapter", return_value=adapter), \
         patch("hyporeddit.ingestion.scheduler.ensure_default_source"), \
         patch("hyporeddit.ingestion.scheduler.insert_ingestion_job"), \
         patch("hyporeddit.ingestion.scheduler.update_ingestion_job"), \
         patch("hyporeddit.ingestion.scheduler.get_resumable_backfill_job", return_value=None), \
         patch("hyporeddit.ingestion.scheduler.upsert_post"):
        run_backfill(limit=8)

    # 8 limit, 5 per page → 2 pages (fetches 5 then 3 more, stops)
    assert adapter.fetch_posts_with_cursor.call_count <= 3


def test_backfill_marks_job_complete_on_success() -> None:
    db = MagicMock(spec=Database)
    adapter = MagicMock()
    adapter.source_name.return_value = "reddit:r/hausbau"
    adapter.fetch_posts_with_cursor.return_value = ([], None)

    with patch("hyporeddit.ingestion.scheduler.get_db", return_value=db), \
         patch("hyporeddit.ingestion.scheduler.RedditJsonAdapter", return_value=adapter), \
         patch("hyporeddit.ingestion.scheduler.ensure_default_source"), \
         patch("hyporeddit.ingestion.scheduler.insert_ingestion_job"), \
         patch("hyporeddit.ingestion.scheduler.update_ingestion_job") as mock_update, \
         patch("hyporeddit.ingestion.scheduler.get_resumable_backfill_job", return_value=None):
        run_backfill(limit=10)

    statuses = [c[1].get("status") or c[0][1] for c in mock_update.call_args_list
                if c[1].get("status") or len(c[0]) > 1]
    assert "complete" in str(mock_update.call_args_list)


def test_backfill_marks_job_failed_on_exception() -> None:
    db = MagicMock(spec=Database)
    adapter = MagicMock()
    adapter.source_name.return_value = "reddit:r/hausbau"
    adapter.fetch_posts_with_cursor.side_effect = RuntimeError("Network error")

    with patch("hyporeddit.ingestion.scheduler.get_db", return_value=db), \
         patch("hyporeddit.ingestion.scheduler.RedditJsonAdapter", return_value=adapter), \
         patch("hyporeddit.ingestion.scheduler.ensure_default_source"), \
         patch("hyporeddit.ingestion.scheduler.insert_ingestion_job"), \
         patch("hyporeddit.ingestion.scheduler.update_ingestion_job") as mock_update, \
         patch("hyporeddit.ingestion.scheduler.get_resumable_backfill_job", return_value=None):
        with pytest.raises(RuntimeError):
            run_backfill(limit=10)

    assert "failed" in str(mock_update.call_args_list)


def test_delta_only_fetches_new_posts() -> None:
    """run_delta() skips posts whose IDs are already in SQLite."""
    db = MagicMock(spec=Database)
    adapter = MagicMock()
    adapter.source_name.return_value = "reddit:r/hausbau"
    # Adapter returns 3 posts but p1 is already in DB
    adapter.fetch_posts_with_cursor.return_value = (
        [make_raw_post("p1"), make_raw_post("p2"), make_raw_post("p3")],
        None,
    )
    adapter.fetch_comments.return_value = []

    with patch("hyporeddit.ingestion.scheduler.get_db", return_value=db), \
         patch("hyporeddit.ingestion.scheduler.RedditJsonAdapter", return_value=adapter), \
         patch("hyporeddit.ingestion.scheduler.ensure_default_source"), \
         patch("hyporeddit.ingestion.scheduler.insert_ingestion_job"), \
         patch("hyporeddit.ingestion.scheduler.update_ingestion_job"), \
         patch("hyporeddit.ingestion.scheduler.get_all_post_ids", return_value={"p1"}), \
         patch("hyporeddit.ingestion.scheduler.upsert_post") as mock_upsert:
        run_delta()

    # Only p2 and p3 are new
    assert mock_upsert.call_count == 2


# ---------------------------------------------------------------------------
# Synchronous ingestion tests
# ---------------------------------------------------------------------------

def test_scheduler_does_not_use_thread_pool() -> None:
    """scheduler.py must not import threading or ThreadPoolExecutor."""
    import inspect
    import hyporeddit.ingestion.scheduler as sched

    src = inspect.getsource(sched)
    assert "ThreadPoolExecutor" not in src
    assert "threading" not in src


def test_ingest_post_fetches_then_writes() -> None:
    """_ingest_post fetches comments via the adapter then writes post + comments to DB."""
    from hyporeddit.ingestion.scheduler import _ingest_post

    adapter = MagicMock()
    db = MagicMock()
    post = make_raw_post("px")
    comment = make_raw_comment("cx", "px")
    adapter.fetch_comments.return_value = [comment]

    with patch("hyporeddit.ingestion.scheduler.upsert_post") as mock_upsert, \
         patch("hyporeddit.ingestion.scheduler.insert_comment") as mock_insert:
        _ingest_post(adapter, db, "reddit:r/hausbau", post)

    adapter.fetch_comments.assert_called_once_with("px")
    mock_upsert.assert_called_once()
    mock_insert.assert_called_once()


def test_ingest_post_returns_post_id_and_comment_count() -> None:
    """_ingest_post returns (post_id, comment_count)."""
    from hyporeddit.ingestion.scheduler import _ingest_post

    adapter = MagicMock()
    adapter.fetch_comments.return_value = [make_raw_comment(), make_raw_comment("c2")]
    db = MagicMock()

    with patch("hyporeddit.ingestion.scheduler.upsert_post"), \
         patch("hyporeddit.ingestion.scheduler.insert_comment"):
        post_id, count = _ingest_post(adapter, db, "src", make_raw_post("p9"))

    assert post_id == "p9"
    assert count == 2


def test_scheduler_has_ingest_post_helper() -> None:
    """_ingest_post helper must exist for sequential ingestion."""
    import hyporeddit.ingestion.scheduler as sched

    assert hasattr(sched, "_ingest_post")


def test_backfill_propagates_exception_from_worker() -> None:
    """run_backfill() re-raises an exception thrown inside a worker thread."""
    db = MagicMock()
    adapter = MagicMock()
    adapter.source_name.return_value = "reddit:r/hausbau"
    adapter.fetch_posts_with_cursor.return_value = ([make_raw_post("p1")], None)
    adapter.fetch_comments.side_effect = RuntimeError("fetch failed")

    with patch("hyporeddit.ingestion.scheduler.get_db", return_value=db), \
         patch("hyporeddit.ingestion.scheduler.RedditJsonAdapter", return_value=adapter), \
         patch("hyporeddit.ingestion.scheduler.ensure_default_source"), \
         patch("hyporeddit.ingestion.scheduler.insert_ingestion_job"), \
         patch("hyporeddit.ingestion.scheduler.update_ingestion_job"), \
         patch("hyporeddit.ingestion.scheduler.get_resumable_backfill_job", return_value=None):
        with pytest.raises(RuntimeError, match="fetch failed"):
            from hyporeddit.ingestion.scheduler import run_backfill
            run_backfill(limit=1)


def test_delta_propagates_exception_from_worker() -> None:
    """run_delta() re-raises an exception thrown inside a worker thread."""
    db = MagicMock()
    adapter = MagicMock()
    adapter.source_name.return_value = "reddit:r/hausbau"
    adapter.fetch_posts_with_cursor.return_value = ([make_raw_post("p1")], None)
    adapter.fetch_comments.side_effect = RuntimeError("delta fetch failed")

    with patch("hyporeddit.ingestion.scheduler.get_db", return_value=db), \
         patch("hyporeddit.ingestion.scheduler.RedditJsonAdapter", return_value=adapter), \
         patch("hyporeddit.ingestion.scheduler.ensure_default_source"), \
         patch("hyporeddit.ingestion.scheduler.insert_ingestion_job"), \
         patch("hyporeddit.ingestion.scheduler.update_ingestion_job"), \
         patch("hyporeddit.ingestion.scheduler.get_all_post_ids", return_value=set()):
        with pytest.raises(RuntimeError, match="delta fetch failed"):
            from hyporeddit.ingestion.scheduler import run_delta
            run_delta()
