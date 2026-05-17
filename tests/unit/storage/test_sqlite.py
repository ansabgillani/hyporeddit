"""Unit tests for storage/sqlite.py — schema creation and basic CRUD.

No env vars required. Uses an in-memory SQLite database per test.
"""

import sqlite3
import time
from collections.abc import Generator

import pytest

from hyporeddit.storage.sqlite import (
    Database,
    get_comments_for_post,
    get_post,
    insert_author,
    insert_comment,
    insert_post,
    upsert_post,
    upsert_source,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_source(id_: str = "reddit:r/hausbau", name: str = "r/hausbau") -> dict:
    return {"id": id_, "name": name, "config": "{}", "created_at": "2024-01-01T00:00:00"}


def make_post(
    id_: str = "abc123",
    source_id: str = "reddit:r/hausbau",
    title: str = "Test Post",
) -> dict:
    return {
        "id": id_,
        "source_id": source_id,
        "title": title,
        "body": "Body text",
        "author": "user1",
        "created_utc": 1_700_000_000,
        "score": 10,
        "upvote_ratio": 0.95,
        "num_comments": 5,
        "flair": None,
        "url": f"https://reddit.com/r/hausbau/comments/{id_}",
        "is_self": 1,
        "edited": 0,
        "fetched_at": "2024-01-01T00:00:00",
    }


def make_comment(
    id_: str = "c1",
    post_id: str = "abc123",
    body: str = "Wir haben 14 Monate gewartet.",
    depth: int = 0,
) -> dict:
    return {
        "id": id_,
        "post_id": post_id,
        "parent_id": post_id,
        "author": "commenter1",
        "body": body,
        "created_utc": 1_700_001_000,
        "score": 5,
        "depth": depth,
        "is_submitter": 0,
        "edited": 0,
        "fetched_at": "2024-01-01T00:00:00",
    }


def make_author(username: str = "user1") -> dict:
    return {
        "username": username,
        "link_karma": 1000,
        "comment_karma": 5000,
        "account_created_utc": 1_600_000_000,
        "captured_at": "2024-01-01T00:00:00",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db() -> Generator[Database, None, None]:
    """Provide a fresh in-memory Database for each test."""
    database = Database(":memory:")
    database.initialize()
    yield database
    database.close()


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_tables_created(db: Database) -> None:
    tables = db.table_names()
    expected = {
        "sources", "posts", "comments", "authors", "chunks",
        "hypotheses", "evaluation_runs", "evidence_classifications", "ingestion_jobs",
    }
    assert expected.issubset(tables)


def test_indexes_created(db: Database) -> None:
    indexes = db.index_names()
    expected = {
        "idx_chunks_source_id",
        "idx_chunks_parent_post_id",
        "idx_evidence_run_id",
        "idx_evaluation_runs_hypothesis",
        "idx_posts_created",
        "idx_comments_post_id",
    }
    assert expected.issubset(indexes)


# ---------------------------------------------------------------------------
# Source tests
# ---------------------------------------------------------------------------

def test_upsert_source_creates(db: Database) -> None:
    upsert_source(db, **make_source())
    row = db.execute("SELECT id, name FROM sources WHERE id='reddit:r/hausbau'").fetchone()
    assert row is not None
    assert row["name"] == "r/hausbau"


def test_upsert_source_idempotent(db: Database) -> None:
    data = make_source()
    upsert_source(db, **data)
    upsert_source(db, **data)  # second call must not raise
    count = db.execute("SELECT COUNT(*) as n FROM sources").fetchone()["n"]
    assert count == 1


# ---------------------------------------------------------------------------
# Post tests
# ---------------------------------------------------------------------------

def test_insert_post(db: Database) -> None:
    upsert_source(db, **make_source())
    insert_post(db, **make_post())
    row = db.execute("SELECT title FROM posts WHERE id='abc123'").fetchone()
    assert row["title"] == "Test Post"


def test_upsert_post_is_idempotent(db: Database) -> None:
    upsert_source(db, **make_source())
    data = make_post()
    upsert_post(db, **data)
    upsert_post(db, **data)
    count = db.execute("SELECT COUNT(*) as n FROM posts").fetchone()["n"]
    assert count == 1


def test_get_post_returns_none_for_missing(db: Database) -> None:
    assert get_post(db, "nonexistent") is None


def test_get_post_returns_row(db: Database) -> None:
    upsert_source(db, **make_source())
    insert_post(db, **make_post())
    row = get_post(db, "abc123")
    assert row is not None
    assert row["title"] == "Test Post"


# ---------------------------------------------------------------------------
# Comment tests
# ---------------------------------------------------------------------------

def test_insert_comment(db: Database) -> None:
    upsert_source(db, **make_source())
    insert_post(db, **make_post())
    insert_comment(db, **make_comment())
    row = db.execute("SELECT body FROM comments WHERE id='c1'").fetchone()
    assert row["body"] == "Wir haben 14 Monate gewartet."


def test_get_comments_for_post_empty(db: Database) -> None:
    upsert_source(db, **make_source())
    insert_post(db, **make_post())
    assert get_comments_for_post(db, "abc123") == []


def test_get_comments_for_post_returns_rows(db: Database) -> None:
    upsert_source(db, **make_source())
    insert_post(db, **make_post())
    insert_comment(db, **make_comment(id_="c1"))
    insert_comment(db, **make_comment(id_="c2", body="Anderer Kommentar"))
    comments = get_comments_for_post(db, "abc123")
    assert len(comments) == 2


def test_insert_comment_multiple_depths(db: Database) -> None:
    upsert_source(db, **make_source())
    insert_post(db, **make_post())
    insert_comment(db, **make_comment(id_="c1", depth=0))
    insert_comment(db, **make_comment(id_="c2", depth=1))
    insert_comment(db, **make_comment(id_="c3", depth=2))
    rows = get_comments_for_post(db, "abc123")
    depths = {r["depth"] for r in rows}
    assert depths == {0, 1, 2}


# ---------------------------------------------------------------------------
# Author tests
# ---------------------------------------------------------------------------

def test_insert_author(db: Database) -> None:
    insert_author(db, **make_author())
    row = db.execute("SELECT username FROM authors WHERE username='user1'").fetchone()
    assert row is not None


def test_insert_author_idempotent(db: Database) -> None:
    data = make_author()
    insert_author(db, **data)
    insert_author(db, **data)
    count = db.execute("SELECT COUNT(*) as n FROM authors WHERE username='user1'").fetchone()["n"]
    assert count == 1


# ---------------------------------------------------------------------------
# Migration idempotency
# ---------------------------------------------------------------------------

def test_initialize_twice_is_safe(db: Database) -> None:
    # Calling initialize on an already-initialized DB must not raise
    db.initialize()
    tables = db.table_names()
    assert "posts" in tables


# ---------------------------------------------------------------------------
# Database.transaction and executemany tests
# ---------------------------------------------------------------------------

def test_transaction_commits_on_success(db: Database) -> None:
    upsert_source(db, **make_source())
    with db.transaction():
        insert_post(db, **make_post())
    row = db.execute("SELECT title FROM posts WHERE id='abc123'").fetchone()
    assert row is not None


def test_transaction_rolls_back_on_exception(db: Database) -> None:
    upsert_source(db, **make_source())
    try:
        with db.transaction():
            # Use execute() directly — repo functions call commit() themselves and bypass rollback
            db.execute(
                """INSERT INTO posts (id, source_id, title, body, author, created_utc, score,
                   upvote_ratio, num_comments, flair, url, is_self, edited, fetched_at)
                   VALUES ('txn_test', 'reddit:r/hausbau', 'T', 'B', 'u', 0, 0, 0, 0, NULL, 'u', 1, 0, '2024-01-01')"""
            )
            raise RuntimeError("abort")
    except RuntimeError:
        pass
    row = db.execute("SELECT id FROM posts WHERE id='txn_test'").fetchone()
    assert row is None


def test_executemany_inserts_multiple_rows(db: Database) -> None:
    upsert_source(db, **make_source())
    rows = [
        (f"p{i}", "reddit:r/hausbau", f"Title {i}", "body", "author",
         1_700_000_000, 5, 0.9, 0, None, f"https://r.com/{i}", 1, 0, "2024-01-01")
        for i in range(3)
    ]
    db.executemany(
        """INSERT OR IGNORE INTO posts
           (id, source_id, title, body, author, created_utc, score, upvote_ratio,
            num_comments, flair, url, is_self, edited, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    db.commit()
    count = db.execute("SELECT COUNT(*) as n FROM posts").fetchone()["n"]
    assert count == 3


def test_close_allows_reconnect(db: Database) -> None:
    db.close()
    db.initialize()
    assert "posts" in db.table_names()


# ---------------------------------------------------------------------------
# Post repository — additional functions
# ---------------------------------------------------------------------------

def test_get_all_post_ids_empty(db: Database) -> None:
    from hyporeddit.storage.sqlite import get_all_post_ids
    assert get_all_post_ids(db) == set()


def test_get_all_post_ids_returns_inserted_ids(db: Database) -> None:
    from hyporeddit.storage.sqlite import get_all_post_ids
    upsert_source(db, **make_source())
    insert_post(db, **make_post(id_="p1"))
    insert_post(db, **make_post(id_="p2"))
    ids = get_all_post_ids(db)
    assert ids == {"p1", "p2"}


def test_get_all_posts_returns_all(db: Database) -> None:
    from hyporeddit.storage.sqlite import get_all_posts
    upsert_source(db, **make_source())
    insert_post(db, **make_post(id_="p1"))
    insert_post(db, **make_post(id_="p2"))
    posts = get_all_posts(db)
    assert len(posts) == 2


# ---------------------------------------------------------------------------
# Chunk repository tests
# ---------------------------------------------------------------------------

def _make_chunk_kwargs(chunk_id: str = "ck1", post_id: str = "abc123") -> dict:
    return {
        "chunk_id": chunk_id,
        "source_type": "comment",
        "source_id": "c1",
        "parent_post_id": post_id,
        "parent_post_title": "Test Post",
        "parent_post_body": "Body text",
        "text_de": "Wir haben 14 Monate gewartet.",
        "text_en": "We waited 14 months.",
        "char_offset": 0,
        "created_at": "2024-01-01T00:00:00",
    }


@pytest.fixture()
def db_with_post(db: Database) -> Database:
    upsert_source(db, **make_source())
    insert_post(db, **make_post())
    return db


def test_insert_chunk_stores_row(db_with_post: Database) -> None:
    from hyporeddit.storage.sqlite import insert_chunk
    insert_chunk(db_with_post, **_make_chunk_kwargs())
    row = db_with_post.execute("SELECT chunk_id FROM chunks WHERE chunk_id='ck1'").fetchone()
    assert row is not None


def test_insert_chunk_is_idempotent(db_with_post: Database) -> None:
    from hyporeddit.storage.sqlite import insert_chunk
    insert_chunk(db_with_post, **_make_chunk_kwargs())
    insert_chunk(db_with_post, **_make_chunk_kwargs())
    count = db_with_post.execute("SELECT COUNT(*) as n FROM chunks WHERE chunk_id='ck1'").fetchone()["n"]
    assert count == 1


def test_get_chunks_returns_matching_rows(db_with_post: Database) -> None:
    from hyporeddit.storage.sqlite import insert_chunk, get_chunks
    insert_chunk(db_with_post, **_make_chunk_kwargs("ck1"))
    insert_chunk(db_with_post, **_make_chunk_kwargs("ck2"))
    rows = get_chunks(db_with_post, ["ck1", "ck2"])
    assert len(rows) == 2
    ids = {r["chunk_id"] for r in rows}
    assert ids == {"ck1", "ck2"}


def test_get_chunks_empty_list_returns_empty(db_with_post: Database) -> None:
    from hyporeddit.storage.sqlite import get_chunks
    assert get_chunks(db_with_post, []) == []


def test_get_all_chunk_ids_excludes_filtered(db_with_post: Database) -> None:
    from hyporeddit.storage.sqlite import insert_chunk, get_all_chunk_ids
    insert_chunk(db_with_post, **_make_chunk_kwargs("ck1"))
    insert_chunk(db_with_post, **_make_chunk_kwargs("ck2"))
    db_with_post.execute("UPDATE chunks SET is_filtered=1 WHERE chunk_id='ck2'")
    db_with_post.commit()
    ids = get_all_chunk_ids(db_with_post)
    assert "ck1" in ids
    assert "ck2" not in ids


def test_mark_chunk_orphaned(db_with_post: Database) -> None:
    from hyporeddit.storage.sqlite import insert_chunk, mark_chunk_orphaned
    insert_chunk(db_with_post, **_make_chunk_kwargs("ck1"))
    mark_chunk_orphaned(db_with_post, "ck1")
    row = db_with_post.execute("SELECT is_orphaned FROM chunks WHERE chunk_id='ck1'").fetchone()
    assert row["is_orphaned"] == 1


def test_get_orphaned_chunks_returns_only_orphaned(db_with_post: Database) -> None:
    from hyporeddit.storage.sqlite import insert_chunk, mark_chunk_orphaned, get_orphaned_chunks
    insert_chunk(db_with_post, **_make_chunk_kwargs("ck1"))
    insert_chunk(db_with_post, **_make_chunk_kwargs("ck2"))
    mark_chunk_orphaned(db_with_post, "ck1")
    orphans = get_orphaned_chunks(db_with_post)
    assert len(orphans) == 1
    assert orphans[0]["chunk_id"] == "ck1"


def test_get_chunk_ids_for_post(db_with_post: Database) -> None:
    from hyporeddit.storage.sqlite import insert_chunk, get_chunk_ids_for_post
    insert_chunk(db_with_post, **_make_chunk_kwargs("ck1", post_id="abc123"))
    insert_chunk(db_with_post, **_make_chunk_kwargs("ck2", post_id="abc123"))
    ids = get_chunk_ids_for_post(db_with_post, "abc123")
    assert ids == {"ck1", "ck2"}


# ---------------------------------------------------------------------------
# Hypothesis repository tests
# ---------------------------------------------------------------------------

def test_insert_hypothesis_stores_row(db: Database) -> None:
    from hyporeddit.storage.sqlite import insert_hypothesis
    insert_hypothesis(db, id="h1", text="Planning is too slow", language="en", created_at="2024-01-01")
    row = db.execute("SELECT id FROM hypotheses WHERE id='h1'").fetchone()
    assert row is not None


def test_insert_hypothesis_is_idempotent(db: Database) -> None:
    from hyporeddit.storage.sqlite import insert_hypothesis
    insert_hypothesis(db, id="h1", text="Planning is too slow", language="en", created_at="2024-01-01")
    insert_hypothesis(db, id="h2", text="Planning is too slow", language="en", created_at="2024-01-01")
    count = db.execute("SELECT COUNT(*) as n FROM hypotheses WHERE text='Planning is too slow'").fetchone()["n"]
    assert count == 1


def test_get_hypothesis_by_text_found(db: Database) -> None:
    from hyporeddit.storage.sqlite import insert_hypothesis, get_hypothesis_by_text
    insert_hypothesis(db, id="h1", text="Planning is too slow", language="en", created_at="2024-01-01")
    row = get_hypothesis_by_text(db, "Planning is too slow")
    assert row is not None
    assert row["id"] == "h1"


def test_get_hypothesis_by_text_not_found(db: Database) -> None:
    from hyporeddit.storage.sqlite import get_hypothesis_by_text
    assert get_hypothesis_by_text(db, "does not exist") is None


# ---------------------------------------------------------------------------
# Evaluation run repository tests
# ---------------------------------------------------------------------------

def _insert_eval_run(db: Database, run_id: str = "run1", hyp_id: str = "h1") -> None:
    from hyporeddit.storage.sqlite import insert_evaluation_run
    insert_evaluation_run(
        db,
        id=run_id,
        hypothesis_id=hyp_id,
        run_at="2024-01-01T00:00:00",
        score=0.75,
        confidence=0.80,
        sample_size=20,
        stance_supports=15,
        stance_contradicts=3,
        stance_neutral=2,
        stance_irrelevant=0,
        synthesis="Strong support.",
        model_classification="claude-haiku",
        model_synthesis="claude-sonnet",
    )


def _setup_hypothesis(db: Database, hyp_id: str = "h1") -> None:
    from hyporeddit.storage.sqlite import insert_hypothesis
    insert_hypothesis(db, id=hyp_id, text="Test hypothesis", language="en", created_at="2024-01-01")


def test_insert_evaluation_run_stores_row(db: Database) -> None:
    _setup_hypothesis(db)
    _insert_eval_run(db)
    row = db.execute("SELECT id FROM evaluation_runs WHERE id='run1'").fetchone()
    assert row is not None


def test_get_evaluation_run_returns_row(db: Database) -> None:
    from hyporeddit.storage.sqlite import get_evaluation_run
    _setup_hypothesis(db)
    _insert_eval_run(db, "run1")
    row = get_evaluation_run("run1", db=db)
    assert row is not None
    assert abs(row["score"] - 0.75) < 1e-6


def test_get_evaluation_run_returns_none_for_missing(db: Database) -> None:
    from hyporeddit.storage.sqlite import get_evaluation_run
    assert get_evaluation_run("nonexistent", db=db) is None


def test_get_evaluation_history_returns_ordered_runs(db: Database) -> None:
    from hyporeddit.storage.sqlite import get_evaluation_history
    _setup_hypothesis(db)
    _insert_eval_run(db, "run1")
    _insert_eval_run(db, "run2")
    history = get_evaluation_history("h1", db=db)
    assert len(history) == 2


def test_get_evaluation_history_empty_for_unknown_hypothesis(db: Database) -> None:
    from hyporeddit.storage.sqlite import get_evaluation_history
    assert get_evaluation_history("unknown", db=db) == []


# ---------------------------------------------------------------------------
# Evidence classification repository tests
# ---------------------------------------------------------------------------

def test_insert_evidence_classifications_stores_rows(db: Database) -> None:
    from hyporeddit.storage.sqlite import insert_evidence_classifications
    _setup_hypothesis(db)
    _insert_eval_run(db)
    upsert_source(db, **make_source())
    insert_post(db, **make_post())
    from hyporeddit.storage.sqlite import insert_chunk
    insert_chunk(db, **_make_chunk_kwargs("ck1"))

    classifications = [
        {
            "id": "ev1",
            "chunk_id": "ck1",
            "stance": "supports",
            "rationale": "relevant",
            "weight": 0.8,
            "retrieval_score": 0.9,
            "created_at": "2024-01-01T00:00:00",
        }
    ]
    insert_evidence_classifications(db, "run1", classifications)

    count = db.execute(
        "SELECT COUNT(*) as n FROM evidence_classifications WHERE run_id='run1'"
    ).fetchone()["n"]
    assert count == 1


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------

def test_get_stats_returns_correct_counts(db: Database) -> None:
    from hyporeddit.storage.sqlite import get_stats
    upsert_source(db, **make_source())
    insert_post(db, **make_post(id_="p1"))
    insert_comment(db, **make_comment(id_="c1", post_id="p1"))

    stats = get_stats(db)

    assert stats["post_count"] == 1
    assert stats["comment_count"] == 1


def test_get_stats_chunk_count_excludes_filtered(db: Database) -> None:
    from hyporeddit.storage.sqlite import get_stats, insert_chunk
    upsert_source(db, **make_source())
    insert_post(db, **make_post())
    insert_chunk(db, **_make_chunk_kwargs("ck1"))
    insert_chunk(db, **_make_chunk_kwargs("ck2"))
    db.execute("UPDATE chunks SET is_filtered=1 WHERE chunk_id='ck2'")
    db.commit()

    stats = get_stats(db)
    assert stats["chunk_count"] == 1


def test_get_stats_returns_none_dates_for_empty_db(db: Database) -> None:
    from hyporeddit.storage.sqlite import get_stats
    stats = get_stats(db)
    assert stats["min_post_utc"] is None
    assert stats["max_post_utc"] is None
    assert stats["last_job"] is None


# ---------------------------------------------------------------------------
# Ingestion job repository tests
# ---------------------------------------------------------------------------

def _insert_job(db: Database, job_id: str = "job1", mode: str = "backfill") -> None:
    from hyporeddit.storage.sqlite import insert_ingestion_job
    insert_ingestion_job(
        db,
        id=job_id,
        mode=mode,
        source_id="reddit:r/hausbau",
        status="running",
        started_at="2024-01-01T00:00:00",
    )


def test_insert_ingestion_job_stores_row(db: Database) -> None:
    _insert_job(db)
    row = db.execute("SELECT id FROM ingestion_jobs WHERE id='job1'").fetchone()
    assert row is not None


def test_update_ingestion_job_changes_status(db: Database) -> None:
    from hyporeddit.storage.sqlite import update_ingestion_job
    _insert_job(db)
    update_ingestion_job(db, id="job1", status="complete")
    row = db.execute("SELECT status FROM ingestion_jobs WHERE id='job1'").fetchone()
    assert row["status"] == "complete"


def test_update_ingestion_job_sets_cursor(db: Database) -> None:
    from hyporeddit.storage.sqlite import update_ingestion_job
    _insert_job(db)
    update_ingestion_job(db, id="job1", status="running", cursor="t3_abc123")
    row = db.execute("SELECT cursor FROM ingestion_jobs WHERE id='job1'").fetchone()
    assert row["cursor"] == "t3_abc123"


def test_update_ingestion_job_sets_posts_fetched(db: Database) -> None:
    from hyporeddit.storage.sqlite import update_ingestion_job
    _insert_job(db)
    update_ingestion_job(db, id="job1", status="complete", posts_fetched=42)
    row = db.execute("SELECT posts_fetched FROM ingestion_jobs WHERE id='job1'").fetchone()
    assert row["posts_fetched"] == 42


def test_update_ingestion_job_sets_error(db: Database) -> None:
    from hyporeddit.storage.sqlite import update_ingestion_job
    _insert_job(db)
    update_ingestion_job(db, id="job1", status="failed", error="Network error")
    row = db.execute("SELECT error FROM ingestion_jobs WHERE id='job1'").fetchone()
    assert row["error"] == "Network error"


def test_get_resumable_backfill_job_returns_running_job(db: Database) -> None:
    from hyporeddit.storage.sqlite import get_resumable_backfill_job
    _insert_job(db, "job1", mode="backfill")
    row = get_resumable_backfill_job(db)
    assert row is not None
    assert row["id"] == "job1"


def test_get_resumable_backfill_job_returns_none_when_complete(db: Database) -> None:
    from hyporeddit.storage.sqlite import get_resumable_backfill_job, update_ingestion_job
    _insert_job(db, "job1", mode="backfill")
    update_ingestion_job(db, id="job1", status="complete")
    assert get_resumable_backfill_job(db) is None


def test_get_resumable_backfill_job_returns_none_for_delta_mode(db: Database) -> None:
    from hyporeddit.storage.sqlite import get_resumable_backfill_job
    _insert_job(db, "job1", mode="delta")
    assert get_resumable_backfill_job(db) is None


# ---------------------------------------------------------------------------
# ensure_default_source tests
# ---------------------------------------------------------------------------

def test_ensure_default_source_creates_source(db: Database) -> None:
    from hyporeddit.config import settings
    from hyporeddit.storage.sqlite import ensure_default_source
    ensure_default_source(db)
    source_id = f"reddit:r/{settings.reddit_subreddit}"
    row = db.execute("SELECT id FROM sources WHERE id=?", (source_id,)).fetchone()
    assert row is not None


def test_ensure_default_source_is_idempotent(db: Database) -> None:
    from hyporeddit.config import settings
    from hyporeddit.storage.sqlite import ensure_default_source
    ensure_default_source(db)
    ensure_default_source(db)
    source_id = f"reddit:r/{settings.reddit_subreddit}"
    count = db.execute("SELECT COUNT(*) as n FROM sources WHERE id=?", (source_id,)).fetchone()["n"]
    assert count == 1
