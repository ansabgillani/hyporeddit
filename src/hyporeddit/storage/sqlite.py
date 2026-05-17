"""SQLite storage layer — schema, connection management, and repository functions.

All direct SQLite access lives here. Use the Database class for connections;
never open sqlite3.connect() elsewhere.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from loguru import logger

from hyporeddit.config import settings

# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    name TEXT,
    config JSON,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES sources(id),
    title TEXT,
    body TEXT,
    author TEXT,
    created_utc INTEGER,
    score INTEGER,
    upvote_ratio REAL,
    num_comments INTEGER,
    flair TEXT,
    url TEXT,
    is_self INTEGER,
    edited INTEGER,
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS comments (
    id TEXT PRIMARY KEY,
    post_id TEXT REFERENCES posts(id),
    parent_id TEXT,
    author TEXT,
    body TEXT,
    created_utc INTEGER,
    score INTEGER,
    depth INTEGER,
    is_submitter INTEGER,
    edited INTEGER,
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS authors (
    username TEXT PRIMARY KEY,
    link_karma INTEGER,
    comment_karma INTEGER,
    account_created_utc INTEGER,
    captured_at TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source_type TEXT,
    source_id TEXT,
    parent_post_id TEXT,
    parent_post_title TEXT,
    parent_post_body TEXT,
    text_de TEXT,
    text_en TEXT,
    char_offset INTEGER,
    is_filtered INTEGER DEFAULT 0,
    is_orphaned INTEGER DEFAULT 0,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    text TEXT UNIQUE,
    language TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id TEXT PRIMARY KEY,
    hypothesis_id TEXT REFERENCES hypotheses(id),
    run_at TEXT,
    score REAL,
    confidence REAL,
    sample_size INTEGER,
    stance_supports INTEGER,
    stance_contradicts INTEGER,
    stance_neutral INTEGER,
    stance_irrelevant INTEGER,
    synthesis TEXT,
    model_classification TEXT,
    model_synthesis TEXT
);

CREATE TABLE IF NOT EXISTS evidence_classifications (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES evaluation_runs(id),
    chunk_id TEXT REFERENCES chunks(chunk_id),
    stance TEXT,
    rationale TEXT,
    weight REAL,
    retrieval_score REAL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id TEXT PRIMARY KEY,
    mode TEXT,
    source_id TEXT,
    status TEXT,
    cursor TEXT,
    posts_fetched INTEGER DEFAULT 0,
    started_at TEXT,
    updated_at TEXT,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent_post_id ON chunks(parent_post_id);
CREATE INDEX IF NOT EXISTS idx_evidence_run_id ON evidence_classifications(run_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_runs_hypothesis ON evaluation_runs(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_utc);
CREATE INDEX IF NOT EXISTS idx_comments_post_id ON comments(post_id);
"""


class Database:
    """Thin wrapper around a sqlite3 connection with dict-row access."""

    def __init__(self, path: str | None = None) -> None:
        self._path = path or settings.sqlite_path
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            if self._path != ":memory:":
                Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    @property
    def conn(self) -> sqlite3.Connection:
        return self._connect()

    def initialize(self) -> None:
        """Create all tables and indexes (idempotent — uses IF NOT EXISTS)."""
        self._connect().executescript(_DDL)
        self._connect().commit()
        logger.debug("SQLite schema initialized at {}", self._path)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params: list[tuple[Any, ...]]) -> sqlite3.Cursor:
        return self.conn.executemany(sql, params)

    def commit(self) -> None:
        self.conn.commit()

    def table_names(self) -> set[str]:
        rows = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return {r["name"] for r in rows}

    def index_names(self) -> set[str]:
        rows = self.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        return {r["name"] for r in rows}

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self) -> Iterator[None]:
        try:
            yield
            self.commit()
        except Exception:
            self.conn.rollback()
            raise


# ---------------------------------------------------------------------------
# Module-level default database instance
# ---------------------------------------------------------------------------

_default_db: Database | None = None


def get_db() -> Database:
    global _default_db
    if _default_db is None:
        _default_db = Database()
        _default_db.initialize()
    return _default_db


# ---------------------------------------------------------------------------
# Source repository
# ---------------------------------------------------------------------------

def upsert_source(
    db: Database,
    id: str,
    name: str,
    config: str,
    created_at: str,
) -> None:
    db.execute(
        "INSERT OR REPLACE INTO sources (id, name, config, created_at) VALUES (?, ?, ?, ?)",
        (id, name, config, created_at),
    )
    db.commit()


def ensure_default_source(db: Database) -> None:
    """Register the configured subreddit source row if not already present."""
    from datetime import datetime, timezone

    subreddit = settings.reddit_subreddit
    source_id = f"reddit:r/{subreddit}"
    existing = db.execute(
        "SELECT id FROM sources WHERE id=?", (source_id,)
    ).fetchone()
    if existing is None:
        upsert_source(
            db,
            id=source_id,
            name=f"r/{subreddit}",
            config="{}",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("Registered default source: {}", source_id)


# ---------------------------------------------------------------------------
# Post repository
# ---------------------------------------------------------------------------

def insert_post(
    db: Database,
    id: str,
    source_id: str,
    title: str,
    body: str | None,
    author: str | None,
    created_utc: int,
    score: int,
    upvote_ratio: float,
    num_comments: int,
    flair: str | None,
    url: str,
    is_self: int,
    edited: int,
    fetched_at: str,
) -> None:
    db.execute(
        """INSERT INTO posts
           (id, source_id, title, body, author, created_utc, score, upvote_ratio,
            num_comments, flair, url, is_self, edited, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (id, source_id, title, body, author, created_utc, score, upvote_ratio,
         num_comments, flair, url, is_self, edited, fetched_at),
    )
    db.commit()


def upsert_post(
    db: Database,
    id: str,
    source_id: str,
    title: str,
    body: str | None,
    author: str | None,
    created_utc: int,
    score: int,
    upvote_ratio: float,
    num_comments: int,
    flair: str | None,
    url: str,
    is_self: int,
    edited: int,
    fetched_at: str,
) -> None:
    db.execute(
        """INSERT OR REPLACE INTO posts
           (id, source_id, title, body, author, created_utc, score, upvote_ratio,
            num_comments, flair, url, is_self, edited, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (id, source_id, title, body, author, created_utc, score, upvote_ratio,
         num_comments, flair, url, is_self, edited, fetched_at),
    )
    db.commit()


def get_post(db: Database, post_id: str) -> sqlite3.Row | None:
    return db.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()


def get_all_post_ids(db: Database) -> set[str]:
    rows = db.execute("SELECT id FROM posts").fetchall()
    return {r["id"] for r in rows}


def get_all_posts(db: Database) -> list[sqlite3.Row]:
    return db.execute("SELECT * FROM posts ORDER BY created_utc").fetchall()


def get_chunk_ids_for_post(db: Database, post_id: str) -> set[str]:
    """Return chunk_ids already stored for a given post (as post chunk or comment chunk)."""
    rows = db.execute(
        "SELECT chunk_id FROM chunks WHERE parent_post_id=?", (post_id,)
    ).fetchall()
    return {r["chunk_id"] for r in rows}


# ---------------------------------------------------------------------------
# Comment repository
# ---------------------------------------------------------------------------

def insert_comment(
    db: Database,
    id: str,
    post_id: str,
    parent_id: str,
    author: str | None,
    body: str,
    created_utc: int,
    score: int,
    depth: int,
    is_submitter: int,
    edited: int,
    fetched_at: str,
) -> None:
    db.execute(
        """INSERT OR IGNORE INTO comments
           (id, post_id, parent_id, author, body, created_utc, score, depth,
            is_submitter, edited, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (id, post_id, parent_id, author, body, created_utc, score, depth,
         is_submitter, edited, fetched_at),
    )
    db.commit()


def get_comments_for_post(db: Database, post_id: str) -> list[sqlite3.Row]:
    return db.execute(
        "SELECT * FROM comments WHERE post_id=? ORDER BY depth, created_utc",
        (post_id,),
    ).fetchall()


# ---------------------------------------------------------------------------
# Author repository
# ---------------------------------------------------------------------------

def insert_author(
    db: Database,
    username: str,
    link_karma: int,
    comment_karma: int,
    account_created_utc: int,
    captured_at: str,
) -> None:
    db.execute(
        """INSERT OR REPLACE INTO authors
           (username, link_karma, comment_karma, account_created_utc, captured_at)
           VALUES (?, ?, ?, ?, ?)""",
        (username, link_karma, comment_karma, account_created_utc, captured_at),
    )
    db.commit()


# ---------------------------------------------------------------------------
# Chunk repository
# ---------------------------------------------------------------------------

def insert_chunk(
    db: Database,
    chunk_id: str,
    source_type: str,
    source_id: str,
    parent_post_id: str,
    parent_post_title: str,
    parent_post_body: str,
    text_de: str,
    text_en: str | None,
    char_offset: int,
    created_at: str,
) -> None:
    db.execute(
        """INSERT OR IGNORE INTO chunks
           (chunk_id, source_type, source_id, parent_post_id, parent_post_title,
            parent_post_body, text_de, text_en, char_offset, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (chunk_id, source_type, source_id, parent_post_id, parent_post_title,
         parent_post_body, text_de, text_en, char_offset, created_at),
    )
    db.commit()


def get_chunks(db: Database, chunk_ids: list[str]) -> list[sqlite3.Row]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    return db.execute(
        f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", tuple(chunk_ids)
    ).fetchall()


def get_all_chunk_ids(db: Database) -> set[str]:
    rows = db.execute(
        "SELECT chunk_id FROM chunks WHERE is_filtered=0"
    ).fetchall()
    return {r["chunk_id"] for r in rows}


def mark_chunk_orphaned(db: Database, chunk_id: str) -> None:
    db.execute("UPDATE chunks SET is_orphaned=1 WHERE chunk_id=?", (chunk_id,))
    db.commit()


def get_orphaned_chunks(db: Database) -> list[sqlite3.Row]:
    return db.execute("SELECT * FROM chunks WHERE is_orphaned=1").fetchall()


# ---------------------------------------------------------------------------
# Hypothesis repository
# ---------------------------------------------------------------------------

def insert_hypothesis(db: Database, id: str, text: str, language: str, created_at: str) -> str:
    db.execute(
        """INSERT OR IGNORE INTO hypotheses (id, text, language, created_at)
           VALUES (?, ?, ?, ?)""",
        (id, text, language, created_at),
    )
    db.commit()
    return id


def get_hypothesis_by_text(db: Database, text: str) -> sqlite3.Row | None:
    return db.execute("SELECT * FROM hypotheses WHERE text=?", (text,)).fetchone()


def get_all_hypotheses_with_latest_run(
    db: Database | None = None,
) -> list[sqlite3.Row]:
    _db = db or get_db()
    return _db.execute(
        """SELECT h.*, er.score, er.confidence, er.run_at
           FROM hypotheses h
           LEFT JOIN evaluation_runs er ON er.hypothesis_id = h.id
             AND er.run_at = (
               SELECT MAX(run_at) FROM evaluation_runs WHERE hypothesis_id = h.id
             )
           ORDER BY COALESCE(er.run_at, h.created_at) DESC"""
    ).fetchall()


# ---------------------------------------------------------------------------
# Evaluation run repository
# ---------------------------------------------------------------------------

def insert_evaluation_run(
    db: Database,
    id: str,
    hypothesis_id: str,
    run_at: str,
    score: float,
    confidence: float,
    sample_size: int,
    stance_supports: int,
    stance_contradicts: int,
    stance_neutral: int,
    stance_irrelevant: int,
    synthesis: str,
    model_classification: str,
    model_synthesis: str,
) -> None:
    db.execute(
        """INSERT INTO evaluation_runs
           (id, hypothesis_id, run_at, score, confidence, sample_size,
            stance_supports, stance_contradicts, stance_neutral, stance_irrelevant,
            synthesis, model_classification, model_synthesis)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (id, hypothesis_id, run_at, score, confidence, sample_size,
         stance_supports, stance_contradicts, stance_neutral, stance_irrelevant,
         synthesis, model_classification, model_synthesis),
    )
    db.commit()


def get_evaluation_run(run_id: str, db: Database | None = None) -> sqlite3.Row | None:
    _db = db or get_db()
    return _db.execute(
        "SELECT * FROM evaluation_runs WHERE id=?", (run_id,)
    ).fetchone()


def get_evaluation_history(hypothesis_id: str, db: Database | None = None) -> list[sqlite3.Row]:  # noqa: E501
    _db = db or get_db()
    return _db.execute(
        "SELECT * FROM evaluation_runs WHERE hypothesis_id=? ORDER BY run_at ASC",
        (hypothesis_id,),
    ).fetchall()


# ---------------------------------------------------------------------------
# Evidence classification repository
# ---------------------------------------------------------------------------

def insert_evidence_classifications(
    db: Database,
    run_id: str,
    classifications: list[dict[str, Any]],
) -> None:
    rows = [
        (
            c["id"],
            run_id,
            c["chunk_id"],
            c["stance"],
            c["rationale"],
            c["weight"],
            c["retrieval_score"],
            c["created_at"],
        )
        for c in classifications
    ]
    db.executemany(
        """INSERT INTO evidence_classifications
           (id, run_id, chunk_id, stance, rationale, weight, retrieval_score, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    db.commit()


# ---------------------------------------------------------------------------
# Stats queries
# ---------------------------------------------------------------------------

def get_stats(db: Database) -> dict[str, Any]:
    post_count = db.execute("SELECT COUNT(*) as n FROM posts").fetchone()["n"]
    comment_count = db.execute("SELECT COUNT(*) as n FROM comments").fetchone()["n"]
    chunk_count = db.execute(
        "SELECT COUNT(*) as n FROM chunks WHERE is_filtered=0"
    ).fetchone()["n"]
    date_range = db.execute(
        "SELECT MIN(created_utc) as min_ts, MAX(created_utc) as max_ts FROM posts"
    ).fetchone()
    last_job = db.execute(
        "SELECT * FROM ingestion_jobs ORDER BY updated_at DESC LIMIT 1"
    ).fetchone()
    return {
        "post_count": post_count,
        "comment_count": comment_count,
        "chunk_count": chunk_count,
        "min_post_utc": date_range["min_ts"],
        "max_post_utc": date_range["max_ts"],
        "last_job": dict(last_job) if last_job else None,
    }


# ---------------------------------------------------------------------------
# Ingestion job repository
# ---------------------------------------------------------------------------

def insert_ingestion_job(
    db: Database,
    id: str,
    mode: str,
    source_id: str,
    status: str,
    started_at: str,
) -> None:
    db.execute(
        """INSERT INTO ingestion_jobs (id, mode, source_id, status, started_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (id, mode, source_id, status, started_at, started_at),
    )
    db.commit()


def update_ingestion_job(
    db: Database,
    id: str,
    status: str,
    cursor: str | None = None,
    posts_fetched: int | None = None,
    error: str | None = None,
    updated_at: str | None = None,
) -> None:
    from datetime import datetime, timezone

    _updated_at = updated_at or datetime.now(timezone.utc).isoformat()
    db.execute(
        """UPDATE ingestion_jobs
           SET status=?, cursor=COALESCE(?, cursor),
               posts_fetched=COALESCE(?, posts_fetched),
               error=COALESCE(?, error),
               updated_at=?
           WHERE id=?""",
        (status, cursor, posts_fetched, error, _updated_at, id),
    )
    db.commit()


def get_resumable_backfill_job(db: Database) -> sqlite3.Row | None:
    return db.execute(
        """SELECT * FROM ingestion_jobs
           WHERE mode='backfill' AND status IN ('running', 'failed')
           ORDER BY started_at DESC LIMIT 1"""
    ).fetchone()


if __name__ == "__main__":
    # Run: python -m hyporeddit.storage.sqlite
    # Env: SQLITE_PATH (optional, defaults to data/sqlite/hyporeddit.db)
    db = Database()
    db.initialize()
    tables = db.table_names()
    logger.info("Tables: {}", tables)
    ensure_default_source(db)
    logger.info("Stats: {}", get_stats(db))
    db.close()
