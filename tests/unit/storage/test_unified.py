"""Unit tests for storage/unified.py — single write path and store verification."""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from hyporeddit.models.ingestion import Chunk, ChunkMetadata
from hyporeddit.storage.sqlite import Database


def _make_chunk(chunk_id: str = "abc123", source_type: str = "comment") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source_type=source_type,
        source_id="t1_xyz",
        text_de="Wir haben lange gewartet.",
        text_en="We waited for a long time.",
        parent_post_id="post1",
        parent_post_title="Bauantrag dauert",
        parent_post_body="Es dauert sehr lange.",
        char_offset=0,
        metadata=ChunkMetadata(
            score=10,
            upvote_ratio=0.95,
            created_utc=1700000000,
            depth=1,
            author_karma=500,
        ),
    )


def _make_db() -> Database:
    db = Database(path=":memory:")
    db.initialize()
    return db


class TestStoreChunk:
    def test_store_chunk_inserts_into_sqlite(self):
        from hyporeddit.storage.unified import store_chunk

        db = _make_db()
        mock_vs = MagicMock()
        chunk = _make_chunk()
        embedding = np.ones(1024, dtype=np.float32)

        store_chunk(chunk, embedding, db=db, vector_store=mock_vs)

        rows = db.execute("SELECT chunk_id FROM chunks WHERE chunk_id=?", (chunk.chunk_id,)).fetchall()
        assert len(rows) == 1

    def test_store_chunk_calls_lance_upsert(self):
        from hyporeddit.storage.unified import store_chunk

        db = _make_db()
        mock_vs = MagicMock()
        chunk = _make_chunk()
        embedding = np.ones(1024, dtype=np.float32)

        store_chunk(chunk, embedding, db=db, vector_store=mock_vs)

        mock_vs.upsert_vector.assert_called_once()
        call_args = mock_vs.upsert_vector.call_args
        assert call_args.args[0] == chunk.chunk_id

    def test_store_chunk_passes_correct_metadata_to_lance(self):
        from hyporeddit.storage.unified import store_chunk

        db = _make_db()
        mock_vs = MagicMock()
        chunk = _make_chunk(source_type="post")
        embedding = np.zeros(1024, dtype=np.float32)

        store_chunk(chunk, embedding, db=db, vector_store=mock_vs)

        metadata = mock_vs.upsert_vector.call_args.args[2]
        assert metadata["source_type"] == "post"
        assert metadata["parent_post_id"] == "post1"

    def test_store_chunk_marks_orphaned_when_lance_fails(self):
        from hyporeddit.storage.unified import store_chunk

        db = _make_db()
        mock_vs = MagicMock()
        mock_vs.upsert_vector.side_effect = RuntimeError("LanceDB unavailable")
        chunk = _make_chunk()
        embedding = np.ones(1024, dtype=np.float32)

        store_chunk(chunk, embedding, db=db, vector_store=mock_vs)

        row = db.execute(
            "SELECT is_orphaned FROM chunks WHERE chunk_id=?", (chunk.chunk_id,)
        ).fetchone()
        assert row is not None
        assert row["is_orphaned"] == 1

    def test_store_chunk_sqlite_inserted_even_if_lance_fails(self):
        from hyporeddit.storage.unified import store_chunk

        db = _make_db()
        mock_vs = MagicMock()
        mock_vs.upsert_vector.side_effect = RuntimeError("LanceDB down")
        chunk = _make_chunk()
        embedding = np.ones(1024, dtype=np.float32)

        store_chunk(chunk, embedding, db=db, vector_store=mock_vs)

        row = db.execute(
            "SELECT chunk_id FROM chunks WHERE chunk_id=?", (chunk.chunk_id,)
        ).fetchone()
        assert row is not None

    def test_store_chunk_idempotent_for_sqlite(self):
        from hyporeddit.storage.unified import store_chunk

        db = _make_db()
        mock_vs = MagicMock()
        chunk = _make_chunk()
        embedding = np.ones(1024, dtype=np.float32)

        store_chunk(chunk, embedding, db=db, vector_store=mock_vs)
        store_chunk(chunk, embedding, db=db, vector_store=mock_vs)

        count = db.execute(
            "SELECT COUNT(*) as n FROM chunks WHERE chunk_id=?", (chunk.chunk_id,)
        ).fetchone()["n"]
        assert count == 1


class TestVerifyStores:
    def test_verify_stores_reports_ok_when_synced(self, capsys):
        from hyporeddit.storage.unified import verify_stores

        db = _make_db()
        mock_vs = MagicMock()
        # SQLite has chunk_ids {"a", "b"}, LanceDB has the same
        db.execute(
            "INSERT INTO chunks (chunk_id, source_type, source_id, parent_post_id, "
            "parent_post_title, parent_post_body, text_de, char_offset, created_at) "
            "VALUES ('a', 'comment', 's1', 'p1', 'T', 'B', 'de', 0, '2024-01-01')"
        )
        db.execute(
            "INSERT INTO chunks (chunk_id, source_type, source_id, parent_post_id, "
            "parent_post_title, parent_post_body, text_de, char_offset, created_at) "
            "VALUES ('b', 'comment', 's2', 'p1', 'T', 'B', 'de', 0, '2024-01-01')"
        )
        db.commit()
        mock_vs.get_all_chunk_ids.return_value = ["a", "b"]

        result = verify_stores(db=db, vector_store=mock_vs)

        assert result["missing_vectors"] == 0
        assert result["orphaned_vectors"] == 0
        assert result["status"] == "OK"

    def test_verify_stores_detects_missing_vectors(self, capsys):
        from hyporeddit.storage.unified import verify_stores

        db = _make_db()
        mock_vs = MagicMock()
        db.execute(
            "INSERT INTO chunks (chunk_id, source_type, source_id, parent_post_id, "
            "parent_post_title, parent_post_body, text_de, char_offset, created_at) "
            "VALUES ('only_in_sqlite', 'comment', 's1', 'p1', 'T', 'B', 'de', 0, '2024-01-01')"
        )
        db.commit()
        mock_vs.get_all_chunk_ids.return_value = []

        result = verify_stores(db=db, vector_store=mock_vs)

        assert result["missing_vectors"] == 1
        assert result["status"] == "OUT OF SYNC"

    def test_verify_stores_detects_orphaned_vectors(self):
        from hyporeddit.storage.unified import verify_stores

        db = _make_db()
        mock_vs = MagicMock()
        # LanceDB has a vector that SQLite doesn't know about
        mock_vs.get_all_chunk_ids.return_value = ["ghost_vector"]

        result = verify_stores(db=db, vector_store=mock_vs)

        assert result["orphaned_vectors"] == 1
        assert result["status"] == "OUT OF SYNC"

    def test_verify_stores_fix_reembeds_orphaned_sqlite_chunks(self):
        from hyporeddit.storage.unified import verify_stores

        db = _make_db()
        mock_vs = MagicMock()
        mock_vs.get_all_chunk_ids.return_value = []

        db.execute(
            "INSERT INTO chunks (chunk_id, source_type, source_id, parent_post_id, "
            "parent_post_title, parent_post_body, text_de, char_offset, created_at, is_orphaned) "
            "VALUES ('orphan1', 'comment', 's1', 'p1', 'T', 'B', 'German text here', 0, '2024-01-01', 1)"
        )
        db.commit()

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.ones((1, 1024), dtype=np.float32)

        verify_stores(db=db, vector_store=mock_vs, fix=True, encoder=mock_encoder)

        mock_vs.upsert_vector.assert_called_once()
        call_chunk_id = mock_vs.upsert_vector.call_args.args[0]
        assert call_chunk_id == "orphan1"
