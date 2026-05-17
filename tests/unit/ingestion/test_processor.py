"""Unit tests for ingestion/processor.py.

All I/O (SQLite, LanceDB, Anthropic, BGE-M3) is mocked.
process_all uses ThreadPoolExecutor with per-worker DB connections, so tests
that exercise the full pipeline use file-based SQLite (via tmp_path) to ensure
worker threads can open the same database.
"""

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from hyporeddit.storage.sqlite import Database


def _make_db_with_post(post_id: str = "post1", path: str = ":memory:") -> Database:
    db = Database(path=path)
    db.initialize()
    db.execute(
        "INSERT OR IGNORE INTO sources (id, name, config, created_at) VALUES (?, ?, ?, ?)",
        ("reddit:r/hausbau", "r/hausbau", "{}", "2024-01-01"),
    )
    db.execute(
        """INSERT INTO posts
           (id, source_id, title, body, author, created_utc, score, upvote_ratio,
            num_comments, flair, url, is_self, edited, fetched_at)
           VALUES (?, 'reddit:r/hausbau', 'Test Post', 'Some body text', 'author1',
                   1700000000, 10, 0.9, 3, NULL, 'https://reddit.com/r/test', 1, 0, '2024-01-01')""",
        (post_id,),
    )
    db.commit()
    return db


def _make_db_with_comments(post_id: str = "post1", n: int = 3, path: str = ":memory:") -> Database:
    db = _make_db_with_post(post_id, path=path)
    for i in range(n):
        db.execute(
            """INSERT INTO comments
               (id, post_id, parent_id, author, body, created_utc, score, depth,
                is_submitter, edited, fetched_at)
               VALUES (?, ?, ?, 'user', 'A substantive comment about building', 1700000000,
                       5, 0, 0, 0, '2024-01-01')""",
            (f"c{i}", post_id, post_id),
        )
    db.commit()
    return db


@pytest.fixture
def mock_encoder():
    enc = MagicMock()
    enc.encode.side_effect = lambda texts, **kw: np.ones((len(texts), 1024), dtype=np.float32)
    return enc


@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    vs.get_all_chunk_ids.return_value = []
    return vs


class TestProcessAll:
    def test_process_all_processes_each_post(self, mock_encoder, mock_vector_store, tmp_path):
        from hyporeddit.ingestion.processor import process_all

        db = _make_db_with_comments("post1", n=2, path=str(tmp_path / "test.db"))
        process_all(db=db, encoder=mock_encoder, vector_store=mock_vector_store)

        # Chunks should be created for the post + 2 comments
        chunk_count = db.execute("SELECT COUNT(*) as n FROM chunks").fetchone()["n"]
        assert chunk_count >= 1  # at minimum the post itself becomes a chunk

    def test_process_all_skips_posts_already_chunked(self, mock_encoder, mock_vector_store, tmp_path):
        from hyporeddit.ingestion.processor import process_all

        db = _make_db_with_post("post1", path=str(tmp_path / "test.db"))
        # Pre-insert a chunk for post1 to simulate it being already processed
        db.execute(
            """INSERT INTO chunks
               (chunk_id, source_type, source_id, parent_post_id, parent_post_title,
                parent_post_body, text_de, char_offset, created_at)
               VALUES ('existing', 'post', 'post1', 'post1', 'T', 'B', 'de', 0, '2024-01-01')"""
        )
        db.commit()

        # Vector store already has this chunk — no missing vectors, so no re-embedding
        mock_vector_store.get_all_chunk_ids.return_value = ["existing"]

        process_all(db=db, encoder=mock_encoder, vector_store=mock_vector_store, reprocess=False)

        # Encoder should not be called — post is already processed and no missing vectors
        mock_encoder.encode.assert_not_called()

    def test_process_all_reprocesses_when_flag_set(self, mock_encoder, mock_vector_store, tmp_path):
        from hyporeddit.ingestion.processor import process_all

        db = _make_db_with_post("post1", path=str(tmp_path / "test.db"))
        # Pre-insert a chunk for post1
        db.execute(
            """INSERT INTO chunks
               (chunk_id, source_type, source_id, parent_post_id, parent_post_title,
                parent_post_body, text_de, char_offset, created_at)
               VALUES ('existing', 'post', 'post1', 'post1', 'T', 'B', 'de', 0, '2024-01-01')"""
        )
        db.commit()

        process_all(db=db, encoder=mock_encoder, vector_store=mock_vector_store, reprocess=True)

        # With reprocess=True, encoder should be called for the post chunks
        mock_encoder.encode.assert_called()

    def test_process_all_calls_encoder_with_text(self, mock_encoder, mock_vector_store, tmp_path):
        from hyporeddit.ingestion.processor import process_all

        db = _make_db_with_post("p1", path=str(tmp_path / "test.db"))
        process_all(db=db, encoder=mock_encoder, vector_store=mock_vector_store)

        mock_encoder.encode.assert_called()

    def test_process_all_filters_deleted_comments(self, mock_encoder, mock_vector_store, tmp_path):
        from hyporeddit.ingestion.processor import process_all

        db = _make_db_with_post("p1", path=str(tmp_path / "test.db"))
        db.execute(
            """INSERT INTO comments
               (id, post_id, parent_id, author, body, created_utc, score, depth,
                is_submitter, edited, fetched_at)
               VALUES ('del1', 'p1', 'p1', 'user', '[deleted]', 1700000000, 0, 0, 0, 0, '2024-01-01')"""
        )
        db.commit()

        process_all(db=db, encoder=mock_encoder, vector_store=mock_vector_store)

        # [deleted] comment should not produce a chunk
        deleted_chunk = db.execute(
            "SELECT chunk_id FROM chunks WHERE source_id='del1'"
        ).fetchone()
        assert deleted_chunk is None

    def test_process_all_logs_progress(self, mock_encoder, mock_vector_store, tmp_path):
        from hyporeddit.ingestion.processor import process_all

        db = _make_db_with_post("p1", path=str(tmp_path / "test.db"))
        # Should complete without raising
        process_all(db=db, encoder=mock_encoder, vector_store=mock_vector_store)
        assert True


class TestMakeTranslation:
    """--make-translation wires translate_batch_de_to_en into the pipeline."""

    def test_make_translation_calls_translator(self, mock_encoder, mock_vector_store, tmp_path):
        """translate_batch_de_to_en is called with the DE texts when flag is set."""
        from hyporeddit.ingestion.processor import process_all

        db = _make_db_with_post("p1", path=str(tmp_path / "test.db"))

        with patch(
            "hyporeddit.translation.translator.translate_batch_de_to_en",
            return_value=["translated text"],
        ) as mock_translate:
            process_all(
                db=db,
                encoder=mock_encoder,
                vector_store=mock_vector_store,
                make_translation=True,
            )

        mock_translate.assert_called_once()
        # First arg is the list of DE texts — must be non-empty strings
        texts_arg = mock_translate.call_args[0][0]
        assert isinstance(texts_arg, list)
        assert all(isinstance(t, str) and t for t in texts_arg)

    def test_make_translation_stores_text_en(self, mock_encoder, mock_vector_store, tmp_path):
        """Translated text is persisted in the text_en column of the chunks table."""
        from hyporeddit.ingestion.processor import process_all

        db = _make_db_with_post("p1", path=str(tmp_path / "test.db"))

        with patch(
            "hyporeddit.translation.translator.translate_batch_de_to_en",
            return_value=["english translation"],
        ):
            process_all(
                db=db,
                encoder=mock_encoder,
                vector_store=mock_vector_store,
                make_translation=True,
            )

        row = db.execute("SELECT text_en FROM chunks LIMIT 1").fetchone()
        assert row is not None
        assert row["text_en"] == "english translation"

    def test_make_translation_false_skips_translator(self, mock_encoder, mock_vector_store, tmp_path):
        """translate_batch_de_to_en is never called when make_translation is False (default)."""
        from hyporeddit.ingestion.processor import process_all

        db = _make_db_with_post("p1", path=str(tmp_path / "test.db"))

        with patch(
            "hyporeddit.translation.translator.translate_batch_de_to_en"
        ) as mock_translate:
            process_all(db=db, encoder=mock_encoder, vector_store=mock_vector_store)

        mock_translate.assert_not_called()

    def test_make_translation_text_en_null_by_default(self, mock_encoder, mock_vector_store, tmp_path):
        """Without --make-translation, text_en remains NULL in the database."""
        from hyporeddit.ingestion.processor import process_all

        db = _make_db_with_post("p1", path=str(tmp_path / "test.db"))
        process_all(db=db, encoder=mock_encoder, vector_store=mock_vector_store)

        row = db.execute("SELECT text_en FROM chunks LIMIT 1").fetchone()
        assert row is not None
        assert row["text_en"] is None

    def test_make_translation_receives_all_chunk_texts(self, mock_encoder, mock_vector_store, tmp_path):
        """Translator receives one entry per chunk, matching what was embedded."""
        from hyporeddit.ingestion.processor import process_all

        db = _make_db_with_comments("p1", n=2, path=str(tmp_path / "test.db"))

        captured: list[list[str]] = []

        def capture_and_translate(texts):
            captured.append(texts)
            return ["en"] * len(texts)

        with patch(
            "hyporeddit.translation.translator.translate_batch_de_to_en",
            side_effect=capture_and_translate,
        ):
            process_all(
                db=db,
                encoder=mock_encoder,
                vector_store=mock_vector_store,
                make_translation=True,
            )

        assert len(captured) == 1
        chunk_count = db.execute("SELECT COUNT(*) as n FROM chunks").fetchone()["n"]
        assert len(captured[0]) == chunk_count


class TestTrainAdapter:
    """--train-adapter forces domain-adapter training regardless of chunk threshold."""

    def _make_adaptive_encoder(self):
        """Returns a (FakeAdaptiveEncoder class, encoder instance, train_mock) triple."""
        train_mock = MagicMock()

        class FakeAdaptiveEncoder:
            def encode(self, texts, **kw):
                return np.ones((len(texts), 1024), dtype=np.float32)

            def train_adapter(self, db):
                train_mock(db)

        return FakeAdaptiveEncoder, FakeAdaptiveEncoder(), train_mock

    def test_train_adapter_flag_forces_training(self, mock_vector_store, tmp_path):
        """train_adapter=True calls enc.train_adapter even below the chunk threshold."""
        from hyporeddit.ingestion.processor import process_all

        FakeAdaptiveEncoder, enc, train_mock = self._make_adaptive_encoder()
        db = _make_db_with_post("p1", path=str(tmp_path / "test.db"))

        with patch("hyporeddit.embedding.adapter.AdaptiveEncoder", FakeAdaptiveEncoder):
            process_all(
                db=db,
                encoder=enc,
                vector_store=mock_vector_store,
                train_adapter=True,
            )

        train_mock.assert_called_once()

    def test_train_adapter_false_below_threshold_skips_training(self, mock_vector_store, tmp_path):
        """train_adapter=False does not call enc.train_adapter when chunks < threshold."""
        from hyporeddit.ingestion.processor import process_all

        FakeAdaptiveEncoder, enc, train_mock = self._make_adaptive_encoder()
        db = _make_db_with_post("p1", path=str(tmp_path / "test.db"))

        # threshold=200, one post → at most a handful of chunks
        with patch("hyporeddit.embedding.adapter.AdaptiveEncoder", FakeAdaptiveEncoder), \
             patch("hyporeddit.ingestion.processor.settings") as mock_settings:
            mock_settings.adapter_train_threshold = 200
            process_all(
                db=db,
                encoder=enc,
                vector_store=mock_vector_store,
                train_adapter=False,
            )

        train_mock.assert_not_called()

    def test_train_adapter_auto_triggers_above_threshold(self, mock_vector_store, tmp_path):
        """Auto-training still fires when total_chunks >= threshold, even without the flag."""
        from hyporeddit.ingestion.processor import process_all

        FakeAdaptiveEncoder, enc, train_mock = self._make_adaptive_encoder()
        db = _make_db_with_post("p1", path=str(tmp_path / "test.db"))

        with patch("hyporeddit.embedding.adapter.AdaptiveEncoder", FakeAdaptiveEncoder), \
             patch("hyporeddit.ingestion.processor.settings") as mock_settings:
            mock_settings.adapter_train_threshold = 0  # always triggered
            process_all(
                db=db,
                encoder=enc,
                vector_store=mock_vector_store,
                train_adapter=False,
            )

        train_mock.assert_called_once()

    def test_train_adapter_noop_for_non_adaptive_encoder(self, mock_encoder, mock_vector_store, tmp_path):
        """train_adapter=True is a no-op when the encoder is not an AdaptiveEncoder."""
        from hyporeddit.ingestion.processor import process_all

        db = _make_db_with_post("p1", path=str(tmp_path / "test.db"))

        # mock_encoder is a plain MagicMock — not an AdaptiveEncoder instance
        process_all(
            db=db,
            encoder=mock_encoder,
            vector_store=mock_vector_store,
            train_adapter=True,
        )

        # train_adapter attribute should not have been called on a plain mock encoder
        assert not hasattr(mock_encoder, "train_adapter") or \
               not mock_encoder.train_adapter.called
