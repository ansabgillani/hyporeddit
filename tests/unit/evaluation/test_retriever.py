"""Unit tests for evaluation/retriever.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hyporeddit.evaluation.retriever import RetrievedChunk, retrieve


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_query_vector() -> np.ndarray:
    rng = np.random.default_rng(42)
    v = rng.standard_normal(1024).astype(np.float32)
    return v / np.linalg.norm(v)


def make_sqlite_row(chunk_id: str = "c1") -> dict:
    return {
        "chunk_id": chunk_id,
        "source_type": "comment",
        "source_id": "s1",
        "text_de": "Wir haben 14 Monate gewartet",
        "text_en": "We waited 14 months",
        "parent_post_id": "p1",
        "parent_post_title": "Bauantrag",
        "parent_post_body": "Post body",
        "char_offset": 0,
        "score": 10,
        "upvote_ratio": 0.9,
        "created_utc": 1_700_000_000,
        "depth": 0,
        "author_karma": 1000,
        "num_comments": 5,
        "awards_count": 0,
    }


# ---------------------------------------------------------------------------
# retrieve() — core behaviour
# ---------------------------------------------------------------------------

def test_retrieve_calls_vector_search(tmp_path) -> None:
    query = make_query_vector()

    with patch("hyporeddit.evaluation.retriever.VectorStore") as MockVS, \
         patch("hyporeddit.evaluation.retriever.get_db") as mock_get_db, \
         patch("hyporeddit.evaluation.retriever.get_chunks") as mock_get_chunks:

        mock_vs = MockVS.return_value
        mock_vs.search.return_value = [("c1", 0.85)]
        mock_get_chunks.return_value = [make_sqlite_row("c1")]

        results = retrieve(query, top_k=10)

    mock_vs.search.assert_called_once_with(query, top_k=10, filters=None)


def test_retrieve_returns_retrieved_chunks(tmp_path) -> None:
    query = make_query_vector()

    with patch("hyporeddit.evaluation.retriever.VectorStore") as MockVS, \
         patch("hyporeddit.evaluation.retriever.get_db"), \
         patch("hyporeddit.evaluation.retriever.get_chunks") as mock_get_chunks:

        mock_vs = MockVS.return_value
        mock_vs.search.return_value = [("c1", 0.85), ("c2", 0.72)]
        mock_get_chunks.return_value = [make_sqlite_row("c1"), make_sqlite_row("c2")]

        results = retrieve(query, top_k=10)

    assert len(results) == 2
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_retrieve_sets_cosine_score() -> None:
    query = make_query_vector()

    with patch("hyporeddit.evaluation.retriever.VectorStore") as MockVS, \
         patch("hyporeddit.evaluation.retriever.get_db"), \
         patch("hyporeddit.evaluation.retriever.get_chunks") as mock_get_chunks:

        mock_vs = MockVS.return_value
        mock_vs.search.return_value = [("c1", 0.93)]
        mock_get_chunks.return_value = [make_sqlite_row("c1")]

        results = retrieve(query)

    assert abs(results[0].cosine_score - 0.93) < 1e-6


def test_retrieve_hydrates_chunk_from_sqlite() -> None:
    query = make_query_vector()

    with patch("hyporeddit.evaluation.retriever.VectorStore") as MockVS, \
         patch("hyporeddit.evaluation.retriever.get_db"), \
         patch("hyporeddit.evaluation.retriever.get_chunks") as mock_get_chunks:

        mock_vs = MockVS.return_value
        mock_vs.search.return_value = [("c1", 0.8)]
        mock_get_chunks.return_value = [make_sqlite_row("c1")]

        results = retrieve(query)

    chunk = results[0].chunk
    assert chunk.chunk_id == "c1"
    assert chunk.text_de == "Wir haben 14 Monate gewartet"
    assert chunk.text_en == "We waited 14 months"
    assert chunk.parent_post_title == "Bauantrag"


def test_retrieve_empty_when_no_vectors() -> None:
    query = make_query_vector()

    with patch("hyporeddit.evaluation.retriever.VectorStore") as MockVS, \
         patch("hyporeddit.evaluation.retriever.get_db"), \
         patch("hyporeddit.evaluation.retriever.get_chunks") as mock_get_chunks:

        mock_vs = MockVS.return_value
        mock_vs.search.return_value = []
        mock_get_chunks.return_value = []

        results = retrieve(query)

    assert results == []


def test_retrieve_passes_filters_to_vector_store() -> None:
    query = make_query_vector()
    filters = {"source_type": "comment"}

    with patch("hyporeddit.evaluation.retriever.VectorStore") as MockVS, \
         patch("hyporeddit.evaluation.retriever.get_db"), \
         patch("hyporeddit.evaluation.retriever.get_chunks") as mock_get_chunks:

        mock_vs = MockVS.return_value
        mock_vs.search.return_value = []
        mock_get_chunks.return_value = []

        retrieve(query, filters=filters)

    mock_vs.search.assert_called_once_with(query, top_k=100, filters=filters)
