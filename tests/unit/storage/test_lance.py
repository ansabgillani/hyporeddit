"""Unit tests for storage/lance.py.

Uses a temporary directory for the LanceDB; no external services.
lancedb must be installed (`pip install lancedb`).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from hyporeddit.storage.lance import VectorStore


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_vector(dim: int = 1024, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)  # unit-normalize


def make_metadata(
    source_type: str = "comment",
    parent_post_id: str = "p1",
    created_utc: int = 1_700_000_000,
    score: int = 5,
) -> dict:
    return {
        "source_type": source_type,
        "parent_post_id": parent_post_id,
        "created_utc": created_utc,
        "score": score,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path: Path) -> VectorStore:
    """Fresh VectorStore in a temp directory per test."""
    vs = VectorStore(path=str(tmp_path))
    vs.initialize()
    return vs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_upsert_and_get_all_ids(store: VectorStore) -> None:
    vec = make_vector()
    store.upsert_vector("chunk1", vec, make_metadata())
    ids = store.get_all_chunk_ids()
    assert "chunk1" in ids


def test_upsert_multiple(store: VectorStore) -> None:
    for i in range(5):
        store.upsert_vector(f"chunk{i}", make_vector(seed=i), make_metadata())
    ids = store.get_all_chunk_ids()
    assert len(ids) == 5


def test_upsert_is_idempotent(store: VectorStore) -> None:
    vec = make_vector()
    store.upsert_vector("chunk1", vec, make_metadata())
    store.upsert_vector("chunk1", vec, make_metadata())
    ids = store.get_all_chunk_ids()
    assert ids.count("chunk1") == 1 if isinstance(ids, list) else len([x for x in ids if x == "chunk1"]) == 1


def test_search_returns_results(store: VectorStore) -> None:
    query = make_vector(seed=99)
    for i in range(10):
        store.upsert_vector(f"chunk{i}", make_vector(seed=i), make_metadata())
    results = store.search(query, top_k=5)
    assert len(results) == 5


def test_search_returns_chunk_id_and_score(store: VectorStore) -> None:
    vec = make_vector(seed=0)
    store.upsert_vector("target", vec, make_metadata())
    results = store.search(vec, top_k=1)
    assert len(results) == 1
    chunk_id, score = results[0]
    assert chunk_id == "target"
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0 + 1e-5  # allow tiny floating-point overshoot


def test_search_nearest_vector_ranks_highest(store: VectorStore) -> None:
    """A query identical to a stored vector should rank it first."""
    target = make_vector(seed=42)
    noise = make_vector(seed=1)
    store.upsert_vector("target", target, make_metadata())
    store.upsert_vector("noise", noise, make_metadata())
    results = store.search(target, top_k=2)
    top_id, _ = results[0]
    assert top_id == "target"


def test_search_with_source_type_filter(store: VectorStore) -> None:
    query = make_vector()
    store.upsert_vector("post1", make_vector(seed=1), make_metadata(source_type="post"))
    store.upsert_vector("comment1", make_vector(seed=2), make_metadata(source_type="comment"))
    results = store.search(query, top_k=10, filters={"source_type": "post"})
    ids = [r[0] for r in results]
    assert "comment1" not in ids
    assert "post1" in ids


def test_get_all_chunk_ids_empty_store(store: VectorStore) -> None:
    ids = store.get_all_chunk_ids()
    assert len(ids) == 0


def test_count(store: VectorStore) -> None:
    for i in range(3):
        store.upsert_vector(f"c{i}", make_vector(seed=i), make_metadata())
    assert store.count() == 3
