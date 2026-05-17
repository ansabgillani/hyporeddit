"""LanceDB vector store wrapper.

Stores BGE-M3 dense embeddings (1024-dim) keyed by chunk_id.
All vector reads and writes from the pipeline go through this class.
"""

from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from hyporeddit.config import settings

_TABLE_NAME = "chunks"
_DIM = 1024


def _open_db(path: str) -> Any:
    import lancedb  # deferred import: optional dependency at Phase 2+
    return lancedb.connect(path)


class VectorStore:
    """Thin wrapper around a LanceDB table for chunk embeddings."""

    def __init__(self, path: str | None = None) -> None:
        self._path = path or settings.lance_path
        self._db: Any = None
        self._table: Any = None

    def initialize(self) -> None:
        """Open or create the LanceDB table."""
        Path(self._path).mkdir(parents=True, exist_ok=True)
        self._db = _open_db(self._path)
        existing = self._db.list_tables().tables
        if _TABLE_NAME in existing:
            self._table = self._db.open_table(_TABLE_NAME)
            logger.debug("Opened existing LanceDB table '{}' at {}", _TABLE_NAME, self._path)
        else:
            self._table = None
            logger.debug("LanceDB at {} — table '{}' will be created on first upsert",
                         self._path, _TABLE_NAME)

    def _ensure_table(self, vector: np.ndarray) -> None:
        """Create the table if it doesn't exist yet (schema inferred from first row)."""
        if self._table is not None:
            return
        import pyarrow as pa

        schema = pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), _DIM)),
            pa.field("source_type", pa.string()),
            pa.field("parent_post_id", pa.string()),
            pa.field("created_utc", pa.int64()),
            pa.field("score", pa.int64()),
        ])
        self._table = self._db.create_table(_TABLE_NAME, schema=schema, mode="overwrite")
        logger.info("Created LanceDB table '{}' at {}", _TABLE_NAME, self._path)

    def upsert_vector(
        self,
        chunk_id: str,
        embedding: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        """Insert or replace a chunk vector."""
        self._ensure_table(embedding)
        row = {
            "chunk_id": chunk_id,
            "vector": embedding.astype(np.float32).tolist(),
            "source_type": metadata.get("source_type", ""),
            "parent_post_id": metadata.get("parent_post_id", ""),
            "created_utc": int(metadata.get("created_utc", 0)),
            "score": int(metadata.get("score", 0)),
        }
        # Delete existing row if present (upsert by chunk_id)
        try:
            self._table.delete(f"chunk_id = '{chunk_id}'")
        except Exception as exc:
            logger.debug("LanceDB delete before upsert failed for {}: {}", chunk_id, exc)
        self._table.add([row])

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, cosine_score) pairs for the top-k nearest neighbors."""
        if self._table is None:
            return []

        query = self._table.search(query_vector.astype(np.float32).tolist()).metric("cosine").limit(top_k)

        if filters:
            for key, value in filters.items():
                if isinstance(value, str):
                    query = query.where(f"{key} = '{value}'")
                else:
                    query = query.where(f"{key} = {value}")

        results = query.to_list()
        # LanceDB cosine metric returns distance (0=identical, 2=opposite)
        # Convert to similarity: 1 - distance/2
        return [(r["chunk_id"], max(0.0, 1.0 - r.get("_distance", 0.0) / 2.0)) for r in results]

    def get_all_chunk_ids(self) -> list[str]:
        """Return all stored chunk_ids."""
        if self._table is None:
            return []
        # Use to_pandas for column selection — compatible with this lancedb version
        df = self._table.to_pandas()
        return df["chunk_id"].tolist() if "chunk_id" in df.columns else []

    def count(self) -> int:
        if self._table is None:
            return 0
        return self._table.count_rows()

    def create_index(self) -> None:
        """Create an HNSW index for fast ANN search (call after bulk load)."""
        if self._table is None or self.count() < 256:
            logger.info("Skipping index creation — table too small or empty")
            return
        self._table.create_index(
            metric="cosine",
            num_partitions=8,
            num_sub_vectors=16,
        )
        logger.info("Created HNSW index on LanceDB table '{}'", _TABLE_NAME)


if __name__ == "__main__":
    # Run: python -m hyporeddit.storage.lance
    # Env: LANCE_PATH (optional)
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        vs = VectorStore(path=tmp)
        vs.initialize()
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(1024).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        vs.upsert_vector("test_chunk", vec, {"source_type": "comment", "parent_post_id": "p1",
                                              "created_utc": 1_700_000_000, "score": 5})
        results = vs.search(vec, top_k=1)
        logger.info("Search result: {}", results)
        assert results[0][0] == "test_chunk"
        logger.info("LanceDB smoke test PASSED")
