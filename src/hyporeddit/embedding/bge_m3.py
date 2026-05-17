"""BGE-M3 encoder wrapper for dense passage and query embeddings.

Uses BAAI/bge-m3 via FlagEmbedding (optional dependency — install with
`pip install FlagEmbedding`). Model is downloaded to ~/.cache/huggingface/
on first use.
"""

from typing import Any

import numpy as np
import torch
from loguru import logger

from hyporeddit.config import settings


def _resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("BGE_M3_DEVICE=cuda requested but CUDA is unavailable — falling back to cpu")
        return "cpu"
    return requested


try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FlagEmbedding is required for embedding. Install with: pip install FlagEmbedding"
    ) from exc

_DIM = 1024


class BGE_M3_Encoder:
    """Wraps BGEM3FlagModel to encode passages and queries into 1024-dim dense vectors."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._device = _resolve_device(device or settings.bge_m3_device)
        self._batch_size = batch_size or settings.bge_m3_batch_size
        logger.info("Loading BGE-M3 model '{}' on device={}", model_name, self._device)
        self._model = BGEM3FlagModel(
            model_name,
            use_fp16=(self._device == "cuda"),
            device=self._device,
        )
        logger.info("BGE-M3 model loaded")

    def encode(self, texts: list[str], progress: Any = None, task_id: Any = None) -> np.ndarray:
        """Encode passages (no query prefix). Returns ndarray of shape (n, 1024)."""
        if not texts:
            return np.empty((0, _DIM), dtype=np.float32)

        results: list[np.ndarray] = []
        batch_size = self._batch_size
        i = 0
        _own_progress = progress is None

        if _own_progress:
            from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
            progress = Progress(
                TextColumn("[cyan]Embedding"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                transient=True,
            )
            task_id = progress.add_task("Embedding", total=len(texts))
            progress.start()

        try:
            while i < len(texts):
                batch = texts[i: i + batch_size]
                try:
                    vecs = self._model.encode(batch, batch_size=len(batch))["dense_vecs"]
                    results.append(np.array(vecs, dtype=np.float32))
                    if progress is not None and task_id is not None:
                        progress.advance(task_id, len(batch))
                    i += batch_size
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                        logger.warning(
                            "OOM during encoding — reducing batch size to {}", batch_size
                        )
                    else:
                        raise
        finally:
            if _own_progress and progress is not None:
                progress.stop()

        return np.vstack(results) if results else np.empty((0, _DIM), dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query using the query instruction prefix. Returns (1024,) ndarray."""
        vecs = self._model.encode_queries([query], batch_size=1)["dense_vecs"]
        return np.array(vecs[0], dtype=np.float32)
