"""Linear adapter layer trained on top of frozen BGE-M3 for domain adaptation.

Weights are persisted to data/model/adapter.pt (configurable via ADAPTER_PATH).
The adapter is identity-initialized — it starts as a strict no-op and only
diverges as training accumulates. Deleting the file resets to the base model.
"""

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from hyporeddit.config import settings
from hyporeddit.embedding.bge_m3 import BGE_M3_Encoder, _DIM

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise ImportError(
        "torch is required for adapter training. Install FlagEmbedding which brings torch as a dependency."
    ) from exc


class _AdapterLayer(nn.Module):
    """Identity-initialized Linear(1024, 1024) adapter."""

    def __init__(self, dim: int = _DIM) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class AdaptiveEncoder:
    """BGE-M3 encoder with a trainable linear adapter layer.

    Drop-in replacement for BGE_M3_Encoder. Loads adapter weights from disk on
    init if they exist; starts as identity (no behavioral change) otherwise.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int | None = None,
        adapter_path: str | None = None,
    ) -> None:
        self._base = BGE_M3_Encoder(model_name=model_name, device=device, batch_size=batch_size)
        self._device = self._base._device
        self._adapter_path = Path(adapter_path or settings.adapter_path)
        self._layer = _AdapterLayer(dim=_DIM).to(self._device)
        self._layer.eval()
        self._meta: dict = {"step": 0, "loss": None, "chunks_seen": 0, "trained_at": None}

        if self._adapter_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public encode interface — same signatures as BGE_M3_Encoder
    # ------------------------------------------------------------------

    def encode(self, texts: list[str], progress: Any = None, task_id: Any = None) -> np.ndarray:
        """Encode passages through BGE-M3 then adapter. Returns (n, 1024) ndarray."""
        base = self._base.encode(texts, progress=progress, task_id=task_id)
        return self._apply_adapter(base)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query through BGE-M3 then adapter. Returns (1024,) ndarray."""
        base = self._base.encode_query(query)
        tensor = torch.from_numpy(base).unsqueeze(0).to(self._device)
        with torch.no_grad():
            out = self._layer(tensor)
        return out.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_adapter(self, db: Any) -> None:
        """Train the adapter on same-thread chunk pairs from SQLite.

        Pre-computes frozen base embeddings once, then trains only the adapter
        layer with InfoNCE loss using in-batch negatives.
        """
        pairs = _sample_pairs(db, settings.adapter_train_pairs_per_thread)
        if len(pairs) < 2:
            logger.warning("adapter-train: only {} pairs — need at least 2, skipping", len(pairs))
            return

        logger.info(
            "adapter-train: {} pairs, {} epochs, lr={}",
            len(pairs), settings.adapter_train_epochs, settings.adapter_train_lr,
        )

        # Pre-compute frozen base embeddings for all unique texts in one batched call
        all_texts = list({t for pair in pairs for t in pair})
        text_index = {t: i for i, t in enumerate(all_texts)}
        base_vecs = self._base.encode(all_texts)
        base_tensor = torch.from_numpy(base_vecs).to(self._device)

        self._layer.train()
        optimizer = torch.optim.AdamW(self._layer.parameters(), lr=settings.adapter_train_lr)
        batch_size = settings.adapter_train_batch_size
        best_loss = float("inf")

        for epoch in range(settings.adapter_train_epochs):
            random.shuffle(pairs)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                if len(batch) < 2:
                    continue  # InfoNCE needs at least 2 pairs per batch
                anchors = torch.stack([base_tensor[text_index[a]] for a, _ in batch])
                positives = torch.stack([base_tensor[text_index[p]] for _, p in batch])

                a_emb = self._layer(anchors)
                p_emb = self._layer(positives)

                loss = _infonce_loss(a_emb, p_emb, temperature=0.07)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            best_loss = min(best_loss, avg)
            logger.info(
                "adapter-train epoch {}/{}: loss={:.4f}",
                epoch + 1, settings.adapter_train_epochs, avg,
            )

        self._layer.eval()
        self._meta["step"] = (self._meta.get("step") or 0) + len(pairs) * settings.adapter_train_epochs
        self._meta["loss"] = best_loss
        self._meta["chunks_seen"] = (self._meta.get("chunks_seen") or 0) + len(all_texts)
        self._meta["trained_at"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def adapter_info(self) -> dict:
        """Return current adapter metadata (path, training state)."""
        return {
            "path": str(self._adapter_path),
            "exists": self._adapter_path.exists(),
            **self._meta,
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _apply_adapter(self, base_vecs: np.ndarray) -> np.ndarray:
        if base_vecs.shape[0] == 0:
            return base_vecs
        tensor = torch.from_numpy(base_vecs).to(self._device)
        with torch.no_grad():
            out = self._layer(tensor)
        return out.cpu().numpy()

    def _save(self) -> None:
        self._adapter_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weights": self._layer.state_dict(), **self._meta}, self._adapter_path)
        logger.info(
            "adapter saved → {} (step={}, loss={:.4f})",
            self._adapter_path,
            self._meta["step"],
            self._meta["loss"] or 0.0,
        )

    def _load(self) -> None:
        ckpt = torch.load(self._adapter_path, map_location=self._device, weights_only=False)
        self._layer.load_state_dict(ckpt["weights"])
        self._meta = {k: ckpt.get(k) for k in ("step", "loss", "chunks_seen", "trained_at")}
        logger.info(
            "adapter loaded from {} (step={}, trained_at={})",
            self._adapter_path,
            self._meta.get("step"),
            self._meta.get("trained_at"),
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sample_pairs(db: Any, pairs_per_thread: int) -> list[tuple[str, str]]:
    """Sample positive pairs: two distinct chunks from the same parent_post_id."""
    rows = db.execute(
        "SELECT parent_post_id, text_de FROM chunks WHERE is_filtered=0 AND text_de IS NOT NULL"
    ).fetchall()

    by_thread: dict[str, list[str]] = {}
    for row in rows:
        pid = row["parent_post_id"]
        by_thread.setdefault(pid, []).append(row["text_de"])

    pairs: list[tuple[str, str]] = []
    for texts in by_thread.values():
        if len(texts) < 2:
            continue
        available = list(texts)
        random.shuffle(available)
        for j in range(0, min(len(available) - 1, pairs_per_thread * 2), 2):
            if available[j] != available[j + 1]:
                pairs.append((available[j], available[j + 1]))

    random.shuffle(pairs)
    return pairs


def _infonce_loss(
    anchors: "torch.Tensor",
    positives: "torch.Tensor",
    temperature: float = 0.07,
) -> "torch.Tensor":
    """InfoNCE with in-batch negatives. Inputs are expected to be L2-normalized."""
    sim = torch.matmul(anchors, positives.T) / temperature  # (B, B)
    labels = torch.arange(sim.shape[0], device=sim.device)
    return F.cross_entropy(sim, labels)
