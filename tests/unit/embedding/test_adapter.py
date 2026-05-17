"""Unit tests for embedding/adapter.py.

Requires torch (installed via FlagEmbedding). BGE_M3_Encoder is mocked so
no model download is required.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk_row(parent_post_id: str, text_de: str) -> dict:
    return {"parent_post_id": parent_post_id, "text_de": text_de}


def _make_db(rows: list[dict]) -> MagicMock:
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = rows
    return db


# ---------------------------------------------------------------------------
# _infonce_loss tests
# ---------------------------------------------------------------------------

class TestInfoNCELoss:
    def test_returns_scalar_tensor(self) -> None:
        from hyporeddit.embedding.adapter import _infonce_loss

        import torch.nn.functional as F
        a = F.normalize(torch.randn(3, 16), dim=-1)
        p = F.normalize(torch.randn(3, 16), dim=-1)

        loss = _infonce_loss(a, p)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0

    def test_is_non_negative(self) -> None:
        from hyporeddit.embedding.adapter import _infonce_loss

        import torch.nn.functional as F
        a = F.normalize(torch.randn(4, 8), dim=-1)
        p = F.normalize(torch.randn(4, 8), dim=-1)

        loss = _infonce_loss(a, p)

        assert loss.item() >= 0.0

    def test_perfect_alignment_gives_lower_loss_than_random(self) -> None:
        """When positives == anchors the loss should be lower than with random positives."""
        from hyporeddit.embedding.adapter import _infonce_loss

        import torch.nn.functional as F
        torch.manual_seed(0)
        a = F.normalize(torch.randn(4, 16), dim=-1)

        perfect_loss = _infonce_loss(a, a.clone())
        random_loss = _infonce_loss(a, F.normalize(torch.randn(4, 16), dim=-1))

        assert perfect_loss.item() <= random_loss.item()

    def test_temperature_affects_loss(self) -> None:
        """Different temperatures must produce different loss values."""
        from hyporeddit.embedding.adapter import _infonce_loss

        import torch.nn.functional as F
        torch.manual_seed(1)
        a = F.normalize(torch.randn(4, 8), dim=-1)
        p = F.normalize(torch.randn(4, 8), dim=-1)

        loss_high = _infonce_loss(a, p, temperature=1.0)
        loss_low = _infonce_loss(a, p, temperature=0.07)

        assert loss_high.item() != loss_low.item()

    def test_gradient_flows_through_loss(self) -> None:
        """Loss must support .backward() for training."""
        from hyporeddit.embedding.adapter import _infonce_loss

        import torch.nn.functional as F
        raw_a = torch.randn(3, 8, requires_grad=True)
        raw_p = torch.randn(3, 8, requires_grad=True)
        a = F.normalize(raw_a, dim=-1)
        p = F.normalize(raw_p, dim=-1)

        loss = _infonce_loss(a, p)
        loss.backward()

        # Gradients should flow back to the leaf tensors
        assert raw_a.grad is not None


# ---------------------------------------------------------------------------
# _sample_pairs tests
# ---------------------------------------------------------------------------

class TestSamplePairs:
    def test_empty_db_returns_no_pairs(self) -> None:
        from hyporeddit.embedding.adapter import _sample_pairs

        pairs = _sample_pairs(_make_db([]), pairs_per_thread=5)
        assert pairs == []

    def test_single_chunk_per_thread_returns_no_pairs(self) -> None:
        from hyporeddit.embedding.adapter import _sample_pairs

        db = _make_db([_make_chunk_row("p1", "Only chunk")])
        pairs = _sample_pairs(db, pairs_per_thread=5)

        assert pairs == []

    def test_two_chunks_same_thread_returns_one_pair(self) -> None:
        from hyporeddit.embedding.adapter import _sample_pairs

        db = _make_db([
            _make_chunk_row("p1", "Text A"),
            _make_chunk_row("p1", "Text B"),
        ])
        pairs = _sample_pairs(db, pairs_per_thread=5)

        assert len(pairs) == 1
        a, p = pairs[0]
        assert isinstance(a, str)
        assert isinstance(p, str)
        assert a != p

    def test_pairs_are_within_same_thread(self) -> None:
        """Chunks from different parent_post_ids must never be paired across threads."""
        from hyporeddit.embedding.adapter import _sample_pairs

        p1_texts = {"P1-A", "P1-B"}
        p2_texts = {"P2-A", "P2-B"}
        db = _make_db([
            _make_chunk_row("p1", "P1-A"),
            _make_chunk_row("p1", "P1-B"),
            _make_chunk_row("p2", "P2-A"),
            _make_chunk_row("p2", "P2-B"),
        ])

        pairs = _sample_pairs(db, pairs_per_thread=5)

        for a, pos in pairs:
            same_thread = (a in p1_texts and pos in p1_texts) or \
                          (a in p2_texts and pos in p2_texts)
            assert same_thread, f"Cross-thread pair detected: ({a!r}, {pos!r})"

    def test_returns_list_of_string_tuples(self) -> None:
        from hyporeddit.embedding.adapter import _sample_pairs

        rows = [_make_chunk_row("p1", f"Text {i}") for i in range(4)]
        db = _make_db(rows)

        pairs = _sample_pairs(db, pairs_per_thread=5)

        for a, pos in pairs:
            assert isinstance(a, str)
            assert isinstance(pos, str)

    def test_pairs_per_thread_limits_output(self) -> None:
        """With pairs_per_thread=1, at most 1 pair per parent_post_id."""
        from hyporeddit.embedding.adapter import _sample_pairs

        rows = [_make_chunk_row("p1", f"Text {i}") for i in range(6)]
        db = _make_db(rows)

        pairs = _sample_pairs(db, pairs_per_thread=1)

        assert len(pairs) <= 1


# ---------------------------------------------------------------------------
# _AdapterLayer tests
# ---------------------------------------------------------------------------

class TestAdapterLayer:
    def test_identity_at_init(self) -> None:
        """With identity weight init, output = normalize(input)."""
        from hyporeddit.embedding.adapter import _AdapterLayer

        import torch.nn.functional as F
        dim = 8
        layer = _AdapterLayer(dim=dim)
        layer.eval()

        v = torch.eye(dim)[:1]  # unit basis vector, shape (1, dim)
        with torch.no_grad():
            out = layer(v)

        expected = F.normalize(v, dim=-1)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_output_is_normalized(self) -> None:
        """forward() must always return unit-norm vectors."""
        from hyporeddit.embedding.adapter import _AdapterLayer

        layer = _AdapterLayer(dim=32)
        layer.eval()

        x = torch.randn(6, 32)
        with torch.no_grad():
            out = layer(x)

        norms = torch.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones(6), atol=1e-5)

    def test_output_shape_matches_input(self) -> None:
        from hyporeddit.embedding.adapter import _AdapterLayer

        dim = 64
        layer = _AdapterLayer(dim=dim)
        layer.eval()

        x = torch.randn(5, dim)
        with torch.no_grad():
            out = layer(x)

        assert out.shape == (5, dim)

    def test_weight_is_square_matrix(self) -> None:
        from hyporeddit.embedding.adapter import _AdapterLayer

        dim = 16
        layer = _AdapterLayer(dim=dim)

        assert layer.proj.weight.shape == (dim, dim)

    def test_no_bias(self) -> None:
        from hyporeddit.embedding.adapter import _AdapterLayer

        layer = _AdapterLayer(dim=8)
        assert layer.proj.bias is None


# ---------------------------------------------------------------------------
# AdaptiveEncoder tests
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_bge():
    """Patch BGE_M3_Encoder so no real model is loaded."""
    with patch("hyporeddit.embedding.adapter.BGE_M3_Encoder") as cls:
        mock = MagicMock()
        mock._device = "cpu"
        mock.encode.side_effect = lambda texts, **kw: np.ones((len(texts), 1024), dtype=np.float32)
        mock.encode_query.return_value = np.ones(1024, dtype=np.float32)
        cls.return_value = mock
        yield mock


class TestAdaptiveEncoderApplyAdapter:
    def test_returns_numpy_array(self, mock_bge) -> None:
        from hyporeddit.embedding.adapter import AdaptiveEncoder

        enc = AdaptiveEncoder(adapter_path="/nonexistent/adapter.pt")
        vecs = np.random.randn(3, 1024).astype(np.float32)

        result = enc._apply_adapter(vecs)

        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_input(self, mock_bge) -> None:
        from hyporeddit.embedding.adapter import AdaptiveEncoder

        enc = AdaptiveEncoder(adapter_path="/nonexistent/adapter.pt")
        vecs = np.random.randn(4, 1024).astype(np.float32)

        result = enc._apply_adapter(vecs)

        assert result.shape == (4, 1024)

    def test_empty_input_returns_empty(self, mock_bge) -> None:
        from hyporeddit.embedding.adapter import AdaptiveEncoder

        enc = AdaptiveEncoder(adapter_path="/nonexistent/adapter.pt")
        empty = np.zeros((0, 1024), dtype=np.float32)

        result = enc._apply_adapter(empty)

        assert result.shape == (0, 1024)

    def test_output_is_normalized(self, mock_bge) -> None:
        """Adapter applies F.normalize — output vectors should be unit-norm."""
        from hyporeddit.embedding.adapter import AdaptiveEncoder

        enc = AdaptiveEncoder(adapter_path="/nonexistent/adapter.pt")
        vecs = np.random.randn(4, 1024).astype(np.float32)

        result = enc._apply_adapter(vecs)

        norms = np.linalg.norm(result, axis=-1)
        np.testing.assert_allclose(norms, np.ones(4), atol=1e-5)


class TestAdaptiveEncoderInfo:
    def test_adapter_info_has_expected_keys(self, mock_bge) -> None:
        from hyporeddit.embedding.adapter import AdaptiveEncoder

        enc = AdaptiveEncoder(adapter_path="/nonexistent/adapter.pt")
        info = enc.adapter_info()

        assert "path" in info
        assert "exists" in info
        assert "step" in info
        assert "loss" in info
        assert "trained_at" in info

    def test_adapter_info_exists_false_when_no_checkpoint(self, mock_bge) -> None:
        from hyporeddit.embedding.adapter import AdaptiveEncoder

        enc = AdaptiveEncoder(adapter_path="/definitely/does/not/exist/adapter.pt")

        assert enc.adapter_info()["exists"] is False

    def test_adapter_info_path_matches_constructor_arg(self, mock_bge) -> None:
        from hyporeddit.embedding.adapter import AdaptiveEncoder

        enc = AdaptiveEncoder(adapter_path="/some/custom/path.pt")

        # Path converts separators on Windows; check the filename is preserved
        assert "path.pt" in enc.adapter_info()["path"]

    def test_adapter_info_initial_step_is_zero_or_none(self, mock_bge) -> None:
        from hyporeddit.embedding.adapter import AdaptiveEncoder

        enc = AdaptiveEncoder(adapter_path="/nonexistent/adapter.pt")

        assert enc.adapter_info()["step"] in (0, None)


class TestAdaptiveEncoderEncodeQuery:
    def test_returns_numpy_array(self, mock_bge) -> None:
        from hyporeddit.embedding.adapter import AdaptiveEncoder

        enc = AdaptiveEncoder(adapter_path="/nonexistent/adapter.pt")
        result = enc.encode_query("Was ist der Bauantrag?")

        assert isinstance(result, np.ndarray)

    def test_output_shape_is_1d_1024(self, mock_bge) -> None:
        from hyporeddit.embedding.adapter import AdaptiveEncoder

        enc = AdaptiveEncoder(adapter_path="/nonexistent/adapter.pt")
        result = enc.encode_query("test query")

        assert result.shape == (1024,)

    def test_calls_base_encode_query(self, mock_bge) -> None:
        from hyporeddit.embedding.adapter import AdaptiveEncoder

        enc = AdaptiveEncoder(adapter_path="/nonexistent/adapter.pt")
        enc.encode_query("Wie lange dauert ein Bauantrag?")

        mock_bge.encode_query.assert_called_once_with("Wie lange dauert ein Bauantrag?")


class TestAdaptiveEncoderEncode:
    def test_returns_numpy_array(self, mock_bge) -> None:
        from hyporeddit.embedding.adapter import AdaptiveEncoder

        enc = AdaptiveEncoder(adapter_path="/nonexistent/adapter.pt")
        result = enc.encode(["Text A", "Text B"])

        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_input_count(self, mock_bge) -> None:
        from hyporeddit.embedding.adapter import AdaptiveEncoder

        enc = AdaptiveEncoder(adapter_path="/nonexistent/adapter.pt")
        result = enc.encode(["A", "B", "C"])

        assert result.shape == (3, 1024)

    def test_calls_base_encode(self, mock_bge) -> None:
        from hyporeddit.embedding.adapter import AdaptiveEncoder

        enc = AdaptiveEncoder(adapter_path="/nonexistent/adapter.pt")
        enc.encode(["Text"])

        mock_bge.encode.assert_called_once()
