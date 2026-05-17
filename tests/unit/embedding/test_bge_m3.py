"""Unit tests for the BGE-M3 encoder.

FlagEmbedding is mocked so no model download is required for these tests.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mock_model(dim: int = 1024):
    """Return a mock BGEM3FlagModel that produces unit vectors."""
    mock = MagicMock()

    def fake_encode(sentences, batch_size=32, **kwargs):
        n = len(sentences)
        vecs = np.ones((n, dim), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return {"dense_vecs": vecs}

    mock.encode.side_effect = fake_encode

    def fake_encode_queries(sentences, batch_size=32, **kwargs):
        n = len(sentences)
        vecs = np.ones((n, dim), dtype=np.float32) * 0.5
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return {"dense_vecs": vecs}

    mock.encode_queries.side_effect = fake_encode_queries
    return mock


@pytest.fixture
def mock_flag_model(monkeypatch):
    model = _make_mock_model()
    with patch("hyporeddit.embedding.bge_m3.BGEM3FlagModel", return_value=model) as mock_cls:
        yield mock_cls, model


class TestBGEM3EncoderEncode:
    def test_encode_returns_ndarray_of_correct_shape(self, mock_flag_model):
        from hyporeddit.embedding.bge_m3 import BGE_M3_Encoder

        _, model = mock_flag_model
        encoder = BGE_M3_Encoder()
        result = encoder.encode(["text one", "text two", "text three"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1024)

    def test_encode_single_text(self, mock_flag_model):
        from hyporeddit.embedding.bge_m3 import BGE_M3_Encoder

        encoder = BGE_M3_Encoder()
        result = encoder.encode(["only one"])
        assert result.shape == (1, 1024)

    def test_encode_empty_list_returns_empty_array(self, mock_flag_model):
        from hyporeddit.embedding.bge_m3 import BGE_M3_Encoder

        encoder = BGE_M3_Encoder()
        result = encoder.encode([])
        assert result.shape == (0, 1024)

    def test_encode_processes_in_batches(self, mock_flag_model):
        from hyporeddit.embedding.bge_m3 import BGE_M3_Encoder

        _, model = mock_flag_model
        encoder = BGE_M3_Encoder(batch_size=2)
        texts = ["a", "b", "c", "d", "e"]
        result = encoder.encode(texts)

        # 5 texts, batch_size=2 → 3 calls: [a,b], [c,d], [e]
        assert model.encode.call_count == 3
        assert result.shape == (5, 1024)

    def test_encode_returns_float32(self, mock_flag_model):
        from hyporeddit.embedding.bge_m3 import BGE_M3_Encoder

        encoder = BGE_M3_Encoder()
        result = encoder.encode(["hello"])
        assert result.dtype == np.float32


class TestBGEM3EncoderQuery:
    def test_encode_query_returns_1d_vector(self, mock_flag_model):
        from hyporeddit.embedding.bge_m3 import BGE_M3_Encoder

        encoder = BGE_M3_Encoder()
        result = encoder.encode_query("what is the planning speed?")
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)

    def test_encode_query_uses_encode_queries_method(self, mock_flag_model):
        from hyporeddit.embedding.bge_m3 import BGE_M3_Encoder

        _, model = mock_flag_model
        encoder = BGE_M3_Encoder()
        encoder.encode_query("test query")
        model.encode_queries.assert_called_once()

    def test_encode_query_does_not_call_encode(self, mock_flag_model):
        from hyporeddit.embedding.bge_m3 import BGE_M3_Encoder

        _, model = mock_flag_model
        encoder = BGE_M3_Encoder()
        encoder.encode_query("a query")
        model.encode.assert_not_called()

    def test_encode_query_returns_float32(self, mock_flag_model):
        from hyporeddit.embedding.bge_m3 import BGE_M3_Encoder

        encoder = BGE_M3_Encoder()
        result = encoder.encode_query("question")
        assert result.dtype == np.float32


class TestBGEM3EncoderOOM:
    def test_oom_auto_reduces_batch_size_and_retries(self, mock_flag_model):
        from hyporeddit.embedding.bge_m3 import BGE_M3_Encoder

        _, model = mock_flag_model
        call_count = 0

        def oom_first_call(sentences, batch_size=32, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("CUDA out of memory")
            n = len(sentences)
            vecs = np.ones((n, 1024), dtype=np.float32)
            return {"dense_vecs": vecs}

        model.encode.side_effect = oom_first_call
        encoder = BGE_M3_Encoder(batch_size=4)
        result = encoder.encode(["a", "b", "c", "d"])
        # Should succeed after auto-reducing batch size
        assert result.shape[1] == 1024

    def test_oom_non_memory_error_propagates(self, mock_flag_model):
        from hyporeddit.embedding.bge_m3 import BGE_M3_Encoder

        _, model = mock_flag_model
        model.encode.side_effect = RuntimeError("some other error")

        encoder = BGE_M3_Encoder()
        with pytest.raises(RuntimeError, match="some other error"):
            encoder.encode(["text"])
