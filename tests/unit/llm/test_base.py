"""Unit tests for llm/base.py."""

import pytest
from hyporeddit.llm.base import LLMClient, StanceResult, ChunkWithContext


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

def test_stance_result_fields() -> None:
    sr = StanceResult(chunk_id="c1", stance="supports", rationale="relevant evidence")
    assert sr.chunk_id == "c1"
    assert sr.stance == "supports"
    assert sr.rationale == "relevant evidence"


def test_chunk_with_context_fields() -> None:
    cwc = ChunkWithContext(
        chunk_id="c1",
        text_de="Wir haben 14 Monate gewartet",
        text_en="We waited 14 months",
        parent_post_title="Bauantrag",
        parent_post_body="Long story.",
        retrieval_score=0.85,
    )
    assert cwc.chunk_id == "c1"
    assert cwc.retrieval_score == 0.85


def test_chunk_with_context_text_en_optional() -> None:
    cwc = ChunkWithContext(
        chunk_id="c1",
        text_de="Test",
        text_en=None,
        parent_post_title="T",
        parent_post_body="B",
        retrieval_score=0.5,
    )
    assert cwc.text_en is None


# ---------------------------------------------------------------------------
# LLMClient is abstract — cannot instantiate directly
# ---------------------------------------------------------------------------

def test_llm_client_is_abstract() -> None:
    with pytest.raises(TypeError):
        LLMClient()  # type: ignore[abstract]


def test_llm_client_subclass_without_methods_is_abstract() -> None:
    class IncompleteClient(LLMClient):
        pass

    with pytest.raises(TypeError):
        IncompleteClient()


def test_llm_client_concrete_subclass_can_be_instantiated() -> None:
    class ConcreteClient(LLMClient):
        def classify_stances(self, hypothesis, chunks, prompt_version="v1"):
            return []

        def synthesize(self, hypothesis, evidence, stats, prompt_version="v1"):
            return ""

        def translate(self, text_de):
            return text_de

        def translate_batch(self, texts_de):
            return [self.translate(t) for t in texts_de]

    client = ConcreteClient()
    assert isinstance(client, LLMClient)
