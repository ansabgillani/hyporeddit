"""LLM abstraction layer — interface and shared dataclasses."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class StanceResult:
    chunk_id: str
    stance: str        # 'supports' | 'contradicts' | 'neutral' | 'irrelevant'
    rationale: str


@dataclass
class ChunkWithContext:
    chunk_id: str
    text_de: str
    text_en: str | None
    parent_post_title: str
    parent_post_body: str
    retrieval_score: float


class LLMClient(ABC):
    @abstractmethod
    def classify_stances(
        self,
        hypothesis: str,
        chunks: list[ChunkWithContext],
        prompt_version: str = "v1",
    ) -> list[StanceResult]:
        """Classify a list of chunks against a hypothesis.

        Handles prompt loading, batching, and response parsing internally.
        """

    @abstractmethod
    def synthesize(
        self,
        hypothesis: str,
        evidence: list[Any],
        stats: dict[str, Any],
        prompt_version: str = "v1",
    ) -> str:
        """Generate a prose synthesis summarising the evidence for a hypothesis."""

    @abstractmethod
    def translate(self, text_de: str) -> str:
        """Translate a German text to English."""

    @abstractmethod
    def translate_batch(self, texts_de: list[str]) -> list[str]:
        """Translate a list of German texts to English in batched API calls."""
