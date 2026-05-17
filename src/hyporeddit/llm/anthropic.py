"""Anthropic-backed LLM client (Claude Haiku for classification, Sonnet for synthesis)."""

import json
from pathlib import Path
from typing import Any

import anthropic
from loguru import logger

from hyporeddit.config import settings
from hyporeddit.llm.base import ChunkWithContext, LLMClient, StanceResult
from hyporeddit.translation.translator import translate_batch_de_to_en, translate_de_to_en

_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    if not path.exists():
        # Fallback: look relative to cwd (for Docker / test environments)
        path = Path("prompts") / name
    return path.read_text(encoding="utf-8")


def _parse_stance_json(raw: str) -> list[dict[str, str]] | None:
    """Extract and parse the JSON array from an LLM response. Returns None on failure."""
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return None


class AnthropicLLMClient(LLMClient):
    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)

    def classify_stances(
        self,
        hypothesis: str,
        chunks: list[ChunkWithContext],
        prompt_version: str = "v1",
    ) -> list[StanceResult]:
        if not chunks:
            return []

        prompt_template = _load_prompt(f"stance_classification_{prompt_version}.txt")
        batch_size = settings.classification_batch_size
        results: list[StanceResult] = []

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_results = self._classify_batch(hypothesis, batch, prompt_template)
            results.extend(batch_results)

        return results

    def _classify_batch(
        self,
        hypothesis: str,
        batch: list[ChunkWithContext],
        prompt_template: str,
    ) -> list[StanceResult]:
        formatted = "\n\n".join(
            f'[{c.chunk_id}] {c.text_en or c.text_de}' for c in batch
        )
        parent_title = batch[0].parent_post_title if batch else ""
        prompt = prompt_template.format(
            hypothesis=hypothesis,
            parent_post_title=parent_title,
            formatted_chunks=formatted,
            subreddit=settings.reddit_subreddit,
        )

        response = self._client.messages.create(
            model=settings.llm_classification_model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        parsed = _parse_stance_json(raw)

        if parsed is None:
            logger.warning("Malformed JSON from classifier — retrying with repair instruction")
            repair_prompt = (
                prompt
                + "\n\nYour previous response was not valid JSON. "
                "Respond with only a JSON array, no other text."
            )
            response = self._client.messages.create(
                model=settings.llm_classification_model,
                max_tokens=1000,
                messages=[{"role": "user", "content": repair_prompt}],
            )
            raw = response.content[0].text
            parsed = _parse_stance_json(raw)

        if parsed is None:
            logger.error("Classifier returned unparseable JSON after retry — skipping batch")
            return []

        return [
            StanceResult(
                chunk_id=item["chunk_id"],
                stance=item.get("stance", "irrelevant"),
                rationale=item.get("rationale", ""),
            )
            for item in parsed
            if "chunk_id" in item
        ]

    def synthesize(
        self,
        hypothesis: str,
        evidence: list[Any],
        stats: dict[str, Any],
        prompt_version: str = "v1",
    ) -> str:
        prompt_template = _load_prompt(f"synthesis_{prompt_version}.txt")

        supporting = [e for e in evidence if getattr(e, "stance", None) == "supports"]
        contradicting = [e for e in evidence if getattr(e, "stance", None) == "contradicts"]
        top_supporting = supporting[:3]
        top_contradicting = contradicting[:3]

        def fmt_evidence(items: list[Any]) -> str:
            if not items:
                return "(none)"
            return "\n".join(
                f'- [{getattr(e, "stance", "?").upper()}] '
                f'{getattr(e, "text_en", None) or getattr(e, "text_de", "")}'
                for e in items
            )

        prompt = prompt_template.format(
            hypothesis=hypothesis,
            score=stats.get("score", 0.5),
            confidence=stats.get("confidence", 0.0),
            sample_size=stats.get("sample_size", 0),
            supports=stats.get("supports", 0),
            contradicts=stats.get("contradicts", 0),
            neutral=stats.get("neutral", 0),
            irrelevant=stats.get("irrelevant", 0),
            supporting_evidence=fmt_evidence(top_supporting),
            contradicting_evidence=fmt_evidence(top_contradicting),
        )

        response = self._client.messages.create(
            model=settings.llm_synthesis_model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def translate(self, text_de: str) -> str:
        return translate_de_to_en(text_de)

    def translate_batch(self, texts_de: list[str]) -> list[str]:
        return translate_batch_de_to_en(texts_de)
