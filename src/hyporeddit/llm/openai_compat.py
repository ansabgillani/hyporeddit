"""OpenAI-compatible LLM client — works with LM Studio / DeepSeek."""

import json
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI

from hyporeddit.config import settings
from hyporeddit.llm.base import ChunkWithContext, LLMClient, StanceResult

_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    if not path.exists():
        path = Path("prompts") / name
    return path.read_text(encoding="utf-8")


def _parse_stance_json(raw: str) -> list[dict[str, str]] | None:
    raw = raw.strip()
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


class OpenAICompatLLMClient(LLMClient):
    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.openai_api_key or "lm-studio",
            base_url=settings.llm_base_url,
        )

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
            results.extend(self._classify_batch(hypothesis, batch, prompt_template))
        return results

    def _classify_batch(
        self,
        hypothesis: str,
        batch: list[ChunkWithContext],
        prompt_template: str,
    ) -> list[StanceResult]:
        formatted = "\n\n".join(f"[{c.chunk_id}] {c.text_en or c.text_de}" for c in batch)
        parent_title = batch[0].parent_post_title if batch else ""
        prompt = prompt_template.format(
            hypothesis=hypothesis,
            parent_post_title=parent_title,
            formatted_chunks=formatted,
            subreddit=settings.reddit_subreddit,
        )
        response = self._client.chat.completions.create(
            model=settings.openai_classification_model,
            max_tokens=10000,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_stance_json(raw)
        if parsed is None:
            logger.warning("Malformed JSON from classifier — retrying with repair instruction")
            repair_prompt = (
                prompt + "\n\nYour previous response was not valid JSON. "
                "Respond with only a JSON array, no other text."
            )
            response = self._client.chat.completions.create(
                model=settings.openai_classification_model,
                max_tokens=10000,
                messages=[{"role": "user", "content": repair_prompt}],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
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

        def fmt_evidence(items: list[Any]) -> str:
            if not items:
                return "(none)"
            lines = []
            for e in items:
                stance = getattr(e, "stance", "?").upper()
                chunk = getattr(e, "chunk", e)
                text = getattr(chunk, "text_en", None) or getattr(chunk, "text_de", "")
                lines.append(f"- [{stance}] {text}")
            return "\n".join(lines)

        prompt = prompt_template.format(
            hypothesis=hypothesis,
            score=stats.get("score", 0.5),
            confidence=stats.get("confidence", 0.0),
            sample_size=stats.get("sample_size", 0),
            supports=stats.get("supports", 0),
            contradicts=stats.get("contradicts", 0),
            neutral=stats.get("neutral", 0),
            irrelevant=stats.get("irrelevant", 0),
            supporting_evidence=fmt_evidence(supporting[:3]),
            contradicting_evidence=fmt_evidence(contradicting[:3]),
        )

        response = self._client.chat.completions.create(
            model=settings.openai_synthesis_model,
            max_tokens=5000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def translate(self, text_de: str) -> str:
        system = (
            "Translate the following German text to English. "
            "Preserve technical construction terms in parentheses after translation. "
            "Return only the translated text."
        )
        response = self._client.chat.completions.create(
            model=settings.openai_classification_model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text_de},
            ],
        )
        return response.choices[0].message.content or ""

    def translate_batch(self, texts_de: list[str]) -> list[str]:
        if not texts_de:
            return []

        batch_size = settings.translation_batch_size
        results: list[str] = []

        for start in range(0, len(texts_de), batch_size):
            sub_batch = texts_de[start : start + batch_size]
            numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(sub_batch))
            system = (
                "You are a German-to-English translator. "
                "You will receive a numbered list of German texts. "
                "Return a JSON array of translated strings in the same order, with no extra text. "
                "Preserve technical construction terms in parentheses after translation."
            )
            response = self._client.chat.completions.create(
                model=settings.openai_classification_model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": numbered},
                ],
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                lines = raw.splitlines()
                raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list) and len(parsed) == len(sub_batch):
                    results.extend(str(t) for t in parsed)
                    continue
            except (json.JSONDecodeError, ValueError):
                pass

            logger.warning(
                "Batch translation failed for sub-batch of {} — falling back to per-item",
                len(sub_batch),
            )
            for text in sub_batch:
                results.append(self.translate(text))

        return results
