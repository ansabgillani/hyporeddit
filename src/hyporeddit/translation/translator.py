"""German → English translation — routes to Anthropic or OpenAI based on LLM_PROVIDER."""

import json
import re
import time
from pathlib import Path
from typing import Any

import anthropic
import openai
from loguru import logger

from hyporeddit.config import settings

_PROMPT_PATH = Path("prompts/translation_v1.txt")
_MAX_RETRIES = 3

_BATCH_SYSTEM_PROMPT = (
    "You are a German-to-English translator. "
    "You will receive a numbered list of German texts. "
    "Output ONLY a raw JSON array of translated strings in the same order. "
    "No markdown, no code fences, no explanation — just the JSON array itself. "
    "Preserve technical construction terms in parentheses after translation. "
    'Example: if input is "1. Hallo\n2. Welt", output exactly: ["Hello", "World"]'
)


def _load_system_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text().strip()
    return (
        "Translate the following German text to English. "
        "Preserve technical construction terms in parentheses after translation. "
        "Return only the translated text."
    )


def translate_de_to_en(text: str, _client: Any = None) -> str:
    if settings.llm_provider == "openai":
        return _translate_openai(text, _client)
    return _translate_anthropic(text, _client)


def _translate_anthropic(text: str, _client: Any = None) -> str:
    client = _client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
    system_prompt = _load_system_prompt()
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            message = client.messages.create(
                model=settings.llm_classification_model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": text}],
            )
            return message.content[0].text
        except anthropic.APIError as exc:
            last_exc = exc
            delay = 2 ** attempt
            logger.warning(
                "Translation API error (attempt {}/{}): {} — retrying in {}s",
                attempt + 1, _MAX_RETRIES, exc, delay,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)

    raise last_exc  # type: ignore[misc]


def _translate_openai(text: str, _client: Any = None) -> str:
    client = _client or openai.OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.llm_base_url,
    )
    system_prompt = _load_system_prompt()
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=settings.openai_classification_model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            return response.choices[0].message.content or ""
        except openai.APIError as exc:
            last_exc = exc
            delay = 2 ** attempt
            logger.warning(
                "Translation API error (attempt {}/{}): {} — retrying in {}s",
                attempt + 1, _MAX_RETRIES, exc, delay,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)

    raise last_exc  # type: ignore[misc]


def _parse_batch_response(raw: str, expected_count: int, input_text: str = "") -> list[str] | None:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    parsed = None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                pass

    if parsed is None:
        logger.warning(
            "Batch translation returned unparseable JSON — falling back\nINPUT:\n{}\nOUTPUT:\n{}",
            input_text, raw,
        )
        return None

    if isinstance(parsed, list) and len(parsed) == expected_count:
        return [str(t) for t in parsed]

    logger.warning(
        "Batch translation returned {} items for {} inputs — falling back\nINPUT:\n{}\nOUTPUT:\n{}",
        len(parsed) if isinstance(parsed, list) else "non-list",
        expected_count,
        input_text,
        raw,
    )
    return None


def _call_batch_anthropic(texts: list[str], client: Any) -> list[str] | None:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            message = client.messages.create(
                model=settings.llm_classification_model,
                max_tokens=4096,
                system=_BATCH_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": numbered},
                    {"role": "assistant", "content": "["},
                ],
            )
            result = _parse_batch_response("[" + message.content[0].text, len(texts), numbered)
            if result is not None:
                return result
            return None
        except anthropic.APIError as exc:
            last_exc = exc
            delay = 2 ** attempt
            logger.warning(
                "Batch translation API error (attempt {}/{}): {} — retrying in {}s",
                attempt + 1, _MAX_RETRIES, exc, delay,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)

    logger.warning("Batch translation failed after retries: {}", last_exc)
    return None


def _call_batch_openai(texts: list[str], client: Any) -> list[str] | None:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=settings.openai_classification_model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": _BATCH_SYSTEM_PROMPT},
                    {"role": "user", "content": numbered},
                ],
            )
            raw = response.choices[0].message.content or ""
            result = _parse_batch_response(raw, len(texts), numbered)
            if result is not None:
                return result
            return None
        except openai.APIError as exc:
            last_exc = exc
            delay = 2 ** attempt
            logger.warning(
                "Batch translation API error (attempt {}/{}): {} — retrying in {}s",
                attempt + 1, _MAX_RETRIES, exc, delay,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)

    logger.warning("Batch translation failed after retries: {}", last_exc)
    return None


def translate_batch_de_to_en(texts: list[str], _client: Any = None) -> list[str]:
    if not texts:
        return []

    if settings.llm_provider == "openai":
        client = _client or openai.OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.llm_base_url,
        )
        call_batch = _call_batch_openai
    else:
        client = _client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
        call_batch = _call_batch_anthropic

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

    batch_size = min(settings.translation_batch_size, 5)
    total = len(texts)
    batches = range(0, total, batch_size)
    num_batches = len(batches)
    results: list[str] = []

    with Progress(
        TextColumn("[cyan]Translating"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Translating", total=total)

        for batch_idx, start in enumerate(batches, 1):
            sub_batch = texts[start : start + batch_size]
            logger.info(
                "Translating batch {}/{} ({} items, {}/{} total) …",
                batch_idx, num_batches, len(sub_batch), start + len(sub_batch), total,
            )
            translated = call_batch(sub_batch, client)
            if translated is not None:
                results.extend(translated)
                progress.advance(task, len(sub_batch))
                logger.info(
                    "Batch {}/{} done — {}/{} texts translated",
                    batch_idx, num_batches, len(results), total,
                )
            else:
                logger.warning(
                    "Batch {}/{} — falling back to per-item translation for {} texts",
                    batch_idx, num_batches, len(sub_batch),
                )
                for item_idx, text in enumerate(sub_batch, 1):
                    logger.info(
                        "  Per-item fallback {}/{} in batch {}/{}",
                        item_idx, len(sub_batch), batch_idx, num_batches,
                    )
                    results.append(translate_de_to_en(text, _client=client))
                    progress.advance(task, 1)

    return results
