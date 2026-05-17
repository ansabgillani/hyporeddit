"""Post and comment → Chunk conversion with sub-chunking.

Sub-chunking strategy:
  - Split on sentence boundaries when text exceeds `chunk_max_words`
  - Use overlapping windows of `chunk_window_words` words with `chunk_overlap_words` overlap
  - chunk_id is a stable hash of (source_id, char_offset)
"""

import hashlib
import re

from hyporeddit.config import settings
from hyporeddit.models.ingestion import Chunk, ChunkMetadata
from hyporeddit.sources.base import RawComment, RawPost

# Simple sentence boundary pattern — split after ., !, ?, …, or German equivalents
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _make_chunk_id(source_id: str, char_offset: int) -> str:
    key = f"{source_id}:{char_offset}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _split_into_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries, preserving content."""
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def _sub_chunk(text: str, source_id: str, base_offset: int) -> list[tuple[str, int]]:
    """Split text into overlapping word windows. Returns (chunk_text, char_offset) pairs."""
    max_w = settings.chunk_max_words
    window = settings.chunk_window_words
    overlap = settings.chunk_overlap_words

    words = text.split()
    if len(words) <= max_w:
        return [(text, base_offset)]

    # Try sentence-boundary splitting first; fall back to raw word windows.
    sentences = _split_into_sentences(text)
    if len(sentences) > 1:
        chunks: list[tuple[str, int]] = []
        current_words: list[str] = []
        current_offset = base_offset

        for sentence in sentences:
            s_words = sentence.split()
            if len(current_words) + len(s_words) > window and current_words:
                chunk_text = " ".join(current_words)
                chunks.append((chunk_text, current_offset))
                carry = current_words[-overlap:] if overlap > 0 else []
                current_offset += len(" ".join(current_words[: len(current_words) - len(carry)])) + 1
                current_words = carry + s_words
            else:
                current_words.extend(s_words)

        if current_words:
            chunks.append((" ".join(current_words), current_offset))

        if len(chunks) > 1:
            return chunks

    # Fallback: sliding word window (no sentence boundaries available)
    result: list[tuple[str, int]] = []
    step = window - overlap
    for start in range(0, len(words), step):
        slice_words = words[start : start + window]
        chunk_text = " ".join(slice_words)
        char_offset = base_offset + len(" ".join(words[:start]))
        if start > 0:
            char_offset += 1  # space before this window
        result.append((chunk_text, char_offset))
        if start + window >= len(words):
            break
    return result if result else [(text, base_offset)]


def chunk_post(post: RawPost) -> list[Chunk]:
    """Convert a RawPost to one or more Chunk objects."""
    body = post.selftext or ""
    if body:
        text = f"{post.title}\n\n{body}"
    else:
        text = post.title

    metadata = ChunkMetadata(
        score=post.score,
        upvote_ratio=post.upvote_ratio,
        created_utc=post.created_utc,
        depth=0,
        author_karma=0,
        num_comments=post.num_comments,
        awards_count=post.awards_count,
    )

    pairs = _sub_chunk(text, post.id, 0)
    return [
        Chunk(
            chunk_id=_make_chunk_id(post.id, offset),
            source_type="post",
            source_id=post.id,
            text_de=chunk_text,
            parent_post_id=post.id,
            parent_post_title=post.title,
            parent_post_body=post.selftext or "",
            char_offset=offset,
            metadata=metadata,
        )
        for chunk_text, offset in pairs
    ]


def chunk_comment(comment: RawComment, post: RawPost) -> list[Chunk]:
    """Convert a RawComment to one or more Chunk objects."""
    text = comment.body or ""

    metadata = ChunkMetadata(
        score=comment.score,
        upvote_ratio=post.upvote_ratio,
        created_utc=comment.created_utc,
        depth=comment.depth,
        author_karma=0,
        num_comments=post.num_comments,
        awards_count=comment.awards_count,
    )

    pairs = _sub_chunk(text, comment.id, 0)
    return [
        Chunk(
            chunk_id=_make_chunk_id(comment.id, offset),
            source_type="comment",
            source_id=comment.id,
            text_de=chunk_text,
            parent_post_id=post.id,
            parent_post_title=post.title,
            parent_post_body=post.selftext or "",
            char_offset=offset,
            metadata=metadata,
        )
        for chunk_text, offset in pairs
    ]


if __name__ == "__main__":
    # Run: python -m hyporeddit.ingestion.chunker
    # Env: none required
    from loguru import logger

    test_post = RawPost(
        id="test1", title="Bauantrag Wartezeit",
        author="user", selftext="Kurzer Text.", score=5, upvote_ratio=0.9,
        num_comments=3, created_utc=1_700_000_000,
        url="https://reddit.com", is_self=True, flair=None, edited=False, awards_count=0,
    )
    chunks = chunk_post(test_post)
    logger.info("Post → {} chunk(s)", len(chunks))
    for c in chunks:
        logger.info("  chunk_id={} offset={} words={}", c.chunk_id, c.char_offset, len(c.text_de.split()))
