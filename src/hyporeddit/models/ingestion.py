"""Pydantic data models for ingested content."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ChunkMetadata:
    score: int = 0
    upvote_ratio: float = 0.0
    created_utc: int = 0
    depth: int = 0
    author_karma: int = 0
    num_comments: int = 0
    awards_count: int = 0


@dataclass
class Chunk:
    chunk_id: str
    source_type: str          # "post" | "comment"
    source_id: str            # reddit post or comment ID
    text_de: str              # German original
    parent_post_id: str
    parent_post_title: str
    parent_post_body: str
    char_offset: int          # byte offset into source text (for sub-chunking)
    metadata: ChunkMetadata
    text_en: str | None = None  # populated after translation
