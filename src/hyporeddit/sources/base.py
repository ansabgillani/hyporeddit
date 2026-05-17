"""SourceAdapter ABC and raw data models for ingestion sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RawPost:
    id: str
    title: str
    author: str | None
    selftext: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: int
    url: str
    is_self: bool
    flair: str | None
    edited: bool
    awards_count: int
    source_name: str = ""


@dataclass
class RawComment:
    id: str
    post_id: str
    parent_id: str          # stripped of t1_/t3_ prefix
    author: str | None
    body: str
    score: int
    depth: int
    created_utc: int
    is_submitter: bool
    edited: bool
    awards_count: int


class SourceAdapter(ABC):
    """Interface for any content source. All ingestion logic is written against this."""

    @abstractmethod
    def fetch_posts(self, limit: int, after: str | None = None) -> list[RawPost]:
        """Fetch a page of posts. `after` is the pagination cursor."""

    @abstractmethod
    def fetch_posts_with_cursor(
        self, limit: int, after: str | None = None
    ) -> tuple[list[RawPost], str | None]:
        """Fetch a page of posts and return (posts, next_cursor)."""

    @abstractmethod
    def fetch_comments(self, post_id: str) -> list[RawComment]:
        """Fetch all comments for a post, returning a flat list with depth."""

    @abstractmethod
    def source_name(self) -> str:
        """Unique identifier for this source (e.g. 'reddit:r/hausbau')."""
