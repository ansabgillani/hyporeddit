"""Reddit unauthenticated JSON adapter implementing SourceAdapter."""

from loguru import logger

from hyporeddit.config import settings
from hyporeddit.ingestion.http_client import PolitHttpClient
from hyporeddit.sources.base import RawComment, RawPost, SourceAdapter


def _strip_prefix(value: str) -> str:
    """Remove Reddit's t1_/t3_ type prefixes from IDs."""
    for prefix in ("t1_", "t2_", "t3_", "t4_", "t5_", "t6_"):
        if value.startswith(prefix):
            return value[3:]
    return value


def _parse_post(child: dict, source_name: str) -> RawPost:
    d = child["data"]
    return RawPost(
        id=d["id"],
        title=d["title"],
        author=d.get("author"),
        selftext=d.get("selftext") or "",
        score=int(d.get("score", 0)),
        upvote_ratio=float(d.get("upvote_ratio", 0.0)),
        num_comments=int(d.get("num_comments", 0)),
        created_utc=int(d.get("created_utc", 0)),
        url=d.get("url", ""),
        is_self=bool(d.get("is_self", False)),
        flair=d.get("link_flair_text"),
        edited=bool(d.get("edited", False)),
        awards_count=int(d.get("total_awards_received", 0)),
        source_name=source_name,
    )


def _flatten_comments(
    children: list[dict],
    post_id: str,
    depth: int = 0,
) -> list[RawComment]:
    """Recursively flatten Reddit's nested comment tree into a flat list."""
    result: list[RawComment] = []
    for child in children:
        if child.get("kind") == "more":
            continue
        if child.get("kind") != "t1":
            continue
        d = child["data"]
        comment = RawComment(
            id=d["id"],
            post_id=post_id,
            parent_id=_strip_prefix(d.get("parent_id", "")),
            author=d.get("author"),
            body=d.get("body", ""),
            score=int(d.get("score", 0)),
            depth=depth,
            created_utc=int(d.get("created_utc", 0)),
            is_submitter=bool(d.get("is_submitter", False)),
            edited=bool(d.get("edited", False)),
            awards_count=int(d.get("total_awards_received", 0)),
        )
        result.append(comment)
        replies = d.get("replies")
        if isinstance(replies, dict):
            nested = replies.get("data", {}).get("children", [])
            result.extend(_flatten_comments(nested, post_id, depth + 1))
    return result


class RedditJsonAdapter(SourceAdapter):
    """Fetches posts and comments from the configured subreddit via unauthenticated Reddit JSON endpoints."""

    def __init__(self) -> None:
        self._client = PolitHttpClient()
        self._subreddit = settings.reddit_subreddit
        self._base = f"https://www.reddit.com/r/{self._subreddit}"

    def source_name(self) -> str:
        return f"reddit:r/{self._subreddit}"

    def fetch_posts(self, limit: int = 100, after: str | None = None) -> list[RawPost]:
        posts, _ = self.fetch_posts_with_cursor(limit=limit, after=after)
        return posts

    def fetch_posts_with_cursor(
        self, limit: int = 100, after: str | None = None
    ) -> tuple[list[RawPost], str | None]:
        url = f"{self._base}/new.json?limit={min(limit, 100)}"
        if after:
            url += f"&after={after}"
        data = self._client.get(url)
        children = data["data"]["children"]
        posts = [_parse_post(c, self.source_name()) for c in children if c.get("kind") == "t3"]
        cursor: str | None = data["data"].get("after")
        logger.debug("Fetched {} posts, next cursor: {}", len(posts), cursor)
        return posts, cursor

    def fetch_comments(self, post_id: str) -> list[RawComment]:
        url = f"{self._base}/comments/{post_id}.json?limit=500&depth=10"
        data = self._client.get(url)
        # Reddit returns a 2-element list: [post_listing, comment_listing]
        comment_listing = data[1] if isinstance(data, list) else data
        children = comment_listing["data"]["children"]
        comments = _flatten_comments(children, post_id, depth=0)
        logger.debug("Fetched {} comments for post {}", len(comments), post_id)
        return comments

    def close(self) -> None:
        self._client.close()


if __name__ == "__main__":
    # Run: python -m hyporeddit.sources.reddit_json
    # Env: none required
    adapter = RedditJsonAdapter()
    adapter._client._request_delay = 0.0
    posts, cursor = adapter.fetch_posts_with_cursor(limit=5)
    logger.info("Fetched {} posts, cursor={}", len(posts), cursor)
    for p in posts:
        logger.info("  [{:>5}↑] {}", p.score, p.title)
    adapter.close()
