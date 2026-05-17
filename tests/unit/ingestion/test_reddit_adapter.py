"""Unit tests for sources/reddit_json.py — JSON parsing and tree flattening.

No network calls. httpx is mocked at the usage site.
"""

from unittest.mock import MagicMock, patch

import pytest

from hyporeddit.sources.reddit_json import RedditJsonAdapter
from hyporeddit.sources.base import RawPost, RawComment


# ---------------------------------------------------------------------------
# Factories — mimic Reddit API response shapes
# ---------------------------------------------------------------------------

def make_reddit_post_child(
    id_: str = "t3_abc123",
    title: str = "Test Post",
    score: int = 10,
    author: str = "user1",
    created_utc: float = 1_700_000_000.0,
    selftext: str = "Body text",
    num_comments: int = 5,
    upvote_ratio: float = 0.95,
    is_self: bool = True,
) -> dict:
    name = id_ if id_.startswith("t3_") else f"t3_{id_}"
    bare_id = name.removeprefix("t3_")
    return {
        "kind": "t3",
        "data": {
            "id": bare_id,
            "name": name,
            "title": title,
            "score": score,
            "author": author,
            "created_utc": created_utc,
            "selftext": selftext,
            "num_comments": num_comments,
            "upvote_ratio": upvote_ratio,
            "is_self": is_self,
            "url": f"https://reddit.com/r/hausbau/comments/{bare_id}",
            "link_flair_text": None,
            "edited": False,
            "total_awards_received": 0,
        },
    }


def make_listing_response(children: list[dict], after: str | None = None) -> dict:
    return {
        "kind": "Listing",
        "data": {
            "after": after,
            "children": children,
        },
    }


def make_comment_child(
    id_: str = "c1",
    body: str = "Test comment",
    author: str = "commenter1",
    score: int = 5,
    created_utc: float = 1_700_001_000.0,
    parent_id: str = "t3_abc123",
    depth: int = 0,
    replies: dict | None = None,
) -> dict:
    return {
        "kind": "t1",
        "data": {
            "id": id_,
            "body": body,
            "author": author,
            "score": score,
            "created_utc": created_utc,
            "parent_id": parent_id,
            "depth": depth,
            "is_submitter": False,
            "edited": False,
            "total_awards_received": 0,
            "replies": replies or "",
        },
    }


def make_comment_tree_response(
    post_id: str = "abc123",
    comments: list[dict] | None = None,
) -> list[dict]:
    """Mimics Reddit's 2-element array response for comments endpoint."""
    post_listing = make_listing_response(
        [make_reddit_post_child(id_=post_id)]
    )
    comment_listing = {
        "kind": "Listing",
        "data": {
            "after": None,
            "children": comments or [],
        },
    }
    return [post_listing, comment_listing]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def adapter() -> RedditJsonAdapter:
    a = RedditJsonAdapter()
    a._client._request_delay = 0.0  # no sleep in unit tests
    return a


# ---------------------------------------------------------------------------
# fetch_posts tests
# ---------------------------------------------------------------------------

def test_fetch_posts_returns_raw_posts(adapter: RedditJsonAdapter) -> None:
    listing = make_listing_response(
        [make_reddit_post_child(id_="abc123", title="My Post")]
    )
    with patch.object(adapter._client, "get", return_value=listing):
        posts = adapter.fetch_posts(limit=1)

    assert len(posts) == 1
    assert isinstance(posts[0], RawPost)
    assert posts[0].id == "abc123"
    assert posts[0].title == "My Post"


def test_fetch_posts_maps_all_required_fields(adapter: RedditJsonAdapter) -> None:
    child = make_reddit_post_child(
        id_="p1",
        title="Title",
        score=42,
        author="redditor",
        created_utc=1_700_000_000.0,
        selftext="Some text",
        upvote_ratio=0.87,
        num_comments=3,
        is_self=True,
    )
    with patch.object(adapter._client, "get", return_value=make_listing_response([child])):
        posts = adapter.fetch_posts(limit=1)

    p = posts[0]
    assert p.score == 42
    assert p.author == "redditor"
    assert p.created_utc == 1_700_000_000
    assert p.selftext == "Some text"
    assert p.upvote_ratio == 0.87
    assert p.num_comments == 3
    assert p.is_self is True


def test_fetch_posts_returns_empty_for_empty_listing(adapter: RedditJsonAdapter) -> None:
    with patch.object(adapter._client, "get", return_value=make_listing_response([])):
        posts = adapter.fetch_posts(limit=100)
    assert posts == []


def test_fetch_posts_passes_after_cursor(adapter: RedditJsonAdapter) -> None:
    with patch.object(adapter._client, "get", return_value=make_listing_response([])) as mock_get:
        adapter.fetch_posts(limit=100, after="t3_cursor123")
    called_url = mock_get.call_args[0][0]
    assert "after=t3_cursor123" in called_url


def test_fetch_posts_returns_pagination_cursor(adapter: RedditJsonAdapter) -> None:
    listing = make_listing_response(
        [make_reddit_post_child()],
        after="t3_nextcursor",
    )
    with patch.object(adapter._client, "get", return_value=listing):
        _, cursor = adapter.fetch_posts_with_cursor(limit=100)
    assert cursor == "t3_nextcursor"


# ---------------------------------------------------------------------------
# fetch_comments tests
# ---------------------------------------------------------------------------

def test_fetch_comments_returns_flat_list(adapter: RedditJsonAdapter) -> None:
    tree = make_comment_tree_response(
        post_id="abc123",
        comments=[
            make_comment_child(id_="c1"),
            make_comment_child(id_="c2"),
        ],
    )
    with patch.object(adapter._client, "get", return_value=tree):
        comments = adapter.fetch_comments("abc123")
    assert len(comments) == 2
    assert all(isinstance(c, RawComment) for c in comments)


def test_fetch_comments_flattens_nested_replies(adapter: RedditJsonAdapter) -> None:
    nested = make_comment_child(
        id_="c2",
        body="Nested reply",
        depth=1,
        parent_id="t1_c1",
    )
    root = make_comment_child(
        id_="c1",
        depth=0,
        replies={
            "kind": "Listing",
            "data": {"children": [nested]},
        },
    )
    tree = make_comment_tree_response(comments=[root])
    with patch.object(adapter._client, "get", return_value=tree):
        comments = adapter.fetch_comments("abc123")
    ids = [c.id for c in comments]
    assert "c1" in ids
    assert "c2" in ids


def test_fetch_comments_skips_more_items(adapter: RedditJsonAdapter) -> None:
    more_item = {"kind": "more", "data": {"children": ["c99"], "count": 1}}
    tree = make_comment_tree_response(
        comments=[make_comment_child(id_="c1"), more_item]
    )
    with patch.object(adapter._client, "get", return_value=tree):
        comments = adapter.fetch_comments("abc123")
    # "more" items are skipped in basic handling (up to 3 expansions are done separately)
    assert all(c.id != "c99" for c in comments)


def test_fetch_comments_maps_depth_correctly(adapter: RedditJsonAdapter) -> None:
    nested = make_comment_child(id_="c2", depth=1, parent_id="t1_c1",
                                replies={"kind": "Listing", "data": {"children": [
                                    make_comment_child(id_="c3", depth=2, parent_id="t1_c2")
                                ]}})
    root = make_comment_child(id_="c1", depth=0, replies={
        "kind": "Listing",
        "data": {"children": [nested]}
    })
    tree = make_comment_tree_response(comments=[root])
    with patch.object(adapter._client, "get", return_value=tree):
        comments = adapter.fetch_comments("abc123")
    depth_map = {c.id: c.depth for c in comments}
    assert depth_map["c1"] == 0
    assert depth_map["c2"] == 1
    assert depth_map["c3"] == 2


def test_source_name(adapter: RedditJsonAdapter) -> None:
    from hyporeddit.config import settings
    assert adapter.source_name() == f"reddit:r/{settings.reddit_subreddit}"
