"""Unit tests for ingestion/chunker.py.

Pure function tests — no env vars, no network, no mocks.
"""

import pytest

from hyporeddit.ingestion.chunker import chunk_comment, chunk_post
from hyporeddit.models.ingestion import Chunk
from hyporeddit.sources.base import RawComment, RawPost


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_raw_post(
    id_: str = "p1",
    title: str = "Bauantrag Erfahrungen",
    body: str = "Wir haben unsere Baugenehmigung endlich bekommen.",
    is_self: bool = True,
) -> RawPost:
    return RawPost(
        id=id_,
        title=title,
        author="user1",
        selftext=body,
        score=10,
        upvote_ratio=0.95,
        num_comments=5,
        created_utc=1_700_000_000,
        url=f"https://reddit.com/r/hausbau/comments/{id_}",
        is_self=is_self,
        flair=None,
        edited=False,
        awards_count=0,
    )


def make_raw_comment(
    id_: str = "c1",
    post_id: str = "p1",
    body: str = "Wir haben 14 Monate auf den Bauantrag gewartet.",
    score: int = 5,
    depth: int = 0,
    author: str = "commenter1",
) -> RawComment:
    return RawComment(
        id=id_,
        post_id=post_id,
        parent_id=post_id,
        author=author,
        body=body,
        score=score,
        depth=depth,
        created_utc=1_700_001_000,
        is_submitter=False,
        edited=False,
        awards_count=0,
    )


def make_long_text(words: int) -> str:
    """Generate a long German-like text with the given word count."""
    base = "Das ist ein langer Text der viele Informationen enthält und immer weiter geht "
    words_list = base.split() * (words // len(base.split()) + 1)
    return " ".join(words_list[:words]) + "."


# ---------------------------------------------------------------------------
# Post chunking tests
# ---------------------------------------------------------------------------

def test_chunk_post_returns_list_of_chunks() -> None:
    post = make_raw_post()
    chunks = chunk_post(post)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_post_text_is_title_plus_body() -> None:
    post = make_raw_post(title="Mein Titel", body="Mein Text")
    chunks = chunk_post(post)
    assert "Mein Titel" in chunks[0].text_de
    assert "Mein Text" in chunks[0].text_de


def test_chunk_post_with_no_body_uses_title_only() -> None:
    post = make_raw_post(body="")
    chunks = chunk_post(post)
    assert len(chunks) >= 1
    assert chunks[0].text_de.strip() != ""


def test_chunk_post_single_chunk_for_short_post() -> None:
    post = make_raw_post(body="Kurzer Text.")
    chunks = chunk_post(post)
    assert len(chunks) == 1


def test_chunk_post_sub_chunks_for_long_post() -> None:
    long_body = make_long_text(450)  # exceeds 400-word threshold
    post = make_raw_post(body=long_body)
    chunks = chunk_post(post)
    assert len(chunks) > 1


def test_chunk_post_stores_source_type_as_post() -> None:
    chunks = chunk_post(make_raw_post())
    assert all(c.source_type == "post" for c in chunks)


def test_chunk_post_stores_source_id() -> None:
    chunks = chunk_post(make_raw_post(id_="mypostid"))
    assert all(c.source_id == "mypostid" for c in chunks)


def test_chunk_post_stable_chunk_id() -> None:
    """Same post produces the same chunk_id."""
    post = make_raw_post()
    chunks1 = chunk_post(post)
    chunks2 = chunk_post(post)
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


def test_chunk_post_distinct_chunk_ids() -> None:
    """Different posts produce different chunk_ids."""
    chunks1 = chunk_post(make_raw_post(id_="p1"))
    chunks2 = chunk_post(make_raw_post(id_="p2"))
    ids1 = {c.chunk_id for c in chunks1}
    ids2 = {c.chunk_id for c in chunks2}
    assert ids1.isdisjoint(ids2)


def test_chunk_post_sub_chunks_have_overlap() -> None:
    """Sub-chunks from a long post should have overlapping text at boundaries."""
    long_body = make_long_text(500)
    post = make_raw_post(body=long_body)
    chunks = chunk_post(post)
    assert len(chunks) >= 2
    # The end of chunk[0] should overlap with the start of chunk[1]
    words0 = chunks[0].text_de.split()
    words1 = chunks[1].text_de.split()
    # Last N words of chunk0 should appear as first words of chunk1 (within margin)
    overlap_region = words0[-60:]  # 50-word overlap, check last 60 words
    first_region = words1[:60]
    common = set(overlap_region) & set(first_region)
    assert len(common) > 0


# ---------------------------------------------------------------------------
# Comment chunking tests
# ---------------------------------------------------------------------------

def test_chunk_comment_returns_list_of_chunks() -> None:
    post = make_raw_post()
    comment = make_raw_comment()
    chunks = chunk_comment(comment, post)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_comment_text_is_comment_body_only() -> None:
    post = make_raw_post(title="Post Title")
    comment = make_raw_comment(body="Kommentar Text")
    chunks = chunk_comment(comment, post)
    assert chunks[0].text_de == "Kommentar Text"


def test_chunk_comment_stores_parent_post_metadata() -> None:
    post = make_raw_post(id_="p1", title="Post Titel", body="Post Body")
    comment = make_raw_comment(post_id="p1")
    chunks = chunk_comment(comment, post)
    chunk = chunks[0]
    assert chunk.parent_post_id == "p1"
    assert chunk.parent_post_title == "Post Titel"
    assert chunk.parent_post_body == "Post Body"


def test_chunk_comment_source_type_is_comment() -> None:
    chunks = chunk_comment(make_raw_comment(), make_raw_post())
    assert all(c.source_type == "comment" for c in chunks)


def test_chunk_comment_sub_chunks_for_long_comment() -> None:
    long_body = make_long_text(450)
    comment = make_raw_comment(body=long_body)
    chunks = chunk_comment(comment, make_raw_post())
    assert len(chunks) > 1


def test_chunk_comment_single_chunk_for_short_comment() -> None:
    comment = make_raw_comment(body="Kurzer Kommentar.")
    chunks = chunk_comment(comment, make_raw_post())
    assert len(chunks) == 1


def test_chunk_comment_metadata_has_score() -> None:
    comment = make_raw_comment(score=42)
    chunks = chunk_comment(comment, make_raw_post())
    assert chunks[0].metadata.score == 42


def test_chunk_comment_metadata_has_depth() -> None:
    comment = make_raw_comment(depth=2)
    chunks = chunk_comment(comment, make_raw_post())
    assert chunks[0].metadata.depth == 2


def test_chunk_comment_char_offset_increments_for_sub_chunks() -> None:
    long_body = make_long_text(500)
    comment = make_raw_comment(body=long_body)
    chunks = chunk_comment(comment, make_raw_post())
    if len(chunks) > 1:
        offsets = [c.char_offset for c in chunks]
        assert offsets[0] < offsets[1]
