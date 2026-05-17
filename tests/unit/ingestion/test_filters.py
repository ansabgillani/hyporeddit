"""Unit tests for ingestion/filters.py.

No env vars, no network. Pure function tests — these run in milliseconds.
"""

import pytest

from hyporeddit.ingestion.filters import apply_filters, is_agreement_token, is_hard_filtered
from hyporeddit.sources.base import RawComment


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_comment(
    body: str = "Wir haben 14 Monate gewartet.",
    author: str = "user1",
    id_: str = "c1",
    post_id: str = "p1",
) -> RawComment:
    return RawComment(
        id=id_,
        post_id=post_id,
        parent_id=post_id,
        author=author,
        body=body,
        score=5,
        depth=0,
        created_utc=1_700_000_000,
        is_submitter=False,
        edited=False,
        awards_count=0,
    )


# ---------------------------------------------------------------------------
# Hard filter tests
# ---------------------------------------------------------------------------

def test_deleted_body_is_filtered() -> None:
    assert is_hard_filtered(make_comment(body="[deleted]")) is True


def test_removed_body_is_filtered() -> None:
    assert is_hard_filtered(make_comment(body="[removed]")) is True


def test_empty_body_is_filtered() -> None:
    assert is_hard_filtered(make_comment(body="")) is True


def test_whitespace_only_body_is_filtered() -> None:
    assert is_hard_filtered(make_comment(body="   \n\t  ")) is True


def test_automoderator_is_filtered() -> None:
    assert is_hard_filtered(make_comment(author="AutoModerator")) is True


def test_none_author_is_filtered() -> None:
    assert is_hard_filtered(make_comment(author=None)) is True  # type: ignore[arg-type]


def test_substantive_comment_not_filtered() -> None:
    assert is_hard_filtered(make_comment(body="Wir haben 14 Monate gewartet.")) is False


def test_short_substantive_german_comment_not_filtered() -> None:
    # "Wir haben 14 Monate auf den Bauantrag gewartet." — perfect evidence, 7 words
    assert is_hard_filtered(make_comment(body="Wir haben 14 Monate gewartet.")) is False


# ---------------------------------------------------------------------------
# Agreement token tests
# ---------------------------------------------------------------------------

def test_danke_is_agreement_token() -> None:
    assert is_agreement_token(make_comment(body="Danke")) is True


def test_danke_case_insensitive() -> None:
    assert is_agreement_token(make_comment(body="DANKE")) is True
    assert is_agreement_token(make_comment(body="danke")) is True


def test_thumbs_up_is_agreement_token() -> None:
    assert is_agreement_token(make_comment(body="👍")) is True


def test_plus_one_is_agreement_token() -> None:
    assert is_agreement_token(make_comment(body="+1")) is True


def test_stimmt_is_agreement_token() -> None:
    assert is_agreement_token(make_comment(body="Stimmt")) is True


def test_agreement_token_not_substring_match() -> None:
    # "Ja, aber wir haben..." starts with "Ja" but is substantive — must NOT be filtered
    assert is_agreement_token(make_comment(body="Ja, aber wir haben 14 Monate gewartet.")) is False


def test_agreement_token_with_trailing_whitespace() -> None:
    # Stripped body must match the token exactly
    assert is_agreement_token(make_comment(body="Danke  ")) is True


def test_substantive_comment_not_agreement_token() -> None:
    assert is_agreement_token(make_comment(body="Das Problem ist komplex.")) is False


def test_genau_is_agreement_token() -> None:
    assert is_agreement_token(make_comment(body="Genau")) is True


def test_jup_is_agreement_token() -> None:
    assert is_agreement_token(make_comment(body="Jup")) is True


# ---------------------------------------------------------------------------
# apply_filters integration
# ---------------------------------------------------------------------------

def test_apply_filters_separates_kept_and_filtered() -> None:
    comments = [
        make_comment(id_="c1", body="[deleted]"),
        make_comment(id_="c2", body="Danke"),
        make_comment(id_="c3", body="Wir haben 14 Monate gewartet."),
        make_comment(id_="c4", body="Das hat uns 80k mehr gekostet als geplant."),
    ]
    kept, filtered = apply_filters(comments)
    kept_ids = {c.id for c in kept}
    filtered_ids = {c.id for c in filtered}
    assert kept_ids == {"c3", "c4"}
    assert filtered_ids == {"c1", "c2"}


def test_apply_filters_all_kept() -> None:
    comments = [
        make_comment(id_="c1", body="Wir haben lange gewartet."),
        make_comment(id_="c2", body="Der Bauantrag war eine Katastrophe."),
    ]
    kept, filtered = apply_filters(comments)
    assert len(kept) == 2
    assert len(filtered) == 0


def test_apply_filters_all_filtered() -> None:
    comments = [
        make_comment(id_="c1", body="[deleted]"),
        make_comment(id_="c2", body="+1"),
    ]
    kept, filtered = apply_filters(comments)
    assert len(kept) == 0
    assert len(filtered) == 2


def test_apply_filters_empty_input() -> None:
    kept, filtered = apply_filters([])
    assert kept == []
    assert filtered == []
