"""Noise filters for raw Reddit comments.

Two filter categories:
  hard filters  — structurally non-informative (deleted, bots)
  agreement tokens — pure acknowledgment with no informational content

Short substantive German comments are intentionally kept — the stance classifier
handles genuine irrelevance.
"""

from loguru import logger

from hyporeddit.config import settings
from hyporeddit.sources.base import RawComment


def is_hard_filtered(comment: RawComment) -> bool:
    """Return True if the comment should always be dropped."""
    if comment.author is None:
        return True
    if comment.author in settings.known_bots:
        return True
    body = (comment.body or "").strip()
    if not body:
        return True
    if body in ("[deleted]", "[removed]"):
        return True
    return False


def is_agreement_token(comment: RawComment) -> bool:
    """Return True if the entire body is a pure agreement/acknowledgment token.

    Match is full-body only (stripped, case-insensitive) — not a substring.
    "Ja, aber..." is NOT matched even though it starts with "Ja".
    """
    body = (comment.body or "").strip().lower()
    return body in {t.lower() for t in settings.agreement_tokens}


def apply_filters(comments: list[RawComment]) -> tuple[list[RawComment], list[RawComment]]:
    """Partition comments into (kept, filtered) lists.

    Filters are applied in order: hard filters first, then agreement tokens.
    Returns both lists so callers can log or inspect what was dropped.
    """
    kept: list[RawComment] = []
    filtered: list[RawComment] = []

    hard_count = 0
    agreement_count = 0

    for comment in comments:
        if is_hard_filtered(comment):
            filtered.append(comment)
            hard_count += 1
        elif is_agreement_token(comment):
            filtered.append(comment)
            agreement_count += 1
        else:
            kept.append(comment)

    logger.info(
        "Filter stats: {} kept | {} hard-filtered | {} agreement-token filtered",
        len(kept), hard_count, agreement_count,
    )
    return kept, filtered


if __name__ == "__main__":
    # Run: python -m hyporeddit.ingestion.filters
    # Env: none required
    samples = [
        RawComment("c1", "p1", "p1", "user", "[deleted]", 0, 0, 0, False, False, 0),
        RawComment("c2", "p1", "p1", "user", "Danke", 0, 0, 0, False, False, 0),
        RawComment("c3", "p1", "p1", "user", "Wir haben 14 Monate gewartet.", 5, 0, 0, False, False, 0),
        RawComment("c4", "p1", "p1", "user", "Ja, aber wir haben lange gewartet.", 3, 0, 0, False, False, 0),
    ]
    kept, filtered_out = apply_filters(samples)
    logger.info("Kept: {}", [c.body for c in kept])
    logger.info("Filtered: {}", [c.body for c in filtered_out])
