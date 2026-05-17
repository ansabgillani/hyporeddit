"""Unit tests for ingestion/stats.py.

print_stats() calls get_db() and get_stats() and renders a Rich table.
Both are mocked — no database file or env vars required.
"""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_stats(
    post_count: int = 10,
    comment_count: int = 50,
    chunk_count: int = 40,
    min_post_utc: int | None = 1_700_000_000,
    max_post_utc: int | None = 1_700_100_000,
    last_job: dict | None = None,
) -> dict:
    return {
        "post_count": post_count,
        "comment_count": comment_count,
        "chunk_count": chunk_count,
        "min_post_utc": min_post_utc,
        "max_post_utc": max_post_utc,
        "last_job": last_job,
    }


# ---------------------------------------------------------------------------
# print_stats tests
# ---------------------------------------------------------------------------

class TestPrintStats:
    def test_runs_without_error(self) -> None:
        from hyporeddit.ingestion.stats import print_stats

        mock_db = MagicMock()
        with patch("hyporeddit.ingestion.stats.get_db", return_value=mock_db), \
             patch("hyporeddit.ingestion.stats.get_stats", return_value=_make_stats()):
            print_stats()

    def test_calls_get_db(self) -> None:
        from hyporeddit.ingestion.stats import print_stats

        mock_db = MagicMock()
        with patch("hyporeddit.ingestion.stats.get_db", return_value=mock_db) as mock_get_db, \
             patch("hyporeddit.ingestion.stats.get_stats", return_value=_make_stats()):
            print_stats()

        mock_get_db.assert_called_once()

    def test_calls_get_stats_with_db(self) -> None:
        from hyporeddit.ingestion.stats import print_stats

        mock_db = MagicMock()
        with patch("hyporeddit.ingestion.stats.get_db", return_value=mock_db), \
             patch("hyporeddit.ingestion.stats.get_stats", return_value=_make_stats()) as mock_get_stats:
            print_stats()

        mock_get_stats.assert_called_once_with(mock_db)

    def test_handles_none_date_range(self) -> None:
        """When no posts exist min/max_post_utc are None — must not raise."""
        from hyporeddit.ingestion.stats import print_stats

        mock_db = MagicMock()
        stats = _make_stats(post_count=0, comment_count=0, chunk_count=0,
                            min_post_utc=None, max_post_utc=None)
        with patch("hyporeddit.ingestion.stats.get_db", return_value=mock_db), \
             patch("hyporeddit.ingestion.stats.get_stats", return_value=stats):
            print_stats()

    def test_handles_last_job_present(self) -> None:
        """When a last ingestion job exists, its fields are rendered without error."""
        from hyporeddit.ingestion.stats import print_stats

        mock_db = MagicMock()
        stats = _make_stats(last_job={
            "mode": "backfill",
            "status": "complete",
            "updated_at": "2026-05-17T10:00:00+00:00",
        })
        with patch("hyporeddit.ingestion.stats.get_db", return_value=mock_db), \
             patch("hyporeddit.ingestion.stats.get_stats", return_value=stats):
            print_stats()

    def test_handles_last_job_none(self) -> None:
        """When no ingestion job has run, last_job=None must not raise."""
        from hyporeddit.ingestion.stats import print_stats

        mock_db = MagicMock()
        stats = _make_stats(last_job=None)
        with patch("hyporeddit.ingestion.stats.get_db", return_value=mock_db), \
             patch("hyporeddit.ingestion.stats.get_stats", return_value=stats):
            print_stats()

    def test_handles_large_corpus(self) -> None:
        from hyporeddit.ingestion.stats import print_stats

        mock_db = MagicMock()
        stats = _make_stats(post_count=50_000, comment_count=2_000_000, chunk_count=1_800_000)
        with patch("hyporeddit.ingestion.stats.get_db", return_value=mock_db), \
             patch("hyporeddit.ingestion.stats.get_stats", return_value=stats):
            print_stats()
