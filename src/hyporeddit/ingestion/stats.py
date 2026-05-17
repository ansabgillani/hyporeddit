"""Corpus statistics — print a summary of what's stored in SQLite."""

from datetime import datetime, timezone

from loguru import logger
from rich.console import Console
from rich.table import Table

from hyporeddit.storage.sqlite import get_db, get_stats

console = Console()


def print_stats() -> None:
    db = get_db()
    s = get_stats(db)

    table = Table(title="hyporeddit Corpus Statistics", show_header=False, min_width=50)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Posts", str(s["post_count"]))
    table.add_row("Comments", str(s["comment_count"]))
    table.add_row("Chunks (embedded)", str(s["chunk_count"]))

    if s["min_post_utc"] and s["max_post_utc"]:
        min_dt = datetime.fromtimestamp(s["min_post_utc"], tz=timezone.utc).strftime("%Y-%m-%d")
        max_dt = datetime.fromtimestamp(s["max_post_utc"], tz=timezone.utc).strftime("%Y-%m-%d")
        table.add_row("Post date range", f"{min_dt} → {max_dt}")
    else:
        table.add_row("Post date range", "—")

    if s["last_job"]:
        job = s["last_job"]
        table.add_row("Last ingestion mode", job.get("mode", "—"))
        table.add_row("Last ingestion status", job.get("status", "—"))
        table.add_row("Last ingestion at", job.get("updated_at", "—")[:19])
    else:
        table.add_row("Last ingestion", "never")

    console.print(table)


if __name__ == "__main__":
    # Run: python -m hyporeddit.ingestion.stats
    # Env: SQLITE_PATH (optional)
    print_stats()
