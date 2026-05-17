"""CLI entry point — Typer command declarations only.

Each command makes exactly one call to the appropriate pipeline/service function.
No business logic lives here.
"""

import typer
from loguru import logger

app = typer.Typer(
    name="hyporeddit",
    help="Hypothesis validation against a Reddit corpus.",
    no_args_is_help=True,
)


@app.command()
def backfill(limit: int = typer.Option(1000, help="Max posts to fetch")) -> None:
    """Fetch historical posts from the configured subreddit (one-time backfill)."""
    from hyporeddit.ingestion.scheduler import run_backfill

    run_backfill(limit=limit)


@app.command()
def ingest() -> None:
    """Fetch new posts since the last run (daily delta)."""
    from hyporeddit.ingestion.scheduler import run_delta

    run_delta()


@app.command()
def stats() -> None:
    """Print corpus statistics."""
    from hyporeddit.ingestion.stats import print_stats

    print_stats()


@app.command()
def evaluate(
    hypothesis: str = typer.Argument(..., help="Hypothesis text to evaluate"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    force_rerun: bool = typer.Option(False, "--force-rerun", help="Bypass evaluation cache"),
    top_k: int = typer.Option(100, help="Number of candidate chunks to retrieve"),
) -> None:
    """Evaluate a hypothesis against the corpus."""
    from hyporeddit.evaluation.pipeline import evaluate_hypothesis

    result = evaluate_hypothesis(hypothesis, top_k=top_k, force_rerun=force_rerun)
    if json_output:
        import json

        typer.echo(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        from hyporeddit.evaluation.display import display_result

        display_result(result)


@app.command(name="show-evaluation")
def show_evaluation(run_id: str = typer.Argument(..., help="Evaluation run ID")) -> None:
    """Re-display a past evaluation run."""
    from hyporeddit.evaluation.display import display_result
    from hyporeddit.storage.sqlite import get_evaluation_run

    result = get_evaluation_run(run_id)
    if result is None:
        logger.error("Run {} not found", run_id)
        raise typer.Exit(1)
    display_result(result)


@app.command()
def history(
    hypothesis_id: str = typer.Argument(..., help="Hypothesis ID"),
) -> None:
    """Show score trend for a hypothesis over time."""
    from hyporeddit.evaluation.display import display_history
    from hyporeddit.storage.sqlite import get_evaluation_history

    runs = get_evaluation_history(hypothesis_id)
    display_history(runs)


@app.command(name="list-hypotheses")
def list_hypotheses() -> None:
    """List all stored hypotheses with their latest evaluation scores."""
    from hyporeddit.evaluation.display import display_hypothesis_list
    from hyporeddit.storage.sqlite import get_all_hypotheses_with_latest_run

    hypotheses = get_all_hypotheses_with_latest_run()
    display_hypothesis_list(hypotheses)


@app.command(name="verify-stores")
def verify_stores(
    fix: bool = typer.Option(False, "--fix", help="Re-embed chunks missing from LanceDB"),
) -> None:
    """Check SQLite and LanceDB are in sync."""
    from hyporeddit.storage.unified import verify_stores as _verify

    _verify(fix=fix)


@app.command()
def process(
    reprocess: bool = typer.Option(False, "--reprocess", help="Re-process already-fetched posts"),
    make_translation: bool = typer.Option(
        False,
        "--make-translation",
        help="Translate chunks DE→EN via LLM after embedding (requires ANTHROPIC_API_KEY)",
    ),
    train_adapter: bool = typer.Option(
        False,
        "--train-adapter",
        help="Force adapter training after storing, regardless of chunk-count threshold",
    ),
) -> None:
    """Process fetched posts into chunks, embeddings, and translations."""
    from hyporeddit.ingestion.processor import process_all

    process_all(reprocess=reprocess, make_translation=make_translation, train_adapter=train_adapter)


if __name__ == "__main__":
    app()
