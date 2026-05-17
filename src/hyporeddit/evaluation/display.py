"""Rich-formatted display helpers for evaluation results."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from hyporeddit.models.evaluation import EvaluationResult

_console = Console()


def _score_bar(score: float, width: int = 30) -> str:
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.2f}"


def display_result(result: EvaluationResult) -> None:
    """Pretty-print a full EvaluationResult using Rich."""
    dist = result.stance_distribution

    _console.print()
    _console.print(Panel(
        f"[bold]{result.hypothesis_text}[/bold]",
        title="Hypothesis",
        border_style="blue",
    ))

    # Score + confidence
    score_color = "green" if result.score >= 0.6 else ("red" if result.score <= 0.4 else "yellow")
    _console.print(f"\n  Score      [{score_color}]{_score_bar(result.score)}[/{score_color}]")
    _console.print(f"  Confidence {_score_bar(result.confidence)}")
    _console.print(f"  Sample     {result.sample_size} evidence chunks\n")

    # Stance distribution table
    table = Table(title="Stance Distribution", box=box.SIMPLE)
    table.add_column("Stance", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("[green]Supports[/green]", str(dist.get("supports", 0)))
    table.add_row("[red]Contradicts[/red]", str(dist.get("contradicts", 0)))
    table.add_row("[yellow]Neutral[/yellow]", str(dist.get("neutral", 0)))
    table.add_row("Irrelevant", str(dist.get("irrelevant", 0)))
    _console.print(table)

    # Top evidence per stance
    supports = [e for e in result.evidence if e.stance == "supports"][:3]
    contradicts = [e for e in result.evidence if e.stance == "contradicts"][:3]

    if supports:
        _console.print("[bold green]Supporting evidence:[/bold green]")
        for item in supports:
            text = item.text_en or item.text_de
            _console.print(f"  • {text[:200]}")
            if item.text_en and item.text_de != item.text_en:
                _console.print(f"    [dim]DE: {item.text_de[:120]}[/dim]")
        _console.print()

    if contradicts:
        _console.print("[bold red]Contradicting evidence:[/bold red]")
        for item in contradicts:
            text = item.text_en or item.text_de
            _console.print(f"  • {text[:200]}")
        _console.print()

    # Synthesis
    _console.print(Panel(result.synthesis, title="Synthesis", border_style="cyan"))
    _console.print(f"\n[dim]Run ID: {result.run_id}[/dim]\n")


def display_history(runs: list) -> None:
    """Display all evaluation runs for a hypothesis as a table."""
    if not runs:
        _console.print("[yellow]No evaluation runs found.[/yellow]")
        return

    table = Table(title="Evaluation History", box=box.SIMPLE)
    table.add_column("Run ID")
    table.add_column("Date")
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Sample", justify="right")

    for run in runs:
        table.add_row(
            run["id"][:8] + "…",
            run["run_at"][:19],
            f"{run['score']:.2f}",
            f"{run['confidence']:.2f}",
            str(run["sample_size"]),
        )
    _console.print(table)


def display_hypothesis_list(hypotheses: list) -> None:
    """Display all hypotheses with their latest evaluation score."""
    if not hypotheses:
        _console.print("[yellow]No hypotheses stored yet.[/yellow]")
        return

    table = Table(title="Stored Hypotheses", box=box.SIMPLE)
    table.add_column("ID")
    table.add_column("Hypothesis")
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Last Run")

    for hyp in hypotheses:
        score = hyp["score"]
        conf = hyp["confidence"]
        run_at = hyp["run_at"]
        table.add_row(
            str(hyp["id"])[:8] + "…",
            str(hyp["text"])[:60],
            f"{score:.2f}" if score is not None else "—",
            f"{conf:.2f}" if conf is not None else "—",
            run_at[:10] if run_at else "never",
        )
    _console.print(table)
