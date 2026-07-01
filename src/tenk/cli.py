"""`tenk` command-line interface (also driven by the Makefile)."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(add_completion=False, help="10-K Intelligence pipeline CLI.")
console = Console()


@app.command()
def ingest():
    """Download filings from EDGAR, parse narrative, and chunk."""
    from tenk.ingest.chunk import chunk_corpus
    from tenk.ingest.edgar import fetch_corpus
    from tenk.ingest.parse import parse_corpus

    manifests = fetch_corpus()
    parse_corpus(manifests)
    chunk_corpus()
    console.print("[green]✓ ingestion complete[/green]")


@app.command()
def index(
    resume: bool = typer.Option(
        False, "--resume", "--incremental",
        help="Only embed chunks not already indexed (resume a partial run / add new filings).",
    ),
):
    """Build the Qdrant hybrid vector index."""
    from tenk.index.build_index import build_index

    build_index(incremental=resume)


@app.command()
def graph(skip_extraction: bool = typer.Option(False, help="Load XBRL facts only, skip LLM extraction")):
    """Build the Neo4j knowledge graph."""
    from tenk.graph.build_graph import build_graph

    build_graph(skip_extraction=skip_extraction)


@app.command()
def ask(
    question: str,
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Stream each pipeline step live as it runs."
    ),
):
    """Answer a question over the indexed filings."""
    from tenk import trace
    from tenk.pipeline import answer_question

    trace.configure_logging(verbose)
    ans = answer_question(question)

    console.print(Panel(ans.text, title=f"[bold]{question}[/bold]", border_style="yellow"))
    console.print(f"[dim]route: {ans.route} · {ans.notes}[/dim]")

    if ans.steps:
        table = Table(title="pipeline trace", title_style="dim", show_edge=False, pad_edge=False)
        table.add_column("", style="cyan", no_wrap=True)
        table.add_column("step", style="bold")
        table.add_column("detail")
        table.add_column("ms", justify="right", style="dim")
        for s in ans.steps:
            table.add_row(trace.ICONS.get(s.name, "•"), s.name, s.detail, f"{s.ms:.0f}")
        total = sum(s.ms for s in ans.steps)
        table.add_row("", "", "[dim]total[/dim]", f"[bold]{total:.0f}[/bold]")
        console.print(table)

    if ans.citations:
        console.print("\n[bold]Sources[/bold]")
        for i, c in enumerate(ans.citations, 1):
            console.print(f"  [{i}] {c.label()}")


@app.command()
def eval(report_dir: str = "eval/reports"):
    """Run the evaluation harness (RAGAS + citation + vector-vs-graph)."""
    try:
        from eval.run_eval import run_eval
    except ModuleNotFoundError as exc:  # `eval/` lives at the repo root, not in the package
        raise SystemExit(
            "Could not import the eval suite — run `make eval` from the repository root."
        ) from exc

    run_eval(report_dir)


if __name__ == "__main__":
    app()
