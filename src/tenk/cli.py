"""`tenk` command-line interface (also driven by the Makefile)."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

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
def index():
    """Build the Qdrant hybrid vector index."""
    from tenk.index.build_index import build_index

    build_index()


@app.command()
def graph(skip_extraction: bool = typer.Option(False, help="Load XBRL facts only, skip LLM extraction")):
    """Build the Neo4j knowledge graph."""
    from tenk.graph.build_graph import build_graph

    build_graph(skip_extraction=skip_extraction)


@app.command()
def ask(question: str):
    """Answer a question over the indexed filings."""
    from tenk.pipeline import answer_question

    ans = answer_question(question)
    console.print(Panel(ans.text, title=f"[bold]{question}[/bold]", border_style="yellow"))
    console.print(f"[dim]route: {ans.route} · {ans.notes}[/dim]")
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
