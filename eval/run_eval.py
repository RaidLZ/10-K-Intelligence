"""Evaluation harness.

Three things, each degrading gracefully if a service/model is unavailable:
  1. **Routing accuracy** — predicted route vs. labelled route (mostly offline / heuristic).
  2. **Generation quality** — citation coverage + optional RAGAS faithfulness/answer-relevancy.
  3. **Vector-vs-Graph head-to-head** — on multi-hop questions, the router (graph) vs. forcing
     vector-only, which is the project's headline result.

Writes `eval/reports/report.md` + `report.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

DATASET = Path(__file__).parent / "dataset.jsonl"


def load_dataset(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _answer_with_route(question: str, route: str):
    """Answer while forcing a retrieval route (for the head-to-head)."""
    from tenk.config import settings
    from tenk.generate.answer import generate_answer
    from tenk.retrieve import graph_retriever, vector_retriever
    from tenk.retrieve.corrective import corrective_retrieve
    from tenk.retrieve.router import detect_tickers, detect_years

    tickers, years = detect_tickers(question), detect_years(question)
    if route == "graph":
        fn = lambda q: graph_retriever.retrieve(q, tickers or settings.tickers, years or settings.years)  # noqa: E731
    else:
        fn = lambda q: vector_retriever.retrieve(q, tickers or None)  # noqa: E731
    ctx, notes = corrective_retrieve(question, fn)
    return generate_answer(question, ctx, route, notes)


def _maybe_ragas(samples: list[dict]) -> dict:
    """Compute faithfulness + answer relevancy if ragas is installed and samples exist."""
    usable = [s for s in samples if s.get("answer") and s.get("contexts")]
    if not usable:
        return {}
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except Exception:
        return {"note": "ragas not installed; skipped (pip install '.[eval]')"}
    try:
        ds = Dataset.from_list(
            [{"question": s["question"], "answer": s["answer"], "contexts": s["contexts"]} for s in usable]
        )
        result = evaluate(ds, metrics=[faithfulness, answer_relevancy])
        return {k: round(float(v), 3) for k, v in result.items()}
    except Exception as exc:  # noqa: BLE001
        return {"note": f"ragas run failed: {exc}"}


def run_eval(report_dir: str = "eval/reports") -> dict:
    from tenk.retrieve.router import classify

    items = load_dataset()
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) routing accuracy
    correct = 0
    for it in items:
        it["pred_route"] = classify(it["question"])["route"]
        it["route_ok"] = it["pred_route"] == it["expected_route"]
        correct += it["route_ok"]
    routing_acc = correct / len(items)

    # 2 & 3) generation + head-to-head (best-effort)
    ragas_samples, h2h = [], []
    answered = errors = 0
    for it in items:
        try:
            from tenk.pipeline import answer_question

            ans = answer_question(it["question"])
            answered += 1
            it["n_citations"] = len(ans.citations)
            ragas_samples.append(
                {"question": it["question"], "answer": ans.text,
                 "contexts": [c.chunk.text for c in ans.contexts]}
            )
            if it["expected_route"] in ("graph", "agentic"):
                vec = _answer_with_route(it["question"], "vector")
                h2h.append({
                    "id": it["id"], "question": it["question"],
                    "router_route": ans.route, "router_citations": len(ans.citations),
                    "vector_only_citations": len(vec.citations),
                })
        except Exception as exc:  # noqa: BLE001
            errors += 1
            it["error"] = str(exc)[:200]

    metrics = {
        "n_items": len(items),
        "routing_accuracy": round(routing_acc, 3),
        "answered": answered,
        "errors": errors,
        "citation_coverage": round(
            sum(1 for it in items if it.get("n_citations", 0) > 0) / max(answered, 1), 3
        ),
        "ragas": _maybe_ragas(ragas_samples),
    }

    (out / "report.json").write_text(
        json.dumps({"metrics": metrics, "items": items, "head_to_head": h2h}, indent=2)
    )
    _write_markdown(out / "report.md", metrics, h2h, items)
    print(json.dumps(metrics, indent=2))
    return metrics


def _write_markdown(path: Path, metrics: dict, h2h: list[dict], items: list[dict]) -> None:
    lines = ["# Evaluation report\n", "## Metrics\n"]
    for k, v in metrics.items():
        if k != "ragas":
            lines.append(f"- **{k}**: {v}")
    lines.append(f"- **ragas**: {metrics.get('ragas') or 'n/a'}\n")

    lines.append("## Vector-vs-Graph head-to-head (multi-hop questions)\n")
    if h2h:
        lines.append("| id | question | router route | router cites | vector-only cites |")
        lines.append("|---|---|---|---|---|")
        for r in h2h:
            lines.append(
                f"| {r['id']} | {r['question'][:60]}… | {r['router_route']} | "
                f"{r['router_citations']} | {r['vector_only_citations']} |"
            )
    else:
        lines.append("_No generation run (services/models offline). Run with Qdrant + Neo4j + Ollama up._")

    lines.append("\n## Routing detail\n| id | type | expected | predicted | ok |")
    lines.append("|---|---|---|---|---|")
    for it in items:
        lines.append(
            f"| {it['id']} | {it['type']} | {it['expected_route']} | "
            f"{it.get('pred_route','?')} | {'✅' if it.get('route_ok') else '❌'} |"
        )
    path.write_text("\n".join(lines) + "\n")
