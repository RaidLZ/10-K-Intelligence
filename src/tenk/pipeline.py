"""End-to-end question answering: route → retrieve (corrective) → generate.

This is the single entry point the CLI, API and UI all call:

    from tenk.pipeline import answer_question
    ans = answer_question("How did Apple's R&D change 2022→2024 vs Microsoft?")
"""
from __future__ import annotations

from tenk.generate.answer import generate_answer
from tenk.llm import get_llm
from tenk.models import Answer, RetrievedContext
from tenk.retrieve import graph_retriever, vector_retriever
from tenk.retrieve.corrective import corrective_retrieve
from tenk.retrieve.router import classify


def _retrieve_fn(route: str, tickers: list[str], years: list[int]):
    if route == "graph":
        return lambda q: graph_retriever.retrieve(q, tickers=tickers, years=years)
    return lambda q: vector_retriever.retrieve(q, tickers=tickers or None)


def _dedup(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
    seen, out = set(), []
    for c in contexts:
        if c.chunk.id not in seen:
            seen.add(c.chunk.id)
            out.append(c)
    return out


def _decompose(query: str) -> list[str]:
    try:
        data = get_llm().json(
            'Break this question into 2-4 atomic sub-questions. Return JSON {"subquestions": [...]}.\n'
            f"QUESTION: {query}",
            system="You decompose complex financial questions into simple retrievable parts.",
        )
        subs = [str(s) for s in data.get("subquestions", []) if str(s).strip()]
        return subs[:4] or [query]
    except Exception:
        return [query]


def answer_question(query: str) -> Answer:
    cls = classify(query)
    route, tickers, years = cls["route"], cls["tickers"], cls["years"]

    if route == "agentic":
        all_ctx: list[RetrievedContext] = []
        trace = []
        for sq in _decompose(query):
            sub = classify(sq)
            fn = _retrieve_fn(sub["route"], sub["tickers"] or tickers, sub["years"] or years)
            ctx, _ = corrective_retrieve(sq, fn)
            all_ctx.extend(ctx)
            trace.append(f"{sq!r}→{sub['route']}")
        notes = f"route=agentic ({cls['reason']}); subq: " + "; ".join(trace)
        return generate_answer(query, _dedup(all_ctx), "agentic", notes)

    fn = _retrieve_fn(route, tickers, years)
    contexts, corr_notes = corrective_retrieve(query, fn)
    notes = f"route={route} ({cls['reason']}); {corr_notes}"
    return generate_answer(query, contexts, route, notes)
