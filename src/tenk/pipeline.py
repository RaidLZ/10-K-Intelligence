"""End-to-end question answering: route → retrieve (corrective) → generate.

This is the single entry point the CLI, API and UI all call:

    from tenk.pipeline import answer_question
    ans = answer_question("How did Apple's R&D change 2022→2024 vs Microsoft?")
"""
from __future__ import annotations

from collections.abc import Callable

from tenk import trace
from tenk.generate.answer import generate_answer
from tenk.llm import get_llm
from tenk.models import Answer, RetrievedContext, TraceStep
from tenk.retrieve import graph_retriever, vector_retriever
from tenk.retrieve.corrective import corrective_retrieve
from tenk.retrieve.router import classify


def _retrieve_fn(route: str, tickers: list[str], years: list[int]):
    if route == "graph":
        return lambda q: graph_retriever.retrieve(q, tickers=tickers, years=years)
    return lambda q: vector_retriever.retrieve(q, tickers=tickers or None, years=years or None)


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
        subs = subs[:4] or [query]
    except Exception:
        subs = [query]
    trace.step("decompose", f"{len(subs)} sub-questions")
    return subs


def _retrieve_with_fallback(query: str, route: str, tickers: list[str], years: list[int]):
    """Retrieve for the chosen route; if a graph route comes back empty (e.g. a narrative
    question misrouted, or shaky LLM Cypher), gracefully fall back to vector search."""
    contexts, notes = corrective_retrieve(query, _retrieve_fn(route, tickers, years))
    if not contexts and route == "graph":
        contexts, vnotes = corrective_retrieve(query, _retrieve_fn("vector", tickers, years))
        if contexts:
            return "vector", contexts, f"graph empty → vector fallback; {vnotes}"
    return route, contexts, notes


def answer_question(query: str, on_step: Callable[[TraceStep], None] | None = None) -> Answer:
    """Answer a question end-to-end.

    Pass ``on_step`` to receive each :class:`TraceStep` as it happens (the UI uses this
    to render a live timeline); the same steps are always attached to ``Answer.steps``.
    """
    tracer = trace.start(on_step=on_step)
    llm = get_llm()
    trace.step("llm", f"provider={llm.provider} · model={llm.model}")

    cls = classify(query)
    route, tickers, years = cls["route"], cls["tickers"], cls["years"]

    if route == "agentic":
        all_ctx: list[RetrievedContext] = []
        subtrace = []
        for sq in _decompose(query):
            sub = classify(sq)
            sub_route, ctx, _ = _retrieve_with_fallback(
                sq, sub["route"], sub["tickers"] or tickers, sub["years"] or years
            )
            all_ctx.extend(ctx)
            subtrace.append(f"{sq!r}→{sub_route}")
        notes = f"route=agentic ({cls['reason']}); subq: " + "; ".join(subtrace)
        ans = generate_answer(query, _dedup(all_ctx), "agentic", notes)
    else:
        route, contexts, corr_notes = _retrieve_with_fallback(query, route, tickers, years)
        notes = f"route={route} ({cls['reason']}); {corr_notes}"
        ans = generate_answer(query, contexts, route, notes)

    ans.steps = tracer.steps
    return ans
