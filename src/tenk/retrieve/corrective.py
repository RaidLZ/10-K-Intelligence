"""Corrective-RAG: grade retrieved context, and rewrite the query when retrieval is weak.

This is what stops the generator from ever seeing irrelevant chunks — the difference
between a confident hallucination and "I don't have enough information to answer that."
"""
from __future__ import annotations

from collections.abc import Callable

from tenk import trace
from tenk.config import settings
from tenk.llm import get_llm
from tenk.models import RetrievedContext

RetrieveFn = Callable[[str], list[RetrievedContext]]

_GRADE_SYSTEM = "You judge whether each context is actually useful for answering the question."
_GRADE_PROMPT = """QUESTION: {q}

CONTEXTS:
{ctx}

Return JSON {{"relevant": [numbers of contexts that genuinely help answer the question]}}."""

_REWRITE_SYSTEM = "You rewrite financial-filing questions to retrieve better evidence."
_REWRITE_PROMPT = (
    'Rewrite this question to be more explicit and keyword-rich for retrieval over SEC 10-K '
    'filings (name the metric, companies, and years if implied). Return JSON {{"query": "..."}}.\n'
    "QUESTION: {q}"
)


def grade(query: str, contexts: list[RetrievedContext]) -> list[int]:
    """Return indices of contexts judged relevant. Falls back to score threshold on error."""
    if not contexts:
        return []
    listing = "\n".join(f"[{i}] {c.chunk.text[:300]}" for i, c in enumerate(contexts))
    try:
        data = get_llm().json(_GRADE_PROMPT.format(q=query, ctx=listing), system=_GRADE_SYSTEM)
        idx = [int(i) for i in data.get("relevant", []) if 0 <= int(i) < len(contexts)]
        trace.step("grade", f"kept {len(idx)}/{len(contexts)} contexts (LLM)")
        return idx
    except Exception:
        # score-based fallback: keep anything above threshold (graph facts score 1.0)
        idx = [i for i, c in enumerate(contexts) if c.score >= settings.relevance_threshold]
        trace.step("grade", f"kept {len(idx)}/{len(contexts)} contexts (score ≥ {settings.relevance_threshold})")
        return idx


def rewrite(query: str) -> str:
    try:
        return get_llm().json(_REWRITE_PROMPT.format(q=query), system=_REWRITE_SYSTEM).get("query", query)
    except Exception:
        return query


def corrective_retrieve(query: str, retrieve_fn: RetrieveFn) -> tuple[list[RetrievedContext], str]:
    """Retrieve → grade → (rewrite + retry once if weak). Returns (contexts, notes)."""
    notes = []
    contexts = retrieve_fn(query)
    relevant = grade(query, contexts)

    # Weak: nothing graded relevant -> rewrite once and retry.
    if not relevant:
        new_q = rewrite(query)
        if new_q and new_q != query:
            trace.step("rewrite", f"weak retrieval → {new_q!r}")
            notes.append(f"corrective: rewrote query → {new_q!r}")
            contexts = retrieve_fn(new_q)
            relevant = grade(new_q, contexts)

    if relevant:
        kept = [contexts[i] for i in relevant]
        notes.append(f"corrective: kept {len(kept)}/{len(contexts)} contexts")
        return kept, " | ".join(notes)

    # Still weak: hand back top contexts (generator is instructed to admit uncertainty).
    notes.append("corrective: no clearly-relevant context; answering cautiously")
    return contexts[: settings.final_k], " | ".join(notes)
