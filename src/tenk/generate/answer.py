"""Generate the final answer with inline [n] citations tied to retrieved contexts.

The prompt forbids using outside knowledge and requires a citation for every claim, so a
reader can trace each number back to a filing, section, and page.
"""
from __future__ import annotations

import re

from tenk.llm import get_llm
from tenk.models import Answer, Citation, RetrievedContext

_SYSTEM = (
    "You are a precise equity-research analyst. Answer ONLY from the numbered context. "
    "Cite every factual claim with [n] matching the context number. If the context does not "
    "contain the answer, say exactly what is missing — never invent figures."
)

_PROMPT = """QUESTION: {q}

CONTEXT:
{ctx}

Write a concise, well-structured answer. Put a [n] citation after each claim that uses context n.
If you compare values, show the numbers. If the context is insufficient, say so plainly."""


def _format_contexts(contexts: list[RetrievedContext]) -> str:
    lines = []
    for i, c in enumerate(contexts, start=1):
        lines.append(f"[{i}] ({c.chunk.citation_label()}) {c.chunk.text[:800]}")
    return "\n\n".join(lines)


def _citations_from_text(text: str, contexts: list[RetrievedContext]) -> list[Citation]:
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(contexts)})
    citations = []
    for n in used:
        ch = contexts[n - 1].chunk
        citations.append(
            Citation(
                ticker=ch.ticker, year=ch.year, section=ch.section, page=ch.page,
                source_url=ch.source_url, snippet=ch.text[:200],
            )
        )
    return citations


def generate_answer(query: str, contexts: list[RetrievedContext], route: str = "vector", notes: str = "") -> Answer:
    if not contexts:
        return Answer(
            question=query, route=route,
            text="I don't have enough indexed filing data to answer that. Try `make ingest` / `make index` first, or rephrase.",
            notes=notes,
        )
    text = get_llm().complete(
        _PROMPT.format(q=query, ctx=_format_contexts(contexts)), system=_SYSTEM
    )
    return Answer(
        question=query, text=text, route=route,
        citations=_citations_from_text(text, contexts), contexts=contexts, notes=notes,
    )
