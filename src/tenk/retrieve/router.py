"""Query router — the piece that decides vector vs. graph vs. agentic.

A fast deterministic heuristic handles the obvious cases (multiple companies, "compare",
year-over-year), and only ambiguous queries fall through to a one-shot LLM classifier.
This keeps routing cheap and explainable — and it's exactly the judgement the project is
meant to demonstrate: knowing *when* the graph is worth it.
"""
from __future__ import annotations

import re

from tenk.config import settings
from tenk.llm import get_llm

# Company aliases for the default corpus (extend alongside TICKERS).
TICKER_ALIASES = {
    "apple": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT",
    "nvidia": "NVDA", "nvda": "NVDA",
    "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL",
    "amazon": "AMZN", "amzn": "AMZN",
    "meta": "META", "facebook": "META",
}

_COMPARE = re.compile(r"\b(compare|versus|vs\.?|differ|relative to|against|both|each)\b", re.I)
_TREND = re.compile(r"\b(from\s+20\d\d\s+to\s+20\d\d|year[-\s]?over[-\s]?year|yoy|trend|chang(e|ed|ing)|grew|growth|over time|since)\b", re.I)
_AGG = re.compile(r"\b(which company|who has the|highest|lowest|most|least|rank|top \d+|across (all|the) companies)\b", re.I)
_CHAINED = re.compile(r"\band then\b|\bafter that\b|;|\bstep by step\b", re.I)


def detect_tickers(query: str) -> list[str]:
    found: list[str] = []
    q = query.lower()
    for alias, ticker in TICKER_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", q) and ticker not in found:
            found.append(ticker)
    # also catch raw uppercase tickers from the corpus
    for t in settings.tickers:
        if re.search(rf"\b{t}\b", query) and t not in found:
            found.append(t)
    return found


def detect_years(query: str) -> list[int]:
    return sorted({int(y) for y in re.findall(r"\b(20\d\d)\b", query)})


def heuristic_route(query: str) -> str | None:
    tickers = detect_tickers(query)
    years = detect_years(query)
    if len(tickers) >= 2:
        return "agentic" if _CHAINED.search(query) else "graph"
    if _AGG.search(query):
        return "graph"
    if _COMPARE.search(query) and (len(tickers) >= 1 or _AGG.search(query)):
        return "graph"
    if _TREND.search(query) or len(years) >= 2:
        return "graph"
    return None  # ambiguous -> ask the LLM


_LLM_SYSTEM = "You classify financial-filing questions by the retrieval strategy they need."
_LLM_PROMPT = """Classify the QUESTION into one route:
- "vector": a single fact/lookup answerable from one filing's text (definitions, what a company said).
- "graph": needs structured reasoning across companies or years (compare, rank, year-over-year, relationships).
- "agentic": several chained sub-questions that must be answered and combined.

Return JSON: {{"route": "...", "reason": "..."}}.
QUESTION: {q}"""


def classify(query: str) -> dict:
    """Return {route, tickers, years, reason}."""
    tickers, years = detect_tickers(query), detect_years(query)
    route = heuristic_route(query)
    reason = "heuristic"
    if route is None:
        try:
            data = get_llm().json(_LLM_PROMPT.format(q=query), system=_LLM_SYSTEM)
            route = data.get("route", "vector")
            reason = "llm: " + str(data.get("reason", ""))[:140]
        except Exception:
            route, reason = "vector", "fallback (default)"
    if route not in {"vector", "graph", "agentic"}:
        route = "vector"
    return {"route": route, "tickers": tickers, "years": years, "reason": reason}
