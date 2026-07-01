"""Vector retrieval — hybrid search + rerank, optionally scoped to detected tickers."""
from __future__ import annotations

from tenk import trace
from tenk.index.vector_store import VectorStore
from tenk.models import RetrievedContext

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def retrieve(
    query: str, tickers: list[str] | None = None, years: list[int] | None = None
) -> list[RetrievedContext]:
    results = _get_store().search(query, tickers=tickers or None, years=years or None)
    scope = ""
    if tickers:
        scope += " " + ",".join(tickers)
    if years:
        scope += " " + ",".join(str(y) for y in years)
    scope = f" scoped to{scope}" if scope else ""
    trace.step("retrieve", f"vector · {len(results)} hits{scope}", retriever="vector", n=len(results))
    return results
