"""Vector retrieval — hybrid search + rerank, optionally scoped to detected tickers."""
from __future__ import annotations

from tenk.index.vector_store import VectorStore
from tenk.models import RetrievedContext

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def retrieve(query: str, tickers: list[str] | None = None) -> list[RetrievedContext]:
    return _get_store().search(query, tickers=tickers or None)
