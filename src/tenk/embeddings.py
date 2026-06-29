"""Embedding + reranking wrappers (FastEmbed, ONNX, CPU-friendly).

Dense vectors for semantic similarity, sparse BM25 vectors for exact-term matches
(tickers, line-item names), and a cross-encoder reranker for final precision.
Models are loaded lazily so importing this module is cheap and offline-safe.
"""
from __future__ import annotations

from functools import lru_cache

from tenk.config import settings


@lru_cache
def _dense():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=settings.embed_model)


@lru_cache
def _sparse():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=settings.sparse_model)


@lru_cache
def _reranker():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=settings.rerank_model)


def embed_dense(texts: list[str]) -> list[list[float]]:
    """Dense embeddings for a batch of texts."""
    return [vec.tolist() for vec in _dense().embed(texts)]


def embed_query_dense(text: str) -> list[float]:
    """Single-query dense embedding (uses query_embed where available)."""
    model = _dense()
    embed_query = getattr(model, "query_embed", None)
    if embed_query is not None:
        return next(iter(embed_query(text))).tolist()
    return next(iter(model.embed([text]))).tolist()


def embed_sparse(texts: list[str]) -> list[dict]:
    """Sparse (BM25) embeddings as {indices, values} dicts for Qdrant."""
    out = []
    for emb in _sparse().embed(texts):
        out.append({"indices": emb.indices.tolist(), "values": emb.values.tolist()})
    return out


def embed_query_sparse(text: str) -> dict:
    return embed_sparse([text])[0]


def rerank(query: str, documents: list[str]) -> list[float]:
    """Cross-encoder relevance scores (higher = more relevant), aligned to `documents`."""
    if not documents:
        return []
    return list(_reranker().rerank(query, documents))


def dense_dim() -> int:
    """Vector dimension of the configured dense model (for Qdrant collection setup)."""
    return len(embed_query_dense("dimension probe"))
