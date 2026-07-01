"""Qdrant hybrid vector store.

Dense (BGE) + sparse (BM25) are fused with Reciprocal Rank Fusion server-side, then a
cross-encoder reranker re-scores the survivors. Hybrid matters in finance: dense catches
paraphrases ("research spending"), BM25 catches exact terms ("R&D expense", tickers).
"""
from __future__ import annotations

import uuid

from tenk.config import settings
from tenk.embeddings import (
    dense_dim,
    embed_dense,
    embed_query_dense,
    embed_query_sparse,
    embed_sparse,
    rerank,
)
from tenk.models import Chunk, RetrievedContext

DENSE = "dense"
SPARSE = "bm25"


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    def __init__(self, url: str | None = None, collection: str | None = None):
        from qdrant_client import QdrantClient

        self.collection = collection or settings.qdrant_collection
        self.client = QdrantClient(url=url or settings.qdrant_url)

    def ensure_collection(self, recreate: bool = False) -> None:
        from qdrant_client import models

        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                self.collection,
                vectors_config={DENSE: models.VectorParams(size=dense_dim(), distance=models.Distance.COSINE)},
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )

    def existing_ids(self) -> set[str]:
        """All point IDs currently in the collection (for incremental indexing)."""
        if not self.client.collection_exists(self.collection):
            return set()
        ids: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, with_payload=False, with_vectors=False,
                limit=1000, offset=offset,
            )
            ids.update(str(p.id) for p in points)
            if offset is None:
                break
        return ids

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 64, incremental: bool = False) -> int:
        from qdrant_client import models

        if incremental:
            # Keep what's there; embed only chunks not already indexed (resume/extend).
            self.ensure_collection(recreate=False)
            have = self.existing_ids()
            before = len(chunks)
            chunks = [c for c in chunks if _point_id(c.id) not in have]
            print(f"  … incremental: {before - len(chunks)} already indexed, {len(chunks)} to add")
        else:
            self.ensure_collection(recreate=True)
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.text for c in batch]
            dense_vecs = embed_dense(texts)
            sparse_vecs = embed_sparse(texts)
            points = []
            for c, dv, sv in zip(batch, dense_vecs, sparse_vecs, strict=True):
                points.append(
                    models.PointStruct(
                        id=_point_id(c.id),
                        vector={
                            DENSE: dv,
                            SPARSE: models.SparseVector(indices=sv["indices"], values=sv["values"]),
                        },
                        payload=c.model_dump(),
                    )
                )
            self.client.upsert(self.collection, points=points)
            total += len(points)
            print(f"  … indexed {total}/{len(chunks)}")
        return total

    def search(
        self, query: str, top_k: int | None = None, final_k: int | None = None,
        tickers: list[str] | None = None, years: list[int] | None = None,
    ) -> list[RetrievedContext]:
        from qdrant_client import models

        top_k = top_k or settings.top_k
        final_k = final_k or settings.final_k

        conditions = []
        if tickers:
            conditions.append(models.FieldCondition(key="ticker", match=models.MatchAny(any=tickers)))
        if years:
            conditions.append(
                models.FieldCondition(key="year", match=models.MatchAny(any=[int(y) for y in years]))
            )
        query_filter = models.Filter(must=conditions) if conditions else None

        sparse_q = embed_query_sparse(query)
        result = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=embed_query_dense(query), using=DENSE, limit=top_k, filter=query_filter),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_q["indices"], values=sparse_q["values"]),
                    using=SPARSE, limit=top_k, filter=query_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        ).points

        candidates = [(Chunk.model_validate(p.payload), p.score) for p in result]
        if not candidates:
            return []

        # Cross-encoder rerank for final precision.
        scores = rerank(query, [c.text for c, _ in candidates])
        ranked = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        return [
            RetrievedContext(chunk=chunk, score=float(score), retriever="vector")
            for (chunk, _fusion), score in ranked[:final_k]
        ]
