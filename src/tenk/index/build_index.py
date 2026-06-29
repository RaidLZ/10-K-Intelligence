"""Build the Qdrant hybrid index from data/processed/chunks.jsonl."""
from __future__ import annotations

from tenk.index.vector_store import VectorStore
from tenk.ingest.chunk import load_chunks


def build_index() -> int:
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("No chunks found. Run `make ingest` first.")
    store = VectorStore()
    n = store.index_chunks(chunks)
    print(f"  ✓ indexed {n} chunks into Qdrant collection '{store.collection}'")
    return n
