"""Turn parsed elements into citable `Chunk`s.

Rules that matter for filings:
  • tables are never split — a table is one chunk (splitting destroys the row/column meaning);
  • narrative is windowed within its section so a chunk never straddles Item boundaries;
  • section + page metadata is copied onto every chunk so answers can cite them.
"""
from __future__ import annotations

import json

from tenk.config import settings
from tenk.models import Chunk

WORDS_PER_CHUNK = 380
OVERLAP_WORDS = 60


def _window(text: str, size: int = WORDS_PER_CHUNK, overlap: int = OVERLAP_WORDS):
    words = text.split()
    if len(words) <= size:
        yield text
        return
    step = size - overlap
    for start in range(0, len(words), step):
        chunk = words[start : start + size]
        if chunk:
            yield " ".join(chunk)
        if start + size >= len(words):
            break


def chunk_elements(ticker: str, year: int, source_url: str, elements: list[dict]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for el in elements:
        section = el.get("section", "")
        page = el.get("page")
        if el.get("is_table"):
            chunks.append(
                Chunk(
                    id=f"{ticker}-{year}-{len(chunks)}",
                    ticker=ticker, year=year, section=section, page=page,
                    is_table=True, text=el["text"], source_url=source_url,
                )
            )
            continue
        for piece in _window(el.get("text", "")):
            if len(piece.split()) < 8:   # drop boilerplate fragments
                continue
            chunks.append(
                Chunk(
                    id=f"{ticker}-{year}-{len(chunks)}",
                    ticker=ticker, year=year, section=section, page=page,
                    is_table=False, text=piece, source_url=source_url,
                )
            )
    return chunks


def chunk_corpus() -> list[Chunk]:
    """Read processed sections + manifests, emit data/processed/chunks.jsonl."""
    corpus_file = settings.raw_dir / "corpus.json"
    manifests = json.loads(corpus_file.read_text()) if corpus_file.exists() else []
    url_by_key = {(m["ticker"], m["year"]): m.get("source_url", "") for m in manifests}

    all_chunks: list[Chunk] = []
    for sections_file in settings.processed_dir.rglob("sections.json"):
        year = int(sections_file.parent.name)
        ticker = sections_file.parent.parent.name
        elements = json.loads(sections_file.read_text())
        url = url_by_key.get((ticker, year), "")
        all_chunks.extend(chunk_elements(ticker, year, url, elements))

    out = settings.processed_dir / "chunks.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(c.model_dump_json() + "\n")
    print(f"  ✓ {len(all_chunks)} chunks → {out}")
    return all_chunks


def load_chunks() -> list[Chunk]:
    path = settings.processed_dir / "chunks.jsonl"
    if not path.exists():
        return []
    return [Chunk.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
