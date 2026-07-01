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
    """Coalesce consecutive same-section narrative into ~WORDS_PER_CHUNK windows.

    A structure-aware parser (Docling) emits many small elements — a lone paragraph or
    list item. Embedding each on its own is slow and gives context-poor chunks, so we
    accumulate them within a section and window the joined text. Tables always stand alone
    (splitting a table destroys its meaning) and break the current run.
    """
    chunks: list[Chunk] = []

    def _emit(section: str, page: int | None, text: str, is_table: bool) -> None:
        chunks.append(
            Chunk(
                id=f"{ticker}-{year}-{len(chunks)}",
                ticker=ticker, year=year, section=section, page=page,
                is_table=is_table, text=text, source_url=source_url,
            )
        )

    run_section: str | None = None
    run_page: int | None = None
    run_parts: list[str] = []

    def _flush() -> None:
        nonlocal run_parts
        if run_parts:
            for piece in _window(" ".join(run_parts)):
                if len(piece.split()) >= 8:   # drop boilerplate fragments
                    _emit(run_section or "", run_page, piece, is_table=False)
        run_parts = []

    for el in elements:
        section = el.get("section", "")
        page = el.get("page")
        text = (el.get("text") or "").strip()
        if el.get("is_table"):
            _flush()
            run_section = None
            if text:
                _emit(section, page, text, is_table=True)
            continue
        if not text:
            continue
        if section != run_section:   # section boundary → never straddle it
            _flush()
            run_section, run_page = section, page
        run_parts.append(text)
    _flush()
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
