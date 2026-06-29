"""Shared data models that flow across ingestion → indexing → retrieval → answer.

Keeping these in one place means a chunk's citation metadata (the page/table it came
from) is carried end-to-end — which is what makes every answer verifiable.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A retrievable unit of a filing, with enough metadata to cite it back."""

    id: str
    ticker: str
    year: int
    form: str = "10-K"
    section: str = ""           # e.g. "Item 7. MD&A", "Item 1A. Risk Factors"
    page: int | None = None
    is_table: bool = False
    text: str
    source_url: str = ""

    def citation_label(self) -> str:
        page = f", p.{self.page}" if self.page else ""
        return f"{self.ticker} {self.year} 10-K · {self.section}{page}"


class Citation(BaseModel):
    ticker: str
    year: int
    section: str = ""
    page: int | None = None
    source_url: str = ""
    snippet: str = ""

    def label(self) -> str:
        page = f", p.{self.page}" if self.page else ""
        return f"{self.ticker} {self.year} 10-K · {self.section}{page}"


class RetrievedContext(BaseModel):
    chunk: Chunk
    score: float = 0.0
    retriever: str = "vector"   # "vector" | "graph"


Route = str  # "vector" | "graph" | "agentic"


class Answer(BaseModel):
    question: str
    text: str
    route: Route = "vector"
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[RetrievedContext] = Field(default_factory=list)
    notes: str = ""             # e.g. corrective-RAG actions taken
