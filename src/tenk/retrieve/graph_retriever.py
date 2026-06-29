"""Graph retrieval over the Neo4j knowledge graph.

Two paths:
  1. **Metric path** (fast, exact): map the question to XBRL line items and fetch
     `REPORTS` values for the relevant companies/years — perfect for compare / trend / rank.
  2. **Relationship path**: guarded text-to-Cypher for questions about segments, risks,
     competitors and people.

Each returned fact becomes a citable context (ticker + year + statement).
"""
from __future__ import annotations

import re

from tenk.config import settings
from tenk.graph.schema import Graph
from tenk.llm import get_llm
from tenk.models import Chunk, RetrievedContext

# Map natural-language concepts to substrings found in XBRL metric labels.
METRIC_SYNONYMS: dict[str, list[str]] = {
    "r&d|research": ["research and development"],
    "revenue|sales|top line": ["revenue", "net sales", "total net sales"],
    "net income|profit|earnings|bottom line": ["net income"],
    "operating income|operating profit": ["operating income"],
    "gross (profit|margin)": ["gross margin", "gross profit"],
    "cost of (sales|revenue|goods)": ["cost of sales", "cost of revenue"],
    "cash|cash and equivalents": ["cash and cash equivalents"],
    "assets": ["total assets"],
    "liabilities": ["total liabilities"],
    "operating expense|opex": ["operating expenses"],
}

_WRITE = re.compile(r"\b(create|merge|delete|set|remove|drop|load\s+csv)\b", re.I)


def metric_keywords(query: str) -> list[str]:
    kws: list[str] = []
    for pattern, labels in METRIC_SYNONYMS.items():
        if re.search(pattern, query, re.I):
            kws.extend(labels)
    return kws


def _facts(graph: Graph, tickers: list[str], years: list[int], keywords: list[str]) -> list[RetrievedContext]:
    rows = graph.run(
        """
        MATCH (c:Company)-[:FILED]->(f:Filing)-[r:REPORTS]->(m:Metric)
        WHERE c.ticker IN $tickers AND f.year IN $years
          AND any(kw IN $kws WHERE toLower(m.name) CONTAINS kw)
        RETURN c.ticker AS ticker, f.year AS year, m.name AS metric,
               r.value AS value, f.source_url AS url
        ORDER BY metric, ticker, year
        """,
        tickers=tickers, years=years, kws=[k.lower() for k in keywords],
    )
    contexts = []
    for row in rows:
        text = f"{row['ticker']} {row['year']} — {row['metric']}: {row['value']:,.0f}"
        chunk = Chunk(
            id=f"graph-{row['ticker']}-{row['year']}-{abs(hash(row['metric'])) % 10000}",
            ticker=row["ticker"], year=row["year"], section="XBRL financials",
            is_table=True, text=text, source_url=row.get("url") or "",
        )
        contexts.append(RetrievedContext(chunk=chunk, score=1.0, retriever="graph"))
    return contexts


_CYPHER_SYSTEM = "You write a single read-only Cypher query for a Neo4j financial-filings graph."
_CYPHER_PROMPT = """Schema:
(:Company {{ticker,name}})-[:FILED]->(:Filing {{year}})-[:REPORTS {{value,year}}]->(:Metric {{name}})
(:Company)-[:HAS_SEGMENT]->(:Segment {{name}})
(:Company)-[:MENTIONS_RISK]->(:Risk {{name}})
(:Company)-[:COMPETES_WITH|DEPENDS_ON|HAS_SUBSIDIARY]->(:Entity {{name}})
(:Company)-[:HAS_EXECUTIVE|HAS_BOARD_MEMBER]->(:Person {{name}})

Write ONE read-only Cypher (MATCH/RETURN only, LIMIT 25) answering the QUESTION.
Return JSON: {{"cypher": "..."}}.
QUESTION: {q}"""


def _relationship_facts(graph: Graph, query: str) -> list[RetrievedContext]:
    try:
        data = get_llm().json(_CYPHER_PROMPT.format(q=query), system=_CYPHER_SYSTEM)
        cypher = data.get("cypher", "")
    except Exception:
        return []
    if not cypher or _WRITE.search(cypher):
        return []
    try:
        rows = graph.run(cypher)
    except Exception as exc:
        print(f"  ! generated Cypher failed: {exc}")
        return []
    contexts = []
    for i, row in enumerate(rows[:25]):
        text = "; ".join(f"{k}={v}" for k, v in dict(row).items())
        contexts.append(
            RetrievedContext(
                chunk=Chunk(id=f"graph-rel-{i}", ticker="", year=0,
                            section="Knowledge graph", text=text),
                score=1.0, retriever="graph",
            )
        )
    return contexts


def retrieve(query: str, tickers: list[str] | None = None, years: list[int] | None = None) -> list[RetrievedContext]:
    tickers = tickers or settings.tickers
    years = years or settings.years
    keywords = metric_keywords(query)
    with Graph() as graph:
        if keywords:
            facts = _facts(graph, tickers, years, keywords)
            if facts:
                return facts
        return _relationship_facts(graph, query)
