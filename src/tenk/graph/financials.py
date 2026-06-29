"""Load XBRL financial statements into the graph as exact, queryable facts.

These are the *ground-truth numbers* — no LLM, no OCR. A `(:Filing)-[:REPORTS {value}]->(:Metric)`
edge per line item lets the graph retriever answer "compare R&D across companies/years" with
real figures it can cite.
"""
from __future__ import annotations

import re

from tenk.graph.schema import Graph

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _to_float(v) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = _NUM_RE.search(v.replace(",", ""))
        if m:
            try:
                return float(m.group())
            except ValueError:
                return None
    return None


def _pick_year_value(cols: dict, year: int) -> float | None:
    """Pick the column whose label references `year`; else the first numeric value."""
    for label, val in cols.items():
        if str(year) in str(label):
            f = _to_float(val)
            if f is not None:
                return f
    for val in cols.values():
        f = _to_float(val)
        if f is not None:
            return f
    return None


def load_financials(graph: Graph, ticker: str, year: int, financials: dict, source_url: str = "") -> int:
    """Create Company, Filing and REPORTS→Metric edges. Returns # of metric edges."""
    filing_id = f"{ticker}-{year}-10-K"
    graph.run(
        """
        MERGE (c:Company {ticker: $ticker})
          ON CREATE SET c.name = $ticker
        MERGE (f:Filing {id: $fid})
          SET f.ticker=$ticker, f.year=$year, f.form='10-K', f.source_url=$url
        MERGE (c)-[:FILED]->(f)
        """,
        ticker=ticker, fid=filing_id, year=year, url=source_url,
    )

    edges = 0
    for statement, lines in (financials or {}).items():
        if not isinstance(lines, dict):
            continue
        for metric_name, cols in lines.items():
            if not isinstance(cols, dict):
                continue
            value = _pick_year_value(cols, year)
            if value is None:
                continue
            graph.run(
                """
                MATCH (f:Filing {id: $fid})
                MERGE (m:Metric {name: $metric})
                  ON CREATE SET m.statement = $statement
                MERGE (f)-[r:REPORTS]->(m)
                  SET r.value = $value, r.year = $year
                """,
                fid=filing_id, metric=str(metric_name)[:120],
                statement=statement, value=value, year=year,
            )
            edges += 1
    return edges
