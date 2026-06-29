"""Build the Neo4j knowledge graph from processed filings.

  • financials.json → exact REPORTS edges (no LLM)
  • Business / Risk-Factors narrative → LLM-extracted qualitative relationships
"""
from __future__ import annotations

import json

from tenk.config import settings
from tenk.graph.extract import extract_relations, write_relations
from tenk.graph.financials import load_financials
from tenk.graph.schema import Graph

# Sections worth extracting relationships from (kept short to bound LLM cost on CPU).
_EXTRACT_SECTIONS = ("Item 1", "Item 1A", "Item 7")


def _narrative_for_extraction(elements: list[dict]) -> str:
    parts = []
    for el in elements:
        if el.get("is_table"):
            continue
        section = el.get("section", "")
        if any(section.startswith(s) for s in _EXTRACT_SECTIONS):
            parts.append(el.get("text", ""))
    return "\n".join(parts)


def build_graph(skip_extraction: bool = False) -> dict:
    stats = {"filings": 0, "metric_edges": 0, "relations": 0}
    with Graph() as graph:
        graph.ensure_schema()
        graph.wipe()
        graph.ensure_schema()

        for sections_file in sorted(settings.processed_dir.rglob("sections.json")):
            year = int(sections_file.parent.name)
            ticker = sections_file.parent.parent.name
            elements = json.loads(sections_file.read_text())

            fin_file = sections_file.parent / "financials.json"
            financials = json.loads(fin_file.read_text()) if fin_file.exists() else {}
            url = ""  # source_url is set on Filing via financials loader if available

            stats["metric_edges"] += load_financials(graph, ticker, year, financials, url)
            stats["filings"] += 1
            print(f"  ✓ {ticker} {year}: financials loaded")

            if not skip_extraction:
                text = _narrative_for_extraction(elements)
                if text.strip():
                    rels = extract_relations(text, ticker)
                    stats["relations"] += write_relations(graph, ticker, rels)
                    print(f"    + {len(rels)} qualitative relations")

    print(f"  ✓ graph built: {stats}")
    return stats
