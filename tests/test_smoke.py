"""Network-free smoke tests.

These never touch Qdrant/Neo4j/Ollama/EDGAR — they exercise the pure-Python logic
(imports, chunking, routing heuristics, metric mapping, citation parsing, JSON extraction)
so CI catches breakage without any services or model downloads.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "data" / "sample" / "AAPL" / "2023" / "sections.json"


def test_light_imports():
    # Importing the pipeline must NOT require heavy/optional deps (all lazy-imported).
    import tenk.cli  # noqa: F401
    import tenk.config  # noqa: F401
    import tenk.pipeline  # noqa: F401


def test_config_parsing():
    from tenk.config import settings

    assert isinstance(settings.tickers, list) and settings.tickers
    assert all(isinstance(y, int) for y in settings.years)


def test_chunking_on_fixture():
    from tenk.ingest.chunk import chunk_elements

    elements = json.loads(FIXTURE.read_text())
    chunks = chunk_elements("AAPL", 2023, "http://example.com/aapl-2023", elements)

    assert len(chunks) >= 4
    assert all(c.ticker == "AAPL" and c.year == 2023 and c.section for c in chunks)
    # the financial table must survive as a single, intact table chunk
    tables = [c for c in chunks if c.is_table]
    assert len(tables) == 1
    assert "Research and development" in tables[0].text
    # every chunk can produce a citation label
    assert chunks[0].citation_label().startswith("AAPL 2023 10-K")


def test_router_heuristics():
    from tenk.retrieve.router import detect_tickers, detect_years, heuristic_route

    assert heuristic_route("Compare Apple's R&D in 2024 to Microsoft's") == "graph"
    assert heuristic_route("How did Apple's R&D change from 2022 to 2024?") == "graph"
    assert heuristic_route("Which company had the highest net income in 2024?") == "graph"
    # a chained, multi-company question routes to agentic
    assert heuristic_route("Compare Apple and NVIDIA R&D; then say who has more supply risk") == "agentic"
    # a plain single-company lookup is ambiguous -> defers to the LLM classifier (None)
    assert heuristic_route("What does Apple say about its services business?") is None

    assert set(detect_tickers("apple vs microsoft")) == {"AAPL", "MSFT"}
    assert detect_years("from 2022 to 2024") == [2022, 2024]


def test_metric_keyword_mapping():
    from tenk.retrieve.graph_retriever import metric_keywords

    kws = metric_keywords("how did R&D and revenue change")
    assert "research and development" in kws
    assert any("revenue" in k or "net sales" in k for k in kws)


def test_citation_parsing():
    from tenk.generate.answer import _citations_from_text
    from tenk.models import Chunk, RetrievedContext

    contexts = [
        RetrievedContext(chunk=Chunk(id="AAPL-2023-0", ticker="AAPL", year=2023, section="Item 7", text="R&D 29,915")),
        RetrievedContext(chunk=Chunk(id="MSFT-2023-0", ticker="MSFT", year=2023, section="Item 7", text="R&D 27,195")),
    ]
    cits = _citations_from_text("Apple spent 29,915 [1] vs Microsoft 27,195 [2].", contexts)
    assert [c.ticker for c in cits] == ["AAPL", "MSFT"]


def test_json_extraction():
    from tenk.llm import _extract_json

    assert _extract_json('```json\n{"route": "graph"}\n```') == {"route": "graph"}
    assert _extract_json('Sure! {"a": 1, "b": [2,3]} done')["b"] == [2, 3]


def test_financials_parsing():
    from tenk.graph.financials import _pick_year_value, _to_float

    assert _to_float("1,234") == 1234.0
    assert _to_float(29915) == 29915.0
    assert _to_float("n/a") is None
    assert _pick_year_value({"2023": 100, "2022": 90}, 2023) == 100.0
    assert _pick_year_value({"FY2022": 90, "FY2023": 100}, 2023) == 100.0
    assert _pick_year_value({"label": "n/a", "other": 5}, 2099) == 5.0  # first numeric fallback


def test_section_normalisation():
    from tenk.ingest.parse import normalise_section

    assert normalise_section("ITEM 1A. Risk Factors") == "Item 1A. Risk Factors"
    assert normalise_section("Item 7 — Management's Discussion").startswith("Item 7")
    assert normalise_section("Company Overview") == "Company Overview"


def test_chunk_window_overlap_and_filters():
    from tenk.ingest.chunk import OVERLAP_WORDS, chunk_elements

    long_text = " ".join(f"word{i}" for i in range(1000))
    elements = [
        {"section": "Item 7. MD&A", "page": 3, "is_table": False, "text": long_text},
        {"section": "Item 7. MD&A", "page": 3, "is_table": False, "text": "too short"},  # dropped (<8 words)
    ]
    chunks = chunk_elements("AAPL", 2023, "url", elements)
    assert len(chunks) > 1  # long text split into multiple windows
    first_words = chunks[0].text.split()
    second_words = chunks[1].text.split()
    assert set(first_words[-OVERLAP_WORDS:]) & set(second_words)  # windows overlap
    assert all(len(c.text.split()) >= 8 for c in chunks)  # short fragment filtered out


def test_relation_map_is_whitelisted():
    from tenk.graph.extract import RELATION_MAP

    labels = {label for label, _rel in RELATION_MAP.values()}
    assert labels <= {"Segment", "Risk", "Person", "Entity"}
    assert all(rel.replace("_", "").isalpha() for _label, rel in RELATION_MAP.values())


def test_classify_offline_heuristics():
    from tenk.retrieve.router import classify

    res = classify("Compare Apple and Microsoft revenue in 2024")
    assert res["route"] == "graph"
    assert set(res["tickers"]) == {"AAPL", "MSFT"}
    assert res["reason"] == "heuristic"


def test_eval_dataset_wellformed():
    path = REPO / "eval" / "dataset.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) >= 12
    for r in rows:
        assert {"id", "question", "type", "expected_route", "companies"} <= r.keys()
        assert r["expected_route"] in {"vector", "graph", "agentic"}
        assert isinstance(r["companies"], list)
