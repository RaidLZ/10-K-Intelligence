"""LLM extraction of qualitative, company-centric relationships from narrative text.

Numbers come from XBRL (see financials.py); this module only extracts the *relationships*
a graph is good at — segments, risks, competitors, dependencies, people. This is the one
batch step that is slow on a CPU 3B model, so `make graph` recommends the API fallback.
"""
from __future__ import annotations

from tenk.graph.schema import Graph
from tenk.llm import LLM, get_llm

# relation -> (target node label, relationship type).  Whitelisted: these strings are
# interpolated into Cypher, so they must never come from model output directly.
RELATION_MAP: dict[str, tuple[str, str]] = {
    "HAS_SEGMENT": ("Segment", "HAS_SEGMENT"),
    "MENTIONS_RISK": ("Risk", "MENTIONS_RISK"),
    "COMPETES_WITH": ("Entity", "COMPETES_WITH"),
    "DEPENDS_ON": ("Entity", "DEPENDS_ON"),
    "HAS_SUBSIDIARY": ("Entity", "HAS_SUBSIDIARY"),
    "HAS_EXECUTIVE": ("Person", "HAS_EXECUTIVE"),
    "HAS_BOARD_MEMBER": ("Person", "HAS_BOARD_MEMBER"),
}

_SYSTEM = "You are a meticulous financial-filings analyst. Extract only facts stated in the text."

_PROMPT = """From this 10-K excerpt for {ticker}, extract company-centric relationships.

Return a JSON array of objects: {{"relation": <REL>, "target": <short name>}} where <REL> is one of:
{relations}.
Rules: target is a short canonical name (e.g. "iPhone", "supply chain disruption", "Tim Cook").
Only include facts explicitly supported by the text. Max 25 items. No duplicates.

EXCERPT:
{excerpt}
"""


def extract_relations(text: str, ticker: str, llm: LLM | None = None) -> list[dict]:
    llm = llm or get_llm()
    excerpt = text[:6000]   # keep prompt bounded for small local models
    try:
        data = llm.json(
            _PROMPT.format(ticker=ticker, relations=", ".join(RELATION_MAP), excerpt=excerpt),
            system=_SYSTEM,
        )
    except Exception as exc:
        print(f"    ! extraction failed for {ticker}: {exc}")
        return []
    items = data if isinstance(data, list) else data.get("relationships", [])
    out = []
    for it in items:
        rel = str(it.get("relation", "")).upper().strip()
        target = str(it.get("target", "")).strip()
        if rel in RELATION_MAP and target:
            out.append({"relation": rel, "target": target[:120]})
    return out


def write_relations(graph: Graph, ticker: str, relations: list[dict]) -> int:
    written = 0
    for r in relations:
        label, rel_type = RELATION_MAP[r["relation"]]
        # label / rel_type are whitelisted; only `target`/`ticker` are parameterised.
        graph.run(
            f"""
            MERGE (c:Company {{ticker: $ticker}})
            MERGE (t:{label} {{name: $target}})
            MERGE (c)-[:{rel_type}]->(t)
            """,
            ticker=ticker, target=r["target"],
        )
        written += 1
    return written
