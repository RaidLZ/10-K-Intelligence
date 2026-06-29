# Architecture

## The thesis
A 10-K is two documents in one: **numbers** (financial statements) and **narrative** (business,
risks, MD&A). Naive RAG flattens both into text chunks and loses the numbers' structure. This system
treats them separately and reunites them at query time.

```
                          SEC EDGAR
                              │ edgartools
              ┌───────────────┴────────────────┐
       narrative HTML                     XBRL financials
              │ Docling (TableFormer)            │  (exact, no OCR)
     section/table-aware elements                │
              │ chunk.py                          │
   ┌──────────┴───────────┐                       │
 dense (BGE)        sparse (BM25)                  │
   └──────────┬───────────┘                        │
              ▼                                     ▼
      Qdrant — hybrid (RRF) + bge-reranker   Neo4j knowledge graph
              │                          (:Company)-[:FILED]->(:Filing)-[:REPORTS]->(:Metric)
              │                          (:Company)-[:HAS_SEGMENT|MENTIONS_RISK|...]->(...)
              └───────────────┬─────────────────────┘
                              ▼
                   Router (router.py)
        single-hop → vector | cross-company/year/rank → graph | chained → agentic
                              ▼
                Corrective-RAG (corrective.py)
            grade context relevance → rewrite query if weak
                              ▼
              Answer with inline [n] citations (answer.py)
                              ▼
                   FastAPI  ·  Streamlit
```

## Why each choice

| Decision | Rationale |
|---|---|
| **Numbers from XBRL, not OCR** | The financial figures are the part you must get exactly right. `edgartools` reads them from structured XBRL, so the graph's `REPORTS` edges are ground-truth — the LLM never transcribes a number from a table image. |
| **Docling for narrative** | Its TableFormer preserves merged cells (Marker splits them), and it emits section headings + page numbers we copy onto every chunk for citations. |
| **Hybrid (dense + BM25)** | Dense catches paraphrase ("research spending" ≈ "R&D"); BM25 catches exact tokens (tickers, line-item names). RRF fuses them; a cross-encoder reranks the survivors. |
| **Graph for multi-hop** | "Compare X vs Y across years" is a traversal, not a similarity search. The graph answers it with exact values; vector RAG would retrieve scattered, hard-to-compare snippets. |
| **Router, not graph-everything** | ~80% of questions are single-hop lookups where vector is faster and just as good. Routing is the senior judgement: *know when the graph is worth it.* |
| **Corrective-RAG** | Grades retrieved context and rewrites weak queries, so the generator either gets good evidence or admits it can't answer — the main hallucination lever. |
| **Local-first, API-optional** | Everything runs on CPU with small models; one env var swaps in an API model for the only slow step (batch graph extraction) or for higher answer quality. |

## When the graph does *not* help
Single-filing lookups ("what does Apple say about its services business?") gain nothing from the
graph and pay extra latency — so the router sends them to vector search. The eval harness
(`eval/run_eval.py`) measures exactly this split: the graph's win on multi-hop questions, and its
neutrality on single-hop ones. Demonstrating that boundary is the point.

## Data flow on disk
```
data/raw/<TICKER>/<YEAR>/         filing.html, financials.json, manifest.json   (gitignored)
data/processed/<TICKER>/<YEAR>/   sections.json, financials.json                 (gitignored)
data/processed/chunks.jsonl       all chunks, ready to index                     (gitignored)
data/sample/                      tiny committed fixture for the offline smoke test
```
