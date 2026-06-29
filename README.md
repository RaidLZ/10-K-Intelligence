<div align="center">

# 10-K Intelligence

**A local-first financial-filings assistant that answers cited, multi-hop questions over SEC EDGAR 10-Ks — with a hybrid vector + GraphRAG retrieval router.**

`Docling` · `edgartools (XBRL)` · `FastEmbed + Qdrant (hybrid)` · `Neo4j (GraphRAG)` · `Corrective-RAG` · `RAGAS eval` · `FastAPI + Streamlit`

</div>

---

> *"How did Apple's R&D spend change from 2022 to 2024, and how does it compare to Microsoft's?"*

That question is **inter-year** (Apple across time) **and cross-company** (Apple vs. Microsoft) — the kind of multi-hop query that plain vector RAG answers badly. This system routes it to a knowledge graph, traverses the relationships, and returns an answer where **every number is cited to a filing, section, and page.**

## Why this isn't 2023-era RAG

Naive RAG (chunk → embed → cosine → done) breaks on real filings: it flattens complex tables and can't reason across documents. This system is the 2026 production shape:

| Problem with naive RAG | What this does instead |
|---|---|
| Tables get flattened into noise | **Numbers come from XBRL** (`edgartools`, exact) — OCR/Docling only for narrative |
| One vector index answers everything | **Query router**: single-hop → vector, multi-hop/cross-company → **graph** |
| Retrieves irrelevant chunks silently | **Corrective-RAG** grades relevance and rewrites weak queries |
| "Trust me" answers | **Inline citations** to page + table on every claim |
| Stops at a demo | **CI eval harness** measuring the vector-vs-graph gap and hallucination rate |

## Architecture

```
EDGAR ──edgartools──> filings + XBRL financials
   narrative │                    │ structured numbers (ground truth)
      Docling parse               │
   section/table-aware chunks     │
        │                         │
   Qdrant (dense BGE + BM25      Neo4j graph
    sparse, hybrid + rerank)     (Company/Filing/Metric/Segment/Risk/Person)
        └───────────┬─────────────┘
                    ▼
         Query Router → vector | graph | agentic
                    ▼
         Corrective-RAG (grade → rewrite if weak)
                    ▼
         Cited answer  →  FastAPI  →  Streamlit
```
See [`docs/architecture.md`](docs/architecture.md) for detail.

## Models (chosen for accuracy on modest, local hardware)

| Stage | Model / tool | Runs locally? |
|---|---|---|
| Filings + financials | `edgartools` (XBRL) | ✅ |
| Parsing / tables | **Docling** (TableFormer) | ✅ CPU |
| Dense embeddings | `BAAI/bge-base-en-v1.5` (FastEmbed) | ✅ CPU |
| Sparse | `Qdrant/bm25` | ✅ CPU |
| Reranker | `BAAI/bge-reranker-base` | ✅ CPU |
| LLM (answers, routing) | **Qwen2.5-3B-Instruct** via Ollama | ✅ CPU/iGPU |
| Graph extraction (batch) | local 3B, or API for speed | ⚙️ API recommended |

Set `LLM_PROVIDER=openai` + `OPENAI_API_KEY` to upgrade any step; nothing else changes.

## Quickstart

```bash
git clone https://github.com/RaidLZ/10-K-Intelligence && cd 10-K-Intelligence
python -m venv .venv && source .venv/bin/activate
make install                       # package + dev/eval extras
cp .env.example .env               # set SEC_IDENTITY; optionally add OPENAI_API_KEY

make up                            # start Qdrant + Neo4j
ollama pull qwen2.5:3b-instruct    # local LLM

make ingest                        # download + parse the 6-company corpus
make index                         # build the hybrid vector index
make graph                         # build the knowledge graph

make ask Q="How did Apple's R&D spend change from 2022 to 2024 vs Microsoft?"
make run                           # FastAPI (:8000) + Streamlit UI
make eval                          # RAGAS + citation + vector-vs-graph report
```

No GPU needed; the MX550-class default runs the small models on CPU. The corpus defaults to
6 mega-caps × 3 years (`TICKERS`/`YEARS` in `.env`) and scales to the S&P 100 by editing that list.

## Results

`make eval` produces `eval/reports/` with the head-to-head. _(Populated after your first run.)_

| Metric | Vector RAG | + GraphRAG router |
|---|---|---|
| Multi-hop accuracy | _tbd_ | _tbd_ |
| Faithfulness (RAGAS) | _tbd_ | _tbd_ |
| Citation correctness | _tbd_ | _tbd_ |
| Hallucination rate | _tbd_ | _tbd_ |

## License
MIT. Built by [Raed Lazreg](https://raidlz.github.io).
