.DEFAULT_GOAL := help
PY ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with all runtime + dev extras
	$(PY) -m pip install -e ".[all,dev]"

up:  ## Start Qdrant + Neo4j
	docker compose up -d

down:  ## Stop infra
	docker compose down

ingest:  ## Download filings + parse narrative sections
	$(PY) -m tenk.cli ingest

index:  ## Build the Qdrant hybrid vector index
	$(PY) -m tenk.cli index

index-resume:  ## Index only chunks not already stored (resume a partial run / add filings)
	$(PY) -m tenk.cli index --resume

graph:  ## Build the Neo4j knowledge graph (API recommended for extraction)
	$(PY) -m tenk.cli graph

graph-facts:  ## Load XBRL financial facts only — fast, no LLM (metric-comparison route)
	$(PY) -m tenk.cli graph --skip-extraction

ask:  ## One-off question, e.g.  make ask Q="How did Apple's R&D change 2022->2024?"
	$(PY) -m tenk.cli ask "$(Q)"

api:  ## Run the FastAPI backend
	uvicorn tenk.api.main:app --reload --port 8000

ui:  ## Run the Streamlit UI
	$(PY) -m streamlit run src/tenk/ui/app.py

run: ## Run API + UI together
	$(MAKE) -j2 api ui

eval:  ## Run the evaluation harness (RAGAS + citation + vector-vs-graph)
	$(PY) -m tenk.cli eval

smoke:  ## Network-free smoke test
	pytest -q

lint:  ## Lint
	ruff check src tests

.PHONY: help install up down ingest index index-resume graph graph-facts ask api ui run eval smoke lint
