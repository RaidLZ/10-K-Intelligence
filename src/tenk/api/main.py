"""FastAPI backend exposing the QA pipeline.

    uvicorn tenk.api.main:app --reload --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"What were Apple'\''s biggest risk factors in 2024?"}' | jq
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from tenk.config import settings
from tenk.models import Answer

app = FastAPI(title="10-K Intelligence", version="0.1.0")


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "tickers": settings.tickers, "years": settings.years}


@app.post("/ask", response_model=Answer)
def ask(req: AskRequest) -> Answer:
    # imported lazily so the app starts even before models are pulled
    from tenk.pipeline import answer_question

    return answer_question(req.question)
