from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import TOP_K
from app.retrieval import retrieve
from app.llm import ConfigError, LLMError
from app.rag import answer_question

app = FastAPI(title="Clinic Assistant")


class SearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=TOP_K, ge=1, le=10)
    raw: bool = False   # True = skip the relevance filter (debugging)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    """Part 1: retrieval only. Part 2 adds /ask, which sends these chunks to an LLM."""
    try:
        hits = retrieve(req.question, req.k, apply_filter=not req.raw)
    except Exception as exc:  # most likely: index not built yet
        raise HTTPException(status_code=503,
                            detail="Index not available. Run: python -m app.ingest") from exc
    return {"question": req.question, "results": hits}

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)

@app.post("/ask")
def ask(req: AskRequest):
    """Full pipeline: guardrails -> retrieval -> grounded LLM answer with citations."""
    try:
        result = answer_question(req.question)
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=f"LLM not configured: {exc}") from exc
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc
    except Exception as exc:  # most likely: index not built yet
        raise HTTPException(status_code=503,
                            detail="Index not available. Run: python -m app.ingest") from exc
    return {"question": req.question, **result}