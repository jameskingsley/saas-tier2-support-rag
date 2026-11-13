import sys
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.generation.generator import SupportCopilotGenerator

app = FastAPI(
    title="Tier-2 B2B SaaS Support Triage Copilot API",
    description="Hybrid RAG + Groq inference microservice for automated support ticket triage.",
    version="1.0.0",
)

# Lazy-loaded singleton for the copilot backend
_copilot: SupportCopilotGenerator | None = None


def get_copilot() -> SupportCopilotGenerator:
    global _copilot
    if _copilot is None:
        _copilot = SupportCopilotGenerator()
    return _copilot


class TriageRequest(BaseModel):
    query: str = Field(..., min_length=5, description="Customer support ticket description or technical inquiry.")
    limit: int = Field(default=4, ge=1, le=10, description="Number of knowledge base cards to retrieve.")


class SourceCard(BaseModel):
    id: str
    score: float
    title: str
    source_type: str
    category: str
    priority: str
    text: str
    metadata: dict[str, Any] = {}


class TriageResponse(BaseModel):
    query: str
    response: str
    retrieved_sources: list[SourceCard]
    token_usage: dict[str, int]


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "tier2-support-triage-copilot"}


@app.post("/api/v1/triage", response_model=TriageResponse)
def triage_ticket(payload: TriageRequest):
    try:
        copilot = get_copilot()
        result = copilot.triage_ticket(user_query=payload.query, limit=payload.limit)
        if copilot.langfuse:
            copilot.langfuse.flush()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Triage pipeline error: {str(e)}")