import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.ingestion.chunker import process_markdown_docs
from app.api import app


def test_settings_loaded():
    """Verify Pydantic settings initialize with expected defaults."""
    assert settings.dense_model == "BAAI/bge-small-en-v1.5"
    assert settings.sparse_model == "Qdrant/bm25"
    assert settings.collection_name != ""


def test_markdown_chunker_parses_guides(tmp_path):
    """Verify technical Markdown guides transform into structured Root-Cause Cards."""
    sample_md = tmp_path / "rate-limit.md"
    sample_md.write_text("# API Rate Limits\n\nExceeding 100 req/min returns HTTP 429.", encoding="utf-8")

    cards = process_markdown_docs(docs_dir=str(tmp_path))
    assert len(cards) == 1
    assert cards[0]["title"] == "API Rate Limits"
    assert cards[0]["source_type"] == "documentation"
    assert "HTTP 429" in cards[0]["text"]


@patch("app.api.get_copilot")
def test_api_triage_endpoint(mock_get_copilot):
    """Verify FastAPI /api/v1/triage endpoint contract and response validation."""
    mock_instance = MagicMock()
    mock_instance.langfuse = None
    mock_instance.triage_ticket.return_value = {
        "query": "Webhook 504 timeout",
        "response": "1. **Identified Root Cause**: 10s gateway timeout.",
        "retrieved_sources": [
            {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "score": 0.95,
                "title": "Webhook Timeout & Retry Limits",
                "source_type": "documentation",
                "category": "API & System Reference",
                "priority": "reference",
                "text": "Asynchronous payload delivery times out after 10 seconds.",
                "metadata": {"filename": "webhooks-timeout.md"},
            }
        ],
        "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }
    mock_get_copilot.return_value = mock_instance

    client = TestClient(app)
    res = client.post("/api/v1/triage", json={"query": "Webhook 504 timeout", "limit": 2})

    assert res.status_code == 200
    body = res.json()
    assert body["query"] == "Webhook 504 timeout"
    assert len(body["retrieved_sources"]) == 1
    assert body["token_usage"]["total_tokens"] == 150


def test_api_health_endpoint():
    """Verify health check endpoint for Render/cloud readiness probes."""
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"