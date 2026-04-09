"""
main.py — FastAPI Application Server for the Cognitive Immune System

This module exposes the CIS pipeline as a REST API with the following endpoints:
    POST /analyze   — Full pipeline analysis (query + context → quarantined/safe claims)
    GET  /memory    — Retrieve the CC-DAG (causal memory) nodes and edges
    GET  /quarantine — List all quarantined claims from the current session
    GET  /stats     — Aggregate statistics across all sessions
    GET  /health    — Health check endpoint

All endpoints are async. CORS is enabled for all origins to support the
Next.js frontend during development and production.

Author: Muhammad Saad, Independent Researcher, Pakistan
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Configure logging BEFORE any imports that use it
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

load_dotenv()

from pipeline import CISPipeline
from database import get_stats as db_get_stats

logger = logging.getLogger("cis.main")

# ---------------------------------------------------------------------------
# Global pipeline instance (initialized at startup)
# ---------------------------------------------------------------------------
pipeline: CISPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    global pipeline
    logger.info("=" * 60)
    logger.info("COGNITIVE IMMUNE SYSTEM — Starting up")
    logger.info("=" * 60)

    db_path = os.getenv("DATABASE_URL", "./cis.db")
    pipeline = CISPipeline(db_path=db_path)

    logger.info("CIS Pipeline ready. Serving requests.")
    yield

    logger.info("CIS shutting down.")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Cognitive Immune System (CIS)",
    description=(
        "Inference-Time Epistemic Quarantine for LLM Hallucination Containment. "
        "A research-grade AI system that intercepts and quarantines contaminated "
        "claims between reasoning steps, preventing downstream contamination. "
        "Paper: 'Inference-Time Epistemic Quarantine: A Governance Primitive "
        "for Contamination Containment in LLM Reasoning Chains' — Muhammad Saad"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for frontend development and deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Request body for the /analyze endpoint."""
    query: str = Field(..., min_length=1, description="The question or prompt to analyze")
    context: str = Field("", description="Optional context to include with the query")


class AnalyzeResponse(BaseModel):
    """Response body for the /analyze endpoint."""
    answer: str
    raw_answer: str = ""
    claims_total: int
    claims_verifiable: int = 0
    contamination_rate: float
    quarantined: list[dict[str, Any]]
    safe: list[dict[str, Any]]
    causal_trace: list[dict[str, Any]]
    latency_ms: int
    session_id: str = ""
    graph_data: dict[str, Any] = {}
    tolerance_set_size: int = 0
    containment_depth_bound: int | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str
    timestamp: str
    version: str = "1.0.0"
    pipeline_ready: bool


class StatsResponse(BaseModel):
    """Response body for the /stats endpoint."""
    total_analyzed: int
    avg_contamination_rate: float
    total_quarantined: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    """Run the full CIS pipeline on a query and return quarantine analysis."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = await pipeline.analyze(
            query=request.query,
            context=request.context,
        )
        return result
    except Exception as e:
        logger.error("Analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/baseline")
async def analyze_baseline(request: AnalyzeRequest) -> dict[str, Any]:
    """Run pure LLM baseline without any CIS interventions."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = await pipeline.analyze_baseline(
            query=request.query,
            context=request.context,
        )
        return result
    except Exception as e:
        logger.error("Baseline analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Baseline analysis failed: {str(e)}")


@app.get("/memory")
async def get_memory() -> dict[str, Any]:
    """Retrieve the CC-DAG (Contamination Causal DAG) structure."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        return pipeline.causal_memory.get_all_nodes_edges()
    except Exception as e:
        logger.error("Memory retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/quarantine")
async def get_quarantine() -> dict[str, Any]:
    """Retrieve all quarantined claims from the current engine state."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        quarantined = pipeline.quarantine_engine.get_quarantined()
        return {
            "quarantined_claims": quarantined,
            "total": len(quarantined),
        }
    except Exception as e:
        logger.error("Quarantine retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats() -> dict[str, Any]:
    """Compute aggregate statistics across all contamination events."""
    try:
        db_path = os.getenv("DATABASE_URL", "./cis.db")
        stats = db_get_stats(db_path=db_path)
        return stats
    except Exception as e:
        logger.error("Stats retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check() -> dict[str, Any]:
    """Health check endpoint — verifies the system is operational."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "pipeline_ready": pipeline is not None,
    }


# ---------------------------------------------------------------------------
# Development server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
