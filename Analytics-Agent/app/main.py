"""app/main.py — Analytics Agent FastAPI service.

Fourth service in the pipeline (port 8003), matching Agent 1/2/3's shape:
a thin FastAPI shell over a LangGraph StateGraph (app/agents/analytics_agent/).
Replaces the old CLI (main.py + argparse), invoked as a subprocess by the
orchestrator — that integration is being rewired in Phase 2 of this rewrite
to a normal HTTP call, same as Agent 1 and Agent 2.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import AGENT_IDENTITY
from app.services.boot_trainer import check_and_retrain_if_needed
from app.routes.analyze import router as analyze_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: run the model-freshness check once at startup.
    Previously ran once per subprocess spawn (main.py::AnalyticsAgent.__init__)
    — now the service is long-lived, so once per process lifetime is enough."""
    logger.info(f"Starting {AGENT_IDENTITY.get('name', 'Analytics Agent')}...")
    check_and_retrain_if_needed()
    logger.info("Analytics Agent ready.")
    yield
    logger.info("Shutting down Analytics Agent.")


app = FastAPI(
    title="Insurance Analytics Agent",
    description=(
        "Answers business questions against an uploaded Insurance dataset — "
        "KPI lookups, variance, root cause, trend, forecast, anomaly detection, "
        "and risk segmentation — using a rule engine, DuckDB, and ML models, "
        "narrated by Groq (or a deterministic template formatter when the LLM "
        "readiness score is below threshold or Groq is unavailable)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(analyze_router)


@app.get("/health", tags=["System"])
async def health_check():
    """Liveness only — Agent 3 has no downstream agents to ping."""
    return {"status": "healthy", "service": "Insurance Analytics Agent"}
