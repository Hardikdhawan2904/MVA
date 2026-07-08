import logging

import httpx
from fastapi import FastAPI

from app.config import settings
from app.routes.pipeline import router as pipeline_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(
    title="Agent Orchestrator",
    description=(
        "Coordinates the multi-agent data pipeline: uploads a dataset to the "
        "Schema Intelligence Layer, feeds its output into the Data Profiling "
        "Engine, and returns both results together. Designed to grow as more "
        "agents are added to the pipeline."
    ),
    version="1.0.0",
)

app.include_router(pipeline_router)


async def _ping(client: httpx.AsyncClient, name: str, url: str) -> dict:
    try:
        resp = await client.get(url, timeout=5.0)
        return {"name": name, "url": url, "reachable": resp.status_code == 200}
    except httpx.HTTPError as e:
        return {"name": name, "url": url, "reachable": False, "error": str(e)}


@app.get("/health")
async def health_check():
    """Health check — also pings each downstream agent so it's obvious what's actually reachable."""
    async with httpx.AsyncClient() as client:
        agents = [
            await _ping(client, "agent1_schema_intelligence", f"{settings.AGENT1_BASE_URL}/health"),
            await _ping(
                client,
                "agent2_data_profiling",
                f"{settings.AGENT2_BASE_URL}{settings.AGENT2_API_PREFIX}/health",
            ),
        ]
    return {
        "status": "healthy",
        "service": "Agent Orchestrator",
        "agents": agents,
    }
