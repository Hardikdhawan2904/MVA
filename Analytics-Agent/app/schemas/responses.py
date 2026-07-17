"""app/schemas/responses.py — API response contract for POST /analyze."""

from pydantic import BaseModel


class AnalysisResponse(BaseModel):
    """Response body for POST /analyze. Field names match what
    Agent-Orchestrator's call_agent3 node already expects from the
    subprocess-era CLI output, so the orchestrator-side change needed to
    consume this over HTTP instead is minimal."""

    status: str
    query: str
    response: str
    conversation_id: str | None = None
    ml_readiness_score_used: float | None = None
    llm_readiness_score_used: float | None = None
