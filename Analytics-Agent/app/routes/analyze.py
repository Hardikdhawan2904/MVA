"""app/routes/analyze.py — POST /analyze, the Analytics Agent's one real
endpoint.

Accepts the uploaded dataset directly (multipart/form-data), rather than a
filesystem path — the difference from the old CLI/subprocess integration
that made Phase 2's orchestrator rewrite (subprocess → httpx call) possible.
"""

import json
import logging

from fastapi import APIRouter, UploadFile, File, Form

from app.agents.analytics_agent.graph import run_analytics_graph
from app.schemas.responses import AnalysisResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analytics"])


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(..., description="Insurance CSV dataset to answer the question against"),
    business_question: str = Form(..., description="The business question to answer"),
    conversation_id: str | None = Form(
        default=None,
        description="Omit for a new conversation (a fresh id is generated and returned in the response). "
                    "Pass back the conversation_id from a prior response to continue it — filter/KPI "
                    "carryover ('what about EMEA?') and the LLM's prior-turn context both depend on this.",
    ),
    ml_readiness: float = Form(
        default=99.75,
        description="ML readiness score from Agent 2 — gates ML model execution vs. the deterministic fallback",
    ),
    llm_readiness: float = Form(
        default=99.75,
        description="LLM readiness score from Agent 2 — gates Groq narration vs. the deterministic template formatter",
    ),
    feature_recommendation: str | None = Form(
        default=None,
        description="Optional JSON string of Agent 2's per-column feature classification for this "
                    "upload — used only to validate the hardcoded ML feature column lists still match "
                    "this dataset's schema; never blocks the request if omitted or malformed.",
    ),
) -> AnalysisResponse:
    try:
        parsed_feature_recommendation = json.loads(feature_recommendation) if feature_recommendation else None
    except json.JSONDecodeError:
        logger.warning("feature_recommendation Form field was not valid JSON — ignoring it.")
        parsed_feature_recommendation = None

    content = await file.read()

    result = run_analytics_graph(
        file_content=content,
        business_question=business_question,
        conversation_id=conversation_id,
        ml_readiness_score=ml_readiness,
        llm_readiness_score=llm_readiness,
        feature_recommendation=parsed_feature_recommendation,
    )

    return AnalysisResponse(**result)
