"""
Pipeline route — orchestrates the handoff between Agent 1 (Schema Intelligence Layer)
and Agent 2 (MVA Data Profiling Engine) via a LangGraph StateGraph (app/agents/orchestration_agent/graph.py).
Each future agent added to the pipeline becomes another node in that same graph.
"""

from fastapi import APIRouter, UploadFile, File, Form

from app.agents.orchestration_agent.graph import run_orchestrator_pipeline

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


@router.post("/run")
async def run_pipeline(
    file: UploadFile = File(..., description="CSV or Excel dataset to run through the full pipeline"),
    sheet_name: str | None = Form(
        default=None,
        description="Required only when the XLSX workbook has more than one non-empty sheet; "
                    "names which sheet Agent 2 should load.",
    ),
    force_reclassify: bool = Form(
        default=False,
        description="If this filename already exists in Agent 1's catalog, re-run its LLM column "
                    "descriptions and domain classification instead of reusing the original result.",
    ),
):
    """
    Runs a dataset through the full agent pipeline:

    1. Agent 1 (Schema Intelligence Layer) — quality gate, metadata extraction,
       LLM column descriptions, business domain classification.
    2. Agent 2 (MVA Data Profiling Engine) — deep structural profiling, quality/readiness
       scoring, chart generation. Seeded with Agent 1's column descriptions and its own
       primary_domain, taken directly from Agent 1's classification (business_domain) —
       no domain input is accepted from the caller.

    If Agent 1 rejects the file at its quality gate, the pipeline stops there —
    Agent 2 is never called. If Agent 1's classification isn't one of Agent 2's
    supported domains (Finance, Payments, Customer, HR), the pipeline stops with a
    clear error rather than guessing.
    """
    content = await file.read()
    filename = file.filename or "upload"
    content_type = file.content_type or "application/octet-stream"

    return await run_orchestrator_pipeline(
        filename=filename,
        content_type=content_type,
        content=content,
        sheet_name=sheet_name,
        force_reclassify=force_reclassify,
    )
