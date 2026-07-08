"""
Pipeline route — orchestrates the handoff between Agent 1 (Schema Intelligence Layer)
and Agent 2 (MVA Data Profiling Engine). Sequential HTTP calls, no branching logic;
each future agent added to the pipeline becomes another step in this same sequence.
"""

import json

import httpx
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

from app.config import settings

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


def _safe_json(resp: httpx.Response):
    try:
        return resp.json()
    except ValueError:
        return {"raw_response": resp.text}


def _strip_raw_rows(agent1_body: dict) -> dict:
    """
    Agent 1's response embeds every row of the uploaded file (`dataframe_records`),
    intended for callers that need the raw data directly. The orchestrator doesn't
    need it (Agent 1 already persists it) and passing it through balloons the
    response to tens of MB for large files, which browser-based tools like Swagger
    struggle to render. Drop it before returning anything to the caller.
    """
    if isinstance(agent1_body, dict) and "dataframe_records" in agent1_body:
        agent1_body = {k: v for k, v in agent1_body.items() if k != "dataframe_records"}
    return agent1_body


async def _call_agent1(
    client: httpx.AsyncClient, filename: str, content_type: str, content: bytes, force_reclassify: bool = False
):
    url = f"{settings.AGENT1_BASE_URL}/upload-dataset"
    try:
        resp = await client.post(
            url,
            files={"file": (filename, content, content_type)},
            params={"force_reclassify": "true"} if force_reclassify else {},
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
        )
        return resp, None
    except httpx.HTTPError as e:
        return None, JSONResponse(
            status_code=502,
            content={
                "stage": "agent1_unreachable",
                "detail": f"Could not reach Agent 1 (Schema Intelligence Layer) at {url}: {e}",
            },
        )


async def _call_agent2(
    client: httpx.AsyncClient,
    filename: str,
    content_type: str,
    content: bytes,
    primary_domain: str,
    schema_metadata_json: str,
    sheet_name: str | None = None,
):
    url = f"{settings.AGENT2_BASE_URL}{settings.AGENT2_API_PREFIX}/profile-runs"
    data = {
        "primary_domain": primary_domain,
        "schema_metadata": schema_metadata_json,
        "request_rules": "[]",
    }
    if sheet_name:
        data["sheet_name"] = sheet_name
    try:
        resp = await client.post(
            url,
            files={"file": (filename, content, content_type)},
            data=data,
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
        )
        return resp, None
    except httpx.HTTPError as e:
        return None, JSONResponse(
            status_code=502,
            content={
                "stage": "agent2_unreachable",
                "detail": f"Could not reach Agent 2 (Data Profiling Engine) at {url}: {e}",
            },
        )


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

    async with httpx.AsyncClient() as client:
        resp1, error = await _call_agent1(client, filename, content_type, content, force_reclassify)
        if error:
            return error

        agent1_body = _strip_raw_rows(_safe_json(resp1))

        if resp1.status_code == 422:
            return JSONResponse(
                status_code=422,
                content={"stage": "agent1_quality_gate", "agent1": agent1_body},
            )

        if resp1.status_code != 200:
            return JSONResponse(
                status_code=resp1.status_code,
                content={"stage": "agent1_error", "agent1": agent1_body},
            )

        primary_domain = agent1_body.get("business_domain")
        if not primary_domain:
            return JSONResponse(
                status_code=422,
                content={
                    "stage": "agent1_classification_missing",
                    "detail": "Agent 1 completed but returned no business_domain to drive Agent 2 with.",
                    "agent1": agent1_body,
                },
            )

        column_descriptions = agent1_body.get("column_descriptions") or {}
        schema_metadata = {
            "columns": [
                {
                    "column_name": col_name,
                    "description": description,
                    "mandatory": False,
                    "expected_unique": False,
                }
                for col_name, description in column_descriptions.items()
            ]
        }

        resp2, error = await _call_agent2(
            client, filename, content_type, content, primary_domain, json.dumps(schema_metadata), sheet_name
        )
        if error:
            return error

        agent2_body = _safe_json(resp2)

        if resp2.status_code != 200:
            note = "Agent 1 completed successfully; Agent 2 rejected the request."
            if resp2.status_code == 422 and isinstance(agent2_body, dict) and \
                    (agent2_body.get("error") or {}).get("code") == "UNSUPPORTED_DOMAIN":
                note = (
                    f"Agent 1 classified this dataset as '{primary_domain}', which Agent 2 does not "
                    f"support (only Finance, Payments, Customer, HR). No override is available through "
                    f"the orchestrator — call Agent 2 directly with an explicit primary_domain if needed."
                )
            return JSONResponse(
                status_code=resp2.status_code,
                content={
                    "stage": "agent2_error",
                    "agent1": agent1_body,
                    "agent2": agent2_body,
                    "note": note,
                },
            )

        run_id = agent2_body["run_id"]
        result_url = f"{settings.AGENT2_BASE_URL}{settings.AGENT2_API_PREFIX}/profile-runs/{run_id}/result"
        try:
            resp3 = await client.get(result_url, timeout=settings.REQUEST_TIMEOUT_SECONDS)
            agent2_full_result = _safe_json(resp3) if resp3.status_code == 200 else agent2_body
        except httpx.HTTPError as e:
            return JSONResponse(
                status_code=502,
                content={
                    "stage": "agent2_result_fetch_failed",
                    "agent1": agent1_body,
                    "agent2": agent2_body,
                    "detail": f"Agent 2 run {run_id} completed, but fetching the full result failed: {e}",
                },
            )

    return {
        "agent1": agent1_body,
        "agent2": agent2_full_result,
        "primary_domain_used": primary_domain,
    }
