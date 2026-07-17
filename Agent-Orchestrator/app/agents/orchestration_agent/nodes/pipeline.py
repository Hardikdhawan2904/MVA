"""Orchestrator graph nodes.

Mirrors the exact HTTP call sequence and error handling the old linear
run_pipeline() function used (app/routes/pipeline.py, pre-graph version) —
this file does not change what gets called or how errors are classified,
it only re-expresses each stage as a graph node with an explicit conditional
edge deciding whether the pipeline stops or continues, instead of a chain of
early `return`s inside one function body.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx

from app.config import settings
from app.agents.orchestration_agent.state import PipelineState


def _safe_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        return resp.json()
    except ValueError:
        return {"raw_response": resp.text}


def _strip_raw_rows(agent1_body: dict) -> dict:
    """Agent 1's response embeds every row of the uploaded file
    (`dataframe_records`). The orchestrator doesn't need it (Agent 1 already
    persists it) and passing it through balloons the response to tens of MB
    for large files. Drop it before it goes anywhere else."""
    if isinstance(agent1_body, dict) and "dataframe_records" in agent1_body:
        agent1_body = {k: v for k, v in agent1_body.items() if k != "dataframe_records"}
    return agent1_body


def _agent3_skip_reason(
    filename: str, primary_domain: str | None, business_question: str | None,
) -> str | None:
    """None means Agent 3 should run; otherwise the string is why it won't."""
    if primary_domain != "Insurance":
        return f"Analytics Agent only supports the Insurance domain (this dataset classified as '{primary_domain}')."
    if not business_question:
        return "Analytics Agent answers one business question at a time — no business_question was supplied."
    if not filename.lower().endswith(".csv"):
        return "Analytics Agent only reads CSV datasets (DuckDB read_csv_auto) — this upload is not a .csv file."
    return None


async def _get_agent2_result(run_id: str) -> tuple[int, dict[str, Any]]:
    """GET Agent 2's already-persisted full result for run_id — no new
    profiling work, just a read. Returns (status_code, body); raises only
    on a network-level failure (connection refused, timeout, ...) — what a
    non-200 status *means* differs by caller (fetch_agent2_result's node
    falls back to the abbreviated agent2_body on anything but 200, since
    run_id is known-valid there; /pipeline/ask treats a 404 as a genuine
    caller error, since the run_id came from outside this request)."""
    result_url = f"{settings.AGENT2_BASE_URL}{settings.AGENT2_API_PREFIX}/profile-runs/{run_id}/result"
    async with httpx.AsyncClient() as client:
        resp = await client.get(result_url, timeout=settings.REQUEST_TIMEOUT_SECONDS)
    return resp.status_code, _safe_json(resp)


async def _analyze_via_agent3(
    business_question: str,
    filename: str,
    content: bytes,
    content_type: str,
    ml_score: float | None,
    llm_score: float | None,
    feature_columns: list,
    ml_breakdown: dict | None = None,
    llm_breakdown: dict | None = None,
) -> dict[str, Any]:
    """The actual call to Agent 3's POST /analyze. Shared by the full
    pipeline's call_agent3 node and the /pipeline/ask route — same request
    shape either way, they just differ in where ml_score/llm_score/
    feature_columns/breakdowns come from (a run just completed vs.
    re-fetched from Agent 2's persisted result by run_id)."""
    url = f"{settings.ANALYTICS_AGENT_BASE_URL}/analyze"
    data = {"business_question": business_question}
    if ml_score is not None:
        data["ml_readiness"] = ml_score
    if llm_score is not None:
        data["llm_readiness"] = llm_score
    if feature_columns:
        data["feature_recommendation"] = json.dumps(feature_columns)
    if ml_breakdown:
        data["ml_readiness_breakdown"] = json.dumps(ml_breakdown)
    if llm_breakdown:
        data["llm_readiness_breakdown"] = json.dumps(llm_breakdown)

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                files={"file": (filename, content, content_type)},
                data=data,
                timeout=settings.ANALYTICS_AGENT_TIMEOUT_SECONDS,
            )
    except httpx.HTTPError as e:
        return {"status": "failed", "reason": f"Could not reach Analytics Agent at {url}: {e}"}

    agent3_body = _safe_json(resp)

    if resp.status_code != 200:
        return {"status": "failed", "reason": f"Analytics Agent returned {resp.status_code}: {agent3_body}"}

    return agent3_body


def _readiness_and_features(
    agent2_full_result: dict,
) -> tuple[float | None, float | None, list, dict | None, dict | None]:
    """Extract (ml_score, llm_score, feature_columns, ml_breakdown, llm_breakdown)
    from Agent 2's full result — shared by call_agent3 and /pipeline/ask,
    same extraction either way.

    ml_breakdown/llm_breakdown are the full readiness assessment dicts
    (strengths, blocking_issues, evidence, recommendations,
    weight_profile_version) — Agent 2's ReadinessEngine already computes
    these, but until now only the bare score float was ever forwarded to
    Agent 3, so it could never explain *why* a gate passed or failed."""
    readiness_assessments = (agent2_full_result or {}).get("readiness_assessments", [])
    ml_assessment = next(
        (r for r in readiness_assessments if r.get("assessment_type") == "ml_readiness"), None,
    )
    llm_assessment = next(
        (r for r in readiness_assessments if r.get("assessment_type") == "llm_readiness"), None,
    )
    ml_score = ml_assessment.get("score") if ml_assessment else None
    llm_score = llm_assessment.get("score") if llm_assessment else None
    feature_columns = (agent2_full_result or {}).get("feature_recommendation", {}).get("feature_columns") or []
    return ml_score, llm_score, feature_columns, ml_assessment, llm_assessment


class OrchestratorGraphNodes:
    """Each node is a bound async method — grouped in a class purely for
    namespacing, no shared mutable state between requests (a fresh
    httpx.AsyncClient is opened per HTTP call, same lifecycle as before)."""

    # ── Node 1: Agent 1 (Schema Intelligence Layer) ─────────────────────────

    async def call_agent1(self, state: PipelineState) -> dict[str, Any]:
        url = f"{settings.AGENT1_BASE_URL}/upload-dataset"
        params = {"force_reclassify": "true"} if state.get("force_reclassify") else {}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    files={"file": (state["filename"], state["content"], state["content_type"])},
                    params=params,
                    timeout=settings.REQUEST_TIMEOUT_SECONDS,
                )
        except httpx.HTTPError as e:
            return {
                "error_status_code": 502,
                "error_content": {
                    "stage": "agent1_unreachable",
                    "detail": f"Could not reach Agent 1 (Schema Intelligence Layer) at {url}: {e}",
                },
            }

        agent1_body = _strip_raw_rows(_safe_json(resp))

        if resp.status_code == 422:
            return {
                "agent1_status_code": resp.status_code,
                "agent1_body": agent1_body,
                "error_status_code": 422,
                "error_content": {"stage": "agent1_quality_gate", "agent1": agent1_body},
            }

        if resp.status_code != 200:
            return {
                "agent1_status_code": resp.status_code,
                "agent1_body": agent1_body,
                "error_status_code": resp.status_code,
                "error_content": {"stage": "agent1_error", "agent1": agent1_body},
            }

        return {"agent1_status_code": resp.status_code, "agent1_body": agent1_body}

    def route_after_agent1(self, state: PipelineState) -> Literal["stop", "continue"]:
        return "stop" if state.get("error_content") is not None else "continue"

    # ── Node 2: derive Agent 2's inputs from Agent 1's classification ──────

    def extract_domain_and_metadata(self, state: PipelineState) -> dict[str, Any]:
        agent1_body = state["agent1_body"]
        primary_domain = agent1_body.get("business_domain")
        if not primary_domain:
            return {
                "error_status_code": 422,
                "error_content": {
                    "stage": "agent1_classification_missing",
                    "detail": "Agent 1 completed but returned no business_domain to drive Agent 2 with.",
                    "agent1": agent1_body,
                },
            }

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
        return {"primary_domain": primary_domain, "schema_metadata": schema_metadata}

    def route_after_domain(self, state: PipelineState) -> Literal["stop", "continue"]:
        return "stop" if state.get("error_content") is not None else "continue"

    # ── Node 3: Agent 2 (Data Profiling Engine) ─────────────────────────────

    async def call_agent2(self, state: PipelineState) -> dict[str, Any]:
        url = f"{settings.AGENT2_BASE_URL}{settings.AGENT2_API_PREFIX}/profile-runs"
        data = {
            "primary_domain": state["primary_domain"],
            "schema_metadata": json.dumps(state["schema_metadata"]),
            "request_rules": "[]",
        }
        if state.get("sheet_name"):
            data["sheet_name"] = state["sheet_name"]
        if state.get("business_question"):
            data["business_question"] = state["business_question"]
        if state.get("target_column"):
            data["target_column"] = state["target_column"]

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    files={"file": (state["filename"], state["content"], state["content_type"])},
                    data=data,
                    timeout=settings.REQUEST_TIMEOUT_SECONDS,
                )
        except httpx.HTTPError as e:
            return {
                "error_status_code": 502,
                "error_content": {
                    "stage": "agent2_unreachable",
                    "detail": f"Could not reach Agent 2 (Data Profiling Engine) at {url}: {e}",
                },
            }

        agent2_body = _safe_json(resp)

        if resp.status_code != 200:
            note = "Agent 1 completed successfully; Agent 2 rejected the request."
            if resp.status_code == 422 and isinstance(agent2_body, dict) and \
                    (agent2_body.get("error") or {}).get("code") == "UNSUPPORTED_DOMAIN":
                note = (
                    f"Agent 1 classified this dataset as '{state['primary_domain']}', which Agent 2 does "
                    f"not support (only Finance, Payments, Customer, HR, Insurance). No override is available through "
                    f"the orchestrator — call Agent 2 directly with an explicit primary_domain if needed."
                )
            return {
                "agent2_status_code": resp.status_code,
                "agent2_body": agent2_body,
                "error_status_code": resp.status_code,
                "error_content": {
                    "stage": "agent2_error",
                    "agent1": state["agent1_body"],
                    "agent2": agent2_body,
                    "note": note,
                },
            }

        return {
            "agent2_status_code": resp.status_code,
            "agent2_body": agent2_body,
            "run_id": agent2_body["run_id"],
        }

    def route_after_agent2(self, state: PipelineState) -> Literal["stop", "continue"]:
        return "stop" if state.get("error_content") is not None else "continue"

    # ── Node 4: fetch Agent 2's full result ─────────────────────────────────

    async def fetch_agent2_result(self, state: PipelineState) -> dict[str, Any]:
        run_id = state["run_id"]
        try:
            status_code, body = await _get_agent2_result(run_id)
            agent2_full_result = body if status_code == 200 else state["agent2_body"]
        except httpx.HTTPError as e:
            return {
                "error_status_code": 502,
                "error_content": {
                    "stage": "agent2_result_fetch_failed",
                    "agent1": state["agent1_body"],
                    "agent2": state["agent2_body"],
                    "detail": f"Agent 2 run {run_id} completed, but fetching the full result failed: {e}",
                },
            }
        return {"agent2_full_result": agent2_full_result}

    def route_after_fetch(self, state: PipelineState) -> Literal["stop", "continue"]:
        return "stop" if state.get("error_content") is not None else "continue"

    # ── Node 5: Agent 3 (Analytics Agent) ────────────────────────────────────
    # Vendored at mva/Analytics-Agent (originally github.com/VirenKhapra/
    # Analytics-agent-for-project-3), now a FastAPI service (port 8003) like
    # Agent 1/2, called the same way as call_agent2 above. Answers one
    # business question at a time over the Insurance dataset via DuckDB +
    # ML/LLM tools. Optional and best-effort: skips cleanly outside its scope
    # (non-Insurance domain, no business_question, non-CSV upload) and never
    # populates error_content/error_status_code on its own failures — Agent 1
    # and Agent 2 already succeeded by the time this node runs, so a broken
    # or unreachable Agent 3 shouldn't fail an otherwise-successful pipeline.

    async def call_agent3(self, state: PipelineState) -> dict[str, Any]:
        business_question = state.get("business_question")
        skip_reason = _agent3_skip_reason(
            state["filename"], state.get("primary_domain"), business_question,
        )
        if skip_reason:
            return {"agent3_body": {"status": "skipped", "reason": skip_reason}}

        ml_score, llm_score, feature_columns, ml_breakdown, llm_breakdown = _readiness_and_features(
            state.get("agent2_full_result"),
        )
        agent3_body = await _analyze_via_agent3(
            business_question, state["filename"], state["content"], state["content_type"],
            ml_score, llm_score, feature_columns, ml_breakdown, llm_breakdown,
        )
        return {"agent3_body": agent3_body}

    # ── Node 6: finalize ─────────────────────────────────────────────────────

    def finalize(self, state: PipelineState) -> dict[str, Any]:
        return {
            "result": {
                "agent1": state["agent1_body"],
                "agent2": state["agent2_full_result"],
                "agent3": state.get("agent3_body"),
                "primary_domain_used": state["primary_domain"],
            }
        }
