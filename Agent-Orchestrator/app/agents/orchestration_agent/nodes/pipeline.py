"""Orchestrator graph nodes.

Mirrors the exact HTTP call sequence and error handling the old linear
run_pipeline() function used (app/routes/pipeline.py, pre-graph version) —
this file does not change what gets called or how errors are classified,
it only re-expresses each stage as a graph node with an explicit conditional
edge deciding whether the pipeline stops or continues, instead of a chain of
early `return`s inside one function body.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
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


def _analytics_agent_python() -> str:
    """ANALYTICS_AGENT_PYTHON overrides; otherwise use this process's own
    interpreter — the Analytics Agent is vendored into mva and shares this
    project's venv, so sys.executable is already the right one."""
    return settings.ANALYTICS_AGENT_PYTHON or sys.executable


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
        result_url = f"{settings.AGENT2_BASE_URL}{settings.AGENT2_API_PREFIX}/profile-runs/{run_id}/result"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(result_url, timeout=settings.REQUEST_TIMEOUT_SECONDS)
            agent2_full_result = _safe_json(resp) if resp.status_code == 200 else state["agent2_body"]
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

    # ── Node 5: Agent 3 (Analytics Agent — CLI subprocess) ──────────────────
    # Vendored at mva/Analytics-Agent (originally github.com/VirenKhapra/
    # Analytics-agent-for-project-3), sharing this project's venv — but still
    # a standalone CLI tool, not an HTTP service like Agent 1/2, so it's run
    # as a subprocess. Answers one business question at a time over the
    # Insurance dataset via DuckDB + ML/LLM tools. Optional and best-effort:
    # skips cleanly outside its scope (non-Insurance domain, no business_question, non-CSV upload) and never
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

        readiness_assessments = (state.get("agent2_full_result") or {}).get("readiness_assessments", [])
        ml_score = next(
            (r.get("score") for r in readiness_assessments if r.get("assessment_type") == "ml_readiness"),
            None,
        )
        llm_score = next(
            (r.get("score") for r in readiness_assessments if r.get("assessment_type") == "llm_readiness"),
            None,
        )
        feature_columns = (state.get("agent2_full_result") or {}).get(
            "feature_recommendation", {}
        ).get("feature_columns") or []

        python_exe = _analytics_agent_python()
        if not Path(python_exe).exists():
            return {
                "agent3_body": {
                    "status": "failed",
                    "reason": f"Analytics Agent python not found at {python_exe} — "
                              f"check ANALYTICS_AGENT_PATH/ANALYTICS_AGENT_PYTHON.",
                },
            }

        tmp_path: str | None = None
        feature_rec_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(state["content"])
                tmp_path = tmp.name

            # Back-date the temp file's mtime so the Analytics Agent's
            # boot_trainer.py — which retrains all 4 ML models (~15s) any
            # time DATASET_PATH looks newer than its shipped .pkl files —
            # doesn't retrain on every single call. A brand-new temp file
            # would otherwise always look "newer" than the committed models.
            old_ts = time.time() - 365 * 86400
            os.utime(tmp_path, (old_ts, old_ts))

            cmd = [python_exe, "main.py", "--query", business_question]
            if ml_score is not None:
                cmd += ["--ml-readiness", str(ml_score)]
            if llm_score is not None:
                cmd += ["--llm-readiness", str(llm_score)]
            if feature_columns:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as fr_tmp:
                    json.dump(feature_columns, fr_tmp)
                    feature_rec_path = fr_tmp.name
                cmd += ["--feature-recommendation-file", feature_rec_path]

            # PYTHONIOENCODING=utf-8: the Analytics Agent prints emoji
            # (e.g. "📋 Execution Plan") — Windows' default cp1252 console
            # encoding crashes on those with UnicodeEncodeError, so this is
            # required for the subprocess to complete at all, not optional.
            env = {**os.environ, "DATASET_PATH": tmp_path, "PYTHONIOENCODING": "utf-8"}

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=settings.ANALYTICS_AGENT_PATH,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=settings.ANALYTICS_AGENT_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {
                    "agent3_body": {
                        "status": "failed",
                        "reason": f"Analytics Agent timed out after {settings.ANALYTICS_AGENT_TIMEOUT_SECONDS}s.",
                    },
                }

            if proc.returncode != 0:
                return {
                    "agent3_body": {
                        "status": "failed",
                        "reason": f"Analytics Agent exited {proc.returncode}: "
                                  f"{stderr.decode(errors='replace')[-2000:]}",
                    },
                }

            return {
                "agent3_body": {
                    "status": "ok",
                    "query": business_question,
                    "ml_readiness_score_used": ml_score,
                    "llm_readiness_score_used": llm_score,
                    "response": stdout.decode(errors="replace"),
                },
            }
        except Exception as e:
            return {"agent3_body": {"status": "failed", "reason": f"Analytics Agent invocation error: {e}"}}
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            if feature_rec_path:
                try:
                    os.unlink(feature_rec_path)
                except OSError:
                    pass

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
