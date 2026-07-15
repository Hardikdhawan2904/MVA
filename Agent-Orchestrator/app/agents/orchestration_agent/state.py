"""Typed state for the orchestrator LangGraph — threads the pipeline handoff
between Agent 1 (Schema Intelligence Layer) and Agent 2 (Data Profiling Engine)
between nodes, replacing the old sequential-with-early-return function body.
"""

from typing import Any, TypedDict


class PipelineState(TypedDict, total=False):
    # ── Inputs ──────────────────────────────────────────────────────────────
    filename: str
    content_type: str
    content: bytes
    sheet_name: str | None
    force_reclassify: bool
    business_question: str | None
    target_column: str | None

    # ── Agent 1 ─────────────────────────────────────────────────────────────
    agent1_status_code: int
    agent1_body: dict[str, Any]

    # ── Derived from Agent 1's classification ──────────────────────────────
    primary_domain: str | None
    schema_metadata: dict[str, Any]

    # ── Agent 2 ─────────────────────────────────────────────────────────────
    agent2_status_code: int
    agent2_body: dict[str, Any]
    run_id: str
    agent2_full_result: dict[str, Any]

    # ── Output ──────────────────────────────────────────────────────────────
    result: dict[str, Any]
    error_status_code: int
    error_content: dict[str, Any]
