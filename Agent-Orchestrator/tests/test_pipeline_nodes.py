"""tests/test_pipeline_nodes.py — unit tests for pure-data helpers in
app/agents/orchestration_agent/nodes/pipeline.py.

Covers:
- _readiness_and_features(): used to extract only the bare score float from
  Agent 2's readiness_assessments, discarding strengths/blocking_issues/
  evidence — the fix threads the full assessment dict through too, so Agent
  3's execution_trace can explain a readiness gate, not just report its score.
- _canonicalize_domain(): Agent 1's classification is open-vocabulary LLM
  output (e.g. "Human Resources"); Agent 2's domain config lookup is an
  exact, case-sensitive match against 5 fixed strings (Finance/Payments/
  Customer/HR/Insurance). Without this, a correct classification could fail
  Agent 2's check purely on wording.
- _dataset_context_fields(): Agent 3 redesign Phase 0 (plan
  "zany-giggling-crayon", Decision A1) — extracts Agent 2's per-column
  semantic classification (column_profiles/hierarchy/charts/
  feature_recommendation) so it can be forwarded to Agent 3, which
  previously received only readiness scores/breakdowns.
- Agent3Capabilities / _agent3_skip_reason (Phase 4.5): Agent 3 became
  domain-agnostic in Phase 4, but the Orchestrator's gate still assumed
  "Agent 3 = Insurance-only" until now. This phase replaces the domain
  check with a real capability check (file type + business_question) and
  confirms _analyze_via_agent3's skipped/failed/error 3-way status
  distinction survives non-Insurance domains actually reaching Agent 3.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import pytest

from app.agents.orchestration_agent.nodes.pipeline import (
    AGENT3_CAPABILITIES, Agent3Capabilities, _agent3_skip_reason, _analyze_via_agent3,
    _canonicalize_domain, _dataset_context_fields, _readiness_and_features,
)


def _agent2_result(ml_score=82.0, llm_score=95.0):
    return {
        "readiness_assessments": [
            {
                "assessment_type": "ml_readiness",
                "score": ml_score,
                "status": "ready",
                "strengths": [{"code": "HIGH_COMPLETENESS", "value": 0.95}],
                "blocking_issues": [],
                "recommendations": [],
                "evidence": [
                    {"dimension": "completeness", "value": 0.95},
                    {"dimension": "feature_coverage", "value": 1.0},
                    {"dimension": "data_freshness", "value": 0.55},
                ],
                "weight_profile_version": "ml-v1",
            },
            {
                "assessment_type": "llm_readiness",
                "score": llm_score,
                "status": "ready",
                "strengths": [],
                "blocking_issues": [],
                "recommendations": [],
                "evidence": [{"dimension": "description_coverage", "value": 0.9}],
                "weight_profile_version": "llm-v1",
            },
        ],
        "feature_recommendation": {"feature_columns": [{"column": "x", "usefulness": "high"}]},
    }


def test_readiness_and_features_returns_scores_and_full_breakdowns():
    ml_score, llm_score, feature_columns, ml_breakdown, llm_breakdown = _readiness_and_features(
        _agent2_result(),
    )

    assert ml_score == 82.0
    assert llm_score == 95.0
    assert feature_columns == [{"column": "x", "usefulness": "high"}]

    assert ml_breakdown["assessment_type"] == "ml_readiness"
    assert ml_breakdown["evidence"][2] == {"dimension": "data_freshness", "value": 0.55}
    assert llm_breakdown["assessment_type"] == "llm_readiness"


def test_readiness_and_features_handles_missing_assessments():
    ml_score, llm_score, feature_columns, ml_breakdown, llm_breakdown = _readiness_and_features(
        {"readiness_assessments": []},
    )
    assert ml_score is None
    assert llm_score is None
    assert feature_columns == []
    assert ml_breakdown is None
    assert llm_breakdown is None


def test_readiness_and_features_handles_none_input():
    ml_score, llm_score, feature_columns, ml_breakdown, llm_breakdown = _readiness_and_features(None)
    assert ml_score is None
    assert llm_score is None
    assert feature_columns == []
    assert ml_breakdown is None
    assert llm_breakdown is None


# ── _canonicalize_domain ────────────────────────────────────────────────────

def test_canonicalize_domain_maps_human_resources_wording_to_hr():
    assert _canonicalize_domain("Human Resources") == "HR"
    assert _canonicalize_domain("human resources") == "HR"
    assert _canonicalize_domain("HR") == "HR"


def test_canonicalize_domain_normalizes_case_for_exact_matches():
    # Agent 2's check is case-sensitive — "insurance"/"finance" lowercase
    # would fail it today exactly like a genuine synonym would.
    assert _canonicalize_domain("insurance") == "Insurance"
    assert _canonicalize_domain("FINANCE") == "Finance"


def test_canonicalize_domain_handles_payments_and_customer_synonyms():
    assert _canonicalize_domain("payment") == "Payments"
    assert _canonicalize_domain("crm") == "Customer"


def test_canonicalize_domain_handles_ecommerce_as_customer_synonym():
    # Agent 1's open-vocabulary classifier labels CRM/loyalty-shaped
    # datasets "E-commerce" as often as "Customer" — same underlying
    # domain, different wording, same treatment as the "crm" synonym above.
    assert _canonicalize_domain("E-commerce") == "Customer"
    assert _canonicalize_domain("ecommerce") == "Customer"


def test_canonicalize_domain_passes_through_unrecognized_domains_unchanged():
    # No synonym for "Healthcare" -- Agent 2's own UNSUPPORTED_DOMAIN error
    # is still the correct outcome, not a silently wrong mapping.
    assert _canonicalize_domain("Healthcare") == "Healthcare"


# ── _dataset_context_fields ─────────────────────────────────────────────────

def test_dataset_context_fields_extracts_all_four_when_present():
    agent2_result = {
        "column_profiles": [{"physical_name": "gross_written_premium_actual", "candidate_column_role": "metric"}],
        "hierarchy": {"status": "accepted", "template_key": "insurance_geographic"},
        "charts": [{"chart_key": "premium_trend", "chart_type": "line"}],
        "feature_recommendation": {"target_column": "underwriting_result_actual", "problem_type": "regression"},
    }
    column_profiles, hierarchy, charts, full_feature_recommendation = _dataset_context_fields(agent2_result)
    assert column_profiles == agent2_result["column_profiles"]
    assert hierarchy == agent2_result["hierarchy"]
    assert charts == agent2_result["charts"]
    assert full_feature_recommendation == agent2_result["feature_recommendation"]


def test_dataset_context_fields_handles_missing_keys():
    column_profiles, hierarchy, charts, full_feature_recommendation = _dataset_context_fields({})
    assert column_profiles is None
    assert hierarchy is None
    assert charts is None
    assert full_feature_recommendation is None


def test_dataset_context_fields_handles_none_input():
    column_profiles, hierarchy, charts, full_feature_recommendation = _dataset_context_fields(None)
    assert column_profiles is None
    assert hierarchy is None
    assert charts is None
    assert full_feature_recommendation is None


def test_dataset_context_fields_treats_empty_lists_and_dicts_as_absent():
    # A quality-gate "lightweight" Agent 2 run can legitimately return
    # empty lists/dicts for these fields (see finalize_lightweight in
    # MVA-use-case-latest-one) — those should forward as None, not as
    # empty-but-present values that look like real (if vacuous) data.
    agent2_result = {"column_profiles": [], "hierarchy": {}, "charts": [], "feature_recommendation": {}}
    column_profiles, hierarchy, charts, full_feature_recommendation = _dataset_context_fields(agent2_result)
    assert column_profiles is None
    assert hierarchy is None
    assert charts is None
    assert full_feature_recommendation is None


# ── Agent3Capabilities / _agent3_skip_reason (Phase 4.5) ────────────────────

def test_agent3_capabilities_defaults_match_agent3s_real_constraints():
    assert AGENT3_CAPABILITIES.supported_file_extensions == frozenset({".csv"})
    assert AGENT3_CAPABILITIES.requires_business_question is True


def test_unsupported_file_reason_flags_non_csv():
    caps = Agent3Capabilities()
    assert caps.unsupported_file_reason("data.xlsx") is not None
    assert caps.unsupported_file_reason("data.csv") is None
    assert caps.unsupported_file_reason("DATA.CSV") is None  # case-insensitive


def test_missing_question_reason_flags_empty_question():
    caps = Agent3Capabilities()
    assert caps.missing_question_reason(None) is not None
    assert caps.missing_question_reason("") is not None
    assert caps.missing_question_reason("Why did X happen?") is None


def test_agent3_skip_reason_no_longer_gates_on_domain():
    """The core Phase 4.5 behavior change: a non-Insurance domain used to
    be the #1 skip reason. It's no longer a parameter at all — the
    function only ever inspects file type and business_question now."""
    assert _agent3_skip_reason("data.csv", "Why did X happen?") is None


@pytest.mark.parametrize("filename,question,should_skip", [
    ("data.xlsx", "Why did X happen?", True),   # unsupported file type
    ("data.csv", None, True),                    # missing business_question
    ("data.csv", "", True),                       # empty business_question
    ("data.csv", "Why did X happen?", False),     # both satisfied
])
def test_agent3_skip_reason_capability_matrix(filename, question, should_skip):
    reason = _agent3_skip_reason(filename, question)
    assert (reason is not None) == should_skip


# ── _analyze_via_agent3: skipped / failed / Agent 3's own error stay distinct ─
#
# Three genuinely different states, all already implemented before Phase
# 4.5 — this just confirms the distinction survives non-Insurance domains
# actually reaching Agent 3 now, rather than assuming it does.

def _run(coro):
    return asyncio.run(coro)


def test_analyze_via_agent3_unreachable_reports_failed_not_skipped_or_error():
    with patch.object(httpx.AsyncClient, "post", new=AsyncMock(side_effect=httpx.ConnectError("refused"))):
        result = _run(_analyze_via_agent3("Why?", "data.csv", b"a,b\n1,2", "text/csv", 90.0, 90.0, []))
    assert result["status"] == "failed"
    assert "Could not reach Analytics Agent" in result["reason"]


def test_analyze_via_agent3_non_200_reports_failed():
    mock_resp = httpx.Response(500, json={"detail": "internal error"}, request=httpx.Request("POST", "http://x"))
    with patch.object(httpx.AsyncClient, "post", new=AsyncMock(return_value=mock_resp)):
        result = _run(_analyze_via_agent3("Why?", "data.csv", b"a,b\n1,2", "text/csv", 90.0, 90.0, []))
    assert result["status"] == "failed"
    assert "returned 500" in result["reason"]


def test_analyze_via_agent3_internal_error_body_passes_through_unchanged():
    """Agent 3 itself catching an internal exception (e.g. a plugin
    construction failure) and returning HTTP 200 with status: "error" is
    NOT the Orchestrator's call to make — it passes the body through
    verbatim, distinct from both "skipped" (never invoked) and "failed"
    (this Orchestrator's own network/HTTP-level verdict)."""
    error_body = {"status": "error", "query": "Why?", "response": "Analytics Agent failed: boom", "conversation_id": "x"}
    mock_resp = httpx.Response(200, json=error_body, request=httpx.Request("POST", "http://x"))
    with patch.object(httpx.AsyncClient, "post", new=AsyncMock(return_value=mock_resp)):
        result = _run(_analyze_via_agent3("Why?", "data.csv", b"a,b\n1,2", "text/csv", 90.0, 90.0, []))
    assert result == error_body
    assert result["status"] == "error"
