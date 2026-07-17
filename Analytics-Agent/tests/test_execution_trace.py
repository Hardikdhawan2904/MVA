"""tests/test_execution_trace.py — Tests for the execution trace/summary
(app/agents/analytics_agent/graph.py::_build_execution_trace).

Most cases are tested by feeding _build_execution_trace crafted final_state
dicts directly — it's a pure function over plain data, so this is faster
and more deterministic than driving full graph runs (and, unlike a live
Groq call, doesn't depend on whether the API happens to be rate-limited
when the suite runs). A handful of real end-to-end cases against the live
dataset confirm the wiring itself (state.py fields actually get threaded
through by narrate()/record_memory()), matching this codebase's practice
of testing against real dependencies rather than mocking everything.
"""

import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.agents.analytics_agent.graph import _build_execution_trace, run_analytics_graph
from app.config import ML_READINESS_THRESHOLD, LLM_READINESS_THRESHOLD

_DATASET = Path(r"C:\Users\dhawa\mva\Schema-Intelligence-Layer\test_data\insurance_variance_data_native.csv")


# ── _build_execution_trace — pure-function unit tests ────────────────────────

def test_ml_gated_happy_path_reports_model_and_passed_gate():
    state = {
        "intent": "forecast",
        "kpi_name": "underwriting_result",
        "evidence": {"kpi": "Underwriting Result"},  # no ml_readiness_blocked key -> model path ran
        "response": "...",
        "ml_readiness_score": 92.0,
        "llm_engine_used": "Groq",
        "llm_readiness_score": 95.0,
        "tools_used": ["RuleEngine", "SQLTool", "MLTool→Prophet/LightGBM", "ExplanationTool"],
    }
    trace, summary = _build_execution_trace(state, elapsed_seconds=1.23)

    ml_step = next(s for s in trace if s["step"] == "forecast")
    assert ml_step["engine"] == "Prophet"
    assert ml_step["gate"] == {"name": "ml_readiness", "score": 92.0, "threshold": ML_READINESS_THRESHOLD, "passed": True}

    assert summary["ml_engine"] == "Prophet"
    assert summary["fallback_used"] is False


def test_ml_gated_fallback_path_reports_fallback_engine_and_failed_gate():
    state = {
        "intent": "anomaly",
        "evidence": {
            "ml_readiness_blocked": True,
            "ml_readiness_score": 40.0,
            "fallback_reason": "ML readiness score (40.00%) below threshold (75.0%).",
            "fallback_applied": "Deterministic Z-Score ratio anomaly detection (analytics_tool.rank)",
        },
        "response": "...",
        "ml_readiness_score": 40.0,
        "llm_engine_used": "Template Formatter",
        "llm_readiness_score": 30.0,
    }
    trace, summary = _build_execution_trace(state, elapsed_seconds=0.5)

    ml_step = next(s for s in trace if s["step"] == "anomaly")
    assert ml_step["engine"] == "Deterministic Z-Score ratio anomaly detection (analytics_tool.rank)"
    assert ml_step["gate"]["passed"] is False
    assert ml_step["reason"] == state["evidence"]["fallback_reason"]

    assert summary["fallback_used"] is True


def test_non_ml_gated_intent_has_no_gate():
    state = {
        "intent": "show_kpi",
        "evidence": {"kpi": "Gross Written Premium", "actual": 100.0},
        "response": "...",
        "llm_engine_used": "Groq",
        "llm_readiness_score": 99.0,
    }
    trace, summary = _build_execution_trace(state, elapsed_seconds=0.2)

    step = next(s for s in trace if s["step"] == "show_kpi")
    assert step["engine"] == "AnalyticsTool"
    assert step["gate"] is None
    assert summary["ml_engine"] is None


def test_root_cause_notes_xgboost_corroboration_when_present():
    state = {
        "intent": "root_cause",
        "evidence": {"kpi": "x", "ml_predicted_driver": "claim_frequency_variance", "ml_confidence": 0.8},
        "response": "...",
        "llm_engine_used": "Groq",
        "llm_readiness_score": 99.0,
    }
    trace, _ = _build_execution_trace(state, elapsed_seconds=0.2)
    step = next(s for s in trace if s["step"] == "root_cause")
    assert step["engine"] == "RootCauseTool"
    assert "XGBoost" in step["reason"]


def test_root_cause_without_ml_evidence_does_not_mention_xgboost():
    state = {
        "intent": "root_cause",
        "evidence": {"kpi": "x"},
        "response": "...",
        "llm_engine_used": "Groq",
        "llm_readiness_score": 99.0,
    }
    trace, _ = _build_execution_trace(state, elapsed_seconds=0.2)
    step = next(s for s in trace if s["step"] == "root_cause")
    assert "XGBoost" not in step["reason"]


def test_narration_groq_error_is_distinct_from_never_attempted():
    """The core nuance this feature exists to capture: a passed llm_readiness
    gate whose Groq call itself failed must not be reported the same way as
    a gate that never passed in the first place."""
    state = {
        "intent": "show_kpi",
        "evidence": {"kpi": "x"},
        "response": "...",
        "llm_engine_used": "Template Formatter (Groq error)",
        "llm_readiness_score": 95.0,  # gate passed
    }
    trace, summary = _build_execution_trace(state, elapsed_seconds=0.2)
    narration = next(s for s in trace if s["step"] == "narration")
    assert narration["gate"]["passed"] is True
    assert narration["engine"] == "Template Formatter (Groq error)"
    assert "Groq call itself failed" in narration["reason"]
    assert summary["fallback_used"] is True


def test_narration_readiness_too_low_is_a_clean_never_attempted():
    state = {
        "intent": "show_kpi",
        "evidence": {"kpi": "x"},
        "response": "...",
        "llm_engine_used": "Template Formatter",
        "llm_readiness_score": 20.0,  # gate failed
    }
    trace, summary = _build_execution_trace(state, elapsed_seconds=0.2)
    narration = next(s for s in trace if s["step"] == "narration")
    assert narration["gate"]["passed"] is False
    assert "below the" in narration["reason"]
    assert summary["fallback_used"] is True


def test_early_exit_response_only_path_produces_short_trace():
    state = {
        "intent": "variance",
        "kpi_name": "nonexistent_kpi",
        "evidence": None,  # handler returned {"response": ...} directly, no evidence dict
        "response": "KPI 'nonexistent_kpi' not found in Rule Engine.",
    }
    trace, summary = _build_execution_trace(state, elapsed_seconds=0.05)

    assert len(trace) == 2  # intent_detection + the early-exit entry, nothing else
    assert trace[1]["step"] == "variance"
    assert trace[1]["engine"] is None
    assert trace[1]["gate"] is None
    assert trace[1]["reason"] == state["response"]
    assert summary["ml_engine"] is None
    assert summary["narration_engine"] is None
    assert summary["fallback_used"] is False


def test_execution_time_is_recorded():
    _, summary = _build_execution_trace({"intent": "show_kpi", "evidence": None, "response": "x"}, elapsed_seconds=1.5)
    assert summary["execution_time_seconds"] == 1.5


# ── Real end-to-end wiring check ──────────────────────────────────────────────

pytestmark_dataset = pytest.mark.skipif(
    not _DATASET.exists(), reason=f"Insurance test dataset not found at {_DATASET}",
)


@pytestmark_dataset
def test_end_to_end_forecast_happy_path_reports_prophet():
    with open(_DATASET, "rb") as f:
        content = f.read()
    result = run_analytics_graph(
        file_content=content,
        business_question="Forecast underwriting result for next 6 months",
        conversation_id=str(uuid.uuid4()),
        ml_readiness_score=99.75,
        llm_readiness_score=99.75,
    )
    assert result["status"] == "ok"
    # Only asserting the ML engine selection here — narration engine depends
    # on Groq's real-time availability (rate limits, outages), which this
    # test isn't about and shouldn't be flaky over. See the narration_engine
    # / fallback_used unit tests above for that behavior, tested deterministically.
    assert result["execution_summary"]["ml_engine"] == "Prophet"
    assert result["execution_summary"]["execution_time_seconds"] > 0
    assert any(step["step"] == "forecast" and step["engine"] == "Prophet" for step in result["execution_trace"])


@pytestmark_dataset
def test_end_to_end_forecast_fallback_path_reports_historical_trend():
    with open(_DATASET, "rb") as f:
        content = f.read()
    result = run_analytics_graph(
        file_content=content,
        business_question="Forecast underwriting result for next 6 months",
        conversation_id=str(uuid.uuid4()),
        ml_readiness_score=40.0,
        llm_readiness_score=99.75,
    )
    assert result["status"] == "ok"
    assert result["execution_summary"]["fallback_used"] is True
    assert "Historical Trend" in result["execution_summary"]["ml_engine"]


# ── AnalysisResponse round-trip ───────────────────────────────────────────────

def test_analysis_response_accepts_trace_and_summary_fields():
    from app.schemas.responses import AnalysisResponse

    resp = AnalysisResponse(
        status="ok", query="q", response="r", conversation_id="c",
        ml_readiness_score_used=99.75, llm_readiness_score_used=99.75,
        execution_trace=[{"step": "intent_detection", "engine": None, "gate": None, "reason": "x"}],
        execution_summary={"intent": "show_kpi", "tools_used": [], "ml_engine": None,
                            "narration_engine": "Groq", "execution_time_seconds": 0.1, "fallback_used": False},
    )
    assert resp.execution_trace[0]["step"] == "intent_detection"
    assert resp.execution_summary["narration_engine"] == "Groq"


def test_analysis_response_defaults_trace_fields_to_none():
    from app.schemas.responses import AnalysisResponse

    resp = AnalysisResponse(status="error", query="q", response="r")
    assert resp.execution_trace is None
    assert resp.execution_summary is None
