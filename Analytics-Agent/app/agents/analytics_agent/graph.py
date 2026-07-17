"""app/agents/analytics_agent/graph.py — Analytics Agent pipeline as a
LangGraph StateGraph.

Topology:

  detect_intent_and_filters ─(route_by_intent)─┬─→ handle_show_kpi ────┐
                                                ├─→ handle_variance ────┤
                                                ├─→ handle_root_cause ──┤
                                                ├─→ handle_trend ───────┤
                                                ├─→ handle_forecast ────┤
                                                ├─→ handle_anomaly ─────┤
                                                └─→ handle_segment ─────┤
                                                                        │
                                     (route_after_handler) ─────────────┤
                    "response" already set (early-exit: KPI not found,  │
                    no data for the filters) ──────────────→ record_memory → END
                    "evidence" set, needs narration ─→ narrate → record_memory → END

Built fresh per request — never a shared singleton. See
nodes/pipeline.py's module docstring for why: every request analyzes a
different uploaded dataset, so SQLTool/MLTool/ExplanationTool all have to be
constructed against this request's own inputs, mirroring
MVA-use-case-latest-one's request-scoped graph construction.
"""

import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, END

from app.agents.analytics_agent.config import get_entry_point
from app.agents.analytics_agent.state import AnalyticsState
from app.agents.analytics_agent.nodes.pipeline import AnalyticsGraphNodes
from app.config import ML_READINESS_THRESHOLD, LLM_READINESS_THRESHOLD

logger = logging.getLogger(__name__)

_HANDLER_INTENTS = ("show_kpi", "variance", "root_cause", "trend", "forecast", "anomaly", "segment")

# intent -> (model used when ml_readiness passes, deterministic fallback name)
# Only the 3 intents that actually gate on ml_readiness appear here — the
# fallback name is descriptive-only (the real "why" comes from the handler's
# own evidence["fallback_reason"], reused verbatim below).
_ML_GATED_ENGINES = {
    "forecast": "Prophet",
    "anomaly": "IsolationForest",
    "segment": "K-Means",
}


def build_analytics_graph(
    dataset_path: str,
    conversation_id: str,
    ml_readiness_score: float = 99.75,
    llm_readiness_score: float = 99.75,
    feature_recommendation: list[dict] | None = None,
):
    """Compile the Analytics Agent StateGraph. Call once per request — see
    module docstring."""
    nodes = AnalyticsGraphNodes(
        dataset_path=dataset_path,
        conversation_id=conversation_id,
        ml_readiness_score=ml_readiness_score,
        llm_readiness_score=llm_readiness_score,
        feature_recommendation=feature_recommendation,
    )

    g = StateGraph(AnalyticsState)

    g.add_node("detect_intent_and_filters", nodes.detect_intent_and_filters)
    g.add_node("handle_show_kpi", nodes.handle_show_kpi)
    g.add_node("handle_variance", nodes.handle_variance)
    g.add_node("handle_root_cause", nodes.handle_root_cause)
    g.add_node("handle_trend", nodes.handle_trend)
    g.add_node("handle_forecast", nodes.handle_forecast)
    g.add_node("handle_anomaly", nodes.handle_anomaly)
    g.add_node("handle_segment", nodes.handle_segment)
    g.add_node("narrate", nodes.narrate)
    g.add_node("record_memory", nodes.record_memory)

    g.set_entry_point(get_entry_point())

    g.add_conditional_edges(
        "detect_intent_and_filters",
        nodes.route_by_intent,
        {intent: f"handle_{intent}" for intent in _HANDLER_INTENTS},
    )

    for intent in _HANDLER_INTENTS:
        g.add_conditional_edges(
            f"handle_{intent}",
            nodes.route_after_handler,
            {"narrate": "narrate", "record_memory": "record_memory"},
        )

    g.add_edge("narrate", "record_memory")
    g.add_edge("record_memory", END)

    return g.compile()


def _build_execution_trace(final_state: dict, elapsed_seconds: float) -> tuple[list[dict], dict]:
    """Derive a step-by-step decision trace and a compact summary from the
    graph's final state, built once after graph.invoke() returns rather
    than having every handler node append its own trace entry inline.

    This works because the handlers already leave enough of a signal behind:
    the 3 ML-gated handlers only ever set evidence["ml_readiness_blocked"]
    on their fallback branch (its absence means the model path ran), and
    narrate()/record_memory() surface llm_engine_used/tools_used into state
    specifically so this function doesn't need direct access to the nodes
    instance itself. Mirrors MVA-use-case-latest-one's
    data_profiling_agent/graph.py::_state_to_pipeline_result — adapt raw
    graph state into a structured result in one place, not scattered
    across nodes.
    """
    intent = final_state.get("intent", "unknown")
    kpi_name = final_state.get("kpi_name")
    evidence = final_state.get("evidence")
    response = final_state.get("response", "")

    trace: list[dict] = [{
        "step": "intent_detection",
        "engine": None,
        "gate": None,
        "reason": f"Detected intent='{intent}'" + (f", kpi='{kpi_name}'" if kpi_name else ""),
    }]

    ml_engine: str | None = None

    if evidence is None:
        # A handler early-exited with just "response" set (KPI not found,
        # no data for the given filters) — that message already explains
        # why, there's nothing further to derive.
        trace.append({"step": intent, "engine": None, "gate": None,
                       "reason": response or "No evidence was produced for this query."})
    else:
        if intent in _ML_GATED_ENGINES:
            blocked = evidence.get("ml_readiness_blocked", False)
            score = final_state.get("ml_readiness_score")
            passed = not blocked
            gate = {"name": "ml_readiness", "score": score, "threshold": ML_READINESS_THRESHOLD, "passed": passed}
            if passed:
                ml_engine = _ML_GATED_ENGINES[intent]
                reason = (f"ML readiness ({score:.1f}%) met the {ML_READINESS_THRESHOLD}% "
                          f"threshold — using the trained {ml_engine} model.")
            else:
                ml_engine = evidence.get("fallback_applied", "deterministic fallback")
                reason = evidence.get("fallback_reason", f"ML readiness below threshold — using {ml_engine}.")
            trace.append({"step": intent, "engine": ml_engine, "gate": gate, "reason": reason})
        else:
            engine = "RootCauseTool" if intent == "root_cause" else "AnalyticsTool"
            reason = "Deterministic calculation — no ML/LLM readiness gate applies to this intent."
            if intent == "root_cause" and "ml_predicted_driver" in evidence:
                reason += " Corroborated by a persisted XGBoost/SHAP driver classification (best-effort)."
            trace.append({"step": intent, "engine": engine, "gate": None, "reason": reason})

        llm_engine = final_state.get("llm_engine_used")
        if llm_engine:
            llm_score = final_state.get("llm_readiness_score")
            llm_passed = llm_score is not None and llm_score >= LLM_READINESS_THRESHOLD
            gate = {"name": "llm_readiness", "score": llm_score, "threshold": LLM_READINESS_THRESHOLD, "passed": llm_passed}
            if llm_engine == "Groq":
                reason = f"LLM readiness ({llm_score:.1f}%) met the {LLM_READINESS_THRESHOLD}% threshold — narrated by Groq."
            elif llm_passed:
                reason = (f"LLM readiness ({llm_score:.1f}%) met the {LLM_READINESS_THRESHOLD}% threshold, "
                          f"but the Groq call itself failed — fell back to the template formatter.")
            else:
                reason = (f"LLM readiness ({llm_score:.1f}%) below the {LLM_READINESS_THRESHOLD}% "
                          f"threshold — used the template formatter instead of Groq.")
            trace.append({"step": "narration", "engine": llm_engine, "gate": gate, "reason": reason})

    llm_engine_used = final_state.get("llm_engine_used")
    summary = {
        "intent": intent,
        "tools_used": final_state.get("tools_used", []),
        "ml_engine": ml_engine,
        "narration_engine": llm_engine_used,
        "execution_time_seconds": round(elapsed_seconds, 3),
        "fallback_used": bool(
            (evidence is not None and intent in _ML_GATED_ENGINES and evidence.get("ml_readiness_blocked", False))
            or (llm_engine_used is not None and llm_engine_used != "Groq")
        ),
    }
    return trace, summary


def run_analytics_graph(
    file_content: bytes,
    business_question: str,
    conversation_id: str | None = None,
    ml_readiness_score: float = 99.75,
    llm_readiness_score: float = 99.75,
    feature_recommendation: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Write the uploaded dataset to a temp CSV, build+invoke the graph against
    it, and return a dict shaped for AnalysisResponse. Always cleans up the
    temp file (the graph-per-request pattern above means nothing else holds
    a reference to it once this returns), and turns any failure into a
    status="error" result instead of raising — app/routes/analyze.py doesn't
    need its own try/except around this call.

    conversation_id ties this call's memory to any prior turns — if omitted,
    a fresh one is generated (new conversation) and always returned in the
    result so the caller can pass it back on the next turn.
    """
    conversation_id = conversation_id or str(uuid.uuid4())
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        graph = build_analytics_graph(
            dataset_path=tmp_path,
            conversation_id=conversation_id,
            ml_readiness_score=ml_readiness_score,
            llm_readiness_score=llm_readiness_score,
            feature_recommendation=feature_recommendation,
        )

        initial_state: AnalyticsState = {
            "business_question": business_question,
            "dataset_path": tmp_path,
            "conversation_id": conversation_id,
            "ml_readiness_score": ml_readiness_score,
            "llm_readiness_score": llm_readiness_score,
            "feature_recommendation": feature_recommendation,
        }

        started = time.perf_counter()
        final_state = graph.invoke(initial_state, config={"recursion_limit": 25})
        elapsed = time.perf_counter() - started

        trace, summary = _build_execution_trace(final_state, elapsed)

        return {
            "status": "ok",
            "query": business_question,
            "response": final_state.get("response", ""),
            "conversation_id": conversation_id,
            "ml_readiness_score_used": ml_readiness_score,
            "llm_readiness_score_used": llm_readiness_score,
            "execution_trace": trace,
            "execution_summary": summary,
        }
    except Exception as e:
        logger.error(f"analytics_graph_failed: {e}", exc_info=True)
        return {
            "status": "error",
            "query": business_question,
            "response": f"Analytics Agent failed: {e}",
            "conversation_id": conversation_id,
            "ml_readiness_score_used": ml_readiness_score,
            "llm_readiness_score_used": llm_readiness_score,
        }
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
