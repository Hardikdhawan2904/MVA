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

import json
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
from app.config import ML_READINESS_THRESHOLD, LLM_READINESS_THRESHOLD, MODEL_REGISTRY_PATH

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

# engine display name -> its key in ml/model_registry.json. Prophet is
# deliberately absent from the "has a training date" story below — it
# refits on every query (ProphetForecaster.fit_and_forecast), so its
# registry entry is a last-run timestamp, not a training date.
_ENGINE_REGISTRY_KEY = {
    "Prophet": "prophet_forecast",
    "IsolationForest": "isolation_forest",
    "K-Means": "kmeans_segmentation",
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


def _load_model_registry() -> dict:
    """Best-effort read of ml/model_registry.json — the same file train.py
    and the ml/*.py modules' _update_registry() calls write to. Returns {}
    on any failure (missing file, bad JSON); never raises, since model
    version metadata is an optional enrichment, same as the readiness
    breakdown below — its absence should never affect the response."""
    try:
        with open(MODEL_REGISTRY_PATH) as f:
            return json.load(f).get("models", {})
    except Exception:
        return {}


def _summarize_breakdown(breakdown: dict | None) -> str | None:
    """From Agent 2's readiness assessment evidence (a list of
    {"dimension": ..., "value": 0-1}), name the strongest and weakest
    dimension — turns "ML readiness is 82%" into "...because feature
    coverage is excellent; the only weak factor is data freshness (55%)."
    Returns None when there's no evidence to summarize (e.g. no breakdown
    was supplied — a direct /analyze call outside the Orchestrator)."""
    if not breakdown:
        return None
    evidence = [e for e in (breakdown.get("evidence") or []) if isinstance(e.get("value"), (int, float))]
    if not evidence:
        return None
    strongest = max(evidence, key=lambda e: e["value"])
    weakest = min(evidence, key=lambda e: e["value"])
    if strongest["dimension"] == weakest["dimension"]:
        return f"Strongest/only factor: {strongest['dimension']} ({strongest['value']:.0%})."
    return (
        f"Strongest on {strongest['dimension']} ({strongest['value']:.0%}); "
        f"weakest on {weakest['dimension']} ({weakest['value']:.0%})."
    )


def _model_version_for_engine(engine: str, registry: dict) -> dict | None:
    """Version metadata for an ML-gated trace entry's engine, read from
    ml/model_registry.json. Two distinct shapes, not one — collapsing them
    would misrepresent what the timestamp means:

    - Prophet refits on every query (verified: ProphetForecaster.fit_and_forecast
      rewrites the registry each call) — it has no fixed training date, so
      it gets "last_run_at", not "trained_at".
    - IsolationForest/K-Means predict against a persisted model trained
      once by train.py — "trained_at" is a real, stable date. Neither has
      an accuracy-style metric (both unsupervised); "accuracy_metric" is
      explicitly None rather than omitted, so that absence is visible
      rather than silent — same reasoning as omitting a fabricated
      confidence score for these two models elsewhere in this system.
    """
    key = _ENGINE_REGISTRY_KEY.get(engine)
    if not key:
        return None
    entry = registry.get(key)
    if not entry:
        return None
    timestamp = entry.get("timestamp")
    if engine == "Prophet":
        return {"refit_per_query": True, "last_run_at": timestamp}
    return {"trained_at": timestamp, "accuracy_metric": None}


def _node_for_step(step: str) -> str:
    """Map a trace entry's "step" back to the LangGraph node name that
    produced it, so per-step durations (measured by node name during
    graph.stream()) can be attached to the right trace entry."""
    if step == "intent_detection":
        return "detect_intent_and_filters"
    if step == "narration":
        return "narrate"
    return f"handle_{step}"


def _build_execution_trace(
    final_state: dict, elapsed_seconds: float, node_durations_ms: dict[str, float] | None = None,
) -> tuple[list[dict], dict]:
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
    registry = _load_model_registry()

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
            breakdown = final_state.get("ml_readiness_breakdown")
            gate = {
                "name": "ml_readiness", "score": score, "threshold": ML_READINESS_THRESHOLD,
                "passed": passed, "breakdown": breakdown,
            }
            if passed:
                ml_engine = _ML_GATED_ENGINES[intent]
                reason = (f"ML readiness ({score:.1f}%) met the {ML_READINESS_THRESHOLD}% "
                          f"threshold — using the trained {ml_engine} model.")
            else:
                ml_engine = evidence.get("fallback_applied", "deterministic fallback")
                reason = evidence.get("fallback_reason", f"ML readiness below threshold — using {ml_engine}.")
            breakdown_summary = _summarize_breakdown(breakdown)
            if breakdown_summary:
                reason += f" {breakdown_summary}"

            entry = {"step": intent, "engine": ml_engine, "gate": gate, "reason": reason}
            if passed:
                entry["model_version"] = _model_version_for_engine(ml_engine, registry)
                if intent == "forecast" and "key_drivers" in evidence:
                    lgbm = registry.get("lightgbm_regressor")
                    if lgbm and "r2" in lgbm:
                        entry["reason"] += (f" Key drivers sourced from a persisted LightGBM model "
                                             f"(r²={lgbm['r2']:.3f}, trained {lgbm.get('timestamp', 'unknown')}).")
            trace.append(entry)
        else:
            engine = "RootCauseTool" if intent == "root_cause" else "AnalyticsTool"
            reason = "Deterministic calculation — no ML/LLM readiness gate applies to this intent."
            if intent == "root_cause" and "ml_predicted_driver" in evidence:
                reason += " Corroborated by a persisted XGBoost/SHAP driver classification (best-effort)."
                xgb = registry.get("xgboost_classifier")
                if xgb and "accuracy" in xgb:
                    reason += f" Classifier accuracy: {xgb['accuracy']:.1%} (trained {xgb.get('timestamp', 'unknown')})."
            trace.append({"step": intent, "engine": engine, "gate": None, "reason": reason})

        llm_engine = final_state.get("llm_engine_used")
        if llm_engine:
            llm_score = final_state.get("llm_readiness_score")
            llm_passed = llm_score is not None and llm_score >= LLM_READINESS_THRESHOLD
            llm_breakdown = final_state.get("llm_readiness_breakdown")
            gate = {
                "name": "llm_readiness", "score": llm_score, "threshold": LLM_READINESS_THRESHOLD,
                "passed": llm_passed, "breakdown": llm_breakdown,
            }
            if llm_engine == "Groq":
                reason = f"LLM readiness ({llm_score:.1f}%) met the {LLM_READINESS_THRESHOLD}% threshold — narrated by Groq."
            elif llm_passed:
                reason = (f"LLM readiness ({llm_score:.1f}%) met the {LLM_READINESS_THRESHOLD}% threshold, "
                          f"but the Groq call itself failed — fell back to the template formatter.")
            else:
                reason = (f"LLM readiness ({llm_score:.1f}%) below the {LLM_READINESS_THRESHOLD}% "
                          f"threshold — used the template formatter instead of Groq.")
            breakdown_summary = _summarize_breakdown(llm_breakdown)
            if breakdown_summary:
                reason += f" {breakdown_summary}"
            trace.append({"step": "narration", "engine": llm_engine, "gate": gate, "reason": reason})

    if node_durations_ms:
        for entry in trace:
            node_name = _node_for_step(entry["step"])
            if node_name in node_durations_ms:
                entry["duration_ms"] = round(node_durations_ms[node_name], 1)

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
    ml_readiness_breakdown: dict | None = None,
    llm_readiness_breakdown: dict | None = None,
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

    ml_readiness_breakdown/llm_readiness_breakdown are optional — Agent 2's
    full readiness assessment (strengths/blocking_issues/evidence), when the
    caller has it (the Orchestrator does; a direct /analyze call may not).
    Surfaced in execution_trace's gate objects so the readiness score can
    explain itself, not just report a number.
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
            "ml_readiness_breakdown": ml_readiness_breakdown,
            "llm_readiness_breakdown": llm_readiness_breakdown,
        }

        # graph.stream(..., stream_mode="updates") yields one {node_name:
        # partial_state_update} chunk per completed node, in the order they
        # run — this gives real per-node wall-clock timing for free, without
        # touching any of the 7 handler methods (same "derive after the
        # fact" pattern as the trace itself). final_state is rebuilt from
        # the same partial updates .invoke() would have merged internally.
        final_state: dict = dict(initial_state)
        node_durations_ms: dict[str, float] = {}
        started = time.perf_counter()
        last_ts = started
        for chunk in graph.stream(initial_state, config={"recursion_limit": 25}, stream_mode="updates"):
            now = time.perf_counter()
            for node_name, partial_update in chunk.items():
                final_state.update(partial_update)
                node_durations_ms[node_name] = (now - last_ts) * 1000
            last_ts = now
        elapsed = time.perf_counter() - started

        trace, summary = _build_execution_trace(final_state, elapsed, node_durations_ms)

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
