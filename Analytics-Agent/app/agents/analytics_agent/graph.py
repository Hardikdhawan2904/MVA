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
import uuid
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, END

from app.agents.analytics_agent.config import get_entry_point
from app.agents.analytics_agent.state import AnalyticsState
from app.agents.analytics_agent.nodes.pipeline import AnalyticsGraphNodes

logger = logging.getLogger(__name__)

_HANDLER_INTENTS = ("show_kpi", "variance", "root_cause", "trend", "forecast", "anomaly", "segment")


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

        final_state = graph.invoke(initial_state, config={"recursion_limit": 25})

        return {
            "status": "ok",
            "query": business_question,
            "response": final_state.get("response", ""),
            "conversation_id": conversation_id,
            "ml_readiness_score_used": ml_readiness_score,
            "llm_readiness_score_used": llm_readiness_score,
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
