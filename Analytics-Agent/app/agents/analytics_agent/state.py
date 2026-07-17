"""app/agents/analytics_agent/state.py — typed state threaded through the
Analytics Agent's LangGraph.

Mirrors the shape of main.py::AnalyticsAgent.process()'s old locals (intent,
kpi_name, filters, evidence) as explicit state fields instead — graph nodes
only communicate through this shared dict, so every intermediate value a
later node needs has to be named here.
"""

from typing import Any, TypedDict


class AnalyticsState(TypedDict, total=False):
    # ── Inputs (set once, at graph invocation) ──────────────────────────────
    business_question: str
    dataset_path: str                    # temp CSV path for this request
    ml_readiness_score: float
    llm_readiness_score: float
    feature_recommendation: list[dict] | None

    # ── Derived by detect_intent_and_filters ────────────────────────────────
    intent: str
    kpi_name: str
    filters: dict[str, Any]

    # ── Built up by whichever handler node the router selects ──────────────
    evidence: dict[str, Any]

    # ── Output ───────────────────────────────────────────────────────────────
    response: str
