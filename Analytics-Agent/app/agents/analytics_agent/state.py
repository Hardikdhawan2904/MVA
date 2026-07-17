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
    conversation_id: str                 # ties this request's memory to prior turns
    ml_readiness_score: float
    llm_readiness_score: float
    feature_recommendation: list[dict] | None
    # Agent 2's full readiness assessments (strengths, blocking_issues,
    # evidence) — optional, since a direct /analyze call outside the
    # Orchestrator has no way to supply them. Read by graph.py's
    # _build_execution_trace to explain *why* a readiness gate passed or
    # failed, not just report the bare score.
    ml_readiness_breakdown: dict | None
    llm_readiness_breakdown: dict | None

    # ── Derived by detect_intent_and_filters ────────────────────────────────
    intent: str
    kpi_name: str
    filters: dict[str, Any]

    # ── Built up by whichever handler node the router selects ──────────────
    evidence: dict[str, Any]

    # ── Output ───────────────────────────────────────────────────────────────
    response: str

    # ── Set by narrate() / record_memory() — read by graph.py's post-hoc
    # execution-trace builder, not consumed inside the graph itself ────────
    llm_engine_used: str          # "Groq" | "Template Formatter" | "Template Formatter (Groq error)"
    tools_used: list[str]         # from _build_plan(intent), same list memory.add_turn() already gets
