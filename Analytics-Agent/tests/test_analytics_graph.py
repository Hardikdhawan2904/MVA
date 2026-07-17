"""tests/test_analytics_graph.py — Routing tests for the Analytics Agent's
LangGraph (app/agents/analytics_agent/graph.py + nodes/pipeline.py).

Focuses on the two router functions that determine graph topology at
runtime (route_by_intent, route_after_handler) plus a real end-to-end
invocation per intent confirming the conditional edges actually wire the
entry node's detected intent to the matching handler — not a full
behavioral test of each handler's evidence-building logic (that's covered
by test_analytics_tool.py, test_ml_persistence.py, test_rule_engine.py, and
test_harness.py's response-content assertions).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from app.agents.analytics_agent.graph import build_analytics_graph
from app.agents.analytics_agent.nodes.pipeline import AnalyticsGraphNodes

_DATASET = Path(r"C:\Users\dhawa\mva\Schema-Intelligence-Layer\test_data\insurance_variance_data_native.csv")

pytestmark = pytest.mark.skipif(
    not _DATASET.exists(), reason=f"Insurance test dataset not found at {_DATASET}"
)


@pytest.fixture(scope="module")
def nodes():
    return AnalyticsGraphNodes(dataset_path=str(_DATASET))


# ── route_by_intent ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("intent,expected_node", [
    ("forecast", "forecast"),
    ("anomaly", "anomaly"),
    ("segment", "segment"),
    ("root_cause", "root_cause"),
    ("variance", "variance"),
    ("trend", "trend"),
    ("show_kpi", "show_kpi"),
    # "ytd" is a real _INTENT_MAP entry, but main.py's if/elif dispatch never
    # special-cased it — it fell through to the `else` branch, i.e. show_kpi.
    ("ytd", "show_kpi"),
    ("something_unrecognized", "show_kpi"),
])
def test_route_by_intent(nodes, intent, expected_node):
    assert nodes.route_by_intent({"intent": intent}) == expected_node


# ── route_after_handler ──────────────────────────────────────────────────────

def test_route_after_handler_response_skips_narrate(nodes):
    assert nodes.route_after_handler({"response": "already answered"}) == "record_memory"


def test_route_after_handler_evidence_goes_to_narrate(nodes):
    assert nodes.route_after_handler({"evidence": {"kpi": "x"}}) == "narrate"


# ── Graph compilation ─────────────────────────────────────────────────────────

def test_graph_builds_and_compiles():
    graph = build_analytics_graph(dataset_path=str(_DATASET))
    # A compiled LangGraph exposes .invoke — presence confirms compile() succeeded.
    assert hasattr(graph, "invoke")


# ── End-to-end: entry node's detected intent reaches the matching handler ────

@pytest.mark.parametrize("query,expected_intent", [
    ("Show Gross Written Premium for FY2025", "show_kpi"),
    ("Show loss ratio variance vs budget for EMEA", "variance"),
    ("Why did underwriting result decline in FY2025?", "root_cause"),
    ("Show the trend of loss ratio over time", "trend"),
    ("Forecast underwriting result for next 6 months", "forecast"),
    ("Detect anomalies in loss ratios", "anomaly"),
    ("Segment portfolio by risk profile", "segment"),
])
def test_end_to_end_intent_routing(query, expected_intent):
    graph = build_analytics_graph(dataset_path=str(_DATASET))
    final_state = graph.invoke(
        {"business_question": query, "dataset_path": str(_DATASET)},
        config={"recursion_limit": 25},
    )
    assert final_state["intent"] == expected_intent
    assert final_state.get("response")
