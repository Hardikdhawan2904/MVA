"""app/agents/analytics_agent/config.py — LangGraph-specific accessors.

Deployment/infra config (secrets, host/port) lives in app/config.py's
Settings-equivalent constants — see that module's docstring for why this
agent keeps YAML loading there too rather than re-parsing agent.yaml a
second time here. This module only owns the one thing genuinely specific
to the graph's own topology: its entry point.
"""

from app.config import EXECUTION_PLANS


def get_entry_point() -> str:
    """The LangGraph entry node name — every query starts here regardless
    of intent, since intent isn't known until this node runs."""
    return "detect_intent_and_filters"


def get_execution_plans() -> dict:
    """Intent → tool-chain display info, as declared in agent.yaml."""
    return EXECUTION_PLANS
