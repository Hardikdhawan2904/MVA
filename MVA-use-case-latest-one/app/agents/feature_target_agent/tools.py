"""Real LangChain tools for the feature-target agent. Built per-run via a
factory function since each run's column profiles/dataframe differ — there's
no shared global state a plain module-level @tool could safely close over.
"""

from typing import Any

import pandas as pd
from langchain_core.tools import tool


def build_feature_target_tools(col_profiles: list, df: pd.DataFrame) -> list:
    """
    Tools available to the feature-target LLM. Both mirror the equivalent
    rule_suggestion_agent tools — the agent already receives every column's
    role/semantic-type/statistics summary up front, these are only for
    drilling into a specific candidate target column before committing to a
    problem_type (e.g. confirming it's genuinely binary/categorical rather
    than continuous).
    """
    profiles_by_name = {p.physical_name: p for p in col_profiles}

    @tool
    def get_column_statistics(column_name: str) -> str:
        """Get the full statistics for a specific column by its physical name."""
        profile = profiles_by_name.get(column_name)
        if profile is None:
            return f"Column '{column_name}' not found. Available columns: {list(profiles_by_name.keys())}"
        return str(profile.to_statistics_dict())

    @tool
    def check_value_distribution(column_name: str) -> str:
        """Get the full distinct-value distribution (value counts) for a
        column — useful before deciding a candidate target's problem_type,
        to see whether it's genuinely binary/categorical (classification) or
        continuous (regression) rather than guessing from samples alone."""
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in the dataset."
        counts = df[column_name].value_counts(dropna=True).head(20)
        return f"Value distribution for '{column_name}' (top 20): {counts.to_dict()}"

    return [get_column_statistics, check_value_distribution]


def tools_to_registry(tools: list) -> dict[str, Any]:
    """Build a name->tool lookup so agent.yaml's agent_tools list (names only)
    can be resolved into actual callables at graph-build time."""
    return {t.name: t for t in tools}
