"""Real LangChain tools for the rule-suggestion agent. Built per-run via a
factory function since each run's column profiles/dataframe/active rules
differ — there's no shared global state a plain module-level @tool could
safely close over.
"""

from typing import Any

import pandas as pd
from langchain_core.tools import tool


def build_rule_suggestion_tools(col_profiles: list, df: pd.DataFrame, existing_rule_keys: list[str]) -> list:
    """
    Tools available to the rule-suggestion LLM. Includes a genuinely new
    capability this pipeline didn't have before: checking which rules already
    exist (YAML-configured + previously approved) so it doesn't propose
    duplicates of rules that are already active for this domain.
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
        """Get the full distinct-value distribution (value counts) for a column
        — useful before proposing an allowed_values rule, to see every distinct
        value rather than just the first few samples."""
        if column_name not in df.columns:
            return f"Column '{column_name}' not found in the dataset."
        counts = df[column_name].value_counts(dropna=True).head(20)
        return f"Value distribution for '{column_name}' (top 20): {counts.to_dict()}"

    @tool
    def check_existing_rules() -> str:
        """List rule_keys that are already active for this dataset's domain
        (from YAML config or previously-approved suggestions). Never propose a
        new rule that duplicates one already in this list."""
        if not existing_rule_keys:
            return "No rules are currently active for this domain."
        return f"Already-active rule keys (do not duplicate these): {existing_rule_keys}"

    return [get_column_statistics, check_value_distribution, check_existing_rules]


def tools_to_registry(tools: list) -> dict[str, Any]:
    """Build a name->tool lookup so agent.yaml's agent_tools list (names only)
    can be resolved into actual callables at graph-build time."""
    return {t.name: t for t in tools}
