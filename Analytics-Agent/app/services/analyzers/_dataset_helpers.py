"""app/services/analyzers/_dataset_helpers.py — small shared helpers used
by several Stage 7 analyzers (plan "zany-giggling-crayon"). Not a public
package API by itself — just avoids the same "pick numeric/categorical
columns from a DatasetContext" and "build a ds/y time series" logic being
copy-pasted across every analyzer family.
"""

from __future__ import annotations

import pandas as pd

from app.services.dataset_context.models import (
    SEMANTIC_ROLE_DIMENSION,
    SEMANTIC_ROLE_METRIC,
    DatasetContext,
)


def numeric_columns(dataset_context: DatasetContext, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    return [c.name for c in dataset_context.columns_with_role(SEMANTIC_ROLE_METRIC) if c.name not in exclude]


def categorical_columns(dataset_context: DatasetContext, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    return [c.name for c in dataset_context.columns_with_role(SEMANTIC_ROLE_DIMENSION) if c.name not in exclude]


def build_time_series(df: pd.DataFrame, temporal_col: str, metric_col: str) -> pd.DataFrame:
    """Generic version of SQLTool.get_time_series()'s grouped-sum shape —
    used when an analyzer receives an already-fetched DataFrame instead of
    issuing its own SQL, so the output ['ds', 'y'] contract stays the same
    regardless of which path built it."""
    ts = df[[temporal_col, metric_col]].copy()
    ts[metric_col] = pd.to_numeric(ts[metric_col], errors="coerce")
    grouped = ts.dropna().groupby(temporal_col, as_index=False)[metric_col].sum()
    grouped = grouped.rename(columns={temporal_col: "ds", metric_col: "y"}).sort_values("ds")
    return grouped.reset_index(drop=True)
