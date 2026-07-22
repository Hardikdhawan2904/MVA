"""app/services/analyzers/trend_analyzer.py — Stage 7 (plan
"zany-giggling-crayon"): TrendAnalyzer.

target_columns convention: [temporal_col, metric_col]. Always deterministic
by design (no capability gate — see ANALYSIS_TYPE_TO_CAPABILITY's
docstring); the sole registry entry wraps AnalyticsTool.trend() unchanged.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.services.analyzers.base import Analyzer, import_class
from app.services.dataset_context.models import DatasetContext
from app.services.models_registry.model_selector import SelectedModel


class TrendAnalyzer(Analyzer):
    analysis_type = "trend"

    def _run(
        self,
        *,
        df: pd.DataFrame,
        dataset_context: DatasetContext,
        target_columns: list[str],
        selected_model: SelectedModel,
        extra_context: dict[str, Any],
    ) -> dict[str, Any]:
        if len(target_columns) < 2:
            return {"error": "TrendAnalyzer requires target_columns=[temporal_col, metric_col]"}
        temporal_col, metric_col = target_columns[0], target_columns[1]
        if temporal_col not in df.columns or metric_col not in df.columns:
            return {"error": f"Trend target columns not found in data: {temporal_col!r}, {metric_col!r}"}

        tool_cls = import_class(selected_model.implementation_class)
        result = tool_cls().trend(df, temporal_col, metric_col)
        if "error" in result:
            return {"error": result["error"]}
        return {"evidence": result}
