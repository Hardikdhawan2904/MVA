"""Chart validator — validates chart specs against available data and policy."""

from typing import Any

from app.core.enums import ChartType
from app.services.charts.candidate_generator import ChartSpec
from app.services.profiling.column_profiler import ColumnProfileResult


class ChartValidator:
    """Validates chart specifications against available columns and policy."""

    def __init__(self, max_dimension_cardinality: int = 30):
        self._max_dim_cardinality = max_dimension_cardinality

    def validate(self, chart: ChartSpec, profiles: list[ColumnProfileResult]) -> list[str]:
        """
        Validate a chart spec. Returns list of warning/error messages.
        Empty list means valid.
        """
        col_map = {p.physical_name: p for p in profiles}
        warnings: list[str] = []

        # Check dimension exists
        if chart.dimension and chart.dimension not in col_map:
            warnings.append(f"Dimension column '{chart.dimension}' not found in dataset")

        # Check metric exists
        if chart.metric and chart.metric not in col_map:
            warnings.append(f"Metric column '{chart.metric}' not found in dataset")

        # Chart-type-specific validation
        if chart.chart_type == ChartType.LINE:
            if not chart.dimension:
                warnings.append("Line chart requires a temporal or ordered dimension")

        elif chart.chart_type == ChartType.SCATTER:
            if not chart.metric or not chart.dimension:
                warnings.append("Scatter chart requires both x and y numeric fields")

        elif chart.chart_type in (ChartType.PIE, ChartType.DONUT):
            if chart.dimension and chart.dimension in col_map:
                cardinality = col_map[chart.dimension].distinct_count
                if cardinality > 10:
                    warnings.append(f"Pie/donut chart dimension has {cardinality} values (max recommended: 10)")

        elif chart.chart_type == ChartType.BAR:
            if chart.dimension and chart.dimension in col_map:
                cardinality = col_map[chart.dimension].distinct_count
                if cardinality > self._max_dim_cardinality:
                    warnings.append(f"Bar chart dimension has {cardinality} values (max: {self._max_dim_cardinality})")

        elif chart.chart_type == ChartType.KPI:
            if not chart.metric:
                warnings.append("KPI chart requires a metric")

        return warnings

    def validate_all(self, charts: list[ChartSpec], profiles: list[ColumnProfileResult]) -> list[ChartSpec]:
        """Validate all charts, attaching warnings. Removes invalid charts."""
        valid_charts: list[ChartSpec] = []
        for chart in charts:
            issues = self.validate(chart, profiles)
            # Only remove charts with critical issues (missing columns)
            critical = [w for w in issues if "not found" in w]
            if critical:
                continue
            chart.warnings.extend(issues)
            valid_charts.append(chart)
        return valid_charts
