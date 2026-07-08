"""Chart ranker — scores and orders chart candidates by relevance."""

from typing import Any

from app.core.enums import ChartCategory
from app.services.charts.candidate_generator import ChartSpec
from app.services.profiling.column_profiler import ColumnProfileResult


class ChartRanker:
    """Ranks chart candidates to select the best subset."""

    def __init__(self, weights: dict[str, float] | None = None):
        self._weights = weights or {
            "domain_relevance": 0.30,
            "data_availability": 0.25,
            "metric_validity": 0.15,
            "dimension_cardinality": 0.10,
            "temporal_coverage": 0.10,
            "hierarchy_compatibility": 0.05,
            "quality_score": 0.05,
        }

    def rank(self, charts: list[ChartSpec], profiles: list[ColumnProfileResult]) -> list[ChartSpec]:
        """
        Rank charts by computed relevance score.
        
        Avoids redundancy — charts communicating the same result are penalized.
        """
        scored: list[tuple[float, ChartSpec]] = []
        seen_metrics: set[str] = set()
        seen_dimensions: set[str] = set()

        for chart in charts:
            score = self._compute_score(chart, profiles, seen_metrics, seen_dimensions)
            scored.append((score, chart))

            # Track used fields for redundancy detection
            if chart.metric:
                seen_metrics.add(chart.metric)
            if chart.dimension:
                seen_dimensions.add(chart.dimension)

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Assign ranks
        ranked = []
        for i, (score, chart) in enumerate(scored):
            chart.rank = i + 1
            ranked.append(chart)

        return ranked

    def _compute_score(self, chart: ChartSpec, profiles: list[ColumnProfileResult],
                       seen_metrics: set[str], seen_dimensions: set[str]) -> float:
        """Compute relevance score for a single chart."""
        score = 0.0
        col_map = {p.physical_name: p for p in profiles}

        # Domain relevance — business charts score higher
        if chart.category == ChartCategory.BUSINESS:
            score += self._weights["domain_relevance"]
        else:
            score += self._weights["domain_relevance"] * 0.5

        # Data availability — both dimension and metric exist with data
        if chart.metric and chart.metric in col_map:
            profile = col_map[chart.metric]
            if profile.null_ratio < 0.5:
                score += self._weights["data_availability"]
            else:
                score += self._weights["data_availability"] * 0.3
        elif chart.data:  # Pre-computed data (profiling charts)
            score += self._weights["data_availability"]

        # Metric validity — numeric parse ratio for metric columns
        if chart.metric and chart.metric in col_map:
            profile = col_map[chart.metric]
            score += self._weights["metric_validity"] * profile.numeric_parse_ratio

        # Dimension cardinality — prefer moderate cardinality
        if chart.dimension and chart.dimension in col_map:
            profile = col_map[chart.dimension]
            if 2 <= profile.distinct_count <= 20:
                score += self._weights["dimension_cardinality"]
            elif profile.distinct_count <= 50:
                score += self._weights["dimension_cardinality"] * 0.5

        # Temporal coverage
        if chart.dimension and chart.dimension in col_map:
            profile = col_map[chart.dimension]
            if profile.datetime_parse_ratio > 0.9:
                score += self._weights["temporal_coverage"]

        # Hierarchy compatibility
        if chart.hierarchy_info:
            score += self._weights["hierarchy_compatibility"]

        # Redundancy penalty
        if chart.metric and chart.metric in seen_metrics:
            score *= 0.6
        if chart.dimension and chart.dimension in seen_dimensions:
            score *= 0.7

        return score
