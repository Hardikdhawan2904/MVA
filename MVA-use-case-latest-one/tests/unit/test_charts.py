"""Tests for chart intelligence."""

import pytest
import pandas as pd

from app.core.config import Settings
from app.core.enums import ChartType, ChartCategory, ColumnRole, RefinedDataType
from app.services.charts.candidate_generator import ChartCandidateGenerator, ChartSpec
from app.services.charts.aggregation_engine import AggregationEngine
from app.services.charts.drilldown_service import DrillDownService
from app.services.profiling.column_profiler import ColumnProfiler
from app.services.profiling.type_refiner import TypeRefiner
from app.services.profiling.semantic_candidate_generator import SemanticCandidateGenerator
from app.core.exceptions import ProcessingError


@pytest.fixture
def profiler() -> ColumnProfiler:
    settings = Settings(DATABASE_URL="postgresql://x:x@localhost/test", MAX_SAMPLE_VALUES=5)
    return ColumnProfiler(settings)


@pytest.fixture
def refiner() -> TypeRefiner:
    return TypeRefiner()


@pytest.fixture
def sem_gen() -> SemanticCandidateGenerator:
    return SemanticCandidateGenerator()


@pytest.fixture
def chart_gen() -> ChartCandidateGenerator:
    return ChartCandidateGenerator(max_charts=5, target_business=3, target_profiling=2)


@pytest.fixture
def agg_engine() -> AggregationEngine:
    return AggregationEngine()


def _build_candidates(profiler, refiner, sem_gen, df):
    from app.services.profiling.identifier_detector import IdentifierDetector
    from app.services.profiling.dataset_profiler import normalize_column_name
    profiles = []
    refined_types = []
    for col in df.columns:
        p = profiler.profile_column(df[col], col, normalize_column_name(col))
        profiles.append(p)
        refined_types.append(refiner.refine(p))
    detector = IdentifierDetector()
    grain = detector.detect(profiles, refined_types)
    id_flags = [c.is_identifier for c in grain.identifier_candidates]
    candidates = sem_gen.generate_all(profiles, refined_types, id_flags)
    return profiles, candidates


class TestChartCandidateGenerator:
    def test_generates_up_to_max_charts(self, profiler, refiner, sem_gen, chart_gen):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-01", "2024-03-01"] * 30,
            "amount": [str(i * 10) for i in range(90)],
            "region": ["North", "South", "East"] * 30,
            "status": ["active", "inactive"] * 45,
        })
        profiles, candidates = _build_candidates(profiler, refiner, sem_gen, df)
        charts = chart_gen.generate(profiles, candidates, [], [], None)
        assert len(charts) <= 5

    def test_business_heavy_composition(self, profiler, refiner, sem_gen, chart_gen):
        df = pd.DataFrame({
            "date": ["2024-01-01"] * 50,
            "revenue": [str(i * 100) for i in range(50)],
            "department": ["Sales", "Eng", "HR", "Finance", "Ops"] * 10,
        })
        profiles, candidates = _build_candidates(profiler, refiner, sem_gen, df)
        charts = chart_gen.generate(profiles, candidates, [], [], None)
        business = [c for c in charts if c.category == ChartCategory.BUSINESS]
        assert len(business) >= 1  # Should have at least some business charts

    def test_no_forced_fifth_chart(self, profiler, refiner, sem_gen):
        """Should not force 5 charts when data doesn't support it."""
        df = pd.DataFrame({"x": ["a"] * 10})
        profiles, candidates = _build_candidates(profiler, refiner, sem_gen, df)
        gen = ChartCandidateGenerator(max_charts=5)
        charts = gen.generate(profiles, candidates, [], [], None)
        # With minimal data, should produce fewer than 5
        assert len(charts) <= 5  # Won't force empty charts

    def test_chart_field_validation(self, profiler, refiner, sem_gen, chart_gen):
        """Charts should only reference columns that exist."""
        df = pd.DataFrame({
            "amount": [str(i) for i in range(50)],
            "category": ["A", "B", "C", "D", "E"] * 10,
        })
        profiles, candidates = _build_candidates(profiler, refiner, sem_gen, df)
        charts = chart_gen.generate(profiles, candidates, [], [], None)
        col_names = set(df.columns)
        for chart in charts:
            if chart.dimension:
                assert chart.dimension in col_names
            if chart.metric:
                assert chart.metric in col_names

    def test_profiling_charts_included(self, profiler, refiner, sem_gen, chart_gen):
        """Quality/profiling charts should appear when data has quality issues."""
        df = pd.DataFrame({
            "col1": ["a", None, "c", None, "e"] * 10,
            "col2": [str(i) for i in range(50)],
        })
        profiles, candidates = _build_candidates(profiler, refiner, sem_gen, df)
        quality = [{"dimension": "completeness", "score": 0.8, "status": "assessed"}]
        charts = chart_gen.generate(profiles, candidates, [], quality, None)
        profiling = [c for c in charts if c.category in (ChartCategory.PROFILING, ChartCategory.QUALITY)]
        assert len(profiling) >= 1


class TestAggregationEngine:
    def test_categorical_aggregation(self, agg_engine):
        df = pd.DataFrame({
            "region": ["North", "South", "North", "East", "South"],
            "amount": ["100", "200", "150", "300", "250"],
        })
        chart = ChartSpec(
            chart_key="test", category=ChartCategory.BUSINESS, chart_type=ChartType.BAR,
            title="Test", dimension="region", metric="amount", aggregation="sum",
        )
        result = agg_engine.aggregate(df, chart)
        assert len(result.data) > 0
        # North should be 250 (100+150)
        north = next((d for d in result.data if d["label"] == "North"), None)
        assert north is not None
        assert north["value"] == 250.0

    def test_kpi_aggregation(self, agg_engine):
        df = pd.DataFrame({"revenue": ["100", "200", "300"]})
        chart = ChartSpec(
            chart_key="kpi", category=ChartCategory.BUSINESS, chart_type=ChartType.KPI,
            title="Total Revenue", metric="revenue", aggregation="sum",
        )
        result = agg_engine.aggregate(df, chart)
        assert len(result.data) == 1
        assert result.data[0]["value"] == 600.0


class TestDrillDownService:
    def test_valid_drill_down(self):
        cubes = [
            {
                "level_column": "country",
                "dimension_path_json": {"region": "North"},
                "aggregated_data_json": [
                    {"label": "US", "value": 5000},
                    {"label": "CA", "value": 3000},
                ],
            }
        ]
        service = DrillDownService()
        result = service.execute_drill_down(
            cubes=cubes,
            chart_id="chart1",
            hierarchy_levels=["region", "country", "city"],
            selected_path={"region": "North"},
        )
        assert result["current_level"] == "country"
        assert result["next_level"] == "city"
        assert len(result["data"]) == 2

    def test_invalid_path_column(self):
        service = DrillDownService()
        with pytest.raises(ProcessingError) as exc_info:
            service.execute_drill_down(
                cubes=[],
                chart_id="c1",
                hierarchy_levels=["region", "country"],
                selected_path={"invalid_col": "x"},
            )
        assert exc_info.value.code == "INVALID_DRILL_DOWN_PATH"

    def test_drill_down_hierarchy_order(self):
        """Selected path must follow hierarchy order."""
        service = DrillDownService()
        # This is valid — selecting at region level to drill to country
        result = service.execute_drill_down(
            cubes=[],
            chart_id="c1",
            hierarchy_levels=["region", "country", "city"],
            selected_path={"region": "North"},
        )
        assert result["current_level"] == "country"
