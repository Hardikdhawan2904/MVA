"""Tests for AI readiness assessments."""

import pytest
import pandas as pd

from app.core.config import Settings
from app.core.enums import ReadinessType, ReadinessStatus
from app.services.profiling.column_profiler import ColumnProfiler
from app.services.readiness.readiness_engine import ReadinessEngine, _determine_status


@pytest.fixture
def profiler() -> ColumnProfiler:
    settings = Settings(DATABASE_URL="postgresql://x:x@localhost/test", MAX_SAMPLE_VALUES=5)
    return ColumnProfiler(settings)


@pytest.fixture
def engine() -> ReadinessEngine:
    return ReadinessEngine()


class TestReadinessThresholds:
    def test_ready(self):
        assert _determine_status(80.0) == ReadinessStatus.READY
        assert _determine_status(100.0) == ReadinessStatus.READY

    def test_partially_ready(self):
        assert _determine_status(60.0) == ReadinessStatus.PARTIALLY_READY
        assert _determine_status(79.99) == ReadinessStatus.PARTIALLY_READY

    def test_not_ready(self):
        assert _determine_status(0.0) == ReadinessStatus.NOT_READY
        assert _determine_status(59.99) == ReadinessStatus.NOT_READY


class TestReadinessEngine:
    def test_all_four_assessments_returned(self, profiler, engine):
        df = pd.DataFrame({
            "id": [f"R{i}" for i in range(100)],
            "amount": [str(i * 10) for i in range(100)],
            "status": ["active", "inactive"] * 50,
            "date": ["2024-01-01"] * 100,
        })
        profiles = profiler.profile_all(df, ["id", "amount", "status", "date"])
        quality = [{"dimension": "completeness", "score": 0.95, "status": "assessed"}]

        results = engine.assess_all(
            profiles=profiles,
            quality_results=quality,
            grain_columns=["id"],
            has_temporal=True,
            metric_count=1,
            dimension_count=1,
            description_coverage=0.5,
            row_count=100,
        )

        types = [r.assessment_type for r in results]
        assert ReadinessType.ANALYTICS in types
        assert ReadinessType.ML in types
        assert ReadinessType.LLM in types
        assert ReadinessType.OVERALL in types

    def test_structured_evidence(self, profiler, engine):
        df = pd.DataFrame({"x": [str(i) for i in range(50)]})
        profiles = profiler.profile_all(df, ["x"])
        results = engine.assess_all(profiles, [], [], False, 0, 0, 0.0, 50)

        for r in results:
            assert isinstance(r.evidence, list)
            assert isinstance(r.strengths, list)
            assert isinstance(r.blocking_issues, list)

    def test_scores_between_0_and_100(self, profiler, engine):
        df = pd.DataFrame({"x": ["a"] * 20})
        profiles = profiler.profile_all(df, ["x"])
        results = engine.assess_all(profiles, [], [], False, 0, 0, 0.0, 20)
        for r in results:
            assert 0.0 <= r.score <= 100.0

    def test_reuses_common_evidence(self, profiler, engine):
        """All readiness types should reference quality scores from same source."""
        quality = [
            {"dimension": "completeness", "score": 0.95, "status": "assessed"},
            {"dimension": "consistency", "score": 0.88, "status": "assessed"},
        ]
        df = pd.DataFrame({"x": [str(i) for i in range(100)]})
        profiles = profiler.profile_all(df, ["x"])
        results = engine.assess_all(profiles, quality, [], True, 2, 3, 0.8, 100)

        analytics = next(r for r in results if r.assessment_type == ReadinessType.ANALYTICS)
        ml = next(r for r in results if r.assessment_type == ReadinessType.ML)
        # Both should reference completeness in evidence
        analytics_dims = [e.get("dimension") for e in analytics.evidence]
        ml_dims = [e.get("dimension") for e in ml.evidence]
        assert "completeness" in analytics_dims
        assert "completeness" in ml_dims
