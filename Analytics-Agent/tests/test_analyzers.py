"""tests/test_analyzers.py — Agent 3 redesign, Phase 3 (plan
"zany-giggling-crayon"): Analyzers + Evidence Builder (Stages 7-8).

Exit criteria per the plan: "Per-analyzer tests against non-Insurance
fixtures; Evidence contract tests." Everything here runs against a
synthetic generic-business fixture (not Insurance) built inline — proving
the analyzers are genuinely domain-agnostic, not just re-tested versions
of Insurance-shaped logic. Not yet wired into the live request path
(that's Phase 4) — every test drives an Analyzer directly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

from app.services.analyzers.base import Analyzer, AnalyzerRegistry, import_class, wrap_if_flat
from app.services.analyzers.anomaly_analyzer import AnomalyAnalyzer
from app.services.analyzers.association_analyzer import AssociationAnalyzer
from app.services.analyzers.classification_analyzer import ClassificationAnalyzer
from app.services.analyzers.clustering_analyzer import ClusteringAnalyzer
from app.services.analyzers.comparative_analyzer import ComparativeAnalyzer
from app.services.analyzers.correlation_analyzer import CorrelationAnalyzer
from app.services.analyzers.distribution_analyzer import DistributionAnalyzer
from app.services.analyzers.feature_importance_analyzer import FeatureImportanceAnalyzer
from app.services.analyzers.forecast_analyzer import ForecastAnalyzer
from app.services.analyzers.outlier_analyzer import OutlierAnalyzer
from app.services.analyzers.ranking_analyzer import RankingAnalyzer
from app.services.analyzers.regression_analyzer import RegressionAnalyzer
from app.services.analyzers.root_cause_analyzer import RootCauseAnalyzer
from app.services.analyzers.segmentation_analyzer import SegmentationAnalyzer
from app.services.analyzers.time_series_analyzer import TimeSeriesAnalyzer
from app.services.analyzers.trend_analyzer import TrendAnalyzer
from app.services.dataset_context.models import (
    SEMANTIC_ROLE_DIMENSION, SEMANTIC_ROLE_IDENTIFIER, SEMANTIC_ROLE_METRIC, SEMANTIC_ROLE_TEMPORAL,
    ColumnContext, DatasetContext,
)
from app.services.evidence.evidence_builder import Evidence, EvidenceBuilder
from app.services.models_registry.model_registry import ModelRegistry
from app.services.models_registry.model_selector import SelectedModel
from app.services.planning.models import PlannedAnalysis
from app.services.scheduling.models import ScheduledAnalysis

pd.set_option("future.no_silent_downcasting", True)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _fixture_df(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """A generic non-Insurance business dataset — no ratio/variance
    columns, no reporting_date, nothing Insurance-shaped."""
    rng = np.random.default_rng(seed)
    revenue = rng.normal(1000, 100, n)
    cost = revenue * 0.6 + rng.normal(0, 20, n)  # correlated with revenue by construction
    return pd.DataFrame({
        "order_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "revenue": revenue,
        "cost": cost,
        "region": rng.choice(["North", "South", "East", "West"], n),
        "status": rng.choice(["success", "failed"], n, p=[0.8, 0.2]),
        "order_id": np.arange(n),
    })


def _fixture_context(df: pd.DataFrame) -> DatasetContext:
    columns = [
        ColumnContext(name="order_date", semantic_role=SEMANTIC_ROLE_TEMPORAL, is_temporal=True),
        ColumnContext(name="revenue", semantic_role=SEMANTIC_ROLE_METRIC, semantic_type="revenue_amount"),
        ColumnContext(name="cost", semantic_role=SEMANTIC_ROLE_METRIC, semantic_type="cost_amount"),
        ColumnContext(name="region", semantic_role=SEMANTIC_ROLE_DIMENSION),
        ColumnContext(name="status", semantic_role=SEMANTIC_ROLE_DIMENSION),
        ColumnContext(name="order_id", semantic_role=SEMANTIC_ROLE_IDENTIFIER, is_identifier=True),
    ]
    return DatasetContext(row_count=len(df), column_count=len(columns), columns=columns, context_source="local_fallback")


def _scheduled(analysis_type: str, target_columns: list[str]) -> ScheduledAnalysis:
    return ScheduledAnalysis(
        sequence=1,
        planned_analysis=PlannedAnalysis(analysis_type=analysis_type, target_columns=target_columns, rationale="test"),
        ml_execution_allowed=True,
    )


def _model_for(purpose: str, algorithm: str) -> SelectedModel:
    """Look up a real AlgorithmSpec from the actual shipped
    config/model_registry.yml by (purpose, algorithm) — exercises the real
    registry content, not a hand-rolled duplicate of it."""
    registry = ModelRegistry()
    specs = [s for s in registry.for_purpose(purpose) if s.algorithm == algorithm]
    assert specs, f"No enabled registry entry for purpose={purpose!r} algorithm={algorithm!r}"
    spec = specs[0]
    return SelectedModel(
        algorithm=spec.algorithm,
        implementation_class=spec.implementation_class,
        requires_ml=spec.requirements.requires_ml,
        cost_tier=spec.cost_tier,
        reasons=["test-selected"],
    )


# ── Evidence / EvidenceBuilder contract ─────────────────────────────────────

def test_evidence_flatten_merges_fallback_metadata_and_evidence_evidence_wins():
    ev = Evidence(evidence={"actual": 100, "shared_key": "from_evidence"}, fallback_metadata={"fallback_applied": "X", "shared_key": "from_fallback"})
    flat = ev.flatten()
    assert flat["actual"] == 100
    assert flat["fallback_applied"] == "X"
    assert flat["shared_key"] == "from_evidence"  # evidence wins on collision


def test_evidence_flatten_with_no_fallback_metadata_is_just_evidence():
    ev = Evidence(evidence={"r2_score": 0.8})
    assert ev.flatten() == {"r2_score": 0.8}


def test_evidence_builder_single_entry_narration_context_is_flat():
    builder = EvidenceBuilder()
    builder.add("trend", "TrendAnalyzer", Evidence(evidence={"direction": "increasing"}))
    assert builder.to_narration_context() == {"direction": "increasing"}


def test_evidence_builder_multi_entry_narration_context_is_nested_by_type():
    builder = EvidenceBuilder()
    builder.add("trend", "TrendAnalyzer", Evidence(evidence={"direction": "increasing"}))
    builder.add("correlation", "CorrelationAnalyzer", Evidence(evidence={"strongest_pair": None}))
    ctx = builder.to_narration_context()
    assert set(ctx["analyses"].keys()) == {"trend", "correlation"}
    assert ctx["analyses"]["trend"]["direction"] == "increasing"


def test_evidence_builder_empty_narration_context_is_empty_dict():
    assert EvidenceBuilder().to_narration_context() == {}
    assert EvidenceBuilder().is_empty()


def test_evidence_builder_by_type_filters_correctly():
    builder = EvidenceBuilder()
    builder.add("trend", "A", Evidence(evidence={"x": 1}))
    builder.add("trend", "B", Evidence(evidence={"x": 2}))
    builder.add("correlation", "C", Evidence(evidence={"x": 3}))
    assert len(builder.by_type("trend")) == 2
    assert len(builder.by_type("correlation")) == 1
    assert len(builder.by_type("nonexistent")) == 0


# ── Analyzer Template Method skeleton (base.py) ─────────────────────────────

class _StubAnalyzer(Analyzer):
    analysis_type = "stub"

    def __init__(self, payload):
        self._payload = payload

    def _run(self, *, df, dataset_context, target_columns, selected_model, extra_context):
        if callable(self._payload):
            return self._payload()
        return self._payload


def test_execute_returns_empty_evidence_when_no_algorithm_selected():
    analyzer = _StubAnalyzer({})
    selected = SelectedModel(algorithm=None, implementation_class=None, requires_ml=False, cost_tier=None, reasons=["no candidates"])
    result = analyzer.execute(pd.DataFrame(), _fixture_context(_fixture_df()), _scheduled("stub", []), selected)
    assert result.evidence == {}
    assert "no candidates" in " ".join(result.reasons)


def test_execute_wraps_run_exceptions_instead_of_raising():
    def _boom():
        raise ValueError("synthetic failure")
    analyzer = _StubAnalyzer(_boom)
    selected = SelectedModel(algorithm="X", implementation_class="app.services.analytics_tool.AnalyticsTool", requires_ml=False, cost_tier="cheap", reasons=[])
    result = analyzer.execute(pd.DataFrame(), _fixture_context(_fixture_df()), _scheduled("stub", []), selected)
    assert result.evidence == {}
    assert any("synthetic failure" in r for r in result.reasons)


def test_execute_ml_path_sets_model_metadata_and_confidence():
    analyzer = _StubAnalyzer({"evidence": {"r2_score": 0.9}})
    selected = SelectedModel(algorithm="Random Forest Regressor", implementation_class="app.services.analyzers.regression_strategies.RandomForestRegressorStrategy", requires_ml=True, cost_tier="moderate", reasons=[])
    result = analyzer.execute(pd.DataFrame(), _fixture_context(_fixture_df()), _scheduled("stub", []), selected, ml_confidence=0.87)
    assert result.model_metadata == {"algorithm": "Random Forest Regressor", "cost_tier": "moderate"}
    assert result.confidence == 0.87
    assert result.fallback_metadata is None


def test_execute_deterministic_path_sets_fallback_metadata_not_model_metadata():
    analyzer = _StubAnalyzer({"evidence": {"trend_slope": 1.2}})
    selected = SelectedModel(algorithm="Linear Trend", implementation_class="app.services.analyzers.forecast_strategies.LinearTrendStrategy", requires_ml=False, cost_tier="cheap", reasons=["Scheduler budget did not allocate an ML slot to this analysis — deterministic only"])
    result = analyzer.execute(pd.DataFrame(), _fixture_context(_fixture_df()), _scheduled("stub", []), selected)
    assert result.model_metadata is None
    assert result.fallback_metadata["fallback_applied"] == "Linear Trend"
    assert result.fallback_metadata["ml_readiness_blocked"] is False
    assert result.confidence is None


def test_execute_deterministic_path_flags_ml_readiness_blocked_when_execution_gate_failed():
    analyzer = _StubAnalyzer({"evidence": {}})
    selected = SelectedModel(algorithm="Z-Score", implementation_class="app.services.analyzers.anomaly_strategies.ZScoreStrategy", requires_ml=False, cost_tier="cheap", reasons=["Scheduler allowed ML, but capability resolution's execution gate did not — deterministic only"])
    result = analyzer.execute(pd.DataFrame(), _fixture_context(_fixture_df()), _scheduled("stub", []), selected)
    assert result.fallback_metadata["ml_readiness_blocked"] is True


# ── AnalyzerRegistry completeness ────────────────────────────────────────────

def test_analyzer_registry_covers_every_enabled_registry_purpose():
    registry = ModelRegistry()
    purposes = {spec.purpose for spec in registry.all()}
    analyzer_types = set(AnalyzerRegistry().all().keys())
    assert purposes == analyzer_types


def test_analyzer_registry_get_returns_none_for_unknown_type():
    assert AnalyzerRegistry().get("not_a_real_analysis_type") is None


# ── Forecast ─────────────────────────────────────────────────────────────────

def test_forecast_moving_average_deterministic():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("forecast", "Moving Average")
    ev = ForecastAnalyzer().execute(df, ctx, _scheduled("forecast", ["order_date", "revenue"]), selected)
    assert len(ev.evidence["forecast"]) == 6
    assert ev.evidence["model"] == "Moving Average"
    assert ev.fallback_metadata["fallback_applied"] == "Moving Average"


def test_forecast_linear_trend_reports_slope():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("forecast", "Linear Trend")
    ev = ForecastAnalyzer().execute(df, ctx, _scheduled("forecast", ["order_date", "revenue"]), selected)
    assert "trend_slope" in ev.evidence
    assert len(ev.evidence["forecast"]) == 6


def test_forecast_missing_target_columns_produces_no_evidence():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("forecast", "Linear Trend")
    ev = ForecastAnalyzer().execute(df, ctx, _scheduled("forecast", ["order_date", "nonexistent_col"]), selected)
    assert ev.evidence == {}
    assert any("not found" in r for r in ev.reasons)


# ── Trend / Ranking / Comparative (AnalyticsTool wrappers) ─────────────────

def test_trend_analyzer_reports_direction():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("trend", "Trend (linear regression over time)")
    ev = TrendAnalyzer().execute(df, ctx, _scheduled("trend", ["order_date", "revenue"]), selected)
    assert ev.evidence["direction"] in {"increasing", "decreasing", "flat"}
    assert ev.evidence["data_points"] == 60


def test_ranking_analyzer_returns_top_n():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("ranking", "Ranking")
    ev = RankingAnalyzer().execute(df, ctx, _scheduled("ranking", ["revenue", "region"]), selected)
    assert len(ev.evidence["ranked_records"]) == 10
    assert ev.evidence["ranked_records"][0]["rank"] == 1


def test_comparative_analyzer_ranks_groups():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("comparative_analysis", "Comparative Groupby")
    ev = ComparativeAnalyzer().execute(df, ctx, _scheduled("comparative_analysis", ["revenue", "region"]), selected)
    assert ev.evidence["group_count"] == 4
    assert ev.evidence["highest"]["revenue"] >= ev.evidence["lowest"]["revenue"]


# ── Correlation ──────────────────────────────────────────────────────────────

def test_correlation_pearson_finds_revenue_cost_relationship():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("correlation", "Pearson")
    ev = CorrelationAnalyzer().execute(df, ctx, _scheduled("correlation", ["revenue", "cost"]), selected)
    assert ev.evidence["strongest_pair"]["column_a"] == "revenue"
    assert ev.evidence["strongest_pair"]["correlation"] > 0.5  # constructed to correlate


def test_correlation_spearman_runs_and_returns_matrix():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("correlation", "Spearman")
    ev = CorrelationAnalyzer().execute(df, ctx, _scheduled("correlation", ["revenue", "cost"]), selected)
    assert ev.evidence["method"] == "Spearman"
    assert ev.evidence["correlation_matrix"]["revenue"]["revenue"] == 1.0


# ── Segmentation / Clustering ────────────────────────────────────────────────

def test_segmentation_quantile_binning():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("segmentation", "Quantile Binning")
    ev = SegmentationAnalyzer().execute(df, ctx, _scheduled("segmentation", ["revenue"]), selected)
    assert sum(ev.evidence["segments"].values()) == 60
    assert len(ev.evidence["segments"]) == 4


def test_segmentation_categorical_grouping():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("segmentation", "Categorical Grouping")
    ev = SegmentationAnalyzer().execute(df, ctx, _scheduled("segmentation", ["region"]), selected)
    assert set(ev.evidence["segments"].keys()) == {"North", "South", "East", "West"}


def test_segmentation_dbscan_flags_noise_points_key():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("segmentation", "DBSCAN")
    ev = SegmentationAnalyzer().execute(df, ctx, _scheduled("segmentation", ["revenue", "cost"]), selected)
    assert "noise_points" in ev.evidence
    assert ev.model_metadata["algorithm"] == "DBSCAN"


def test_clustering_analyzer_quantile_binning_reuses_segmentation_strategy():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("clustering", "Quantile Binning (clustering)")
    ev = ClusteringAnalyzer().execute(df, ctx, _scheduled("clustering", ["revenue"]), selected)
    assert sum(ev.evidence["segments"].values()) == 60


# ── Anomaly / Outlier ────────────────────────────────────────────────────────

def test_anomaly_z_score_flags_injected_outlier():
    df = _fixture_df()
    df.loc[0, "revenue"] = 100000  # egregious outlier
    ctx = _fixture_context(df)
    selected = _model_for("anomaly_detection", "Z-Score")
    ev = AnomalyAnalyzer().execute(df, ctx, _scheduled("anomaly_detection", ["revenue", "cost"]), selected)
    assert ev.evidence["anomaly_count"] >= 1
    flagged_scores = [a["revenue"] for a in ev.evidence["anomalies"]]
    assert 100000 in flagged_scores


def test_anomaly_iqr_and_modified_z_score_run_without_error():
    df = _fixture_df()
    df.loc[0, "revenue"] = 100000
    ctx = _fixture_context(df)
    for algorithm in ("IQR", "Modified Z-Score (MAD)"):
        selected = _model_for("anomaly_detection", algorithm)
        ev = AnomalyAnalyzer().execute(df, ctx, _scheduled("anomaly_detection", ["revenue", "cost"]), selected)
        assert ev.evidence.get("anomaly_count") is not None, f"{algorithm} produced no evidence: {ev.reasons}"


def test_anomaly_local_outlier_factor_ml_path():
    df = _fixture_df()
    df.loc[0, "revenue"] = 100000
    ctx = _fixture_context(df)
    selected = _model_for("anomaly_detection", "Local Outlier Factor")
    ev = AnomalyAnalyzer().execute(df, ctx, _scheduled("anomaly_detection", ["revenue", "cost"]), selected, ml_confidence=0.9)
    assert ev.model_metadata["algorithm"] == "Local Outlier Factor"
    assert ev.confidence == 0.9


def test_outlier_analyzer_is_distinct_analysis_type_reusing_zscore():
    df = _fixture_df()
    ctx = _fixture_context(df)
    selected = _model_for("outlier_detection", "Z-Score (outlier_detection)")
    ev = OutlierAnalyzer().execute(df, ctx, _scheduled("outlier_detection", ["revenue", "cost"]), selected)
    assert "anomaly_count" in ev.evidence
    assert OutlierAnalyzer.analysis_type != AnomalyAnalyzer.analysis_type


# ── Classification / Regression ─────────────────────────────────────────────

def test_classification_majority_class_baseline():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("classification", "Majority Class Baseline")
    ev = ClassificationAnalyzer().execute(df, ctx, _scheduled("classification", ["status"]), selected)
    assert ev.evidence["predicted_class"] == "success"
    assert ev.evidence["accuracy"] == pytest.approx(0.8, abs=0.1)


def test_classification_logistic_regression_ml_path():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("classification", "Logistic Regression")
    ev = ClassificationAnalyzer().execute(df, ctx, _scheduled("classification", ["status", "revenue", "cost"]), selected, ml_confidence=0.95)
    assert "accuracy" in ev.evidence
    assert ev.model_metadata["algorithm"] == "Logistic Regression"


def test_regression_linear_regression_deterministic():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("regression", "Linear Regression")
    ev = RegressionAnalyzer().execute(df, ctx, _scheduled("regression", ["revenue", "cost"]), selected)
    assert ev.evidence["r2_score"] > 0.2  # revenue/cost constructed to correlate
    assert ev.fallback_metadata["fallback_applied"] == "Linear Regression"


def test_regression_random_forest_ml_path():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("regression", "Random Forest Regressor")
    ev = RegressionAnalyzer().execute(df, ctx, _scheduled("regression", ["revenue", "cost", "region"]), selected, ml_confidence=0.8)
    assert "r2_score" in ev.evidence
    assert ev.model_metadata["algorithm"] == "Random Forest Regressor"


# ── Feature Importance ───────────────────────────────────────────────────────

def test_feature_importance_correlation_based():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("feature_importance", "Correlation-Based Importance")
    ev = FeatureImportanceAnalyzer().execute(df, ctx, _scheduled("feature_importance", ["revenue", "cost"]), selected)
    assert ev.evidence["feature_contributions"][0]["feature"] == "cost"


# ── Distribution / Time Series / Association ────────────────────────────────

def test_distribution_analysis_reports_percentiles():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("distribution_analysis", "Descriptive Distribution")
    ev = DistributionAnalyzer().execute(df, ctx, _scheduled("distribution_analysis", ["revenue"]), selected)
    assert "p50" in ev.evidence["percentiles"]
    assert ev.evidence["count"] == 60


def test_time_series_analysis_reports_summary():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("time_series_analysis", "Time Series Descriptive")
    ev = TimeSeriesAnalyzer().execute(df, ctx, _scheduled("time_series_analysis", ["order_date", "revenue"]), selected)
    assert ev.evidence["data_points"] == 60
    assert "seasonality_flag" in ev.evidence


def test_association_rules_contingency_table():
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("association_rules", "Contingency Table Co-occurrence")
    ev = AssociationAnalyzer().execute(df, ctx, _scheduled("association_rules", ["region", "status"]), selected)
    assert ev.evidence["pairs_analysed"] == 1
    assert ev.evidence["results"][0]["column_a"] == "region"


# ── Root Cause: the plan's explicitly flagged risk (labeled vs. correlation) ─

def test_root_cause_labeled_mode_used_when_driver_columns_available():
    """Mirrors Insurance's exact preserved behavior — when the Planner
    supplies known driver columns (from a domain plugin), RootCauseTool's
    labeled mode runs unchanged."""
    df = pd.DataFrame({
        "underwriting_result_actual": [100.0, 200.0],
        "exposure_growth_variance": [40.0, 10.0],
        "premium_rate_variance": [-10.0, 90.0],
    })
    ctx = _fixture_context(_fixture_df())
    selected = _model_for("root_cause", "Deterministic Driver Decomposition")
    ev = RootCauseAnalyzer().execute(
        df, ctx,
        _scheduled("root_cause", ["underwriting_result_actual", "exposure_growth_variance", "premium_rate_variance"]),
        selected,
    )
    assert ev.evidence["mode"] == "labeled_driver"
    assert ev.evidence["primary_driver"]["driver"] in {"exposure_growth_variance", "premium_rate_variance"}


def test_root_cause_correlation_mode_used_when_no_driver_columns_known():
    """The new generic default (plan's Stage 7 section) — no plugin/labels
    means correlate every other numeric column against the outcome metric
    instead of erroring out or silently doing nothing."""
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("root_cause", "Deterministic Driver Decomposition")
    ev = RootCauseAnalyzer().execute(df, ctx, _scheduled("root_cause", ["revenue"]), selected)
    assert ev.evidence["mode"] == "correlation_based"
    assert ev.evidence["primary_driver"]["driver"] == "cost"  # revenue/cost constructed to correlate
    assert "amount" in ev.evidence["primary_driver"]  # same key the template formatter already reads


def test_root_cause_falls_back_to_correlation_mode_when_driver_columns_not_in_data():
    """Driver columns were requested (e.g. a stale plugin config) but
    aren't actually present in this dataset — correlation mode is the
    safe default rather than an error."""
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("root_cause", "Deterministic Driver Decomposition")
    ev = RootCauseAnalyzer().execute(df, ctx, _scheduled("root_cause", ["revenue", "totally_made_up_variance_col"]), selected)
    assert ev.evidence["mode"] == "correlation_based"


def test_root_cause_correlation_mode_reports_candidate_pool_used():
    """Transparency flag so a caller can tell whether the heuristic
    driver-named-column narrowing fired or it fell back to every numeric
    column — no column in this fixture is driver-shaped, so it's the
    all-numeric-columns fallback."""
    df, ctx = _fixture_df(), _fixture_context(_fixture_df())
    selected = _model_for("root_cause", "Deterministic Driver Decomposition")
    ev = RootCauseAnalyzer().execute(df, ctx, _scheduled("root_cause", ["revenue"]), selected)
    assert ev.evidence["candidate_pool"] == "all_numeric_columns"


def test_root_cause_correlation_mode_narrows_to_heuristic_driver_named_columns():
    """A real column ('cost') correlates just as strongly with 'revenue' as
    the driver-named one, but once ANY column looks driver-shaped by name
    (contains variance/driver/contribution/impact), only those are
    considered — the whole point being a more targeted, less noisy answer
    than dumping every numeric column into the correlation, without ever
    entering labeled mode (still discovered statistically, not assumed)."""
    df = _fixture_df()
    df["unrelated_metric"] = df["revenue"] * 2 + 1  # would dominate if considered
    df["region_variance"] = df["cost"] * 0.5  # driver-named, correlates with revenue via cost
    ctx = _fixture_context(df)
    ctx.columns.append(ColumnContext(name="unrelated_metric", semantic_role=SEMANTIC_ROLE_METRIC))
    ctx.columns.append(ColumnContext(name="region_variance", semantic_role=SEMANTIC_ROLE_METRIC))

    selected = _model_for("root_cause", "Deterministic Driver Decomposition")
    ev = RootCauseAnalyzer().execute(df, ctx, _scheduled("root_cause", ["revenue"]), selected)

    assert ev.evidence["candidate_pool"] == "heuristic_driver_named_columns"
    drivers = {d["driver"] for d in ev.evidence["driver_breakdown"]}
    assert drivers == {"region_variance"}  # "cost" and "unrelated_metric" excluded once a driver-named column exists


def test_kpi_discovered_root_cause_falls_through_to_correlation_not_a_trivial_self_explanation():
    """Regression test for the exact bug caught via live testing against a
    Finance dataset with no domain plugin: a Stage-2-discovered KPI's
    non-primary source columns (e.g. a budget/reference column) are KPI
    *inputs*, not a real driver decomposition — treating a lone such
    column as "the driver" trivially "explains" 100% of itself. The
    Planner now only ever proposes target_columns=[primary_metric] for a
    generically-discovered KPI (planning_rules.rule_kpi_grounded_root_cause),
    so RootCauseAnalyzer has no driver tail to attempt labeled mode with."""
    df = _fixture_df()
    ctx = _fixture_context(df)
    selected = _model_for("root_cause", "Deterministic Driver Decomposition")
    # Single-column target_columns, exactly what the fixed planning rule now produces.
    ev = RootCauseAnalyzer().execute(df, ctx, _scheduled("root_cause", ["revenue"]), selected)
    assert ev.evidence["mode"] == "correlation_based"
    # The bug: labeled mode would have summed a lone "driver" column (e.g.
    # a KPI's own reference/budget input) and trivially reported it as
    # 100% of the variance, purely because it was the only column present
    # — not because it was statistically meaningful. mode ==
    # "correlation_based" (not "labeled_driver") is the regression guard.
    assert ev.evidence["primary_driver"]["driver"] != "revenue"
