"""tests/test_analytics_tool.py — Unit tests for the Analytics Tool"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from app.services.analytics_tool import AnalyticsTool


def _sample_df():
    return pd.DataFrame({
        "region":   ["EMEA", "APAC", "Africa"],
        "actual":   [1000.0, 800.0, 600.0],
        "budget":   [1100.0, 750.0, 650.0],
    })


def test_aggregate_sum():
    tool = AnalyticsTool()
    df = _sample_df()
    result = tool.aggregate(df, "actual", "sum")
    assert result == 2400.0


def test_aggregate_mean():
    tool = AnalyticsTool()
    df = _sample_df()
    result = tool.aggregate(df, "actual", "mean")
    assert abs(result - 800.0) < 0.01


def test_variance_vs_budget_favorable():
    tool = AnalyticsTool()
    result = tool.variance_vs_budget(1100.0, 1000.0)
    assert result["variance_amount"] == 100.0
    assert result["direction"] == "favorable"


def test_variance_vs_budget_unfavorable():
    tool = AnalyticsTool()
    result = tool.variance_vs_budget(900.0, 1000.0)
    assert result["variance_amount"] == -100.0
    assert result["direction"] == "unfavorable"


def test_yoy_growth():
    tool = AnalyticsTool()
    result = tool.yoy_growth(1100.0, 1000.0)
    assert abs(result["growth_pct"] - 10.0) < 0.01


def test_trend_increasing():
    tool = AnalyticsTool()
    df = pd.DataFrame({
        "date": ["2024-01", "2024-02", "2024-03", "2024-04"],
        "val":  [100, 110, 120, 130],
    })
    result = tool.trend(df, "date", "val")
    assert result["direction"] == "increasing"
    assert result["data_points"] == 4


def test_ranking():
    tool = AnalyticsTool()
    df = _sample_df()
    ranked = tool.rank(df, "actual", "region", top_n=2)
    assert ranked.iloc[0]["region"] == "EMEA"
    assert len(ranked) == 2


def test_contribution():
    tool = AnalyticsTool()
    df = _sample_df()
    contrib = tool.contribution(df, "actual", "region")
    assert abs(contrib["contribution_pct"].sum() - 100.0) < 0.01


def test_loss_ratio_calculation():
    tool = AnalyticsTool()
    lr = tool.calculate_loss_ratio(750.0, 1000.0)
    assert lr == 75.0


def test_combined_ratio():
    tool = AnalyticsTool()
    cr = tool.calculate_combined_ratio(75.0, 28.0)
    assert cr == 103.0


if __name__ == "__main__":
    test_aggregate_sum()
    test_aggregate_mean()
    test_variance_vs_budget_favorable()
    test_variance_vs_budget_unfavorable()
    test_yoy_growth()
    test_trend_increasing()
    test_ranking()
    test_contribution()
    test_loss_ratio_calculation()
    test_combined_ratio()
    print("✅ All Analytics Tool tests passed")
