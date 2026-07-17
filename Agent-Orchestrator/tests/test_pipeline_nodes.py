"""tests/test_pipeline_nodes.py — unit tests for pure-data helpers in
app/agents/orchestration_agent/nodes/pipeline.py.

_readiness_and_features() is the one function these cover: it used to
extract only the bare score float from Agent 2's readiness_assessments,
discarding strengths/blocking_issues/evidence — the fix threads the full
assessment dict through too, so Agent 3's execution_trace can explain a
readiness gate, not just report its score.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agents.orchestration_agent.nodes.pipeline import _readiness_and_features


def _agent2_result(ml_score=82.0, llm_score=95.0):
    return {
        "readiness_assessments": [
            {
                "assessment_type": "ml_readiness",
                "score": ml_score,
                "status": "ready",
                "strengths": [{"code": "HIGH_COMPLETENESS", "value": 0.95}],
                "blocking_issues": [],
                "recommendations": [],
                "evidence": [
                    {"dimension": "completeness", "value": 0.95},
                    {"dimension": "feature_coverage", "value": 1.0},
                    {"dimension": "data_freshness", "value": 0.55},
                ],
                "weight_profile_version": "ml-v1",
            },
            {
                "assessment_type": "llm_readiness",
                "score": llm_score,
                "status": "ready",
                "strengths": [],
                "blocking_issues": [],
                "recommendations": [],
                "evidence": [{"dimension": "description_coverage", "value": 0.9}],
                "weight_profile_version": "llm-v1",
            },
        ],
        "feature_recommendation": {"feature_columns": [{"column": "x", "usefulness": "high"}]},
    }


def test_readiness_and_features_returns_scores_and_full_breakdowns():
    ml_score, llm_score, feature_columns, ml_breakdown, llm_breakdown = _readiness_and_features(
        _agent2_result(),
    )

    assert ml_score == 82.0
    assert llm_score == 95.0
    assert feature_columns == [{"column": "x", "usefulness": "high"}]

    assert ml_breakdown["assessment_type"] == "ml_readiness"
    assert ml_breakdown["evidence"][2] == {"dimension": "data_freshness", "value": 0.55}
    assert llm_breakdown["assessment_type"] == "llm_readiness"


def test_readiness_and_features_handles_missing_assessments():
    ml_score, llm_score, feature_columns, ml_breakdown, llm_breakdown = _readiness_and_features(
        {"readiness_assessments": []},
    )
    assert ml_score is None
    assert llm_score is None
    assert feature_columns == []
    assert ml_breakdown is None
    assert llm_breakdown is None


def test_readiness_and_features_handles_none_input():
    ml_score, llm_score, feature_columns, ml_breakdown, llm_breakdown = _readiness_and_features(None)
    assert ml_score is None
    assert llm_score is None
    assert feature_columns == []
    assert ml_breakdown is None
    assert llm_breakdown is None
