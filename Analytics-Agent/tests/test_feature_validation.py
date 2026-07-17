"""tests/test_feature_validation.py — Tests for ml/feature_validation.py,
added to cross-check Agent 3's hardcoded per-model feature column lists
against Agent 2's per-upload column classification at boot time."""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pytest

from ml.feature_validation import validate_feature_columns
from config import ISO_FOREST_CFG, KMEANS_CFG, XGBOOST_CFG, LGBM_CFG


def _complete_fixture() -> list[dict]:
    """A feature-recommendation covering every column every hardcoded list
    expects, all correctly roled — the "nothing wrong" baseline other tests
    mutate one entry of."""
    cols = []
    for col in ISO_FOREST_CFG["feature_columns"]:
        cols.append({"column": col, "role": "metric"})
    for col in KMEANS_CFG["feature_columns"]:
        if col not in {c["column"] for c in cols}:
            cols.append({"column": col, "role": "metric"})
    for col in XGBOOST_CFG["feature_columns"]:
        if col not in {c["column"] for c in cols}:
            cols.append({"column": col, "role": "metric"})
    for col in LGBM_CFG["categorical_features"]:
        cols.append({"column": col, "role": "dimension"})
    return cols


def _write(tmp_path: Path, data) -> str:
    p = tmp_path / "feature_recommendation.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def test_no_path_is_a_clean_noop(caplog):
    with caplog.at_level(logging.INFO):
        validate_feature_columns(None)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    assert "skipping column validation" in caplog.text


def test_nonexistent_file_warns_but_does_not_raise(caplog):
    # An unreadable file (missing, permission error, ...) is reported as a
    # warning — distinct from the None case, which is the expected/silent
    # "no recommendation was ever supplied" path — but never raises, since
    # this check must never block the agent from starting.
    with caplog.at_level(logging.WARNING):
        validate_feature_columns(r"C:\definitely\does\not\exist.json")
    assert any("Could not read feature recommendation" in r.message for r in caplog.records)


def test_all_correct_produces_no_warnings(tmp_path, caplog):
    path = _write(tmp_path, _complete_fixture())
    with caplog.at_level(logging.INFO):
        validate_feature_columns(path)
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)
    assert "all hardcoded model columns match" in caplog.text


def test_missing_column_warns_with_model_and_column_named(tmp_path, caplog):
    fixture = [c for c in _complete_fixture() if c["column"] != "loss_ratio_actual"]
    path = _write(tmp_path, fixture)
    with caplog.at_level(logging.WARNING):
        validate_feature_columns(path)
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("loss_ratio_actual" in w and "doesn't have it" in w for w in warnings)
    # Both IsolationForest and KMeans use this column — expect both named.
    assert any("IsolationForest" in w and "loss_ratio_actual" in w for w in warnings)
    assert any("KMeans" in w and "loss_ratio_actual" in w for w in warnings)


def test_wrong_role_warns_with_expected_vs_actual(tmp_path, caplog):
    fixture = _complete_fixture()
    for c in fixture:
        if c["column"] == "loss_ratio_actual":
            c["role"] = "dimension"
    path = _write(tmp_path, fixture)
    with caplog.at_level(logging.WARNING):
        validate_feature_columns(path)
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "loss_ratio_actual" in w and "'metric'" in w and "'dimension'" in w
        for w in warnings
    )


def test_malformed_json_is_a_clean_noop(tmp_path, caplog):
    p = tmp_path / "broken.json"
    p.write_text("{not valid json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        validate_feature_columns(str(p))
    assert any("Could not read feature recommendation" in r.message for r in caplog.records)
