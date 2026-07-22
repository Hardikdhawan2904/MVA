"""app/services/analyzers/feature_importance_strategies.py — Stage 7 (plan
"zany-giggling-crayon"): feature-importance strategies registered in
config/model_registry.yml alongside SHAP (VarianceClassifier).
"""

from __future__ import annotations

import pandas as pd


class PermutationImportanceStrategy:
    """scikit-learn permutation_importance on a quick RandomForest fit —
    model-agnostic, doesn't need a persisted classifier/regressor like
    SHAP's TreeExplainer path does."""

    def compute(self, df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> dict:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.preprocessing import LabelEncoder

        cols = [c for c in feature_cols if c in df.columns]
        if not cols or target_col not in df.columns:
            return {"error": f"Feature importance target/feature columns not found: target={target_col!r}, features={cols}"}

        data = df[cols + [target_col]].copy()
        for col in cols:
            if data[col].dtype == object:
                data[col] = LabelEncoder().fit_transform(data[col].astype(str))
            data[col] = pd.to_numeric(data[col], errors="coerce")

        is_numeric_target = pd.api.types.is_numeric_dtype(df[target_col])
        if is_numeric_target:
            data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
        else:
            data[target_col] = LabelEncoder().fit_transform(data[target_col].astype(str))
        data = data.dropna()

        if len(data) < 20:
            return {"error": f"Insufficient rows for Permutation Importance: {len(data)} (need >= 20)"}

        X, y = data[cols], data[target_col]
        model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
        model.fit(X, y)

        result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        fi_df = pd.DataFrame({
            "feature": cols,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        fi_df["rank"] = fi_df.index + 1

        return {
            "model": "Permutation Importance",
            "target": target_col,
            "rows_used": len(data),
            "feature_contributions": fi_df.round(4).to_dict(orient="records"),
        }


class CorrelationBasedImportanceStrategy:
    """|Pearson correlation| with the target as a proxy for feature
    importance — cheap, deterministic, target must be numeric."""

    def compute(self, df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> dict:
        cols = [c for c in feature_cols if c in df.columns]
        if not cols or target_col not in df.columns:
            return {"error": f"Feature importance target/feature columns not found: target={target_col!r}, features={cols}"}

        data = df[cols + [target_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(data) < 5:
            return {"error": f"Insufficient rows for Correlation-Based Importance: {len(data)} (need >= 5)"}

        correlations = data[cols].corrwith(data[target_col]).abs()
        fi_df = correlations.sort_values(ascending=False).reset_index()
        fi_df.columns = ["feature", "importance"]
        fi_df["rank"] = fi_df.index + 1

        return {
            "model": "Correlation-Based Importance",
            "target": target_col,
            "rows_used": len(data),
            "feature_contributions": fi_df.round(4).to_dict(orient="records"),
        }
