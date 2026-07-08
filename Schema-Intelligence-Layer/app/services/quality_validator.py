"""
Data Quality Validation Service.
Executes configured quality checks on a sample of the dataset.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Union, Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

# Resolve config path relative to service location
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "quality_threshold.json",
)


class BaseQualityCheck:
    """Interface for modular dataset quality checks."""

    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        """
        Execute check on sample DataFrame.
        
        Args:
            df_sample: The sampled Pandas DataFrame to inspect.
            weight: Max weight/score allocated for this check in configuration.
            limits: Dictionary containing limit rules, thresholds, and parameters.
            
        Returns:
            A tuple of (achieved_score, optional_warning_string).
        """
        raise NotImplementedError("Check must implement validate method")


# 1. Column Count Check
class ColumnCountCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        min_cols = limits.get("min_columns", 1)
        col_count = len(df_sample.columns)
        if col_count < min_cols:
            warning = f"Dataset column count {col_count} is below the minimum limit of {min_cols}."
            return 0.0, warning
        return weight, None


# 2. Missing Values Check
class MissingValuesCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if df_sample.size == 0:
            return weight, None
        total_missing = df_sample.isna().sum().sum()
        pct_missing = (total_missing / df_sample.size) * 100
        
        # Linear score deduction based on missing cells
        achieved_score = weight * (1.0 - (pct_missing / 100.0))
        
        warning = None
        max_allowed = limits.get("max_missing_values_pct", 20.0)
        if pct_missing > max_allowed:
            warning = f"Missing values density ({pct_missing:.1f}%) exceeds threshold of {max_allowed}%."
            
        return max(0.0, achieved_score), warning


# 3. Duplicate Columns Check
class DuplicateColumnsCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if len(df_sample.columns) == 0:
            return weight, None
        
        # Check duplicate column names
        dup_cols = df_sample.columns.duplicated().sum()
        pct_dup = (dup_cols / len(df_sample.columns)) * 100
        
        achieved_score = weight * (1.0 - (dup_cols / len(df_sample.columns)))
        
        warning = None
        if dup_cols > 0:
            warning = f"Detected {dup_cols} duplicate column names ({pct_dup:.1f}% duplicate rate)."
            
        return max(0.0, achieved_score), warning


# 4. Duplicate Rows Check
class DuplicateRowsCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if len(df_sample) == 0:
            return weight, None
        
        # Count duplicate rows
        dup_rows = df_sample.duplicated().sum()
        pct_dup = (dup_rows / len(df_sample)) * 100
        
        achieved_score = weight * (1.0 - (dup_rows / len(df_sample)))
        
        warning = None
        max_allowed = limits.get("max_duplicate_rows_pct", 10.0)
        if pct_dup > max_allowed:
            warning = f"Duplicate rows ({pct_dup:.1f}%) exceeds threshold of {max_allowed}%."
            
        return max(0.0, achieved_score), warning


# 5. Empty Columns Check
class EmptyColumnsCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if len(df_sample.columns) == 0:
            return weight, None
        
        empty_cols = sum(df_sample[col].isna().all() for col in df_sample.columns)
        pct_empty = (empty_cols / len(df_sample.columns)) * 100
        
        achieved_score = weight * (1.0 - (empty_cols / len(df_sample.columns)))
        
        warning = None
        max_allowed = limits.get("max_empty_columns_pct", 10.0)
        if pct_empty > max_allowed:
            warning = f"Entirely empty columns ({pct_empty:.1f}%) exceeds threshold of {max_allowed}%."
            
        return max(0.0, achieved_score), warning


# 6. Datatype Consistency Check
class DatatypeConsistencyCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if len(df_sample.columns) == 0:
            return weight, None
        
        col_consistencies = []
        for col in df_sample.columns:
            non_null = df_sample[col].dropna()
            if non_null.empty:
                col_consistencies.append(1.0)
                continue
            
            # Categorize python basic groups
            type_groups = []
            for val in non_null:
                if isinstance(val, bool):
                    type_groups.append("bool")
                elif isinstance(val, (int, float)):
                    type_groups.append("numeric")
                elif isinstance(val, (pd.Timestamp, datetime)):
                    type_groups.append("datetime")
                else:
                    type_groups.append("string")
            
            dominant_count = max(type_groups.count(t) for t in set(type_groups))
            col_consistencies.append(dominant_count / len(type_groups))
            
        overall_consistency = np.mean(col_consistencies) if col_consistencies else 1.0
        achieved_score = weight * overall_consistency
        
        warning = None
        min_allowed = limits.get("min_datatype_consistency_pct", 80.0)
        if (overall_consistency * 100.0) < min_allowed:
            warning = f"Datatype consistency ({overall_consistency*100.0:.1f}%) is below minimum requirements of {min_allowed}%."
            
        return max(0.0, achieved_score), warning


# 7. Corrupted Values Check
class CorruptedValuesCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if df_sample.size == 0:
            return weight, None
        
        placeholders = [p.lower() for p in limits.get("corrupted_placeholders", [])]
        corrupted_count = 0
        total_valid = 0
        
        for col in df_sample.columns:
            for val in df_sample[col]:
                if pd.notna(val):
                    total_valid += 1
                    val_str = str(val).strip().lower()
                    if val_str in placeholders:
                        corrupted_count += 1
                        
        if total_valid == 0:
            return weight, None
            
        pct_corrupted = (corrupted_count / total_valid) * 100
        achieved_score = weight * (1.0 - (corrupted_count / total_valid))
        
        warning = None
        max_allowed = limits.get("max_corrupted_values_pct", 5.0)
        if pct_corrupted > max_allowed:
            warning = f"Corrupted placeholders ({pct_corrupted:.1f}%) exceeds threshold of {max_allowed}%."
            
        return max(0.0, achieved_score), warning


# 8. Null-heavy Rows Check
class NullHeavyRowsCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if len(df_sample) == 0:
            return weight, None
            
        threshold = limits.get("null_heavy_row_pct_threshold", 50.0)
        null_heavy_count = 0
        
        for _, row in df_sample.iterrows():
            row_null_pct = row.isna().mean() * 100
            if row_null_pct > threshold:
                null_heavy_count += 1
                
        pct_null_heavy = (null_heavy_count / len(df_sample)) * 100
        achieved_score = weight * (1.0 - (null_heavy_count / len(df_sample)))
        
        warning = None
        max_allowed = limits.get("max_null_heavy_rows_pct", 10.0)
        if pct_null_heavy > max_allowed:
            warning = f"Null-heavy rows ({pct_null_heavy:.1f}%) exceeds threshold of {max_allowed}%."
            
        return max(0.0, achieved_score), warning


# 9. Cell Length Outliers Check
class CellLengthOutliersCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if df_sample.size == 0:
            return weight, None
            
        outliers_count = 0
        total_strings = 0
        
        for col in df_sample.columns:
            non_null = df_sample[col].dropna()
            lengths = [len(str(val)) for val in non_null if isinstance(val, str)]
            if len(lengths) < 3:
                continue
                
            total_strings += len(lengths)
            mean_len = np.mean(lengths)
            std_len = np.std(lengths)
            
            if std_len > 0:
                # Count lengths outside 3 standard deviations
                outliers_count += sum(abs(l - mean_len) > 3 * std_len for l in lengths)
                
        if total_strings == 0:
            return weight, None
            
        pct_outliers = (outliers_count / total_strings) * 100
        achieved_score = weight * (1.0 - (outliers_count / total_strings))
        
        warning = None
        max_allowed = limits.get("max_cell_length_outliers_pct", 5.0)
        if pct_outliers > max_allowed:
            warning = f"Cell length outliers ({pct_outliers:.1f}%) exceeds threshold of {max_allowed}%."
            
        return max(0.0, achieved_score), warning


# 10. Mixed Data Formats Check
class MixedFormatsCheck(BaseQualityCheck):
    def validate(
        self, df_sample: pd.DataFrame, weight: float, limits: Dict[str, Any]
    ) -> Tuple[float, Optional[str]]:
        if len(df_sample.columns) == 0:
            return weight, None
            
        casing_consistencies = []
        for col in df_sample.columns:
            non_null = df_sample[col].dropna()
            strings = [str(x) for x in non_null if isinstance(x, str) and x.strip() != ""]
            if not strings:
                continue
                
            casing_types = []
            for s in strings:
                if s.isupper():
                    casing_types.append("UPPER")
                elif s.islower():
                    casing_types.append("lower")
                elif s.istitle():
                    casing_types.append("Title")
                else:
                    casing_types.append("Mixed")
                    
            majority_count = max(casing_types.count(c) for c in set(casing_types))
            casing_consistencies.append(majority_count / len(strings))
            
        overall_consistency = np.mean(casing_consistencies) if casing_consistencies else 1.0
        achieved_score = weight * overall_consistency
        
        pct_inconsistent = (1.0 - overall_consistency) * 100
        warning = None
        max_allowed = limits.get("max_mixed_formats_pct", 10.0)
        if pct_inconsistent > max_allowed:
            warning = f"Casing format inconsistency ({pct_inconsistent:.1f}%) exceeds threshold of {max_allowed}%."
            
        return max(0.0, achieved_score), warning


# Registry containing all modular validator checks
QUALITY_CHECKS: Dict[str, BaseQualityCheck] = {
    "column_count": ColumnCountCheck(),
    "missing_values": MissingValuesCheck(),
    "duplicate_columns": DuplicateColumnsCheck(),
    "duplicate_rows": DuplicateRowsCheck(),
    "empty_columns": EmptyColumnsCheck(),
    "datatype_consistency": DatatypeConsistencyCheck(),
    "corrupted_values": CorruptedValuesCheck(),
    "null_heavy_rows": NullHeavyRowsCheck(),
    "cell_length_outliers": CellLengthOutliersCheck(),
    "mixed_formats": MixedFormatsCheck(),
}


class DataQualityValidator:
    """Configuration-driven data quality validator runner."""

    def __init__(self, config_path: str = CONFIG_PATH):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse the JSON configuration file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found at {self.config_path}. Using fallback defaults.")
            return self._get_fallback_config()
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading quality config: {e}. Using fallback defaults.")
            return self._get_fallback_config()

    def _get_fallback_config(self) -> Dict[str, Any]:
        """Provide fallback configuration if config file reading fails."""
        return {
            "sampling": {"first_rows": 10, "last_rows": 10},
            "passing_score": 75,
            "weights": {
                "column_count": 15,
                "missing_values": 20,
                "duplicate_columns": 5,
                "duplicate_rows": 5,
                "empty_columns": 10,
                "datatype_consistency": 15,
                "corrupted_values": 10,
                "null_heavy_rows": 10,
                "cell_length_outliers": 5,
                "mixed_formats": 5,
            },
            "limits": {
                "min_columns": 1,
                "max_missing_values_pct": 20.0,
                "max_duplicate_rows_pct": 10.0,
                "max_empty_columns_pct": 10.0,
                "min_datatype_consistency_pct": 80.0,
                "max_corrupted_values_pct": 5.0,
                "max_null_heavy_rows_pct": 10.0,
                "max_cell_length_outliers_pct": 5.0,
                "max_mixed_formats_pct": 10.0,
                "null_heavy_row_pct_threshold": 50.0,
                "corrupted_placeholders": ["?", "n/a", "na", "null", "none", "undefined", "NaN"],
            },
        }

    def run_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all configured validation checks on a head and tail sample.
        
        Args:
            df: The complete input DataFrame to validate.
            
        Returns:
            A structured dict matching the QualityReport output schema.
        """
        # Determine sampling settings
        sampling = self.config.get("sampling", {})
        first_rows = sampling.get("first_rows", 10)
        last_rows = sampling.get("last_rows", 10)
        
        # Extract sample without scanning full dataset (take head/tail only)
        total_rows = len(df)
        if total_rows <= (first_rows + last_rows):
            df_sample = df
        else:
            # Take first N and last N rows directly
            df_sample = pd.concat([df.head(first_rows), df.tail(last_rows)])
            
        # Execute checks
        weights = self.config.get("weights", {})
        limits = self.config.get("limits", {})
        passing_score = self.config.get("passing_score", 75)
        
        check_scores: Dict[str, float] = {}
        warnings: List[str] = []
        total_score = 0.0
        
        for check_name, weight in weights.items():
            check_obj = QUALITY_CHECKS.get(check_name)
            if check_obj:
                try:
                    score, warning = check_obj.validate(df_sample, float(weight), limits)
                    check_scores[check_name] = round(score, 2)
                    total_score += score
                    if warning:
                        warnings.append(warning)
                except Exception as e:
                    logger.error(f"Error executing validation check '{check_name}': {e}")
                    check_scores[check_name] = 0.0
                    warnings.append(f"Validation check '{check_name}' failed to execute: {str(e)}")
            else:
                logger.warning(f"Configured check '{check_name}' not found in registry.")
                check_scores[check_name] = 0.0

        decision = "PASS" if total_score >= passing_score else "FAIL"
        
        return {
            "dataset_score": round(total_score, 2),
            "passing_score": passing_score,
            "decision": decision,
            "summary": {
                "rows_analyzed": len(df_sample),
                "columns": len(df.columns),
            },
            "checks": check_scores,
            "warnings": warnings,
        }
