"""
tools/root_cause_tool.py — Tool 4: Variance Root Cause Analyser

Determines WHY a metric changed by:
1. Loading pre-computed variance driver columns.
2. Measuring each driver's contribution.
3. Ranking contributors.
4. Detecting the dominant driver.

Never guesses. Only returns evidence from the data.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# The 14 pre-computed variance driver columns in the dataset
VARIANCE_DRIVER_COLUMNS = [
    "exposure_growth_variance",
    "premium_rate_variance",
    "policy_mix_variance",
    "claim_frequency_variance",
    "claim_severity_variance",
    "catastrophe_loss_variance",
    "fraud_leakage_variance",
    "premium_nonpayment_variance",
    "fx_translation_variance",
    "reinsurance_recovery_variance",
    "reserve_development_variance",
    "acquisition_cost_variance",
    "claims_processing_delay_variance",
    "one_off_variance_amount",
]

DRIVER_LABELS = {
    "exposure_growth_variance":        "Exposure Growth",
    "premium_rate_variance":           "Premium Rate",
    "policy_mix_variance":             "Policy Mix",
    "claim_frequency_variance":        "Claim Frequency",
    "claim_severity_variance":         "Claim Severity",
    "catastrophe_loss_variance":       "Catastrophe Loss",
    "fraud_leakage_variance":          "Fraud Leakage",
    "premium_nonpayment_variance":     "Premium Non-Payment",
    "fx_translation_variance":         "FX Translation",
    "reinsurance_recovery_variance":   "Reinsurance Recovery",
    "reserve_development_variance":    "Reserve Development",
    "acquisition_cost_variance":       "Acquisition Cost",
    "claims_processing_delay_variance":"Claims Processing Delay",
    "one_off_variance_amount":         "One-Off / Exceptional",
}


class RootCauseTool:
    """
    Decomposes variance into its constituent drivers and ranks them
    by magnitude and contribution percentage.
    """

    def analyse(
        self,
        df: pd.DataFrame,
        total_variance: float | None = None,
    ) -> dict:
        """
        Main root cause analysis entry point.

        Args:
            df             : DataFrame from SQLTool.get_variance_drivers()
            total_variance : If provided, validates driver sum against total.

        Returns:
            {
              "driver_breakdown": [{driver, label, amount, contribution_pct, direction}],
              "primary_driver": {...},
              "secondary_driver": {...},
              "explained_variance": float,
              "unexplained_variance": float | None,
              "pre_tagged_driver": str | None,
              "flags": {...},
            }
        """
        driver_results = []

        for col in VARIANCE_DRIVER_COLUMNS:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            amount = series.sum()
            driver_results.append({
                "driver":  col,
                "label":   DRIVER_LABELS.get(col, col),
                "amount":  round(amount, 2),
            })

        if not driver_results:
            return {"error": "No variance driver columns found in the provided DataFrame."}

        # Total absolute variance for contribution calculation
        total_abs = sum(abs(d["amount"]) for d in driver_results)

        for d in driver_results:
            d["contribution_pct"] = (
                round(abs(d["amount"]) / total_abs * 100, 2) if total_abs > 0 else 0.0
            )
            d["direction"] = "favorable" if d["amount"] > 0 else (
                "unfavorable" if d["amount"] < 0 else "neutral"
            )

        # Sort by absolute contribution
        driver_results.sort(key=lambda x: abs(x["amount"]), reverse=True)

        explained_variance = sum(d["amount"] for d in driver_results)
        unexplained = None
        if total_variance is not None:
            unexplained = round(total_variance - explained_variance, 2)

        primary   = driver_results[0] if len(driver_results) >= 1 else None
        secondary = driver_results[1] if len(driver_results) >= 2 else None

        # Pre-tagged root cause from the dataset
        pre_tagged_driver = None
        if "primary_variance_driver" in df.columns:
            tag_series = df["primary_variance_driver"].dropna()
            if not tag_series.empty:
                pre_tagged_driver = tag_series.mode().iloc[0]

        # Flag metadata
        flags = {}
        for flag_col in ["structural_vs_timing_flag", "recurring_flag", "controllable_flag"]:
            if flag_col in df.columns:
                tag_series = df[flag_col].dropna()
                if not tag_series.empty:
                    flags[flag_col] = tag_series.mode().iloc[0]

        result = {
            "driver_breakdown":    driver_results,
            "primary_driver":      primary,
            "secondary_driver":    secondary,
            "explained_variance":  round(explained_variance, 2),
            "unexplained_variance":unexplained,
            "pre_tagged_driver":   pre_tagged_driver,
            "flags":               flags,
            "drivers_analysed":    len(driver_results),
        }

        logger.info(
            f"Root Cause: primary='{primary['label'] if primary else 'N/A'}' "
            f"({primary['contribution_pct']:.1f}%)" if primary else "Root Cause: no drivers found"
        )
        return result
