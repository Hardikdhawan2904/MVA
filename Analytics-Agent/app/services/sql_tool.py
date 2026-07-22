"""
app/services/sql_tool.py — DuckDB In-Process Query Engine

Executes SQL against the insurance CSV dataset via DuckDB.
The Analytics Agent uses this ONLY to retrieve required records.
Never performs calculations here — that is the Analytics Tool's job.
"""

import logging
from pathlib import Path

import duckdb
import pandas as pd

from app.config import DATASET_PATH

logger = logging.getLogger(__name__)

# Agent 3 redesign, Phase 2 (plan "zany-giggling-crayon") — these were
# hardcoded directly inside get_variance_drivers()/get_ratio_data() before;
# now they're the *default* argument values, sourced from the Insurance
# domain plugin (domain_plugins/insurance/plugin.py) at call sites that
# want a different domain's columns, with the exact same content as
# before when a caller (e.g. today's Insurance-only pipeline.py) doesn't
# override them — byte-identical behavior, genuinely parameterizable.
DEFAULT_VARIANCE_DRIVER_COLUMNS = [
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
    "variance_vs_budget_amount",
    "variance_vs_prior_year_amount",
    "primary_variance_driver",
    "secondary_variance_driver",
    "variance_category",
    "structural_vs_timing_flag",
    "recurring_flag",
    "controllable_flag",
    "root_cause_tag",
]

DEFAULT_RATIO_COLUMNS = [
    "reporting_date", "year", "quarter",
    "region", "country_name", "insurance_segment", "line_of_business",
    "loss_ratio_actual", "loss_ratio_budget", "loss_ratio_prior_year",
    "expense_ratio_actual", "expense_ratio_budget",
    "combined_ratio_actual", "combined_ratio_budget",
    "claim_frequency_actual", "claim_frequency_budget",
    "average_claim_severity_actual", "average_claim_severity_budget",
    "renewal_rate_actual",
    "policy_lapse_rate_actual",
    "reserve_adequacy_ratio_actual",
    "premium_collection_rate_actual",
]


class SQLTool:
    """
    DuckDB-backed SQL executor.

    Rules:
    - Always returns a pandas DataFrame.
    - Never does calculations — only filtering, grouping, aggregation for retrieval.
    - Logs every query for auditability.

    One connection per instance, bound to `dataset_path` at construction —
    deliberately NOT a process-wide cached singleton. As a subprocess
    invoked fresh per upload this used to be safe to cache at module scope;
    as a long-lived FastAPI service handling many different uploads over
    its lifetime, a shared connection would silently keep answering every
    request from whichever dataset happened to be loaded first.
    """

    def __init__(self, dataset_path: str | Path | None = None, view_name: str = "insurance"):
        csv_path = str(dataset_path or DATASET_PATH)
        self.view_name = view_name
        self.conn = duckdb.connect(database=":memory:")
        self.conn.execute(f"""
            CREATE VIEW {view_name} AS
            SELECT * FROM read_csv_auto('{csv_path}', ALL_VARCHAR=FALSE, HEADER=TRUE)
        """)
        logger.info(f"DuckDB view '{view_name}' registered from: {csv_path}")

    # ── Core Executor ─────────────────────────────────────────────────────────

    def query(self, sql: str, parameters: list | None = None) -> pd.DataFrame:
        """Execute raw SQL and return a DataFrame. `parameters`, if given,
        binds to `?` placeholders in `sql` (DuckDB parameterized query) —
        used by the filter-building methods below so filter *values* never
        get string-interpolated directly into the SQL text."""
        logger.info(f"SQL >> {sql.strip()[:200]}")
        try:
            result = (
                self.conn.execute(sql, parameters) if parameters is not None
                else self.conn.execute(sql)
            ).fetchdf()
            logger.info(f"SQL returned {len(result)} rows, {len(result.columns)} columns")
            return result
        except Exception as e:
            logger.error(f"SQL error: {e}\nQuery: {sql}")
            raise

    # ── Convenience Builders ─────────────────────────────────────────────────

    def get_kpi_data(
        self,
        columns: list[str],
        filters: dict | None = None,
        group_by: list[str] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Generic KPI retrieval with optional filters and grouping.

        Args:
            columns   : List of column names to SELECT (can include SUM(x) etc.)
            filters   : Dict of {column: value} equality filters.
            group_by  : List of columns for GROUP BY.
            order_by  : ORDER BY clause string.
            limit     : Row limit.
        """
        select_clause = ", ".join(columns)
        sql = f"SELECT {select_clause} FROM {self.view_name}"

        # Column names come from code (KPI/hierarchy definitions), not user
        # input, so they stay as identifiers in the SQL text — only filter
        # *values* (which do trace back to query-derived dimension matches)
        # are bound as `?` parameters rather than string-interpolated.
        params: list = []
        if filters:
            conditions = []
            for col, val in filters.items():
                if val is None:
                    conditions.append(f"{col} IS NULL")
                elif isinstance(val, (list, tuple)):
                    conditions.append(f"{col} IN ({', '.join(['?'] * len(val))})")
                    params.extend(val)
                else:
                    conditions.append(f"{col} = ?")
                    params.append(val)
            sql += " WHERE " + " AND ".join(conditions)

        if group_by:
            sql += " GROUP BY " + ", ".join(group_by)

        if order_by:
            sql += f" ORDER BY {order_by}"

        if limit:
            sql += f" LIMIT {limit}"

        return self.query(sql, parameters=params or None)

    def get_period_data(
        self,
        columns: list[str],
        year: int | None = None,
        quarter: str | None = None,
        month: int | None = None,
        fiscal_year: str | None = None,
        ytd_only: bool = False,
        extra_filters: dict | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve data filtered by time period.

        Args:
            columns     : Columns to SELECT.
            year        : e.g. 2024
            quarter     : e.g. 'Q1'
            month       : e.g. 3
            fiscal_year : e.g. 'FY2025'
            ytd_only    : If True, adds ytd_flag = 'Y' filter.
            extra_filters: Additional equality filters.
        """
        filters = extra_filters.copy() if extra_filters else {}
        if year:
            filters["year"] = str(year)
        if quarter:
            filters["quarter"] = quarter
        if month:
            filters["month"] = str(month)
        if fiscal_year:
            filters["fiscal_year"] = fiscal_year
        if ytd_only:
            filters["ytd_flag"] = "Y"
        return self.get_kpi_data(columns, filters=filters)

    def get_variance_drivers(
        self,
        filters: dict | None = None,
        group_by: list[str] | None = None,
        variance_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Retrieve all pre-computed variance driver columns.

        variance_columns defaults to DEFAULT_VARIANCE_DRIVER_COLUMNS
        (Insurance's exact list, unchanged) — pass a different domain's
        columns to reuse this query shape elsewhere."""
        variance_cols = list(variance_columns) if variance_columns is not None else list(DEFAULT_VARIANCE_DRIVER_COLUMNS)
        dim_cols = group_by or []
        select_cols = list(set(dim_cols + variance_cols))

        if group_by:
            agg_cols = [
                f"SUM(TRY_CAST({c} AS DOUBLE)) AS {c}"
                for c in variance_cols
                if "flag" not in c and "driver" not in c
                and "category" not in c and "tag" not in c
            ]
            cat_cols = [
                f"MODE({c}) AS {c}"
                for c in variance_cols
                if any(x in c for x in ["flag", "driver", "category", "tag"])
            ]
            select_cols = group_by + agg_cols + cat_cols

        return self.get_kpi_data(select_cols, filters=filters, group_by=group_by)

    def get_ratio_data(self, filters: dict | None = None, ratio_columns: list[str] | None = None) -> pd.DataFrame:
        """Retrieve all pre-computed ratio columns for anomaly detection.

        ratio_columns defaults to DEFAULT_RATIO_COLUMNS (Insurance's exact
        list, unchanged) — pass a different domain's columns to reuse this
        query shape elsewhere."""
        ratio_cols = list(ratio_columns) if ratio_columns is not None else list(DEFAULT_RATIO_COLUMNS)
        return self.get_kpi_data(ratio_cols, filters=filters)

    def get_time_series(
        self,
        metric_column: str,
        group_by_col: str = "reporting_date",
        filters: dict | None = None,
    ) -> pd.DataFrame:
        """
        Return a time-series DataFrame for forecasting.
        DuckDB sorts by date automatically.
        """
        params = list(filters.values()) if filters else []
        where_clause = (
            "WHERE " + " AND ".join(f"{k} = ?" for k in filters) if filters else ""
        )
        sql = f"""
            SELECT
                {group_by_col} AS ds,
                SUM(TRY_CAST("{metric_column}" AS DOUBLE)) AS y
            FROM {self.view_name}
            {where_clause}
            GROUP BY {group_by_col}
            ORDER BY {group_by_col}
        """
        return self.query(sql, parameters=params or None)
