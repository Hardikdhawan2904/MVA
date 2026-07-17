"""
main.py — Analytics Agent Orchestrator

Entry point for the Enterprise Insurance Analytics Agent.

Usage:
    python main.py --query "Show Gross Written Premium for FY2025"
    python main.py --query "Why did underwriting result decline in Q2 2025?"
    python main.py --query "Forecast underwriting result for next 6 months"
    python main.py --query "Detect anomalies in loss ratios"
    python main.py --query "Segment portfolio by risk profile"
    python main.py --interactive

The agent:
1. Parses and understands the query intent.
2. Creates an execution plan.
3. Calls the required tools in sequence.
4. Returns a structured, evidence-backed answer.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from config import (
    LOG_LEVEL, LOG_FORMAT,
    AGENT_IDENTITY, AGENT_ROLE, MEMORY_CONFIG,
    EXECUTION_PLANS, SYSTEM_PROMPT_CFG,
)
from memory import MemoryManager

# Single-file import also works:
# from tools import RuleEngine, SQLTool, AnalyticsTool, ...

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger("AnalyticsAgent")

# ── Tool Imports ───────────────────────────────────────────────────────────────
from tools.rule_engine           import RuleEngine
from tools.sql_tool              import SQLTool
from tools.analytics_tool        import AnalyticsTool
from tools.root_cause_tool       import RootCauseTool
from tools.explanation_tool      import ExplanationTool
from tools.knowledge_update_tool import KnowledgeUpdateTool
from tools.ml_tool               import MLTool
from boot_trainer                import check_and_retrain_if_needed
from ml.feature_validation       import validate_feature_columns

# ── Intent Keywords ────────────────────────────────────────────────────────────
_INTENT_MAP = {
    "forecast":   ["forecast", "predict", "projection", "next quarter", "future", "expected"],
    "anomaly":    ["anomaly", "anomalies", "irregular", "outlier", "unusual", "spike", "detect"],
    "root_cause": ["why", "reason", "cause", "driver", "drove", "driven", "factors", "impact", "contributing", "explain why"],
    "variance":   ["variance", "vs budget", "vs prior", "vs plan", "deviation", "gap", "difference"],
    "trend":      ["trend", "over time", "qoq", "yoy", "mom", "quarter over quarter", "growth"],
    "segment":    ["segment portfolio", "segment risk", "segment by", "cluster", "group risk", "group by risk", "portfolio risk", "risk tier", "segment by risk"],
    "ytd":        ["ytd", "year to date", "year-to-date"],
    "show_kpi":   ["show", "what is", "total", "amount", "value", "display", "give me"],
}

_PERIOD_KEYWORDS = {
    "quarter": ["q1", "q2", "q3", "q4"],
}
# Fiscal year / calendar year are matched by pattern, not a fixed list, so
# this doesn't silently stop working (unfiltered, with no error) the moment
# a query mentions a year outside whatever was hardcoded here.
_FISCAL_YEAR_RE = re.compile(r"\bfy(20\d{2})\b")
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

_DIMENSION_KEYWORDS = {
    "region": {
        "emea": "EMEA", "apac": "APAC", "africa": "Africa", "middle east": "Middle East",
    },
    "insurance_segment": {
        "general insurance": "General Insurance", "health": "Health",
        "specialty": "Commercial Specialty", "commercial": "Commercial Insurance", "life": "Life",
    },
    "country_name": {
        "uk": "United Kingdom", "united kingdom": "United Kingdom",
        "singapore": "Singapore", "india": "India",
        "australia": "Australia", "uae": "United Arab Emirates",
        "south africa": "South Africa",
    },
    # "health", "specialty", and "life" deliberately omitted here even
    # though the dataset has matching line_of_business values — they're
    # already mapped under insurance_segment above, and matching both
    # simultaneously silently double-filters a query (can zero out results
    # if a row doesn't happen to satisfy both dimensions at once).
    "line_of_business": {
        "motor": "Motor Insurance", "property": "Property Insurance",
        "travel": "Travel Insurance",
    },
}

# agent_config.yml's execution_plans use lowercase tool ids ("rule_engine",
# "sql_tool", ...) — this maps them to the friendly names shown in the
# "Execution Plan" line printed for every query.
_TOOL_DISPLAY_NAMES = {
    "rule_engine": "RuleEngine",
    "sql_tool": "SQLTool",
    "analytics_tool": "AnalyticsTool",
    "root_cause_tool": "RootCauseTool",
    "ml_tool": "MLTool",
    "knowledge_update_tool": "KnowledgeUpdateTool",
    "explanation_tool": "ExplanationTool",
}



class AnalyticsAgent:
    """
    Orchestrates all analytics tools to answer business questions.
    Identity, role, and memory config are all loaded from agent_config.yml.
    """

    def __init__(self, ml_readiness_score: float = 99.75, llm_readiness_score: float = 99.75):
        # ── Identity from YAML ─────────────────────────────────────────────
        self.name       = AGENT_IDENTITY.get("name", "Analytics Agent")
        self.version    = AGENT_IDENTITY.get("version", "1.0.0")
        self.domain     = AGENT_IDENTITY.get("domain", "Insurance")
        self.persona    = AGENT_ROLE.get("persona", "")
        self.core_rules = SYSTEM_PROMPT_CFG.get("core_rules", [])
        self.exec_plans = EXECUTION_PLANS

        logger.info(f"Initialising {self.name} v{self.version} [{self.domain}]...")

        # ── Boot-Time Auto-Retrain Check ────────────────────────────────────
        # If the dataset CSV is newer than the saved ML models, retrain automatically.
        check_and_retrain_if_needed()

        # ── Tools ───────────────────────────────────────────────────────────
        self.rule_engine  = RuleEngine()
        self.sql          = SQLTool()
        self.analytics    = AnalyticsTool()
        self.root_cause   = RootCauseTool()
        self.explanation  = ExplanationTool(llm_readiness_score)
        self.knowledge    = KnowledgeUpdateTool(self.rule_engine)
        self.ml           = MLTool(ml_readiness_score)

        # ── Memory (from YAML) ────────────────────────────────────────────
        self.memory = MemoryManager()

        logger.info(f"{self.name} ready ✓ (memory: max_turns={self.memory.max_turns})")

    # ── Query Processing ───────────────────────────────────────────────────────

    def process(self, query: str) -> str:
        """
        Main entry point. Parses query, plans, executes, and returns narrative.
        """
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}\n")

        query_lower = query.lower()

        # ── Step 1: Detect intent ────────────────────────────────────────────
        intent = self._detect_intent(query_lower)
        # Prior turn's filters are defaults; whatever this query explicitly
        # mentions overrides/adds to them — lets a bare follow-up like
        # "what about EMEA?" carry over the KPI/period from the last turn.
        filters = {**self.memory.get_last_filters(), **self._extract_filters(query_lower)}
        kpi_name = self._detect_kpi(query_lower) or self.memory.get_last_kpi() or "underwriting_result"

        # Give the LLM narrator awareness of the recent conversation so
        # follow-up answers read coherently instead of each turn being
        # answered in isolation.
        self.explanation.set_conversation_context(self.memory.get_context_for_llm())

        logger.info(f"Intent={intent} | KPI={kpi_name} | Filters={filters}")
        print(f"📋 Execution Plan: [{', '.join(self._build_plan(intent))}]")
        print()

        # ── Step 2: Route to appropriate handler ─────────────────────────────
        tools_used = self._build_plan(intent)
        try:
            if intent == "forecast":
                response = self._handle_forecast(query, kpi_name, filters)
            elif intent == "anomaly":
                response = self._handle_anomaly(query, filters)
            elif intent == "segment":
                response = self._handle_segment(query, filters)
            elif intent == "root_cause":
                response = self._handle_root_cause(query, kpi_name, filters)
            elif intent == "variance":
                response = self._handle_variance(query, kpi_name, filters)
            elif intent == "trend":
                response = self._handle_trend(query, kpi_name, filters)
            else:
                response = self._handle_show_kpi(query, kpi_name, filters)
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=True)
            response = f"❌ Agent Error: {str(e)}\nPlease check your query and try again."

        # ── Step 3: Record turn in memory ────────────────────────────────────
        self.memory.add_turn(
            user_query       = query,
            intent           = intent,
            kpi_name         = kpi_name,
            filters          = filters,
            tools_used       = tools_used,
            evidence         = {},
            response_summary = response[:300],
        )

        return response


    # ── Intent Detection ───────────────────────────────────────────────────────

    def _detect_intent(self, query: str) -> str:
        for intent, keywords in _INTENT_MAP.items():
            if any(kw in query for kw in keywords):
                return intent
        return "show_kpi"

    def _build_plan(self, intent: str) -> list[str]:
        """Tool sequence for display — derived from agent_config.yml's
        execution_plans (self.exec_plans), not a second hardcoded copy of
        it, so the two can no longer drift apart."""
        plan_cfg = self.exec_plans.get(intent)
        if not plan_cfg:
            return ["RuleEngine", "SQLTool", "AnalyticsTool", "ExplanationTool"]

        ml_model = plan_cfg.get("ml_model")
        display = []
        for tool_id in plan_cfg.get("chain", []):
            name = _TOOL_DISPLAY_NAMES.get(tool_id, tool_id)
            if tool_id == "ml_tool" and ml_model:
                name = f"{name}→{ml_model}"
                if intent == "forecast":
                    # Prophet drives the forecast itself; LightGBM's feature
                    # importances are also surfaced as "key drivers" evidence.
                    name += "/LightGBM"
            display.append(name)
        return display

    # ── Filter Extraction ──────────────────────────────────────────────────────

    def _extract_filters(self, query: str) -> dict:
        filters = {}
        for col, keywords in _PERIOD_KEYWORDS.items():
            for kw in keywords:
                if kw in query:
                    filters[col] = kw.upper()

        fy_match = _FISCAL_YEAR_RE.search(query)
        if fy_match:
            filters["fiscal_year"] = f"FY{fy_match.group(1)}"
        year_match = _YEAR_RE.search(query)
        if year_match:
            filters["year"] = year_match.group(1)

        for col, kw_map in _DIMENSION_KEYWORDS.items():
            for kw, val in kw_map.items():
                if re.search(rf"\b{re.escape(kw)}\b", query):
                    filters[col] = val

        if "ytd" in query or "year to date" in query:
            filters["ytd_flag"] = "Y"
        return filters

    def _detect_kpi(self, query: str) -> str:
        kpi_aliases = {
            "gross written premium": "gross_written_premium",
            "gwp": "gross_written_premium",
            "net written premium": "net_written_premium",
            "nwp": "net_written_premium",
            "loss ratio": "loss_ratio",
            "combined ratio": "combined_ratio",
            "expense ratio": "expense_ratio",
            "underwriting result": "underwriting_result",
            "uwr": "underwriting_result",
            "earned premium": "earned_premium",
            "net earned premium": "net_earned_premium",
            "nep": "net_earned_premium",
            "claim frequency": "claim_frequency",
            "claims frequency": "claim_frequency",
            "claim severity": "average_claim_severity",
            "renewal rate": "renewal_rate",
            "lapse rate": "policy_lapse_rate",
            "reserve adequacy": "reserve_adequacy_ratio",
            "ibnr": "ibnr_reserve",
            "premium collection": "premium_collection_rate",
        }
        for alias, key in kpi_aliases.items():
            if alias in query:
                return key
        return None  # No explicit KPI mentioned — caller decides the fallback

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _handle_show_kpi(self, query: str, kpi_name: str, filters: dict) -> str:
        # Step 1: Rule Engine
        kpi = self.rule_engine.get_kpi(kpi_name)
        if not kpi:
            logger.info(f"KPI '{kpi_name}' missing — triggering Knowledge Update Tool")
            update_result = self.knowledge.update(kpi_name)
            if update_result["success"]:
                kpi = self.rule_engine.get_kpi(kpi_name)
            else:
                return f"KPI '{kpi_name}' not found and could not be auto-generated.\n{update_result['reason']}"

        # Step 2: SQL Tool
        actual_col = kpi.get("actual_column")
        budget_col = kpi.get("budget_column")
        cols = [c for c in [actual_col, budget_col] if c]
        df = self.sql.get_period_data(cols, extra_filters=filters)

        if df.empty:
            return f"No data found for '{kpi['label']}' with the given filters: {filters}"

        # Step 3: Analytics Tool
        unit = kpi.get("unit", "")
        method = "mean" if unit in ("%", "ratio", "rate") or kpi.get("category") == "ratio" else "sum"

        evidence = {"kpi": kpi["label"], "unit": kpi.get("unit", ""), "filters": filters}
        if actual_col and actual_col in df.columns:
            evidence["actual"] = self.analytics.aggregate(df, actual_col, method=method)
        if budget_col and budget_col in df.columns:
            evidence["budget"] = self.analytics.aggregate(df, budget_col, method=method)
        if "actual" in evidence and "budget" in evidence:
            var = self.analytics.variance_vs_budget(evidence["actual"], evidence["budget"])
            evidence.update(var)

        # Step 4: Explanation Tool
        return self.explanation.narrate(evidence, query)

    def _handle_variance(self, query: str, kpi_name: str, filters: dict) -> str:
        kpi = self.rule_engine.get_kpi(kpi_name)
        if not kpi:
            return f"KPI '{kpi_name}' not found in Rule Engine."

        actual_col     = kpi.get("actual_column")
        budget_col     = kpi.get("budget_column")
        prior_yr_col   = kpi.get("prior_year_column")
        ytd_actual_col = kpi.get("ytd_actual_column")
        ytd_budget_col = kpi.get("ytd_budget_column")

        cols = [c for c in [actual_col, budget_col, prior_yr_col, ytd_actual_col, ytd_budget_col] if c]
        df   = self.sql.get_period_data(cols, extra_filters=filters)

        if df.empty:
            return f"No data for '{kpi['label']}' with filters: {filters}"

        # Determine aggregation method
        unit = kpi.get("unit", "")
        method = "mean" if unit in ("%", "ratio", "rate") or kpi.get("category") == "ratio" else "sum"

        evidence = {"kpi": kpi["label"], "unit": kpi.get("unit"), "filters": filters}

        if actual_col and actual_col in df.columns:
            evidence["actual"] = self.analytics.aggregate(df, actual_col, method=method)
        if budget_col and budget_col in df.columns:
            evidence["budget"] = self.analytics.aggregate(df, budget_col, method=method)
            if "actual" in evidence:
                v = self.analytics.variance_vs_budget(evidence["actual"], evidence["budget"])
                evidence["variance_vs_budget_amount"] = v["variance_amount"]
                evidence["variance_vs_budget_pct"]    = v["variance_pct"]
                evidence["direction"]                  = v["direction"]
        if prior_yr_col and prior_yr_col in df.columns:
            evidence["prior_year"] = self.analytics.aggregate(df, prior_yr_col, method=method)
            if "actual" in evidence:
                v = self.analytics.variance_vs_prior_year(evidence["actual"], evidence["prior_year"])
                evidence["variance_vs_prior_year_amount"] = v["variance_amount"]
                evidence["variance_vs_prior_year_pct"]    = v["variance_pct"]

        return self.explanation.narrate(evidence, query)

    def _handle_root_cause(self, query: str, kpi_name: str, filters: dict) -> str:
        kpi = self.rule_engine.get_kpi(kpi_name)
        if not kpi:
            return f"KPI '{kpi_name}' not found."

        # Get overall variance first
        actual_col   = kpi.get("actual_column")
        budget_col   = kpi.get("budget_column")
        metric_cols  = [c for c in [actual_col, budget_col] if c]
        df_summary   = self.sql.get_period_data(metric_cols, extra_filters=filters)

        unit = kpi.get("unit", "")
        method = "mean" if unit in ("%", "ratio", "rate") or kpi.get("category") == "ratio" else "sum"

        total_variance = None
        evidence = {"kpi": kpi["label"], "filters": filters}
        if actual_col and actual_col in df_summary.columns and budget_col and budget_col in df_summary.columns:
            actual = self.analytics.aggregate(df_summary, actual_col, method=method)
            budget = self.analytics.aggregate(df_summary, budget_col, method=method)
            v = self.analytics.variance_vs_budget(actual, budget)
            evidence.update(v)
            total_variance = v["variance_amount"]

        # Get variance driver data
        group_by = [c for c in ["region", "insurance_segment", "line_of_business"] if c not in filters]
        df_drivers = self.sql.get_variance_drivers(filters=filters)

        if df_drivers.empty:
            return f"No variance driver data found for filters: {filters}"

        # Root Cause Tool
        rc_result = self.root_cause.analyse(df_drivers, total_variance=total_variance)
        evidence.update(rc_result)

        # Driver descriptions from Rule Engine
        for d in evidence.get("driver_breakdown", []):
            d["description"] = self.rule_engine.get_variance_driver_description(d["driver"])

        # ML-predicted driver (XGBoost + SHAP) as corroborating evidence
        # alongside the deterministic breakdown above — best-effort, never
        # blocks the response if the model isn't available or errors.
        try:
            ml_evidence = self.ml.get_root_cause_ml_evidence(df_drivers)
            if ml_evidence and "error" not in ml_evidence:
                evidence["ml_predicted_driver"] = ml_evidence["predicted_driver"]
                evidence["ml_confidence"] = ml_evidence["confidence"]
                evidence["ml_top_features"] = ml_evidence["top_features"]
        except Exception as e:
            logger.warning(f"ML root-cause evidence unavailable: {e}")

        return self.explanation.narrate(evidence, query)

    def _handle_trend(self, query: str, kpi_name: str, filters: dict) -> str:
        kpi = self.rule_engine.get_kpi(kpi_name)
        if not kpi:
            return f"KPI '{kpi_name}' not found."

        actual_col = kpi.get("actual_column")
        ts_df = self.sql.get_time_series(actual_col, group_by_col="reporting_date", filters=filters)

        if ts_df.empty or len(ts_df) < 2:
            return f"Insufficient time-series data for '{kpi['label']}'."

        trend_result = self.analytics.trend(ts_df, "ds", "y")
        evidence = {"kpi": kpi["label"], "unit": kpi.get("unit"), "filters": filters}
        evidence.update(trend_result)

        return self.explanation.narrate(evidence, query)

    def _handle_forecast(self, query: str, kpi_name: str, filters: dict) -> str:
        kpi = self.rule_engine.get_kpi(kpi_name)
        if not kpi:
            return f"KPI '{kpi_name}' not found."

        actual_col = kpi.get("actual_column")
        ts_df = self.sql.get_time_series(actual_col, group_by_col="reporting_date", filters=filters)

        # Check ML readiness fallback
        status = self.ml.get_readiness_status()
        if not status["ml_authorized"]:
            logger.warning("ML readiness score below threshold. Executing deterministic fallback.")
            trend_result = self.analytics.trend(ts_df, "ds", "y")
            evidence = {
                "kpi": kpi["label"],
                "unit": kpi.get("unit"),
                "filters": filters,
                "ml_readiness_blocked": True,
                "ml_readiness_score": status["ml_readiness_score"],
                "fallback_reason": f"ML readiness score ({status['ml_readiness_score']:.2f}%) below threshold ({status['threshold']}%).",
                "fallback_applied": "Historical Trend Analysis (analytics_tool.trend)",
            }
            evidence.update(trend_result)
            return self.explanation.narrate(evidence, query)

        ml_result = self.ml.forecast_timeseries(ts_df, periods=6)
        evidence  = {"kpi": kpi["label"], "unit": kpi.get("unit"), "filters": filters}
        evidence.update(ml_result)

        # LightGBM's persisted feature importances as "what typically drives
        # this KPI" evidence alongside Prophet's trend — best-effort, never
        # blocks the response if no persisted model exists for this target.
        try:
            driver_evidence = self.ml.get_forecast_key_drivers(actual_col)
            if driver_evidence and "error" not in driver_evidence:
                evidence["key_drivers"] = driver_evidence["key_drivers"]
        except Exception as e:
            logger.warning(f"Forecast key-driver evidence unavailable: {e}")

        return self.explanation.narrate(evidence, query)

    def _handle_anomaly(self, query: str, filters: dict) -> str:
        df = self.sql.get_ratio_data(filters=filters or None)
        if df.empty:
            return "No ratio data available for anomaly detection."

        status = self.ml.get_readiness_status()
        if not status["ml_authorized"]:
            logger.warning("ML readiness score below threshold. Executing deterministic anomaly detection fallback.")
            evidence = {
                "ml_readiness_blocked": True,
                "ml_readiness_score": status["ml_readiness_score"],
                "fallback_reason": f"ML readiness score ({status['ml_readiness_score']:.2f}%) below threshold ({status['threshold']}%).",
                "fallback_applied": "Deterministic Z-Score ratio anomaly detection (analytics_tool.rank)",
            }
            # Find anomalies where loss_ratio_actual is > mean + 2*std
            anomalies = []
            for col in ["loss_ratio_actual", "combined_ratio_actual"]:
                if col in df.columns:
                    mean = df[col].mean()
                    std = df[col].std()
                    if std > 0:
                        df_anom = df[df[col] > mean + 2 * std]
                        for _, row in df_anom.iterrows():
                            anomalies.append({
                                "reporting_date": str(row.get("reporting_date")),
                                "region": row.get("region"),
                                "country_name": row.get("country_name"),
                                "insurance_segment": row.get("insurance_segment"),
                                "metric": col.replace("_actual", "").replace("_", " ").title(),
                                "value": float(row[col]),
                                "mean": float(mean),
                                "std_dev": float(std),
                                "deviation_stds": float((row[col] - mean) / std)
                            })
            evidence["anomalies_detected"] = anomalies[:10]  # top 10 anomalies
            return self.explanation.narrate(evidence, query)

        label_cols = ["reporting_date", "region", "country_name", "insurance_segment"]
        ml_result  = self.ml.detect_anomalies(
            df,
            label_cols=[c for c in label_cols if c in df.columns],
        )
        return self.explanation.narrate(ml_result, query)

    def _handle_segment(self, query: str, filters: dict) -> str:
        df = self.sql.get_ratio_data(filters=filters or None)
        if df.empty:
            return "No data available for portfolio segmentation."

        status = self.ml.get_readiness_status()
        if not status["ml_authorized"]:
            logger.warning("ML readiness score below threshold. Executing deterministic risk segmentation fallback.")
            evidence = {
                "ml_readiness_blocked": True,
                "ml_readiness_score": status["ml_readiness_score"],
                "fallback_reason": f"ML readiness score ({status['ml_readiness_score']:.2f}%) below threshold ({status['threshold']}%).",
                "fallback_applied": "Deterministic rule-based risk segmentation",
            }
            low_count = int((df["combined_ratio_actual"] < 90).sum())
            med_count = int(((df["combined_ratio_actual"] >= 90) & (df["combined_ratio_actual"] <= 100)).sum())
            high_count = int((df["combined_ratio_actual"] > 100).sum())
            evidence["segments"] = {
                "Low Risk (<90% Combined Ratio)": low_count,
                "Medium Risk (90-100% Combined Ratio)": med_count,
                "High Risk (>100% Combined Ratio)": high_count,
            }
            evidence["total_records"] = len(df)
            return self.explanation.narrate(evidence, query)

        label_cols = ["region", "country_name", "insurance_segment", "line_of_business"]
        ml_result  = self.ml.segment_portfolio(
            df,
            label_cols=[c for c in label_cols if c in df.columns],
        )
        return self.explanation.narrate(ml_result, query)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Enterprise Insurance Analytics Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --query "Show Gross Written Premium for FY2025"
  python main.py --query "Why did underwriting result decline in Q2 2025?"
  python main.py --query "Forecast underwriting result for next 6 months"
  python main.py --query "Detect anomalies in financial ratios"
  python main.py --query "Show loss ratio variance vs budget for EMEA"
  python main.py --interactive
        """,
    )
    parser.add_argument("--query", type=str, help="Business question to answer")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive mode")
    parser.add_argument(
        "--ml-readiness", type=float, default=99.75,
        help="ML readiness score from upstream agents (default: 99.75)"
    )
    parser.add_argument(
        "--llm-readiness", type=float, default=99.75,
        help="LLM readiness score from upstream agents — gates Groq narration "
             "vs. the deterministic template formatter (default: 99.75)"
    )
    parser.add_argument(
        "--feature-recommendation-file", type=str, default=None,
        help="Path to a JSON file of Agent 2's per-column feature classification "
             "for this upload — used only to validate the hardcoded ML feature "
             "column lists still match this dataset's schema (default: skip check)"
    )
    args = parser.parse_args()

    validate_feature_columns(args.feature_recommendation_file)

    agent = AnalyticsAgent(ml_readiness_score=args.ml_readiness, llm_readiness_score=args.llm_readiness)

    if args.interactive:
        print("\n🏦 Enterprise Insurance Analytics Agent")
        print("   Type 'exit' or 'quit' to stop.\n")
        while True:
            try:
                query = input("You: ").strip()
                if query.lower() in ("exit", "quit", "q"):
                    print("Goodbye.")
                    break
                if not query:
                    continue
                response = agent.process(query)
                print(f"\n{response}\n")
            except KeyboardInterrupt:
                print("\nGoodbye.")
                break
    elif args.query:
        response = agent.process(args.query)
        print(response)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
