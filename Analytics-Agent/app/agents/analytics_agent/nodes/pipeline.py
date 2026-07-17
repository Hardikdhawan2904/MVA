"""app/agents/analytics_agent/nodes/pipeline.py — LangGraph node functions
for the Analytics Agent.

`AnalyticsGraphNodes` is constructed fresh per request (see graph.py) — not
a shared singleton. Every request analyzes a different uploaded dataset, so
`SQLTool`'s DuckDB connection, `MLTool`'s readiness-scored models, and
`ExplanationTool`'s LLM gate all have to be bound to this request's inputs,
exactly like MVA-use-case-latest-one's request-scoped node construction.

Each handler is a direct translation of the old CLI's
`main.py::AnalyticsAgent._handle_*` method: same tool calls, same evidence
shape, same early-exit strings — just "read from state, write partial state
back" instead of "return a string directly". Two behaviors carried over
verbatim from main.py::process() are worth flagging since they look like
bugs but are existing, unchanged behavior:
  - memory.add_turn() is always called with evidence={}, never the
    real evidence dict.
  - _handle_forecast() never checks ts_df for emptiness before use,
    unlike _handle_trend() which does.
"""

import logging
import re

from app.agents.analytics_agent.config import get_execution_plans
from app.agents.analytics_agent.state import AnalyticsState
from app.services.rule_engine import RuleEngine
from app.services.sql_tool import SQLTool
from app.services.analytics_tool import AnalyticsTool
from app.services.root_cause_tool import RootCauseTool
from app.services.explanation_tool import ExplanationTool
from app.services.knowledge_update_service import KnowledgeUpdateTool
from app.services.ml_tool import MLTool
from app.services.memory import MemoryManager
from app.services.ml.feature_validation import validate_feature_columns

logger = logging.getLogger(__name__)

# ── Intent Keywords (ported unchanged from the old CLI's main.py) ───────────
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

# agent.yaml's execution_plans use lowercase tool ids ("rule_engine",
# "sql_tool", ...) — this maps them to the friendly names shown in the
# "Execution Plan" line / recorded in memory.
_TOOL_DISPLAY_NAMES = {
    "rule_engine": "RuleEngine",
    "sql_tool": "SQLTool",
    "analytics_tool": "AnalyticsTool",
    "root_cause_tool": "RootCauseTool",
    "ml_tool": "MLTool",
    "knowledge_update_tool": "KnowledgeUpdateTool",
    "explanation_tool": "ExplanationTool",
}

_KPI_ALIASES = {
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

_AGENT_ERROR_PREFIX = "❌ Agent Error: "


class AnalyticsGraphNodes:
    """One bound method per LangGraph node. Constructed once per request
    with everything that request needs, exactly the way
    `main.py::AnalyticsAgent.__init__` used to construct one tool set per
    subprocess invocation."""

    def __init__(
        self,
        dataset_path: str,
        ml_readiness_score: float = 99.75,
        llm_readiness_score: float = 99.75,
        feature_recommendation: list[dict] | None = None,
    ):
        validate_feature_columns(feature_recommendation)

        self.rule_engine = RuleEngine()
        self.sql         = SQLTool(dataset_path)
        self.analytics   = AnalyticsTool()
        self.root_cause  = RootCauseTool()
        self.explanation = ExplanationTool(llm_readiness_score)
        self.knowledge   = KnowledgeUpdateTool(self.rule_engine)
        self.ml          = MLTool(ml_readiness_score)
        self.memory      = MemoryManager()
        self.exec_plans  = get_execution_plans()

    # ── Intent / Filter Detection (entry node) ──────────────────────────────

    def detect_intent_and_filters(self, state: AnalyticsState) -> dict:
        query_lower = state["business_question"].lower()

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
        return {"intent": intent, "kpi_name": kpi_name, "filters": filters}

    def _detect_intent(self, query: str) -> str:
        for intent, keywords in _INTENT_MAP.items():
            if any(kw in query for kw in keywords):
                return intent
        return "show_kpi"

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

    def _detect_kpi(self, query: str) -> str | None:
        for alias, key in _KPI_ALIASES.items():
            if alias in query:
                return key
        return None  # No explicit KPI mentioned — caller decides the fallback

    def _build_plan(self, intent: str) -> list[str]:
        """Tool sequence recorded in memory — derived from agent.yaml's
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

    # ── Handlers ─────────────────────────────────────────────────────────────
    # Each either returns {"evidence": {...}} for the shared `narrate` node
    # to turn into a response, or {"response": "..."} directly for the
    # early-exit cases main.py never routed through Groq/the template
    # formatter at all (KPI not found, no data for the given filters).

    def handle_show_kpi(self, state: AnalyticsState) -> dict:
        try:
            kpi_name, filters = state["kpi_name"], state["filters"]

            kpi = self.rule_engine.get_kpi(kpi_name)
            if not kpi:
                logger.info(f"KPI '{kpi_name}' missing — triggering Knowledge Update Tool")
                update_result = self.knowledge.update(kpi_name)
                if update_result["success"]:
                    kpi = self.rule_engine.get_kpi(kpi_name)
                else:
                    return {"response": f"KPI '{kpi_name}' not found and could not be auto-generated.\n{update_result['reason']}"}

            actual_col = kpi.get("actual_column")
            budget_col = kpi.get("budget_column")
            cols = [c for c in [actual_col, budget_col] if c]
            df = self.sql.get_period_data(cols, extra_filters=filters)

            if df.empty:
                return {"response": f"No data found for '{kpi['label']}' with the given filters: {filters}"}

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

            return {"evidence": evidence}
        except Exception as e:
            logger.error(f"handle_show_kpi error: {e}", exc_info=True)
            return {"response": f"{_AGENT_ERROR_PREFIX}{e}\nPlease check your query and try again."}

    def handle_variance(self, state: AnalyticsState) -> dict:
        try:
            kpi_name, filters = state["kpi_name"], state["filters"]

            kpi = self.rule_engine.get_kpi(kpi_name)
            if not kpi:
                return {"response": f"KPI '{kpi_name}' not found in Rule Engine."}

            actual_col     = kpi.get("actual_column")
            budget_col     = kpi.get("budget_column")
            prior_yr_col   = kpi.get("prior_year_column")
            ytd_actual_col = kpi.get("ytd_actual_column")
            ytd_budget_col = kpi.get("ytd_budget_column")

            cols = [c for c in [actual_col, budget_col, prior_yr_col, ytd_actual_col, ytd_budget_col] if c]
            df   = self.sql.get_period_data(cols, extra_filters=filters)

            if df.empty:
                return {"response": f"No data for '{kpi['label']}' with filters: {filters}"}

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

            return {"evidence": evidence}
        except Exception as e:
            logger.error(f"handle_variance error: {e}", exc_info=True)
            return {"response": f"{_AGENT_ERROR_PREFIX}{e}\nPlease check your query and try again."}

    def handle_root_cause(self, state: AnalyticsState) -> dict:
        try:
            kpi_name, filters = state["kpi_name"], state["filters"]

            kpi = self.rule_engine.get_kpi(kpi_name)
            if not kpi:
                return {"response": f"KPI '{kpi_name}' not found."}

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

            df_drivers = self.sql.get_variance_drivers(filters=filters)

            if df_drivers.empty:
                return {"response": f"No variance driver data found for filters: {filters}"}

            rc_result = self.root_cause.analyse(df_drivers, total_variance=total_variance)
            evidence.update(rc_result)

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

            return {"evidence": evidence}
        except Exception as e:
            logger.error(f"handle_root_cause error: {e}", exc_info=True)
            return {"response": f"{_AGENT_ERROR_PREFIX}{e}\nPlease check your query and try again."}

    def handle_trend(self, state: AnalyticsState) -> dict:
        try:
            kpi_name, filters = state["kpi_name"], state["filters"]

            kpi = self.rule_engine.get_kpi(kpi_name)
            if not kpi:
                return {"response": f"KPI '{kpi_name}' not found."}

            actual_col = kpi.get("actual_column")
            ts_df = self.sql.get_time_series(actual_col, group_by_col="reporting_date", filters=filters)

            if ts_df.empty or len(ts_df) < 2:
                return {"response": f"Insufficient time-series data for '{kpi['label']}'."}

            trend_result = self.analytics.trend(ts_df, "ds", "y")
            evidence = {"kpi": kpi["label"], "unit": kpi.get("unit"), "filters": filters}
            evidence.update(trend_result)

            return {"evidence": evidence}
        except Exception as e:
            logger.error(f"handle_trend error: {e}", exc_info=True)
            return {"response": f"{_AGENT_ERROR_PREFIX}{e}\nPlease check your query and try again."}

    def handle_forecast(self, state: AnalyticsState) -> dict:
        try:
            kpi_name, filters = state["kpi_name"], state["filters"]

            kpi = self.rule_engine.get_kpi(kpi_name)
            if not kpi:
                return {"response": f"KPI '{kpi_name}' not found."}

            actual_col = kpi.get("actual_column")
            ts_df = self.sql.get_time_series(actual_col, group_by_col="reporting_date", filters=filters)

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
                return {"evidence": evidence}

            ml_result = self.ml.forecast_timeseries(ts_df, periods=6)
            evidence  = {"kpi": kpi["label"], "unit": kpi.get("unit"), "filters": filters}
            evidence.update(ml_result)

            # LightGBM's persisted feature importances as "what typically
            # drives this KPI" evidence alongside Prophet's trend —
            # best-effort, never blocks the response.
            try:
                driver_evidence = self.ml.get_forecast_key_drivers(actual_col)
                if driver_evidence and "error" not in driver_evidence:
                    evidence["key_drivers"] = driver_evidence["key_drivers"]
            except Exception as e:
                logger.warning(f"Forecast key-driver evidence unavailable: {e}")

            return {"evidence": evidence}
        except Exception as e:
            logger.error(f"handle_forecast error: {e}", exc_info=True)
            return {"response": f"{_AGENT_ERROR_PREFIX}{e}\nPlease check your query and try again."}

    def handle_anomaly(self, state: AnalyticsState) -> dict:
        try:
            filters = state["filters"]
            df = self.sql.get_ratio_data(filters=filters or None)
            if df.empty:
                return {"response": "No ratio data available for anomaly detection."}

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
                return {"evidence": evidence}

            label_cols = ["reporting_date", "region", "country_name", "insurance_segment"]
            ml_result  = self.ml.detect_anomalies(
                df,
                label_cols=[c for c in label_cols if c in df.columns],
            )
            return {"evidence": ml_result}
        except Exception as e:
            logger.error(f"handle_anomaly error: {e}", exc_info=True)
            return {"response": f"{_AGENT_ERROR_PREFIX}{e}\nPlease check your query and try again."}

    def handle_segment(self, state: AnalyticsState) -> dict:
        try:
            filters = state["filters"]
            df = self.sql.get_ratio_data(filters=filters or None)
            if df.empty:
                return {"response": "No data available for portfolio segmentation."}

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
                return {"evidence": evidence}

            label_cols = ["region", "country_name", "insurance_segment", "line_of_business"]
            ml_result  = self.ml.segment_portfolio(
                df,
                label_cols=[c for c in label_cols if c in df.columns],
            )
            return {"evidence": ml_result}
        except Exception as e:
            logger.error(f"handle_segment error: {e}", exc_info=True)
            return {"response": f"{_AGENT_ERROR_PREFIX}{e}\nPlease check your query and try again."}

    # ── Routers ──────────────────────────────────────────────────────────────

    # The only 6 intents main.py's if/elif dispatched to a dedicated
    # handler — everything else (including "ytd", which _INTENT_MAP can
    # produce but main.py never special-cased) fell through to its `else`
    # branch, i.e. show_kpi. Kept as the same fallthrough here.
    _HANDLER_INTENTS = {"forecast", "anomaly", "segment", "root_cause", "variance", "trend"}

    def route_by_intent(self, state: AnalyticsState) -> str:
        intent = state["intent"]
        return intent if intent in self._HANDLER_INTENTS else "show_kpi"

    def route_after_handler(self, state: AnalyticsState) -> str:
        """A handler either wrote "response" directly (early-exit — KPI not
        found, no data for the filters) or "evidence" for the shared
        narrate node to turn into a response."""
        return "record_memory" if "response" in state else "narrate"

    # ── Narration / Memory (shared tail nodes) ──────────────────────────────

    def narrate(self, state: AnalyticsState) -> dict:
        response = self.explanation.narrate(state["evidence"], state["business_question"])
        return {"response": response}

    def record_memory(self, state: AnalyticsState) -> dict:
        """Runs for every path — narrated or an early-exit error string —
        exactly like main.py::process() recorded every turn regardless of
        which branch produced the response. `evidence={}` here (not the
        real evidence dict) matches that method's existing behavior."""
        response = state.get("response", "")
        self.memory.add_turn(
            user_query       = state["business_question"],
            intent           = state["intent"],
            kpi_name         = state["kpi_name"],
            filters          = state["filters"],
            tools_used       = self._build_plan(state["intent"]),
            evidence         = {},
            response_summary = response[:300],
        )
        return {}
