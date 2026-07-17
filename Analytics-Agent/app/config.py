"""
app/config.py — Central Configuration for the Analytics Agent

Unlike the other three services in this monorepo, this file deliberately
blends two roles the others keep separate (a pure-env `Settings` class in
`app/config.py`, YAML-driven agent config in `app/agents/<name>/config.py`):
Agent 3's business logic (`app/services/*.py`) is so uniformly config-driven
(KPI definitions, business rules, ML hyperparameters, feature columns) that
splitting the loader across two files would just mean every service imports
from two places for one coherent set of settings. `app/agents/analytics_agent/config.py`
still exists and still owns the LangGraph-specific accessor (`get_entry_point()`),
matching the other agents' structure — it just reads through this module
rather than re-parsing `agent.yaml` a second time.

Priority order for settings:
  1. app/agents/analytics_agent/agent.yaml — agent identity, role, memory, execution plans
  2. config/ml_config.yml                  — all ML model settings (swappable)
  3. config/business_rules.yml             — predefined insurance rules + new rules
  4. .env                                  — secrets (API keys), HOST/PORT
  5. Hardcoded defaults below              — last resort fallback
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

load_dotenv()

# ── Base Paths ────────────────────────────────────────────────────────────────
# This file lives at app/config.py, one level deeper than the old top-level
# config.py — .parent.parent to get back to the Analytics-Agent repo root.
BASE_DIR   = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR   = BASE_DIR / "data"
ML_DIR     = BASE_DIR / "ml"
RULES_DIR  = CONFIG_DIR / "rules"
AGENT_YAML_PATH = BASE_DIR / "app" / "agents" / "analytics_agent" / "agent.yaml"

# Primary dataset — only relevant to scripts/cli.py and train.py now; the
# FastAPI route (app/routes/analyze.py) always overrides this per-request
# with the uploaded file's own temp path.
DATASET_PATH = Path(os.getenv(
    "DATASET_PATH",
    "/Users/virenkhapra/Downloads/insurance_variance_data_native.csv"
))

# ── Service host/port ──────────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8003"))

# ── YAML Loaders ──────────────────────────────────────────────────────────────
def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

AGENT_CONFIG = _load_yaml(AGENT_YAML_PATH)
ML_CONFIG    = _load_yaml(CONFIG_DIR / "ml_config.yml")
RULES_CONFIG = _load_yaml(CONFIG_DIR / "business_rules.yml")

# ── Agent Config Accessors ────────────────────────────────────────────────────
AGENT_IDENTITY    = AGENT_CONFIG.get("agent", {})
AGENT_ROLE        = AGENT_CONFIG.get("role", {})
MEMORY_CONFIG     = AGENT_CONFIG.get("memory", {})
TOOL_REGISTRY     = AGENT_CONFIG.get("tools", [])
EXECUTION_PLANS   = AGENT_CONFIG.get("execution_plans", {})
DATASET_CTX       = AGENT_CONFIG.get("dataset_context", {})
SYSTEM_PROMPT_CFG = AGENT_CONFIG.get("system_prompt", {})

# ── ML Config Accessors ───────────────────────────────────────────────────────
ML_SETTINGS         = ML_CONFIG.get("ml_settings", {})
ML_READINESS_THRESHOLD  = float(ML_SETTINGS.get("readiness_threshold", 75.0))
LLM_READINESS_THRESHOLD = float(ML_SETTINGS.get("llm_readiness_threshold", 75.0))
ML_FALLBACK_ENABLED = ML_SETTINGS.get("fallback_on_low_readiness", True)
ML_MODEL_SAVE_DIR   = BASE_DIR / ML_SETTINGS.get("model_save_dir", "ml/trained")

PROPHET_CFG    = ML_CONFIG.get("prophet", {})
LGBM_CFG       = ML_CONFIG.get("lightgbm", {})
ISO_FOREST_CFG = ML_CONFIG.get("isolation_forest", {})
XGBOOST_CFG    = ML_CONFIG.get("xgboost", {})
KMEANS_CFG     = ML_CONFIG.get("kmeans", {})
LLM_FALLBACK   = ML_CONFIG.get("llm_fallback", {})

# ── Business Rules Accessors ──────────────────────────────────────────────────
PREDEFINED_RULES  = RULES_CONFIG.get("predefined_rules", [])
KPI_THRESHOLDS    = RULES_CONFIG.get("thresholds", {})
VARIANCE_DRIVERS  = RULES_CONFIG.get("variance_drivers", {})
FLAG_DECODING     = RULES_CONFIG.get("flags", {})
NEW_RULES         = RULES_CONFIG.get("new_rules", [])
SUPPORTED_OPS     = RULES_CONFIG.get("supported_operations", [])

# ── Rules Files (JSON) ─────────────────────────────────────────────────────────
KPI_DEFINITIONS_PATH      = RULES_DIR / "kpi_definitions.json"
DRILL_DOWN_HIERARCHY_PATH = RULES_DIR / "drill_down_hierarchy.json"
BUSINESS_RULES_YAML_PATH  = CONFIG_DIR / "business_rules.yml"

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Groq LLM Settings ─────────────────────────────────────────────────────────
# Sourced from ml_config.yml's llm_fallback.model — one source of truth
# instead of a separately hardcoded string.
GROQ_MODEL       = LLM_FALLBACK.get("model", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = 0.0
GROQ_MAX_TOKENS  = 1024

# ── ML Runtime Settings ───────────────────────────────────────────────────────
ANOMALY_CONTAMINATION = float(
    ISO_FOREST_CFG.get("hyperparameters", {}).get("contamination", 0.05)
)
FORECAST_PERIODS = int(
    PROPHET_CFG.get("forecast_horizons", {}).get("default_periods", 6)
)

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# ── Reporting ──────────────────────────────────────────────────────────────────
REPORTING_CURRENCY = "USD"
AMOUNT_UNIT        = "USD thousands"

# ── Model Registry ─────────────────────────────────────────────────────────────
MODEL_REGISTRY_PATH = ML_DIR / "model_registry.json"
