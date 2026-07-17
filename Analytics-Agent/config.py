"""
config.py — Central Configuration for the Analytics Agent

Priority order for settings:
  1. config/agent_config.yml   — Agent identity, role, memory, tool registry
  2. config/ml_config.yml      — All ML model settings (swappable)
  3. config/business_rules.yml — Predefined insurance rules + new rules
  4. .env                      — Secrets (API keys)
  5. Hardcoded defaults below  — Last resort fallback
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml

load_dotenv()

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR   = BASE_DIR / "data"
ML_DIR     = BASE_DIR / "ml"
RULES_DIR  = CONFIG_DIR / "rules"          # Moved to config/rules/

# Primary dataset
DATASET_PATH = Path(os.getenv(
    "DATASET_PATH",
    "/Users/virenkhapra/Downloads/insurance_variance_data_native.csv"
))

# ── YAML Loaders ──────────────────────────────────────────────────────────────
def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

# Load all three config YAMLs
AGENT_CONFIG   = _load_yaml(CONFIG_DIR / "agent_config.yml")
ML_CONFIG      = _load_yaml(CONFIG_DIR / "ml_config.yml")
RULES_CONFIG   = _load_yaml(CONFIG_DIR / "business_rules.yml")

# ── Agent Config Accessors ────────────────────────────────────────────────────
AGENT_IDENTITY    = AGENT_CONFIG.get("agent", {})
AGENT_ROLE        = AGENT_CONFIG.get("role", {})
MEMORY_CONFIG     = AGENT_CONFIG.get("memory", {})
TOOL_REGISTRY     = AGENT_CONFIG.get("tools", [])
EXECUTION_PLANS   = AGENT_CONFIG.get("execution_plans", {})
ML_READINESS_CFG  = AGENT_CONFIG.get("ml_readiness", {})
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

# ── Rules Files (JSON — legacy compatibility) ─────────────────────────────────
KPI_DEFINITIONS_PATH      = RULES_DIR / "kpi_definitions.json"
DRILL_DOWN_HIERARCHY_PATH = RULES_DIR / "drill_down_hierarchy.json"
BUSINESS_RULES_YAML_PATH  = CONFIG_DIR / "business_rules.yml"

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Groq LLM Settings ─────────────────────────────────────────────────────────
# Sourced from ml_config.yml's llm_fallback.model — was a separate hardcoded
# string here, duplicating that value with nothing to catch the two drifting.
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
