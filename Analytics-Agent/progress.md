# Analytics Agent — Progress Tracker

> Last Updated: 2026-07-15 | Status: ✅ All phases complete — Agent live

---

## Build Phases

| Phase | Name | Status |
|---|---|---|
| Phase 1 | Foundation (config, requirements, rules) | ✅ Done |
| Phase 2 | Data Layer (DuckDB SQL Tool) | ✅ Done |
| Phase 3 | Analytics Core | ✅ Done |
| Phase 4 | ML Layer | ✅ Done |
| Phase 5 | Intelligence Layer (Groq LLM) | ✅ Done |
| Phase 6 | Orchestrator (main.py) | ✅ Done |
| Phase 7 | Tests & Polish | ✅ Done |

---

## File → Component Mapping & Working Description

| File | Component | Description & Working | Status |
|---|---|---|---|
| `main.py` | **Agent Orchestrator** | Entrypoint. Parses inputs, identifies user intent, sequences the execution plan, handles conversation memory, and coordinates tool execution. | ✅ |
| `train.py` | **ML Trainer** | Offline script that loads raw CSV data, trains XGBoost (classifier), LightGBM (regressor), K-Means (segmentation), and Isolation Forest (anomalies), and saves them to disk. | ✅ |
| `config.py` | **Config System Loader** | Bootstraps the application by parsing the configuration YAML files, checking environment variables, and establishing paths. | ✅ |
| `requirements.txt` | **Dependencies** | Declares standard Python packages (e.g. `pydantic`, `pandas`, `duckdb`, `prophet`, `xgboost`, `lightgbm`, `scikit-learn`). | ✅ |
| `config/agent_config.yml` | **Orchestration Config** | Defines the agent's identity, system prompts, conversation memory limitations, execution paths, and intent keywords. | ✅ |
| `config/ml_config.yml` | **ML Config** | Configures parameters for ML models, feature/target columns, readiness score parameters, and fallbacks. | ✅ |
| `config/business_rules.yml` | **Rules Config** | Houses the 18 predefined business rules, alerts thresholds, controllable flag mappings, and dynamic runtime rules. | ✅ |
| `config/rules/kpi_definitions.json` | **KPI Mappings** | Connects user terms (like GWP, loss ratio) to database columns, units, and aggregation logic. | ✅ |
| `config/rules/drill_down_hierarchy.json` | **Hierarchy Config** | Map defining how to zoom in on dimensions (e.g., Region $\rightarrow$ Country for Geography; Year $\rightarrow$ Quarter for Time). | ✅ |
| `tools/schemas.py` | **Validation Layer** | Defines Pydantic v2 schemas (`KPIDefinitionsRoot`, `BusinessRulesRoot`) to validate all YAML/JSON config files on startup. | ✅ |
| `tools/rule_engine.py` | **Rules Evaluator** | Reads the rules and checks if aggregated metric values cross business thresholds (e.g., loss ratio > 100%). | ✅ |
| `tools/sql_tool.py` | **Database Query Engine** | Dynamically constructs SQL queries and queries the CSV data using DuckDB. Returns filtered Pandas DataFrames. | ✅ |
| `tools/analytics_tool.py` | **Mathematical Calculator** | Computes math formulas (sums, averages, growth, differences) on DataFrames returned from the SQL query. | ✅ |
| `tools/root_cause_tool.py` | **Decomposition Engine** | Analyzes the 14 variance driver columns to calculate exact percentage contributions and identify the root cause of drops or increases. | ✅ |
| `tools/ml_tool.py` | **ML Orchestrator** | Performs data readiness verification. Routes the workflow to ML models (if ready) or fallback mathematical models (if readiness is low). | ✅ |
| `tools/explanation_tool.py` | **LLM Narrator** | Sends structured mathematical results (JSON) to Groq's Llama 3.3 model to write a business-ready executive explanation. | ✅ |
| `tools/knowledge_update_tool.py` | **Dynamic Learner** | Discovers missing KPIs in user requests, creates definitions via LLM, validates them, and appends them to configs at runtime. | ✅ |
| `ml/forecaster.py` | **TimeSeries/Regression Models** | Houses Prophet forecasting code and LightGBM regression logic for predicting underwriting result actuals. | ✅ |
| `ml/anomaly_detector.py` | **Outlier Models** | Implements Isolation Forest to scan financial ratio metrics and flag unusual data points. | ✅ |
| `ml/classifier.py` | **Classification/Clustering Models** | Implements XGBoost to classify the primary variance drivers and K-Means for risk segmentation. | ✅ |
| `ml/model_registry.json` | **ML Registry** | Stores metadata (timestamps, training parameters, accuracy metrics) for all trained ML model artifacts. | ✅ |


---

## Algorithm Registry

| Algorithm | Library | Use Case | File |
|---|---|---|---|
| **Prophet** | `prophet` | Monthly underwriting result forecasting | `ml/forecaster.py` |
| **LightGBM Regressor** | `lightgbm` | Multi-feature KPI prediction | `ml/forecaster.py` |
| **Isolation Forest** | `scikit-learn` | Financial ratio anomaly detection | `ml/anomaly_detector.py` |
| **XGBoost Classifier** | `xgboost` | Variance driver classification | `ml/classifier.py` |
| **K-Means** | `scikit-learn` | Risk profile segmentation | `ml/classifier.py` |
| **DuckDB SQL** | `duckdb` | In-process CSV querying | `tools/sql_tool.py` |
| **Groq LLM** | `groq` | Evidence to business language | `tools/explanation_tool.py` |

---

## ML Model Rationale

### Prophet — Why?
- Business time series with quarterly seasonality (insurance peaks at Q4/year-end)
- Handles missing months without imputation
- Built-in uncertainty intervals required for regulated reporting
- Interpretable trend + seasonality decomposition

### LightGBM — Why?
- 141 mixed-type columns; handles categoricals natively
- 10-100x faster than XGBoost on this row count
- Feature importance maps directly to business drivers (GWP, claims, expenses)
- Used when forecast depends on many input variables

### Isolation Forest — Why?
- Unsupervised (no labelled anomalies needed in dataset)
- Detects ratio outliers: loss_ratio > 100%, combined_ratio > 105%
- Fast on 2,000 rows; zero hyperparameter tuning needed
- Returns contamination score for ranking severity

### XGBoost Classifier — Why?
- primary_variance_driver and root_cause_tag are pre-labelled (supervised learning)
- Gold standard for multi-class tabular classification
- SHAP values provide post-hoc explainability per prediction
- Every prediction backed by feature weights — no hallucination

### K-Means — Why?
- Unsupervised risk-tier grouping
- Cluster centroids = interpretable risk profiles
- Invoked only on segmentation/portfolio queries

---

## Groq Model Config

| Setting | Value |
|---|---|
| Provider | Groq API |
| Model | llama-3.3-70b-versatile |
| Temperature | 0.0 (deterministic) |
| Role | Explanation Tool only |
| Constraint | Must NOT calculate or infer — only narrate supplied evidence |

---

## Decisions Log

| Date | Decision |
|---|---|
| 2026-07-15 | Groq API chosen for faster inference |
| 2026-07-15 | CLI interface confirmed (python main.py --query) |
| 2026-07-15 | ~~CSV at /Users/virenkhapra/Downloads/insurance_variance_data_native.csv~~ — superseded 2026-07-17: vendored into `mva/Analytics-Agent/`, `DATASET_PATH` now overridden per-request by the orchestrator subprocess call (see below) |
| 2026-07-15 | DuckDB chosen for SQL — no database server needed |
| 2026-07-15 | All monetary values in USD (reporting_currency_code = USD) |
| 2026-07-17 | Vendored into the `mva` monorepo as Agent 3, wired into `Agent-Orchestrator` as a subprocess (real HTTP service integration doesn't apply — this stays a CLI tool, invoked with `--ml-readiness`/`--llm-readiness`/`--feature-recommendation-file` sourced from Agent 2's per-upload output) |
| 2026-07-17 | Fixed `eval()`-based rule evaluation (security hole + a dormant bug — 4 rules using uppercase `AND` had silently never fired since inception) with an AST-whitelist evaluator |
| 2026-07-17 | Added real model persistence (`ml/persistence.py`) — IsolationForest/KMeans/XGBoost/LightGBM now predict against the `train.py`-fit models instead of refitting from scratch on every query |
| 2026-07-17 | Wired in the previously dead `MemoryManager` (KPI/filter carryover + LLM context on follow-up queries) and closed several config/code duplication gaps |
| 2026-07-17 | Added `llm_readiness` gating (mirrors `ml_readiness`) and `ml/feature_validation.py` (boot-time check that hardcoded ML feature columns still match Agent 2's per-upload column classification) |
| 2026-07-17 | Removed dead code found by a full audit: `tools.py` (unreachable duplicate of `tools/__init__.py`), the stale/diverged `config/rules/business_rules.json`, and several orphaned methods in `analytics_tool.py`/`root_cause_tool.py`/`knowledge_update_tool.py` with zero callers anywhere |
