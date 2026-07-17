# Analytics Agent (Agent 3)

Answers one Insurance business question at a time — KPI lookup, variance vs. budget/prior year, root-cause decomposition, trend, forecast, anomaly detection, and portfolio risk segmentation — against an uploaded dataset, using a rule engine, DuckDB, ML models, and Groq for narration.

Originally a separate project by a colleague (github.com/VirenKhapra/Analytics-agent-for-project-3), vendored into this monorepo and rebuilt as a FastAPI + LangGraph service to match Agent 1/2/3's shared architecture — see [Decisions Log](#decisions-log) below for how it got here.

## Architecture

```
POST /analyze (file + business_question + ml_readiness + llm_readiness)
  → detect_intent_and_filters (regex intent/KPI/filter detection, memory carryover)
  → route by intent
      → handle_show_kpi / handle_variance / handle_root_cause / handle_trend
      → handle_forecast / handle_anomaly / handle_segment
        (ml_readiness < threshold → deterministic fallback instead of the ML model)
  → narrate (Groq, or a deterministic template formatter if llm_readiness < threshold
             or Groq is unavailable/rate-limited)
  → record_memory (sliding-window conversation history, in-process)
  → response
```

A thin `app/main.py`/`app/routes/analyze.py` shell over a LangGraph `StateGraph` (`app/agents/analytics_agent/`), same shape as Agent 1/2. Unlike the other two data-intake agents, this graph is built **fresh per request** rather than once at import time — every request answers a question about a *different* uploaded dataset, so its DuckDB connection, rule engine, and ML-readiness-gated tools all have to be bound to that request's own inputs.

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze` | Answer one business question against an uploaded CSV. Params: `file` (multipart upload), `business_question` (str), `ml_readiness` (float, default 99.75), `llm_readiness` (float, default 99.75), `feature_recommendation` (optional JSON string — Agent 2's per-column feature classification for this upload, used only to cross-check the hardcoded ML feature-column lists still match this dataset's schema; never blocks the request). |
| GET | `/health` | Liveness check — no downstream agents to ping. |

### Response shape

```json
{
  "status": "ok",
  "query": "Show Gross Written Premium for FY2025",
  "response": "## Summary\n...",
  "ml_readiness_score_used": 99.75,
  "llm_readiness_score_used": 99.75
}
```

Normally called by [`Agent-Orchestrator`](../Agent-Orchestrator) as the pipeline's optional third stage (Insurance domain + CSV + `business_question` only) — see the [root API reference](../API_REFERENCE.md#agent-3--analytics-agent-optional-third-stage) for that integration. Can also be called directly, or exercised locally without an HTTP server via `scripts/cli.py` (see below).

## Local Setup

This project shares one virtual environment with the rest of the pipeline — set up once from the **repo root** (see the [root README](../README.md)), not from inside this folder:

```bash
cd ..
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cd Analytics-Agent

# Copy env file and add your Groq API key
cp .env.example .env

..\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8003
```

No database — conversation memory is in-process only (resets on restart), and there's nothing else to persist across requests.

### Standalone CLI (no HTTP server needed)

```bash
..\venv\Scripts\python.exe scripts/cli.py --query "Show Gross Written Premium for FY2025"
..\venv\Scripts\python.exe scripts/cli.py --interactive
```

Reads the dataset from `DATASET_PATH` (see Environment Variables) instead of an upload. Each query calls the graph fresh, same as a real HTTP request — conversation memory does not carry over between turns in `--interactive` mode, unlike the retired CLI's `AnalyticsAgent` object, which stayed alive for the whole session.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| GROQ_API_KEY | *(empty)* | Groq API key — narration falls back to a deterministic template formatter if unset or rate-limited. |
| HOST | 0.0.0.0 | Bind host. |
| PORT | 8003 | Bind port. |
| DATASET_PATH | *(colleague's original Mac path)* | **Local testing only** — used by `scripts/cli.py` and `train.py`. Has no effect on `POST /analyze`, which always uses the uploaded file. |

Domain config (KPI definitions, business rules, ML hyperparameters, feature columns) lives in `config/*.yml`/`config/rules/*.json`, not environment variables — see `app/config.py`'s module docstring for the full priority order.

## Model Training

```bash
..\venv\Scripts\python.exe train.py                      # train all 4 models
..\venv\Scripts\python.exe train.py --model xgboost       # train one model
..\venv\Scripts\python.exe train.py --confusion-matrix    # show XGBoost confusion matrix
```

Trains Prophet-free LightGBM, IsolationForest, XGBoost, and K-Means against `DATASET_PATH`, saving artifacts to `ml/trained/*.pkl` (query-time code predicts against these persisted models instead of refitting per request). `app/main.py`'s startup lifespan runs `app/services/boot_trainer.py`'s freshness check once per process lifetime — if `DATASET_PATH` is newer than the saved models (or any are missing), it retrains automatically before the service starts accepting requests.

## Running Tests

```bash
..\venv\Scripts\python.exe -m pytest tests/ -v
```

Covers the rule engine, analytics tool, ML persistence (including the AST-safe rule-condition evaluator), feature-column validation, LangGraph routing, and a 12-case end-to-end harness exercising every intent (real Groq calls where a key is configured — slower and rate-limit-prone, but confirms the whole path works, not just its parts).

## Known Limitations

- Only understands the Insurance dataset's specific column schema (hardcoded, matching `Schema-Intelligence-Layer/test_data/insurance_variance_data_native.csv`) and only reads CSV, not Excel.
- No authentication in v1.
- ML feature-column lists (`config/ml_config.yml`) are hand-curated Insurance-domain expertise, not derived from Agent 2's generic `feature_recommendation` — `feature_recommendation` is used only as a diagnostic cross-check (see `POST /analyze`'s `feature_recommendation` param above), never to swap which columns a model actually uses.
- Conversation memory is in-process and non-persistent — restarting the service (or, for `scripts/cli.py`, each individual query) starts a fresh conversation.

## Decisions Log

| Date | Decision |
|---|---|
| 2026-07-15 | Groq API chosen for faster inference; DuckDB chosen for SQL (no database server needed); all monetary values in USD. |
| 2026-07-17 | Vendored into the `mva` monorepo as Agent 3, initially wired into `Agent-Orchestrator` as a CLI subprocess. |
| 2026-07-17 | Fixed `eval()`-based rule evaluation (security hole + a dormant bug — 4 rules using uppercase `AND` had silently never fired since inception) with an AST-whitelist evaluator; added real model persistence (`ml/persistence.py`); wired in the previously-dead `MemoryManager`; added `llm_readiness` gating (mirrors `ml_readiness`) and `ml/feature_validation.py`. |
| 2026-07-17 | Rebuilt as a full FastAPI + LangGraph service (`app/main.py`, `app/routes/`, `app/agents/analytics_agent/`, `app/services/`) to match Agent 1/2/3's shared architecture, and `Agent-Orchestrator`'s `call_agent3` rewired from a subprocess invocation to a plain `httpx` call — the same shape as its call to Agent 2. The old `main.py`/`tools/`/flat `ml/*.py` CLI structure was retired once the new service was verified working end-to-end. |
