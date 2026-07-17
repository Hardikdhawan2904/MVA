# Analytics Agent (Agent 3)

Answers one Insurance business question at a time — KPI lookup, variance vs. budget/prior year, root-cause decomposition, trend, forecast, anomaly detection, and portfolio risk segmentation — against an uploaded dataset, using a rule engine, DuckDB, ML models, and Groq for narration.

Originally a separate project by a colleague (github.com/VirenKhapra/Analytics-agent-for-project-3), vendored into this monorepo and rebuilt as a FastAPI + LangGraph service to match Agent 1/2/3's shared architecture — see [Decisions Log](#decisions-log) below for how it got here.

## Architecture

```
POST /analyze (file + business_question + conversation_id + ml_readiness + llm_readiness)
  → detect_intent_and_filters (regex intent/KPI/filter detection, memory carryover)
  → route by intent
      → handle_show_kpi / handle_variance / handle_root_cause / handle_trend
      → handle_forecast / handle_anomaly / handle_segment
        (ml_readiness < threshold → deterministic fallback instead of the ML model)
  → narrate (Groq, or a deterministic template formatter if llm_readiness < threshold
             or Groq is unavailable/rate-limited)
  → record_memory (sliding-window conversation history, persisted to Postgres
                    by conversation_id — see Conversation Memory below)
  → response
```

A thin `app/main.py`/`app/routes/analyze.py` shell over a LangGraph `StateGraph` (`app/agents/analytics_agent/`), same shape as Agent 1/2. Unlike the other two data-intake agents, this graph is built **fresh per request** rather than once at import time — every request answers a question about a *different* uploaded dataset, so its DuckDB connection, rule engine, and ML-readiness-gated tools all have to be bound to that request's own inputs.

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/analyze` | Answer one business question against an uploaded CSV. Params: `file` (multipart upload), `business_question` (str), `conversation_id` (optional str — omit for a new conversation; pass back a prior response's `conversation_id` to continue it), `ml_readiness` (float, default 99.75), `llm_readiness` (float, default 99.75), `feature_recommendation` (optional JSON string — Agent 2's per-column feature classification for this upload, used only to cross-check the hardcoded ML feature-column lists still match this dataset's schema; never blocks the request), `ml_readiness_breakdown` / `llm_readiness_breakdown` (optional JSON strings — Agent 2's full readiness assessment, i.e. strengths/blocking_issues/evidence, for the respective gate; surfaced in `execution_trace` so the gate can explain *why*, not just report a score; never blocks the request if omitted or malformed). |
| GET | `/health` | Liveness check — no downstream agents to ping. |

### Response shape

```json
{
  "status": "ok",
  "query": "Forecast underwriting result for next 6 months",
  "response": "## Summary\n...",
  "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "ml_readiness_score_used": 82.0,
  "llm_readiness_score_used": 95.77,
  "execution_trace": [
    {"step": "intent_detection", "engine": null, "gate": null, "reason": "Detected intent='forecast', kpi='underwriting_result'", "duration_ms": 4.6},
    {"step": "forecast", "engine": "Prophet", "gate": {"name": "ml_readiness", "score": 82.0, "threshold": 75.0, "passed": true, "breakdown": {"evidence": [{"dimension": "feature_coverage", "value": 0.96}], "strengths": [], "blocking_issues": []}}, "reason": "ML readiness (82.0%) met the 75.0% threshold — using the trained Prophet model. Strongest/only factor: feature_coverage (96%).", "duration_ms": 360.3, "model_version": {"refit_per_query": true, "last_run_at": "2026-07-17T22:24:02"}},
    {"step": "narration", "engine": "Groq", "gate": {"name": "llm_readiness", "score": 95.77, "threshold": 75.0, "passed": true, "breakdown": null}, "reason": "LLM readiness (95.8%) met the 75.0% threshold — narrated by Groq.", "duration_ms": 270.7}
  ],
  "execution_summary": {
    "intent": "forecast", "tools_used": ["RuleEngine", "SQLTool", "MLTool→Prophet/LightGBM", "ExplanationTool"],
    "ml_engine": "Prophet", "narration_engine": "Groq", "execution_time_seconds": 0.663, "fallback_used": false
  }
}
```

`execution_trace`/`execution_summary` are built once from the LangGraph run's final state — `null` on `status: "error"` rather than fabricating an explanation for a genuine crash. `gate.breakdown` is `null` when the caller didn't supply `ml_readiness_breakdown`/`llm_readiness_breakdown` (a direct `/analyze` call with just the bare scores). `model_version` only appears on an ML-gated step (`forecast`/`anomaly`/`segment`) that actually ran a model — never on the deterministic-fallback path. Prophet reports `last_run_at` rather than a training date since it refits on every query; IsolationForest/K-Means report `trained_at` plus an explicitly-`null` `accuracy_metric` (both unsupervised — no fabricated number, same reasoning as the LightGBM/K-Means confidence field being omitted rather than invented). `duration_ms` on every step is real per-node wall-clock time from `graph.stream()`, not estimated.

Normally called by [`Agent-Orchestrator`](../Agent-Orchestrator) as the pipeline's optional third stage (Insurance domain + CSV + `business_question` only) — see the [root API reference](../API_REFERENCE.md#agent-3--analytics-agent-optional-third-stage) for that integration, including the readiness-breakdown forwarding. The orchestrator never passes `conversation_id` (it's stateless — every `/pipeline/run` or `/pipeline/ask` call gets a fresh conversation), so multi-turn memory only applies when calling `POST /analyze` directly. Can also be exercised locally without an HTTP server via `scripts/cli.py` (see below).

## Conversation Memory

Backed by Postgres — one table (`agent3.conversation_turns`) on the same [`Shared-Postgres`](../Shared-Postgres) instance Agent 1/2 already use, isolated in its own schema. This matters more than it might look: `run_analytics_graph()` builds a brand-new graph (and therefore a brand-new `MemoryManager`) on *every single HTTP request* — without real persistence there was no way for two separate `POST /analyze` calls to share history at all, restart or no restart. `conversation_id` is what ties requests back to the same history.

- Pass the same `conversation_id` on the next call to continue a conversation — filter/KPI carryover ("what about EMEA?" after an FY2025 GWP question) and the LLM narrator's prior-turn context both depend on this.
- History survives service restarts (it didn't before — the retired CLI's in-process `AnalyticsAgent` object was the only thing keeping turns alive, and even that reset on every subprocess spawn).
- **Non-fatal if Postgres is down**: `init_db()` at startup logs an error and continues rather than crashing the service; `MemoryManager` degrades to in-process-only, non-persistent history for that request if a load or save fails. A question always gets answered even if memory can't be read or written.
- Schema/table are created automatically at startup (`app/services/database.py::init_db()`, called from `app/main.py`'s lifespan) — nothing to migrate by hand.

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

Needs the shared Postgres running for conversation memory (`cd ../Shared-Postgres && docker compose up -d`) — see [Conversation Memory](#conversation-memory) above. Not a hard requirement: if it's unreachable, the service still starts and answers questions, just without memory persisting.

### Standalone CLI (no HTTP server needed)

```bash
..\venv\Scripts\python.exe scripts/cli.py --query "Show Gross Written Premium for FY2025"
..\venv\Scripts\python.exe scripts/cli.py --interactive
```

Reads the dataset from `DATASET_PATH` (see Environment Variables) instead of an upload. Each query calls the graph fresh, same as a real HTTP request — but `--interactive` mode generates one `conversation_id` at startup and reuses it across turns, so memory carries forward within a session the same way a real client passing the same `conversation_id` back on each `POST /analyze` call would.

## Environment Variables

`GROQ_API_KEY` and the `POSTGRES_*` connection details default from the shared **repo-root** `.env` (`../.env`) now — see the [root README](../README.md#quick-start). This service's own `.env` only needs to set them if overriding the shared value specifically for Agent 3.

| Variable | Default | Description |
|----------|---------|-------------|
| GROQ_API_KEY *(root `.env`)* | *(empty)* | Groq API key — narration falls back to a deterministic template formatter if unset or rate-limited. |
| HOST | 0.0.0.0 | Bind host. |
| PORT | 8003 | Bind port. |
| DATASET_PATH | *(colleague's original Mac path)* | **Local testing only** — used by `scripts/cli.py` and `train.py`. Has no effect on `POST /analyze`, which always uses the uploaded file. |
| POSTGRES_HOST *(root `.env`)* | localhost | Shared Postgres host — same instance as Agent 1/2. |
| POSTGRES_PORT *(root `.env`)* | 5433 | Shared Postgres port. |
| POSTGRES_DB *(root `.env`)* | mva_pipeline | Shared database name. |
| POSTGRES_USER *(root `.env`)* | postgres | Connects as the `postgres` superuser, like Agent 1 — not a dedicated role like Agent 2's `mva_user`, since `agent3`'s schema is created idempotently by this service's own `init_db()` at every startup. |
| POSTGRES_PASSWORD *(root `.env`)* | postgres | |

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

Covers the rule engine, analytics tool, ML persistence (including the AST-safe rule-condition evaluator), feature-column validation, LangGraph routing, Postgres-backed conversation memory (round-trip persistence and DB-down degradation, against the real shared instance), the execution trace/summary builder (`tests/test_execution_trace.py` — readiness-breakdown summarization, per-step timing, model versioning, all against the real `ml/model_registry.json` rather than hardcoded values), and a 12-case end-to-end harness exercising every intent (real Groq calls where a key is configured — slower and rate-limit-prone, but confirms the whole path works, not just its parts).

## Known Limitations

- Only understands the Insurance dataset's specific column schema (hardcoded, matching `Schema-Intelligence-Layer/test_data/insurance_variance_data_native.csv`) and only reads CSV, not Excel.
- No authentication in v1.
- ML feature-column lists (`config/ml_config.yml`) are hand-curated Insurance-domain expertise, not derived from Agent 2's generic `feature_recommendation` — `feature_recommendation` is used only as a diagnostic cross-check (see `POST /analyze`'s `feature_recommendation` param above), never to swap which columns a model actually uses.
- Multi-turn conversation memory only works when a caller explicitly passes `conversation_id` back on each request — `Agent-Orchestrator` doesn't do this today (it's intentionally stateless), so pipeline-driven questions each start a fresh conversation. Only a direct `POST /analyze` caller (or `scripts/cli.py --interactive`) gets continuity.

## Decisions Log

| Date | Decision |
|---|---|
| 2026-07-15 | Groq API chosen for faster inference; DuckDB chosen for SQL (no database server needed); all monetary values in USD. |
| 2026-07-17 | Vendored into the `mva` monorepo as Agent 3, initially wired into `Agent-Orchestrator` as a CLI subprocess. |
| 2026-07-17 | Fixed `eval()`-based rule evaluation (security hole + a dormant bug — 4 rules using uppercase `AND` had silently never fired since inception) with an AST-whitelist evaluator; added real model persistence (`ml/persistence.py`); wired in the previously-dead `MemoryManager`; added `llm_readiness` gating (mirrors `ml_readiness`) and `ml/feature_validation.py`. |
| 2026-07-17 | Rebuilt as a full FastAPI + LangGraph service (`app/main.py`, `app/routes/`, `app/agents/analytics_agent/`, `app/services/`) to match Agent 1/2/3's shared architecture, and `Agent-Orchestrator`'s `call_agent3` rewired from a subprocess invocation to a plain `httpx` call — the same shape as its call to Agent 2. The old `main.py`/`tools/`/flat `ml/*.py` CLI structure was retired once the new service was verified working end-to-end. |
| 2026-07-18 | Added a Postgres schema (`agent3`, same shared instance as Agent 1/2) and moved conversation memory from an in-process, always-empty-per-request list to real cross-request persistence keyed by `conversation_id` — this is what makes the sliding-window filter/KPI carryover actually work in the HTTP world, which it structurally couldn't before. `init_db()` failure is deliberately non-fatal, unlike Agent 1's — memory is an enhancement, not core to answering a question. |
| 2026-07-18 | Fixed `AnalyticsTool.variance()` computing favorable/unfavorable purely from the arithmetic sign, never reading each KPI's own `higher_is_better` flag (defined in `config/rules/kpi_definitions.json`, required by the schema, but read nowhere in computation code) — every "lower is better" ratio KPI (loss ratio, expense ratio, combined ratio, ...) had its variance direction backwards. |
| 2026-07-18 | Added `execution_trace`/`execution_summary` to every `POST /analyze` response — a step-by-step decision log (intent → ML gate/engine → LLM gate/engine) built once from the graph's final state rather than touching each of the 7 handler methods. The one real gap it closes: a passed `llm_readiness` gate whose Groq call itself fails is now reported distinctly from a gate that never passed (`ExplanationTool.last_engine_used`, a new observable, makes this possible). |
| 2026-07-18 | Enriched the trace further: Agent 2's full readiness assessment (not just the bare score) now flows through `Agent-Orchestrator` into `execution_trace`'s gate objects, summarized into a strongest/weakest-factor sentence; real per-step `duration_ms` via `graph.stream(..., stream_mode="updates")` instead of `graph.invoke()`, with zero handler changes; and `model_version` metadata read from `ml/model_registry.json` — Prophet reported as `refit_per_query` (it has no fixed training date) rather than mislabeled with a fake one, IsolationForest/K-Means with an explicit `null` accuracy metric (unsupervised — no number to report) rather than a fabricated one. |
