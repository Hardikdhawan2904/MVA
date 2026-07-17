# MVA Pipeline — API Reference

Two data-intake agents and one orchestrator, each a separate FastAPI service with its own Swagger UI, plus one optional CLI-based agent (vendored into this repo, sharing its venv) invoked as a subprocess. This doc lists every real endpoint, what it's for, and how to open each service's interactive docs locally.

**Flow:** Upload → Agent 1 *(classify + quality-gate)* → Orchestrator *(relay)* → Agent 2 *(profile + score)* → Agent 3 *(optional, Insurance Q&A)* → Combined result

---

## Connect

| Service | Port | Swagger UI | Health check |
|---|---|---|---|
| **Agent 1 — Schema Intelligence** | `8000` | http://127.0.0.1:8000/docs | `GET /health` |
| **Agent 2 — Data Profiling Engine** | `8001` | http://127.0.0.1:8001/docs | `GET /api/v1/health` |
| **Orchestrator** | `8002` | http://127.0.0.1:8002/docs | `GET /health` |

**Agent 1** validates uploads, runs the 10-check quality gate, and classifies business domain via LLM.
**Agent 2** does deep column profiling, quality scoring, hierarchy detection, chart + rule generation.
**Orchestrator** runs a file through Agent 1 then Agent 2 in one call, no manual domain entry needed.

---

## Starting all three locally

1. Postgres must be running first — the shared instance lives in `Shared-Postgres/` (`docker compose up -d`), listening on port `5433`.
2. Each service uses the shared virtual environment at the repo root (`venv/`) and its own `app.main:app` entry point:

```bash
cd Schema-Intelligence-Layer   && ..\venv\Scripts\python -m uvicorn app.main:app --port 8000 --reload
cd MVA-use-case-latest-one     && ..\venv\Scripts\python -m uvicorn app.main:app --port 8001 --reload
cd Agent-Orchestrator          && ..\venv\Scripts\python -m uvicorn app.main:app --port 8002 --reload
```

3. Or run `start-all.ps1` from the repo root — it launches all three (plus Postgres) in separate windows in one shot.
4. Once a service prints `Application startup complete`, its `/docs` URL above is live — open it in a browser to try requests directly, no separate client needed.

---

## Agent 1 — Schema Intelligence Layer

Base URL: `http://127.0.0.1:8000`

First stop for any raw upload. Validates the file, scores it against 10 deterministic quality checks, and — for new files or ones explicitly flagged — classifies the business domain and describes every column via an LLM. Its classification is what downstream agents rely on; there's no manual override.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/upload-dataset` | Upload a CSV or Excel file. Runs the quality gate (aborts with 422 on FAIL), classifies the domain, catalogs it in Postgres, and caches the parsed rows in memory. Params: `file` (multipart upload), `force_reclassify` (bool — re-run LLM classification instead of reusing a cached result) |
| `GET` | `/datasets` | List every cataloged dataset with its domain, row/column counts, and quality score. |
| `GET` | `/datasets/{dataset_id}` | Full catalog record for one dataset — classification, column descriptions, full quality report. |
| `GET` | `/datasets/{dataset_id}/dataframe` | The cached row data as JSON records — only available while the server that processed it hasn't restarted. Param: `limit` (optional, cap rows returned) |
| `GET` | `/health` | Liveness check — returns `{"status": "healthy"}` when the service is up. |

---

## Agent 2 — Data Profiling Engine

Base URL: `http://127.0.0.1:8001/api/v1`

Takes a raw file plus the primary domain Agent 1 already determined, and produces column-level profiling, secondary classification, hierarchy detection, quality scoring across 9 dimensions, chart candidates, and business-rule evaluation in one pass.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/profile-runs` | Start a profiling run. Runs synchronously and returns once complete. Params: `file` (multipart upload), `primary_domain` (must be Finance, Payments, Customer, HR, or Insurance), `sheet_name` (required only for multi-sheet Excel workbooks) |
| `GET` | `/profile-runs/{run_id}` | Run summary — status, domain, row/column counts, timestamps. |
| `GET` | `/profile-runs/{run_id}/result` | The complete result — every column profile, quality dimension, chart, hierarchy edge, and rule evaluation in one payload. |
| `GET` | `/profile-runs/{run_id}/columns` | Just the column-level profiles (types, stats, semantic classification) for this run. |
| `GET` | `/profile-runs/{run_id}/quality` | Quality assessment scores across all 9 dimensions plus the overall weighted score. |
| `GET` | `/profile-runs/{run_id}/readiness` | Analytics / ML / LLM readiness assessments — is this dataset actually usable downstream. |
| `GET` | `/profile-runs/{run_id}/hierarchy` | The detected dimensional hierarchy (e.g. region → country → branch) with per-edge confidence. |
| `GET` | `/profile-runs/{run_id}/rule-evaluations` | Results of evaluating this domain's business rules against the uploaded data. |
| `GET` | `/profile-runs/{run_id}/charts` | Generated chart specs — domain-specific where the data supports them, generic otherwise — with aggregated data attached. |
| `POST` | `/profile-runs/{run_id}/charts/{chart_id}/drill-down` | Drill into one level of a hierarchy chart (e.g. from country down into its cities) for a specific path. |
| `GET` | `/rule-suggestions` | List all LLM-proposed business rules across runs, optionally filtered by approval status. Param: `status` (optional, e.g. proposed / approved / rejected) |
| `GET` | `/rule-suggestions/{suggestion_id}` | A single proposed rule's full detail. |
| `POST` | `/rule-suggestions/{suggestion_id}/approve` | Approve a proposed rule so it's evaluated against future uploads in this domain. |
| `POST` | `/rule-suggestions/{suggestion_id}/reject` | Reject a proposed rule. |
| `GET` | `/health` | Liveness + database connectivity check. |

---

## Orchestrator

Base URL: `http://127.0.0.1:8002`

The one call to make if you just want a file profiled end to end. Sends the upload to Agent 1, takes whatever domain it decides on, and forwards straight into Agent 2 — no domain has to be picked by hand. Stops cleanly with a clear error at whichever stage fails.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/pipeline/run` | Runs a file through both agents and returns all results together under one response. Params: `file` (multipart upload), `sheet_name` (required only for multi-sheet Excel workbooks), `force_reclassify` (bool — re-run Agent 1's LLM classification), `business_question` (optional — also drives Agent 3, see below), `target_column` (optional) |
| `GET` | `/health` | Liveness check that also reports whether Agent 1 and Agent 2 are currently reachable. |

`/pipeline/run`'s response body is `{"agent1": {...}, "agent2": {...}, "agent3": {...} | null, "primary_domain_used": "..."}`.

### Agent 3 — Analytics Agent (optional third stage)

Vendored into this repo at `Analytics-Agent/` (originally a colleague's separate project, github.com/VirenKhapra/Analytics-agent-for-project-3) — a CLI tool, not an HTTP service, so the orchestrator shells out to it as a subprocess instead of calling it over `httpx`. It installs from the same root `requirements.txt` and runs under the same shared venv as the other three folders. It answers one business question at a time over an Insurance dataset (KPI lookup, variance, root-cause, forecast, anomaly detection, segmentation).

Runs only when **all** of: `primary_domain == "Insurance"`, the upload is a `.csv`, and `business_question` was supplied. Fed Agent 2's `ml_readiness` score (`agent2.readiness_assessments[].score` where `assessment_type == "ml_readiness"`) as `--ml-readiness`. Response shapes:

```jsonc
// Ran successfully:
"agent3": {"status": "ok", "query": "...", "ml_readiness_score_used": 39.48, "response": "<narrative text>"}

// Outside its scope (wrong domain / no question / not CSV):
"agent3": {"status": "skipped", "reason": "..."}

// Invoked but errored/timed out — never fails the overall pipeline:
"agent3": {"status": "failed", "reason": "..."}
```

Configured via `Agent-Orchestrator/.env`: `ANALYTICS_AGENT_PATH` (defaults to `Analytics-Agent/` inside this repo), `ANALYTICS_AGENT_PYTHON` (optional override; defaults to the orchestrator's own interpreter, since Agent 3 shares the same venv), `ANALYTICS_AGENT_TIMEOUT_SECONDS`.

---

Ports assume the default local setup (Agent 1 → 8000, Agent 2 → 8001, Orchestrator → 8002) with Postgres on 5433. If your friend is running on a different machine, swap `127.0.0.1` for that machine's address and make sure the ports are reachable — Swagger itself needs nothing beyond the running service.
