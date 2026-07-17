# MVA — Multi-Agent Data Pipeline

**One system** that ingests a CSV/Excel dataset and takes it end-to-end: quality gating, schema classification, structural profiling, hierarchy inference, business-rule validation, AI-readiness scoring, chart generation, and AI-proposed rule suggestions with a human approve/reject loop.

Internally it's organized as four cooperating services plus a shared database — not because they're separate projects, but because each stage (classification, profiling, orchestration, Insurance Q&A) is cleanly separable and independently testable. One repo, one dependency set, one way to run it.

## Architecture

```
                     ┌────────────────────┐
   Upload ─────────▶ │  Agent Orchestrator │
                     └─────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                  ▼
   ┌─────────────────────┐          ┌─────────────────────────┐
   │  Agent 1             │          │  Agent 2                │
   │  Schema Intelligence  │ ───────▶ │  MVA Data Profiling      │
   │  Layer                │  domain  │  Engine                  │
   │  (quality gate,       │  +      │  (profiling, quality,    │
   │  classification,      │  column │  hierarchy, readiness,   │
   │  column descriptions) │  descrs │  charts, rule suggestions)│
   └──────────┬───────────┘          └────────────┬─────────────┘
              │                                    │
              └──────────────┬─────────────────────┘
                              ▼
                   ┌─────────────────────┐
                   │   Shared Postgres    │
                   │  (agent1 / agent2    │
                   │   schemas, one DB)   │
                   └──────────┬──────────┘
                              │ (Insurance domain +
                              │  business_question only)
                              ▼
                   ┌─────────────────────┐
                   │  Agent 3 (optional)  │
                   │  Analytics Agent     │
                   │  — CLI subprocess,   │
                   │  shared venv         │
                   └─────────────────────┘
```

Each service runs as its own FastAPI process (so they can be started, stopped, and observed independently), but they share one virtual environment, one dependency list, and one repo history. The orchestrator is the only thing that chains them together.

Agent 3 is the one exception to "FastAPI process": it's a CLI tool (originally from a colleague's project, now vendored into this repo at `Analytics-Agent/`), not a web service, so the orchestrator shells out to it as a subprocess rather than calling it over HTTP — but it installs from the same root `requirements.txt` and runs under the same shared venv as everything else. See "Agent 3 — Analytics Agent" below.

## What's inside

| Folder | Role | Port |
|---|---|---|
| [`Schema-Intelligence-Layer`](./Schema-Intelligence-Layer) | Quality gate, LLM column descriptions, business domain classification | 8000 |
| [`MVA-use-case-latest-one`](./MVA-use-case-latest-one) | Deep structural profiling, quality/readiness scoring, hierarchy inference, chart generation, AI rule suggestions | 8001 |
| [`Agent-Orchestrator`](./Agent-Orchestrator) | Chains Agent 1 → Agent 2 → (optionally) Agent 3 into one call | 8002 |
| [`Shared-Postgres`](./Shared-Postgres) | The one Postgres server everything persists to, schema-isolated per service | 5433 |
| [`Analytics-Agent`](./Analytics-Agent) | Agent 3 — Insurance-domain Q&A (KPI/variance/root-cause/forecast/anomaly/segment), invoked as a CLI subprocess, not a web service | — |

Each has its own README going deeper on that piece specifically — this file is the map, not a duplicate.

## Quick Start

1. One virtual environment for the whole thing, from the repo root:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt -r requirements-dev.txt
   ```
2. Each of `Schema-Intelligence-Layer`, `MVA-use-case-latest-one`, `Agent-Orchestrator`, and `Analytics-Agent` still needs its own `.env` file (copy from `.env.example` in each) — Agent 1, Agent 2, and Agent 3 need a Groq API key for LLM features; the orchestrator needs none.
3. Start everything at once:

```powershell
powershell -File start-all.ps1
```

This starts the shared Postgres container plus all three services (using the one shared venv), each in its own terminal window. Then:

- Full pipeline (recommended entry point): `http://127.0.0.1:8002/docs`
- Agent 1 alone: `http://127.0.0.1:8000/docs`
- Agent 2 alone: `http://127.0.0.1:8001/docs`

## The pipeline, end to end

1. **Upload** a CSV/XLSX file to the orchestrator's `/pipeline/run`.
2. **Agent 1** runs a configurable quality gate (10 checks — nulls, duplicates, corrupted values, etc.). Files that fail stop here with a `422`.
3. Passing files get LLM-generated column descriptions and a business-domain classification.
4. **Agent 2** receives Agent 1's output — including that domain classification, applied automatically with no manual input — and runs its full profiling pipeline: structural stats, semantic type detection, secondary-domain classification, hierarchy inference, business-rule evaluation (both YAML-configured and previously human-approved rules), quality/readiness scoring, and chart generation.
5. The LLM also proposes up to 5 candidate business rules from what it saw in that run's columns. These sit as `proposed` until a human approves or rejects them via Agent 2's API — approved rules then automatically apply to every future upload in that domain, closing the loop.
6. If the caller passed a `business_question` **and** Agent 1 classified the file as `Insurance` **and** it's a CSV, **Agent 3** answers that one question using Agent 2's ML-readiness score — see below. Otherwise this step is skipped.
7. All agents' results come back together in one response (`agent1`, `agent2`, `agent3`).

## Agent 3 — Analytics Agent (optional, Insurance only)

Unlike Agent 1/2, Agent 3 (`Analytics-Agent/`) is a **CLI tool**, not an HTTP service — it answers one business question at a time (KPI lookup, variance, root-cause, forecast, anomaly detection, portfolio segmentation) against an Insurance dataset via DuckDB + ML/LLM tools. It shares this repo's dependencies and venv like the other three folders; the orchestrator just invokes it as a subprocess instead of calling it over HTTP, since it has no server to call:

- **Runs only when**: `primary_domain == "Insurance"`, the upload is a `.csv`, and `business_question` was supplied. Otherwise `agent3` in the response is `{"status": "skipped", "reason": "..."}`.
- **Inputs it's given**: the same uploaded rows (written to a temp CSV per request), and Agent 2's `ml_readiness` score (from `agent2.readiness_assessments`) passed as `--ml-readiness`.
- **Best-effort**: if it errors out or times out, `agent3.status == "failed"` with a reason — this never fails Agent 1/2's already-successful result.
- **Setup**: covered by the normal Quick Start above (`pip install -r requirements.txt` includes its deps — `duckdb`, `prophet`, `lightgbm`, `xgboost`, `scikit-learn`, `shap` — and it needs its own `Analytics-Agent/.env` with `GROQ_API_KEY`, same as Agent 1/2). Nothing separate to clone or install.
- Configured via `Agent-Orchestrator/.env`: `ANALYTICS_AGENT_PATH` (defaults to `Analytics-Agent/` inside this repo) and optionally `ANALYTICS_AGENT_PYTHON` to point at a different interpreter — left blank, it uses the orchestrator's own (shared venv) interpreter.
- Originally built as a separate project by a colleague (github.com/VirenKhapra/Analytics-agent-for-project-3) and vendored in here; its own repo still exists independently if contributing changes back upstream.

## Known constraints worth knowing

- Agent 2 supports 5 primary domains (`Finance`, `Payments`, `Customer`, `HR`, `Insurance`) — each backed by a real config file defining its secondary domains, hierarchy templates, chart templates, and business rules. Agent 1's classification is open-vocabulary (14+ suggested domains) — only ones that land on an exact match to Agent 2's 5 will make it through the full pipeline; anything else stops with a clear error rather than a guess.
- LLM features (column descriptions, domain classification, rule suggestions) require a Groq API key and degrade gracefully to deterministic fallbacks if the key is missing or rate-limited — the pipeline never hard-fails because of the LLM.
- Agent 3 only understands the Insurance dataset's specific column schema (hardcoded, matching `Schema-Intelligence-Layer/test_data/insurance_variance_data_native.csv`) and only reads CSV, not Excel.
