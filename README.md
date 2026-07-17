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
                   │  FastAPI + LangGraph │
                   └─────────────────────┘
```

Each service runs as its own FastAPI process (so they can be started, stopped, and observed independently), but they share one virtual environment, one dependency list, and one repo history. The orchestrator is the only thing that chains them together — Agent 3 included, called over HTTP exactly like Agent 1/2. See "Agent 3 — Analytics Agent" below.

## What's inside

| Folder | Role | Port |
|---|---|---|
| [`Schema-Intelligence-Layer`](./Schema-Intelligence-Layer) | Quality gate, LLM column descriptions, business domain classification | 8000 |
| [`MVA-use-case-latest-one`](./MVA-use-case-latest-one) | Deep structural profiling, quality/readiness scoring, hierarchy inference, chart generation, AI rule suggestions | 8001 |
| [`Agent-Orchestrator`](./Agent-Orchestrator) | Chains Agent 1 → Agent 2 → (optionally) Agent 3 into one call | 8002 |
| [`Shared-Postgres`](./Shared-Postgres) | The one Postgres server everything persists to, schema-isolated per service | 5433 |
| [`Analytics-Agent`](./Analytics-Agent) | Agent 3 — Insurance-domain Q&A (KPI/variance/root-cause/forecast/anomaly/segment) | 8003 |

Each has its own README going deeper on that piece specifically — this file is the map, not a duplicate.

## Quick Start

1. One virtual environment for the whole thing, from the repo root:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt -r requirements-dev.txt
   ```
2. Two layers of `.env` files:
   - **Root** (`cp .env.example .env`, fill in `GROQ_API_KEY`): shared values genuinely identical across Agent 1, Agent 3, and the Orchestrator — `GROQ_API_KEY`, the `POSTGRES_*` connection details, `LOG_LEVEL`. Each of those three services loads this as a fallback *underneath* its own local `.env` (local always wins on any key both define).
   - **Per-service** (`cp .env.example .env` inside `Schema-Intelligence-Layer/`, `Agent-Orchestrator/`, and `Analytics-Agent/`): only what's genuinely local to that service — e.g. Agent 1's `GROQ_MODEL` override, Agent 3's `DATASET_PATH`/`HOST`/`PORT`, the Orchestrator's `AGENT1_BASE_URL`/`AGENT2_BASE_URL`/etc.
   - `MVA-use-case-latest-one` (Agent 2) is the exception — it has its own differently-shaped config (`DATABASE_URL`, `LLM_API_KEY`, a separate Postgres role) and keeps its own fully self-contained `.env`, untouched by the root file.
3. Start everything at once:

```powershell
powershell -File start-all.ps1
```

This starts the shared Postgres container plus all four services (using the one shared venv), each in its own terminal window. Then:

- Full pipeline (recommended entry point): `http://127.0.0.1:8002/docs`
- Agent 1 alone: `http://127.0.0.1:8000/docs`
- Agent 2 alone: `http://127.0.0.1:8001/docs`
- Agent 3 alone: `http://127.0.0.1:8003/docs`

## The pipeline, end to end

1. **Upload** a CSV/XLSX file to the orchestrator's `/pipeline/run`.
2. **Agent 1** runs a configurable quality gate (10 checks — nulls, duplicates, corrupted values, etc.). Files that fail stop here with a `422`.
3. Passing files get LLM-generated column descriptions and a business-domain classification.
4. **Agent 2** receives Agent 1's output — including that domain classification, applied automatically with no manual input — and runs its full profiling pipeline: structural stats, semantic type detection, secondary-domain classification, hierarchy inference, business-rule evaluation (both YAML-configured and previously human-approved rules), quality/readiness scoring, and chart generation.
5. The LLM also proposes up to 5 candidate business rules from what it saw in that run's columns. These sit as `proposed` until a human approves or rejects them via Agent 2's API — approved rules then automatically apply to every future upload in that domain, closing the loop.
6. If the caller passed a `business_question` **and** Agent 1 classified the file as `Insurance` **and** it's a CSV, **Agent 3** answers that one question using Agent 2's ML-readiness score — see below. Otherwise this step is skipped.
7. All agents' results come back together in one response (`agent1`, `agent2`, `agent3`).
8. **Follow-up questions** don't need to repeat steps 1-5 — `POST /pipeline/ask` (with the earlier response's `agent2.run_id`) re-asks Agent 3 alone, skipping Agent 1's quality gate and Agent 2's full profiling entirely.

## Agent 3 — Analytics Agent (optional, Insurance only)

Agent 3 (`Analytics-Agent/`, port 8003) is a FastAPI service like Agent 1/2 — a thin `app/main.py`/`app/routes/` shell over a LangGraph `StateGraph` (`app/agents/analytics_agent/`). It answers one business question at a time (KPI lookup, variance, root-cause, forecast, anomaly detection, portfolio segmentation) against an uploaded Insurance dataset via DuckDB + ML/LLM tools (`app/services/`), and the orchestrator calls its `POST /analyze` over httpx exactly like it calls Agent 1/2:

- **Runs only when**: `primary_domain == "Insurance"`, the upload is a `.csv`, and `business_question` was supplied. Otherwise `agent3` in the response is `{"status": "skipped", "reason": "..."}`.
- **Inputs it's given**: the same uploaded file (posted straight through, no temp file on the orchestrator's side anymore), and Agent 2's `ml_readiness`/`llm_readiness` scores (from `agent2.readiness_assessments`) as Form fields.
- **Best-effort**: if it's unreachable or returns a non-200, `agent3.status == "failed"` with a reason — this never fails Agent 1/2's already-successful result.
- **Setup**: covered by the normal Quick Start above (`pip install -r requirements.txt` includes its deps — `duckdb`, `prophet`, `lightgbm`, `xgboost`, `scikit-learn`, `shap` — and it needs its own `Analytics-Agent/.env` with `GROQ_API_KEY`, same as Agent 1/2). Nothing separate to clone or install.
- Configured via `Agent-Orchestrator/.env`: `ANALYTICS_AGENT_BASE_URL` (defaults to `http://127.0.0.1:8003`).
- A standalone local-testing CLI (no HTTP server needed) is still available at `Analytics-Agent/scripts/cli.py --query "..."`.
- Originally built as a separate project by a colleague (github.com/VirenKhapra/Analytics-agent-for-project-3) and vendored in here; its own repo still exists independently if contributing changes back upstream.
- **Asking a follow-up question?** Use `POST /pipeline/ask` instead of `/pipeline/run` — re-uploads the file (Agent 3 needs real rows to query; nothing durably stores them elsewhere) but skips Agent 1 and Agent 2's actual pipelines, reusing Agent 2's already-persisted readiness scores by `run_id`. See [`Agent-Orchestrator/README.md`](./Agent-Orchestrator/README.md#re-asking-agent-3-without-re-running-the-whole-pipeline).

## Known constraints worth knowing

- Agent 2 supports 5 primary domains (`Finance`, `Payments`, `Customer`, `HR`, `Insurance`) — each backed by a real config file defining its secondary domains, hierarchy templates, chart templates, and business rules. Agent 1's classification is open-vocabulary (14+ suggested domains) — only ones that land on an exact match to Agent 2's 5 will make it through the full pipeline; anything else stops with a clear error rather than a guess.
- LLM features (column descriptions, domain classification, rule suggestions) require a Groq API key and degrade gracefully to deterministic fallbacks if the key is missing or rate-limited — the pipeline never hard-fails because of the LLM.
- Agent 3 only understands the Insurance dataset's specific column schema (hardcoded, matching `Schema-Intelligence-Layer/test_data/insurance_variance_data_native.csv`) and only reads CSV, not Excel.
