# MVA — Multi-Agent Data Pipeline

**One system** that ingests a CSV/Excel dataset and takes it end-to-end: quality gating, schema classification, structural profiling, hierarchy inference, business-rule validation, AI-readiness scoring, chart generation, and AI-proposed rule suggestions with a human approve/reject loop.

Internally it's organized as three cooperating services plus a shared database — not because they're separate projects, but because each stage (classification, profiling, orchestration) is cleanly separable and independently testable. One repo, one dependency set, one way to run it.

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
                   └─────────────────────┘
```

Each service runs as its own FastAPI process (so they can be started, stopped, and observed independently), but they share one virtual environment, one dependency list, and one repo history. The orchestrator is the only thing that chains them together, and it's deliberately simple (sequential HTTP calls, no workflow framework).

## What's inside

| Folder | Role | Port |
|---|---|---|
| [`Schema-Intelligence-Layer`](./Schema-Intelligence-Layer) | Quality gate, LLM column descriptions, business domain classification | 8000 |
| [`MVA-use-case-latest-one`](./MVA-use-case-latest-one) | Deep structural profiling, quality/readiness scoring, hierarchy inference, chart generation, AI rule suggestions | 8001 |
| [`Agent-Orchestrator`](./Agent-Orchestrator) | Chains the two above into one call | 8002 |
| [`Shared-Postgres`](./Shared-Postgres) | The one Postgres server everything persists to, schema-isolated per service | 5433 |

Each has its own README going deeper on that piece specifically — this file is the map, not a duplicate.

## Quick Start

1. One virtual environment for the whole thing, from the repo root:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt -r requirements-dev.txt
   ```
2. Each of `Schema-Intelligence-Layer`, `MVA-use-case-latest-one`, and `Agent-Orchestrator` still needs its own `.env` file (copy from `.env.example` in each) — Agent 1 and Agent 2 need a Groq API key for LLM features; the orchestrator needs none.
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
6. Both agents' results come back together in one response.

## Known constraints worth knowing

- Agent 2 only supports 4 primary domains (`Finance`, `Payments`, `Customer`, `HR`) — each backed by a real config file defining its secondary domains, hierarchy templates, chart templates, and business rules. Agent 1's classification is open-vocabulary (14+ suggested domains) — only ones that land on an exact match to Agent 2's 4 will make it through the full pipeline; anything else stops with a clear error rather than a guess.
- LLM features (column descriptions, domain classification, rule suggestions) require a Groq API key and degrade gracefully to deterministic fallbacks if the key is missing or rate-limited — the pipeline never hard-fails because of the LLM.
