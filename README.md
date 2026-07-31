<div align="center">

# 🤖 MVA — Multi-Agent Data Pipeline

**One upload. End-to-end intelligence.**

*Quality gating → Schema classification → Deep profiling → Hierarchy inference → AI-readiness scoring → Business-rule validation → Chart generation → Natural language analytics Q&A*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-FF6B6B?style=for-the-badge)](https://langchain-ai.github.io/langgraph)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?style=for-the-badge)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Services at a Glance](#-services-at-a-glance)
- [The Pipeline, Step by Step](#-the-pipeline-step-by-step)
- [Quick Start](#-quick-start)
- [Agent 3 — Analytics Agent](#-agent-3--analytics-agent-optional)
- [How Scoring Works](#-how-scoring-works)
- [ML Models Inside Agent 3](#-ml-models-inside-agent-3)
- [Known Constraints](#-known-constraints)
- [Project Structure](#-project-structure)

---

## 🔍 Overview

MVA ingests a **CSV or Excel dataset** and takes it through a fully automated, multi-stage intelligence pipeline — no manual domain entry, no configuration per file. Internally, the system is four cooperating FastAPI microservices sharing one database, one virtual environment, and one dependency list. Each stage is cleanly separable and independently testable; the Orchestrator is the only thing that chains them together.

> **Key idea:** Upload once → get quality gating, schema classification, structural profiling, hierarchy inference, business-rule evaluation, AI-readiness scoring, chart generation, and natural language Q&A — all in one API response.

---

## 🏗 Architecture

```
                     ┌──────────────────────────────────────────┐
   Upload ─────────▶ │          Agent Orchestrator  :8002        │
                     │  Stage 0A · Dataset Registry              │
                     │  (SHA-256 fingerprint — duplicate uploads │
                     │   short-circuit straight to Agent 3,      │
                     │   skipping Agent 1 + 2 entirely)          │
                     └─────────────┬────────────────────────────┘
                                   │  new content / new version /
                                   │  force_revalidate=true
               ┌───────────────────┴──────────────────┐
               ▼                                       ▼
   ┌───────────────────────┐          ┌───────────────────────────┐
   │  Agent 1  :8000        │          │  Agent 2  :8001            │
   │  Schema Intelligence   │─────────▶│  MVA Data Profiling Engine │
   │                        │  domain  │                            │
   │  · Quality gate (10✓) │  +       │  · Structural profiling    │
   │  · LLM column descrs  │  column  │  · Quality / readiness     │
   │  · Domain classif.    │  descrs  │  · Hierarchy inference     │
   │                        │          │  · Charts + rule suggest   │
   └───────────┬────────────┘          └────────────┬──────────────┘
               │                                     │
               └──────────────────┬──────────────────┘
                                  ▼
                       ┌─────────────────────┐
                       │   Shared Postgres    │
                       │   mva_pipeline  :5433│
                       │  (agent1 / agent2    │
                       │   schemas, one DB)   │
                       └──────────┬───────────┘
                                  │  .csv upload + business_question
                                  ▼
                       ┌─────────────────────┐
                       │  Agent 3  :8003      │
                       │  Analytics Agent     │
                       │  (optional)          │
                       │  FastAPI + LangGraph │
                       └─────────────────────┘
```

---

## 📦 Services at a Glance

| Service | Folder | Port | Role |
|---|---|:---:|---|
| **Agent 1** — Schema Intelligence Layer | [`Schema-Intelligence-Layer/`](./Schema-Intelligence-Layer) | `8000` | Quality gate (10 checks), LLM column descriptions, business-domain classification |
| **Agent 2** — Data Profiling Engine | [`Data-Profiling-Agent/`](./Data-Profiling-Agent) | `8001` | Deep structural profiling, quality/readiness scoring, hierarchy inference, chart generation, AI rule suggestions |
| **Orchestrator** | [`Agent-Orchestrator/`](./Agent-Orchestrator) | `8002` | Chains Agent 1 → Agent 2 → Agent 3; Dataset Registry (caching, deduplication) |
| **Shared Postgres** | [`Shared-Postgres/`](./Shared-Postgres) | `5433` | One Postgres server, schema-isolated per service (`agent1` / `agent2`) |
| **Agent 3** — Analytics Agent *(optional)* | [`Analytics-Agent/`](./Analytics-Agent) | `8003` | Domain-agnostic analytics engine — KPI discovery, forecasting, anomaly detection, root-cause analysis, natural language Q&A |

Each service has its own `README.md` going deeper on that piece — this file is the map, not a duplicate.

---

## 🔄 The Pipeline, Step by Step

```
1. Upload CSV/XLSX  →  POST /pipeline/run
      │
      ▼
2. Stage 0A · Dataset Registry
   ├─ Duplicate? (SHA-256 match)  →  serve cached Agent 1+2 result immediately
   └─ New / force_revalidate?     →  continue ↓

3. Agent 1 · Quality Gate (10 weighted checks, threshold = 75)
   ├─ FAIL  →  stop, return 422 with full quality report
   └─ PASS  →  LLM generates column descriptions + business domain classification

4. Orchestrator canonicalizes domain  →  Agent 2

5. Agent 2 · Full Profiling Pipeline
   ├─ Structural stats + semantic type detection
   ├─ Secondary domain classification
   ├─ Dimensional hierarchy inference
   ├─ Business rule evaluation (YAML-configured + human-approved rules)
   ├─ Quality scoring (9 dimensions) + AI-readiness scoring
   ├─ Chart generation
   └─ LLM proposes up to 5 candidate business rules (human approve/reject loop)

6. Agent 3 (if .csv + business_question supplied)
   └─ Answers the question using Agent 2's readiness scores + ML/DuckDB engine

7. Combined response:  agent1 + agent2 + agent3 + fingerprint/copy_id/was_cached
```

> **Follow-up questions?** Use `POST /pipeline/ask` (with the earlier `agent2.run_id`) — skips Agent 1 & 2 entirely, re-asks Agent 3 alone.

---

## 🚀 Quick Start

### 1 · Create the Virtual Environment

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt -r requirements-dev.txt
```

### 2 · Configure Environment Variables

Two layers of `.env` files:

```bash
# Root — shared across Agent 1, Agent 3, and Orchestrator
cp .env.example .env
# Fill in: GROQ_API_KEY, POSTGRES_* connection details, LOG_LEVEL

# Per-service — only what's local to that agent
cp Schema-Intelligence-Layer/.env.example  Schema-Intelligence-Layer/.env
cp Agent-Orchestrator/.env.example         Agent-Orchestrator/.env
cp Analytics-Agent/.env.example            Analytics-Agent/.env
# Data-Profiling-Agent has its own self-contained .env (DATABASE_URL / LLM_API_KEY)
cp Data-Profiling-Agent/.env.example       Data-Profiling-Agent/.env
```

> 💡 Each service loads the root `.env` as a **fallback** — local always wins on any key present in both.

### 3 · Bootstrap the Database *(first time only)*

```powershell
# Windows — native PostgreSQL 17 install, port 5433
& "C:\Program Files\PostgreSQL\17\bin\initdb.exe"  -D "C:\PGData\mva-pipeline" -U postgres --pwprompt
& "C:\Program Files\PostgreSQL\17\bin\pg_ctl.exe"  -D "C:\PGData\mva-pipeline" start
& "C:\Program Files\PostgreSQL\17\bin\createdb.exe" -h localhost -p 5433 -U postgres mva_pipeline
& "C:\Program Files\PostgreSQL\17\bin\psql.exe"    -h localhost -p 5433 -U postgres -d mva_pipeline `
    -f "Shared-Postgres\init\01-create-agent-schemas.sql"

# Agent 2 migrations (Alembic)
cd Data-Profiling-Agent
..\venv\Scripts\python.exe -m alembic upgrade head
cd ..
```

> ✅ Agent 1 and Agent 3 create their own schema/tables idempotently at startup — no manual step needed for them.  
> ✅ Already-initialized instances skip this step entirely — `start-all.ps1` just starts the existing instance.

### 4 · Start Everything

```powershell
powershell -File start-all.ps1
```

This starts the shared Postgres instance (idempotently) plus all four services, each in its own terminal window.

| Endpoint | URL |
|---|---|
| 🗺 Full pipeline *(recommended entry point)* | http://127.0.0.1:8002/docs |
| Agent 1 Swagger UI | http://127.0.0.1:8000/docs |
| Agent 2 Swagger UI | http://127.0.0.1:8001/docs |
| Agent 3 Swagger UI | http://127.0.0.1:8003/docs |

---

## 🧠 Agent 3 — Analytics Agent *(optional)*

Agent 3 (`Analytics-Agent/`, port `8003`) is a FastAPI + LangGraph service — a domain-agnostic, dataset-driven analytics engine that answers one business question per call.

### When It Runs

| Condition | Result |
|---|---|
| Upload is `.csv` **and** `business_question` supplied | ✅ Agent 3 runs |
| Upload is `.xlsx` / `.xls` | ⏭ Skipped — DuckDB reads CSV only |
| No `business_question` | ⏭ Skipped — Agent 3 answers one question at a time |

Works for any of Agent 2's 5 supported domains (`Finance`, `Payments`, `Customer`, `HR`, `Insurance`). A dataset with no matching domain plugin still gets a **real generic report** (trend/forecast/correlation/anomaly/segmentation on whatever columns the dataset actually has) — not a skip.

### 10-Stage Analytics Pipeline

```
Stage 0 · Build Dataset Context   →  Stage 1 · Resolve Capabilities
Stage 2 · Discover KPIs           →  Stage 3 · Interpret Question
Stage 4 · Plan Analytics          →  Stage 5 · Schedule Analyses
Stage 6 · Execute (ML engine)     →  Stage 7 · Execute (LLM engine)
Stage 8 · Collect Evidence        →  Stage 9 · Narrate + Record Memory
```

### What It Explains

Every `status: "ok"` response carries:
- **`execution_trace`** — step-by-step: intent → ML gate → LLM gate, with real per-step timing and model version/accuracy from `ml/model_registry.json`
- **`execution_summary`** — compact rollup of what ran and why

### Re-asking Without Re-running

```bash
POST /pipeline/ask
  run_id=<earlier agent2.run_id>
  file=<same CSV>
  business_question=<new question>
```

Reuses Agent 2's already-persisted readiness scores — skips Agent 1 and Agent 2's full pipelines entirely.

### Standalone CLI *(no HTTP server needed)*

```bash
python Analytics-Agent/scripts/cli.py --query "What is the loss ratio trend?"
```

---

## 📊 How Scoring Works

| Score | Computed by | Formula | Full detail |
|---|---|---|---|
| **Quality gate** (pass/fail) | Agent 1 | 10 weighted checks, weights sum to 100; `passing_score = 75` | [`Schema-Intelligence-Layer/README.md`](./Schema-Intelligence-Layer/README.md) |
| **Data quality score** | Agent 2 | `Σ(weight × score) / Σ(weight)` — `not_assessable` dimensions excluded from both sides | [`Data-Profiling-Agent/README.md`](./Data-Profiling-Agent/README.md) |
| **AI readiness** (analytics / ml / llm / overall) | Agent 2 | Point-additions, 0–100; `≥80` ready · `≥60` partially ready · `<60` not ready | [`Data-Profiling-Agent/README.md`](./Data-Profiling-Agent/README.md) |
| **Capability resolution** | Agent 3 | `ml_readiness_score ≥ 75.0` threshold — Agent 2's number reused directly, never recomputed | [`Analytics-Agent/README.md`](./Analytics-Agent/README.md) |

> **Note:** Agent 3 receives Agent 2's blended composite `.score` — not the `dataset_score` / `task_compatibility_score` split. Only the composite makes the trip.

---

## 🤖 ML Models Inside Agent 3

| Model | Library | Use Case |
|---|---|---|
| **Prophet** | `prophet` | Monthly time-series forecasting with seasonality |
| **LightGBM Regressor** | `lightgbm` | Multi-feature KPI prediction (141 mixed-type columns) |
| **Isolation Forest** | `scikit-learn` | Financial ratio anomaly detection (unsupervised) |
| **XGBoost Classifier** | `xgboost` | Variance driver classification with SHAP explainability |
| **K-Means** | `scikit-learn` | Risk profile segmentation |
| **DuckDB SQL** | `duckdb` | In-process CSV querying (no database server needed) |

> Models are pre-trained against the **Insurance reference dataset** (`train.py`). Non-Insurance domain uploads fit fresh models per request.

---

## ⚠️ Known Constraints

| Constraint | Detail |
|---|---|
| **Agent 2 domains** | Supports exactly 5: `Finance` · `Payments` · `Customer` · `HR` · `Insurance`. Agent 1's classification is open-vocabulary (14+ suggested domains); the Orchestrator canonicalizes known synonyms before forwarding. A genuinely unsupported domain stops the pipeline with a clear error. |
| **Agent 3 file type** | CSV only — DuckDB's `read_csv_auto`. Excel uploads skip Agent 3 with an explicit reason. |
| **LLM dependency** | All LLM calls (column descriptions, domain classification, rule suggestions) use Groq, degrading gracefully to deterministic fallbacks if the API key is missing, rate-limited, or unreachable. The pipeline **never hard-fails** because of the LLM. |
| **ML model scope** | Agent 3's pre-trained models (Prophet/LightGBM/IsolationForest/XGBoost/K-Means) are trained against the Insurance dataset only. Other domains fit fresh per-request models. |
| **Follow-up re-upload** | `POST /pipeline/ask` requires re-uploading the file — Agent 3 queries live CSV rows via DuckDB; nothing durably stores raw file bytes after the original request. |
| **startup script** | `start-all.ps1` is Windows PowerShell with hardcoded native Postgres paths. macOS/Linux users need to start services manually. |

---

## 🗂 Project Structure

```
📦 mva/
├── 📁 Schema-Intelligence-Layer/   # Agent 1 — Quality gate + classification  (port 8000)
│   ├── app/
│   │   ├── agents/                 # LangGraph StateGraph nodes
│   │   ├── config.py
│   │   ├── models/
│   │   ├── prompts/
│   │   ├── routes/
│   │   └── services/
│   ├── config/                     # quality_threshold.json + domain prompts
│   └── tests/
│
├── 📁 Data-Profiling-Agent/        # Agent 2 — Deep profiling + readiness     (port 8001)
│   ├── app/
│   │   ├── agents/                 # LangGraph StateGraph + 2 ReAct sub-agents
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   ├── models/
│   │   ├── repositories/
│   │   ├── schemas/
│   │   └── services/               # charts/ · classification/ · domains/ ·
│   │                               # hierarchy/ · profiling/ · quality/ ·
│   │                               # readiness/ · rules/ · llm/
│   ├── config/                     # Per-domain YAML configs (5 domains)
│   ├── migrations/                 # Alembic migrations
│   └── tests/
│
├── 📁 Agent-Orchestrator/          # Orchestrator — chains agents + registry   (port 8002)
│   ├── app/
│   │   ├── agents/
│   │   │   └── orchestration_agent/
│   │   │       ├── graph.py
│   │   │       ├── nodes/pipeline.py   # All pipeline nodes + domain canonicalization
│   │   │       └── state.py
│   │   ├── config.py
│   │   ├── routes/
│   │   └── services/
│   │       └── dataset_registry/   # SHA-256 fingerprint + Postgres cache
│   └── tests/
│
├── 📁 Analytics-Agent/             # Agent 3 — NL analytics Q&A               (port 8003)
│   ├── app/
│   │   ├── agents/analytics_agent/ # 10-stage LangGraph pipeline
│   │   ├── routes/
│   │   └── services/
│   │       ├── analyzers/
│   │       ├── capability_resolution/
│   │       ├── domain_plugins/     # Insurance (full) + Generic fallback + 4 thin plugins
│   │       ├── ml/                 # forecaster · anomaly_detector · classifier
│   │       ├── planning/
│   │       └── tools/
│   ├── config/                     # ml_config.yml · business_rules.yml · KPI definitions
│   ├── ml/                         # Trained model artifacts + registry.json
│   ├── scripts/cli.py              # Standalone test CLI
│   └── train.py                    # Offline model trainer
│
├── 📁 Shared-Postgres/             # DB init scripts + README                  (port 5433)
│   └── init/01-create-agent-schemas.sql
│
├── 📁 test_data/                   # Sample CSVs/XLSXs (Banking, Insurance, Retail …)
│
├── requirements.txt                # Single dependency set for all services
├── requirements-dev.txt            # Test dependencies (pytest, factory-boy …)
├── start-all.ps1                   # One-command launcher (Windows PowerShell)
├── .env.example                    # Root shared config template
├── README.md                       # ← you are here
└── API_REFERENCE.md                # Full endpoint reference for all 4 services
```

---

## 📄 Further Reading

| Document | Contents |
|---|---|
| [`API_REFERENCE.md`](./API_REFERENCE.md) | Every endpoint across all 4 services, request shapes, and response fields |
| [`Schema-Intelligence-Layer/README.md`](./Schema-Intelligence-Layer/README.md) | Quality gate formulas, classification logic, Agent 1 API detail |
| [`Data-Profiling-Agent/README.md`](./Data-Profiling-Agent/README.md) | Profiling dimensions, readiness scoring, rule suggestion lifecycle, Agent 2 API detail |
| [`Agent-Orchestrator/README.md`](./Agent-Orchestrator/README.md) | Dataset Registry design, caching lifecycle, `/pipeline/ask` detail |
| [`Analytics-Agent/README.md`](./Analytics-Agent/README.md) | 10-stage pipeline, domain plugin architecture, ML model rationale, Agent 3 API detail |
| [`Shared-Postgres/README.md`](./Shared-Postgres/README.md) | Schema layout, role/permission setup, `search_path` gotcha and fix |

---

<div align="center">

*Built as a research project into multi-agent LLM pipeline design.*  
*Agent 3 originally developed by [@VirenKhapra](https://github.com/VirenKhapra/Analytics-agent-for-project-3) — vendored and integrated here.*

</div>
