# MVA Data Profiling Engine

**Multi-Variance Analysis — Schema, Quality, Hierarchy, Readiness, and Chart Intelligence**

A production-structured backend that profiles CSV/XLSX datasets, classifies them into domains, validates hierarchy structures, assesses data quality, evaluates AI-readiness, and generates typed chart specifications.

## Architecture

```
File Upload → Validation → Profiling → Type Refinement → Semantic Candidates
    → Schema Intelligence → Domain Classification → Category Classification
    → Hierarchy Inference → Business Rules → Quality Assessment
    → AI Readiness → Chart Generation → Persist Results → Cleanup
```

Every stage produces typed results. Non-critical failures do not destroy successful upstream results.

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 16 — runs from a shared server used by all agents in the pipeline (see below), not from this repo directly

### Local Development

This project shares one virtual environment with the rest of the pipeline — see the [root README](../README.md). From the repo root:

```bash
# Start the shared Postgres server (once)
cd Shared-Postgres
docker compose up -d
cd ..

# Create the shared environment and install (once, for all three services)
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cd MVA-use-case-latest-one

# Copy env file
cp .env.example .env
# Defaults already match the shared Postgres server (localhost:5433, mva_pipeline db,
# this project's tables live in the `agent2` schema) — no edits needed for local dev

# Run migrations
alembic upgrade head

# Start the server
uvicorn app.main:app --reload --port 8001

# Run tests
python -m pytest tests/ -v
```

Note: this repo's own `docker-compose.yml` is a comment-only pointer to the shared
server above — running `docker-compose up` directly in this directory does nothing.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/profile-runs` | Create profiling run. Accepts an optional `sheet_name` form field, required only when the uploaded XLSX workbook has more than one non-empty sheet — names which sheet to load. |
| GET | `/api/v1/profile-runs/{id}` | Run summary |
| GET | `/api/v1/profile-runs/{id}/result` | Full result |
| GET | `/api/v1/profile-runs/{id}/columns` | Column profiles |
| GET | `/api/v1/profile-runs/{id}/quality` | Quality assessments |
| GET | `/api/v1/profile-runs/{id}/readiness` | AI readiness |
| GET | `/api/v1/profile-runs/{id}/hierarchy` | Hierarchy chain |
| GET | `/api/v1/profile-runs/{id}/rule-evaluations` | Business rule evaluation results (per-rule pass/fail counts and score, persisted for every run) |
| GET | `/api/v1/profile-runs/{id}/charts` | Chart specs |
| POST | `/api/v1/profile-runs/{id}/charts/{cid}/drill-down` | Drill-down |
| GET | `/api/v1/rule-suggestions` | List AI-proposed rule suggestions (optional `?status=` filter) |
| GET | `/api/v1/rule-suggestions/{id}` | Get a single suggestion |
| POST | `/api/v1/rule-suggestions/{id}/approve` | Approve a suggestion — materializes it into an active rule definition that future runs in the same primary domain will evaluate |
| POST | `/api/v1/rule-suggestions/{id}/reject` | Reject a suggestion — no rule definition is created |

## Example Usage

```bash
curl -X POST http://localhost:8001/api/v1/profile-runs \
  -F "file=@payments.csv" \
  -F "primary_domain=Payments" \
  -F 'schema_metadata={"columns":[{"column_name":"amount","description":"Payment amount","mandatory":true}]}'
```

## Supported Domains

| Primary | Secondary Domains |
|---------|-------------------|
| Payments | Authorization, Clearing, Settlement, Fraud |
| Customer | CRM, Customer Satisfaction, Loyalty |
| HR | Employee, Payroll, Recruitment |
| Finance | Revenue, P&L, Forecasting |

## Adding a New Domain

1. Create `config/domains/insurance.yaml` following the existing structure
2. Define secondary domains with keywords and semantic roles
3. Add hierarchy templates
4. Add chart templates
5. Add business rules

**No Python code changes required.** The engine loads configuration dynamically.

## Configuration

All domain-specific behavior is in `config/` YAML files:

- `config/domains/*.yaml` — domain definitions, secondary domains, templates, rules
- `config/quality_weights.yaml` — quality dimension weights (actively used by `calculate_overall_score`)
- `config/readiness_weights.yaml` — documents the intended AI readiness weight profiles, but **`ReadinessEngine` does not currently load this file** — its weights are hardcoded directly in `app/services/readiness/readiness_engine.py`. Editing this YAML has no effect today.
- `config/hierarchy_thresholds.yaml` — FD validation thresholds
- `config/chart_policy.yaml` — chart generation policy
- `config/application.yaml` — global thresholds

## Key Design Principles

### Deterministic Before LLM
- Physical types: Pandas + parse ratios (never LLM)
- Statistics: NumPy/Pandas (never LLM)
- Identifier detection: cardinality analysis (never LLM)
- Rule enforcement: typed engine (never LLM)
- FD validation: groupby aggregation (never LLM)

### LLM Only For Semantic Reasoning
- Confirm/override semantic types
- Classify ambiguous secondary domains
- Propose business rule candidates
- Generate recommendation text

### Raw Data Lifecycle
- Uploaded file → temp directory (UUID-scoped)
- Loaded into DataFrame transiently
- Processed through pipeline
- Temp file deleted on success AND failure
- Only derived metadata persisted to PostgreSQL
- No raw rows in database

## Data Quality Dimensions

| Dimension | Formula | When Not Assessable |
|-----------|---------|---------------------|
| Completeness | 1 - null_count/total for mandatory cols | No mandatory columns defined |
| Uniqueness | 1 - dupes/total for expected-unique cols | No expected-unique columns |
| Validity | pass/checked for range/allowed rules | No validity rules configured |
| Conformity | pass/checked for regex rules | No conformity rules |
| Consistency | 1 - contradictions/checked | No cross-field rules |
| Business Rules | pass/total across all active rules | No rules evaluated |
| Timeliness | Requires SLA config | Always (v1) |
| Integrity | Requires reference data | Always (v1) |
| Accuracy | Requires trusted reference | Always (v1) |
| Semantic Quality | Weighted avg of confidences | No SI results |

**Overall score** = `Σ(weight × score) / Σ(weight)` for assessed dimensions only.

## AI Readiness

All three assessments reuse the same quality evidence with different weight profiles:

- **Analytics**: completeness, dimensions, metrics, grain, temporal fields
- **ML**: completeness, feature coverage, identifier contamination, row count
- **LLM**: description coverage, semantic quality, schema clarity

Thresholds: ≥80 ready, ≥60 partially_ready, <60 not_ready

## AI Rule Suggestions

After profiling, the pipeline asks the LLM to propose up to 5 candidate business rules based on that run's column profiles (null ratios, distinct counts, sample values) — e.g. *"this column is never null, add a non-null rule"* or *"only two values observed, add an allowed-values rule."* Suggestions are always structured, engine-compatible rules (one of the 7 types `RuleEngine` supports), never free text.

- Suggestions are persisted per-run with `status: proposed` and never auto-activate.
- `GET /rule-suggestions` / `GET /rule-suggestions/{id}` — review them (also embedded directly in `GET /profile-runs/{id}/result` under `rule_suggestions`, so no separate call is required to see their IDs).
- `POST /rule-suggestions/{id}/approve` — materializes the suggestion into an active `rule_definitions` row, scoped to that run's primary domain.
- From that point on, **every future upload in the same primary domain is automatically checked against it** — no code change, no re-approval. Results are persisted per-run and queryable via `GET /profile-runs/{id}/rule-evaluations`.
- `POST /rule-suggestions/{id}/reject` — no rule gets created.

Generation degrades gracefully: if the LLM call fails or is rate-limited, `rule_suggestions` comes back as an empty list rather than failing the run.

## Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific phase
python -m pytest tests/unit/test_quality.py -v

# With coverage
python -m pytest tests/ --cov=app --cov-report=term-missing
```

## Running Migrations

```bash
# Apply all migrations
alembic upgrade head

# Generate new migration after model changes
alembic revision --autogenerate -m "description"

# Rollback one step
alembic downgrade -1
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | postgresql://... | PostgreSQL connection |
| MAX_UPLOAD_SIZE_MB | 25 | Max file size |
| MAX_DATASET_ROWS | 200000 | Max rows |
| MAX_DATASET_COLUMNS | 200 | Max columns |
| PROCESSING_TIMEOUT_SECONDS | 120 | Pipeline timeout |
| MIN_CUBE_GROUP_SIZE | 5 | Small-group suppression |
| LLM_PROVIDER | local | LLM backend (`local` or `groq`) |
| LLM_API_KEY | | Groq API key |
| LLM_BURST_COOLDOWN_SECONDS | 5 | Delay before the rule-suggestion LLM call, after the schema-intelligence LLM call — Groq's free tier enforces a burst limit tight enough that firing both back-to-back gets rate-limited even with unused quota |
| LOG_LEVEL | INFO | Logging level |

## Known Limitations

- Drill-down cubes not yet persisted to PostgreSQL (in-memory for demo)
- LLM integration requires API key; without it, deterministic fallback is used
- No authentication/authorization in v1
- Background-thread processing (job abstraction ready for async/queue migration)
- XLSX workbooks with multiple non-empty sheets require the `sheet_name` form field to disambiguate which sheet to load; omitting it on a multi-sheet file returns a `MULTIPLE_XLSX_SHEETS` error
- `config/readiness_weights.yaml` documents intended AI-readiness weights but isn't actually loaded by `ReadinessEngine` — the weights are hardcoded in Python instead (see Configuration section above)
- Individual rule-suggestion generation calls are bounded to the first 30 columns and 5 suggestions per run; not every column gets considered on very wide datasets
