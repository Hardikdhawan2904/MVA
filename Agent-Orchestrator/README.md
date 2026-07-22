# Agent Orchestrator

Coordinates the multi-agent data pipeline: uploads a dataset once, sends it through **Agent 1 (Schema Intelligence Layer)** for classification and quality gating, then feeds Agent 1's output into **Agent 2 (MVA Data Profiling Engine)** for deep profiling — and, when the upload is Insurance data with a business question attached, forwards it on to **Agent 3 (Analytics Agent)** to answer that question. Returns all results together in a single response.

Expressed as a LangGraph `StateGraph` (`app/agents/orchestration_agent/`) rather than a chain of early returns — mostly sequential HTTP calls with conditional stop-on-error edges (no LLM calls or tools of its own), plus one best-effort optional stage for Agent 3. Each future agent added to the pipeline becomes another node in the same graph.

## Architecture

```
Upload → Agent 1 (classify + quality gate) → Agent 2 (profile, using Agent 1's classification)
       → Agent 3 (optional — Insurance domain + business_question + CSV only)
       → combined response
```

If Agent 1 rejects the file at its quality gate, the pipeline stops there — Agent 2 (and therefore Agent 3) is never called.

**`primary_domain` is never accepted from the caller.** It's taken directly from Agent 1's `business_domain` classification and forwarded to Agent 2 automatically. If Agent 1's classification isn't one of Agent 2's supported domains (`Finance`, `Payments`, `Customer`, `HR`, `Insurance`), the pipeline stops with a clear `agent1_classification_missing` or domain-mismatch error rather than guessing.

Agent 3 is best-effort and never fails an otherwise-successful pipeline: outside its scope it's cleanly skipped, and if it's unreachable or errors, `agent3.status == "failed"` with a reason while `agent1`/`agent2` still return normally.

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/pipeline/run` | Runs a file through the pipeline. Accepts `file` (required), `sheet_name` (optional — required only for multi-sheet XLSX workbooks), `force_reclassify` (optional boolean, re-runs Agent 1's classification if the filename already exists in its catalog), `business_question` (optional — also drives Agent 3, see below), `target_column` (optional — explicit override for Agent 2's target-column guessing). |
| POST | `/pipeline/ask` | Re-ask Agent 3 a *different* `business_question` against a dataset that already went through `/pipeline/run`, without repeating Agent 1's quality gate or Agent 2's full profiling. See below. |
| GET | `/health` | Health check — also pings Agent 1 and Agent 2 so it's obvious what's actually reachable (Agent 3 isn't pinged here since it's optional/best-effort). |

### Response shape

```json
{
  "agent1": { "...": "Agent 1's full response (raw row data stripped out)" },
  "agent2": { "...": "Agent 2's full profiling result, including rule_suggestions" },
  "agent3": { "...": "Agent 3's answer, or {\"status\": \"skipped\", ...} outside its scope" },
  "primary_domain_used": "Finance"
}
```

`agent1.dataframe_records` (the raw uploaded rows) is deliberately stripped from the response before returning — Agent 1 already persists it, and passing it through balloons responses to tens of MB on large files, which Swagger struggles to render.

### Agent 3 (Analytics Agent)

Runs only when **all** of: `primary_domain == "Insurance"`, the upload is a `.csv`, and `business_question` was supplied. Called over `httpx` — the same shape as the call to Agent 2 — posting the file straight through along with Agent 2's `ml_readiness`/`llm_readiness` scores, their full readiness breakdown (`agent2.readiness_assessments[]` — strengths/blocking_issues/evidence, not just the score, extracted by `_readiness_and_features()` in `app/agents/orchestration_agent/nodes/pipeline.py`), and (if present) its `feature_recommendation.feature_columns`. Agent 3's response — including its `execution_trace`/`execution_summary` — is forwarded back through `agent3_body` untouched. See the [root API reference](../API_REFERENCE.md#agent-3--analytics-agent-optional-third-stage) for the full response shapes, and [`Analytics-Agent/README.md`](../Analytics-Agent/README.md) for what it does internally.

### Re-asking Agent 3 without re-running the whole pipeline

`POST /pipeline/ask` — for asking a follow-up question against a dataset already processed by `/pipeline/run`. Takes `file` (re-upload — required, see why below), `business_question` (the new question), and `run_id` (Agent 2's `run_id`, copied from the earlier `/pipeline/run` response's `agent2.run_id`). It fetches Agent 2's *already-persisted* result with one lightweight `GET` (not a new profiling run) and calls Agent 3 directly — Agent 1's quality gate and Agent 2's full profiling pipeline never re-run.

```json
{"agent3": {"status": "ok", "query": "...", "response": "...", "ml_readiness_score_used": 29.47, "llm_readiness_score_used": 95.77,
            "execution_trace": [ "..." ], "execution_summary": { "...": "..." }},
 "primary_domain_used": "Insurance"}
```

`agent3.status` works exactly like `/pipeline/run` (`ok` / `skipped` / `failed`, never a hard error). Two error cases are specific to this endpoint, since they're caller mistakes rather than "outside Agent 3's scope": `404` if `run_id` isn't known to Agent 2, `502` if Agent 2 is unreachable.

**Why the file still has to be re-uploaded:** neither Agent 1 nor Agent 2 durably stores the raw dataset rows anywhere — Agent 1's dataframe cache is in-memory only (wiped on restart), and Agent 2 deletes its temp-uploaded copy at the end of every run. Agent 3 needs real rows to query via DuckDB, and there's nowhere durable this endpoint can fetch them back from on its own. Only the readiness scores and feature recommendation (which *are* durably persisted) get reused by `run_id`.

## Local Setup

This project shares one virtual environment with the rest of the pipeline — set up once from the **repo root** (see the [root README](../README.md)), not from inside this folder:

```bash
cd ..
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cd Agent-Orchestrator

# Copy env file and adjust if Agent 1 / Agent 2 run on different hosts/ports
cp .env.example .env

..\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8002
```

Agent 1, Agent 2, and (if you want the Insurance Q&A stage) Agent 3 must already be running (see their own READMEs) — the orchestrator only coordinates between them, it has no database or LLM of its own.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| AGENT1_BASE_URL | http://127.0.0.1:8000 | Base URL for Agent 1 |
| AGENT2_BASE_URL | http://127.0.0.1:8001 | Base URL for Agent 2 |
| AGENT2_API_PREFIX | /api/v1 | Agent 2's API prefix |
| REQUEST_TIMEOUT_SECONDS | 120.0 | Timeout for calls to Agent 1/Agent 2 |
| ANALYTICS_AGENT_BASE_URL | http://127.0.0.1:8003 | Base URL for Agent 3 |
| ANALYTICS_AGENT_TIMEOUT_SECONDS | 120.0 | Timeout for the call to Agent 3 |

No API keys or secrets live here — LLM credentials belong to Agent 1, Agent 2, and Agent 3 individually.

## Running Tests

```bash
cd ..
..\venv\Scripts\python.exe -m pytest Agent-Orchestrator/tests/ -v
```

Uses the shared root venv (this project doesn't carry its own `pytest`) — covers the pure-data helpers in `app/agents/orchestration_agent/nodes/pipeline.py`, currently `_readiness_and_features()` (extracting scores *and* full readiness breakdowns from Agent 2's result, including the missing-assessments and `None`-input edge cases).

## Example

```bash
curl -X POST http://localhost:8002/pipeline/run \
  -F "file=@payments.csv"
```

No `primary_domain` field — it's derived automatically from Agent 1's classification.

```bash
# Insurance dataset with a business question — also invokes Agent 3
curl -X POST http://localhost:8002/pipeline/run \
  -F "file=@insurance_variance_data_native.csv" \
  -F "business_question=Show Gross Written Premium for FY2025"
```

```bash
# Follow-up question against the same dataset — only Agent 3 runs
curl -X POST http://localhost:8002/pipeline/ask \
  -F "file=@insurance_variance_data_native.csv" \
  -F "business_question=Why did underwriting result decline in FY2025?" \
  -F "run_id=<agent2.run_id from the /pipeline/run response above>"
```

## Known Limitations

- No retry/circuit-breaker logic on the calls to Agent 1/Agent 2/Agent 3 — a transient failure surfaces immediately as a `502` (Agent 1/2) or `agent3.status == "failed"` (Agent 3) rather than being retried
- No authentication in v1
- Domain auto-derivation canonicalizes known synonyms of Agent 1's broader vocabulary onto Agent 2's 5 supported domains (`_canonicalize_domain()` / `_DOMAIN_SYNONYMS` in `app/agents/orchestration_agent/nodes/pipeline.py` — e.g. `"Human Resources"` → `"HR"`, case variants), but the map is finite: an unrecognized synonym still fails Agent 2's exact-match check. Notably, Agent 1's classification prompt doesn't suggest `"Payments"` or `"Customer"` as domain names at all (see `Schema-Intelligence-Layer/app/prompts/llm_service_prompt.py`), so those two domains rarely get classified into by Agent 1 in the first place — no amount of synonym-mapping here fixes that; it would need a prompt change in Agent 1.
- `POST /pipeline/ask` requires re-uploading the file — there's no durable storage of raw dataset rows anywhere in the pipeline today (see "Why the file still has to be re-uploaded" above). Adding that (e.g. Agent 1 writing bytes to S3/Postgres keyed by `dataset_id`, plus a `dataset_id` column on Agent 2's `profile_runs`) would let a future version drop this requirement.
