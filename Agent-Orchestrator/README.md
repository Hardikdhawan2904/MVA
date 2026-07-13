# Agent Orchestrator

Coordinates the multi-agent data pipeline: uploads a dataset once, sends it through **Agent 1 (Schema Intelligence Layer)** for classification and quality gating, then feeds Agent 1's output into **Agent 2 (MVA Data Profiling Engine)** for deep profiling — and returns both results together in a single response.

Deliberately simple: sequential HTTP calls with conditional stop-on-error edges, expressed as a LangGraph `StateGraph` (`app/agents/orchestration_agent/`) rather than a chain of early returns — no LLM calls, no tools, just a deterministic relay. Each future agent added to the pipeline becomes another node in the same graph.

## Architecture

```
Upload → Agent 1 (classify + quality gate) → Agent 2 (profile, using Agent 1's classification)
       → combined response
```

If Agent 1 rejects the file at its quality gate, the pipeline stops there — Agent 2 is never called.

**`primary_domain` is never accepted from the caller.** It's taken directly from Agent 1's `business_domain` classification and forwarded to Agent 2 automatically. If Agent 1's classification isn't one of Agent 2's four supported domains (`Finance`, `Payments`, `Customer`, `HR`), the pipeline stops with a clear `agent1_classification_missing` or domain-mismatch error rather than guessing.

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/pipeline/run` | Runs a file through the full pipeline. Accepts `file` (required), `sheet_name` (optional — required only for multi-sheet XLSX workbooks), `force_reclassify` (optional boolean, re-runs Agent 1's classification if the filename already exists in its catalog). |
| GET | `/health` | Health check — also pings both downstream agents so it's obvious what's actually reachable. |

### Response shape

```json
{
  "agent1": { "...": "Agent 1's full response (raw row data stripped out)" },
  "agent2": { "...": "Agent 2's full profiling result, including rule_suggestions" },
  "primary_domain_used": "Finance"
}
```

`agent1.dataframe_records` (the raw uploaded rows) is deliberately stripped from the response before returning — Agent 1 already persists it, and passing it through balloons responses to tens of MB on large files, which Swagger struggles to render.

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

Agent 1 and Agent 2 must already be running (see their own READMEs) — the orchestrator only coordinates between them, it has no database or LLM of its own.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| AGENT1_BASE_URL | http://127.0.0.1:8000 | Base URL for Agent 1 |
| AGENT2_BASE_URL | http://127.0.0.1:8001 | Base URL for Agent 2 |
| AGENT2_API_PREFIX | /api/v1 | Agent 2's API prefix |
| REQUEST_TIMEOUT_SECONDS | 120.0 | Timeout for calls to either agent |

No API keys or secrets live here — LLM credentials belong to Agent 1 and Agent 2 individually.

## Example

```bash
curl -X POST http://localhost:8002/pipeline/run \
  -F "file=@payments.csv"
```

No `primary_domain` field — it's derived automatically from Agent 1's classification.

## Known Limitations

- No retry/circuit-breaker logic on the calls to Agent 1/Agent 2 — a transient failure on either surfaces immediately as a `502` rather than being retried
- No authentication in v1
- Domain auto-derivation only forwards Agent 1's classification as-is; it doesn't attempt to map Agent 1's broader vocabulary (14+ possible domain names) onto Agent 2's 4 supported domains beyond an exact string match
