# Schema Intelligence Layer

The **Schema Intelligence Layer** acts as the intelligent quality gateway and schema classifier of the pipeline. It validates incoming datasets against configurable data quality rules, extracts technical schemas, generates column descriptions using LLMs, and stores metadata in a persistent PostgreSQL catalog. 

If the dataset passes quality validation, it is cached in-memory as a Pandas DataFrame, making it immediately available for downstream agents like the **Data Profiling Layer**.

---

## 1. System Architecture & Directory Structure

```
Schema_Intelligence_layer/
├── app/
│   ├── main.py                    # FastAPI application setup and database lifecycle hooks
│   ├── config.py                  # Pydantic-settings configuration (.env loader)
│   ├── models/
│   │   └── schemas.py             # Pydantic data schemas (DatasetMetadata, QualityReport, etc.)
│   ├── routes/
│   │   └── upload.py              # API endpoints for ingestion, catalogs, and registry queries
│   ├── services/
│   │   ├── quality_validator.py   # Config-driven Data Quality Validator (10 modular checks)
│   │   ├── validator.py           # Supported file extensions helper
│   │   ├── metadata_extractor.py  # Column types, names, counts, and sample extraction
│   │   ├── llm_service.py         # Groq LLM client (domain classification and descriptions)
│   │   └── database.py            # PostgreSQL connection, schema initializations, and CRUD queries
│   ├── prompts/
│   │   └── llm_service_prompt.py  # System and user prompts for Groq LLM tasks
│   └── datastore/
│       └── registry.py            # RAM-based DataFrame cache registry
├── config/
│   └── quality_threshold.json     # Configuration file for quality checks, limits, and weights
├── test_data/                     # Directory containing sample Excel and CSV datasets
├── test_local.py                  # Standalone CLI test pipeline script
└── README.md                      # Project documentation

# Dependencies (requirements.txt / requirements-dev.txt) live at the repo root,
# shared with the other two services — see ../README.md
```

### Detailed File Responsibilities

*   **[app/main.py](app/main.py)**: Bootstraps the FastAPI framework application. Sets up global CORS settings, integrates logging, and registers the database lifespans. On application startup, it automatically invokes `init_db()` to ensure PostgreSQL tables and schemas are initialized before incoming HTTP client calls are handled.
*   **[app/config.py](app/config.py)**: Implements settings management using Pydantic Settings. Dynamically parses environment variables from the `.env` configuration file (such as `GROQ_API_KEY`, custom database paths, and API port setups) and enforces type-safety checks across them.
*   **[app/models/schemas.py](app/models/schemas.py)**: Manages Pydantic validation schemas. Declares data models for structural verification of APIs:
    *   `DatasetMetadata`: Represents the schema of the cataloged metadata.
    *   `QualityReport` / `QualitySummary`: Defines the validation response schema, housing scores, passing statuses, and warning lists.
    *   `UploadResponse`: Outlines the API response payload returned upon a successful ingestion, containing metadata, quality details, and full row records.
    *   `DatasetListItem`: Dictates the fields returned when listing datasets.
*   **[app/routes/upload.py](app/routes/upload.py)**: Exposes API endpoints and coordinates the execution flow of the dataset ingestion pipeline:
    *   `/upload-dataset` (POST): Accepts file streams, invokes loading utilities, triggers quality validator gates, queries LLM services on PASS (aborts with `HTTP 422` on FAIL), caches files in-memory, and commits records to PostgreSQL. Handles incremental row appending when files with identical filenames are uploaded. Accepts an optional `force_reclassify` query parameter — when appending to an existing dataset, this re-runs LLM column descriptions and domain classification instead of reusing the original result (useful for recovering a dataset stuck with a failed/fallback classification).
    *   `/datasets` (GET): Queries PostgreSQL to list all cataloged metadata records.
    *   `/datasets/{dataset_id}` (GET): Fetches the full metadata catalog record.
    *   `/datasets/{dataset_id}/dataframe` (GET): Retrieves the loaded DataFrame from the memory registry cache, returning row lists as JSON.
*   **[app/services/quality_validator.py](app/services/quality_validator.py)**: Executes the core weighted scoring and quality checks. It reads configuration parameters from `config/quality_threshold.json` and runs 10 independent checks against the **entire uploaded dataset** — Column Count, Missing Values, Duplicate Columns, Duplicate Rows, Empty Columns, Datatype Consistency, Corrupted Values, Null-Heavy Rows, Cell Length Outliers, and Mixed Formats. All 10 checks are implemented with vectorized pandas operations (no Python-level row loops), so scanning the full dataset is fast regardless of size — earlier versions sampled only the first/last 10 rows to keep this gate fast, which is no longer necessary now that the checks themselves are fast.
*   **[app/services/validator.py](app/services/validator.py)**: Declares the set of supported file extensions (CSV, XLS, XLSX) and a helper to extract a filename's extension. File parsing and the non-empty/well-formed checks now live in the Schema Intelligence Agent's `validate_and_load` LangGraph node (`app/agents/schema_intelligence_agent/nodes/pipeline.py`).
*   **[app/services/metadata_extractor.py](app/services/metadata_extractor.py)**: Extracts technical structural parameters. Calculates row counts, column counts, lists column names, extracts column Pandas data types (mapping them to serializable string names), and slices a clean JSON-safe data sample of the dataset.
*   **[app/services/llm_service.py](app/services/llm_service.py)**: Communicates with the Groq API endpoint using the `llama-3.1-8b-instant` model. Generates natural language explanations for columns based on names and samples, and classifies the dataset's business domain (e.g. Finance, Logistics, HR) with confidence scores and logic reasoning. Column description generation is batched — datasets wider than `MAX_COLUMNS_PER_LLM_CALL` (30) columns are split into multiple smaller LLM calls and merged, since a single prompt covering every column in a very wide dataset can exceed Groq's per-request token limit. Domain classification similarly caps the columns it includes in its prompt to the same limit, using them as a representative sample rather than requiring exhaustive coverage.
*   **[app/services/database.py](app/services/database.py)**: Manages PostgreSQL CRUD query executions. Initializes the database catalog, runs `ADD COLUMN IF NOT EXISTS` schema migrations to support additions (like `quality_score` and `quality_report`), and serializes/deserializes JSON objects for storage.
*   **[app/prompts/llm_service_prompt.py](app/prompts/llm_service_prompt.py)**: Organizes prompt engineering systems and user instructions. Renders variables inside template formats, enforcing JSON output formatting for LLM responses.
*   **[app/datastore/registry.py](app/datastore/registry.py)**: Implements the RAM cache. Exposes a global cache dictionary mapping `dataset_id` to Pandas DataFrame instances. This allows subsequent profiling or analytics layers running in the same process to retrieve datasets immediately using `get_dataframe(dataset_id)` without reading files back from disk.

---

## 2. Ingestion Pipeline & Data Flow

Below is the execution flow diagram of the ingestion pipeline.

### Visual Architecture Flowchart
```
                  ┌───────────────────────────────┐
                  │       1. Upload File          │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ 2. File Ext & Integrity Check │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ 3. Load into Pandas DataFrame │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │ 4. Execute 10 Quality Checks  │
                  │  on the full dataset (vector- │
                  │  ized, quality_validator.py)  │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                                 / \
                                /   \
                               /     \
                              / Score \
                             <  >=75   >
                              \ Passing/
                               \  ?   /
                                \   /
                                 \ /
                                  │
                       ┌──────────┴──────────┐
                    No │                 Yes │
                       ▼                     ▼
        ┌────────────────────────┐  ┌────────────────────────┐
        │     FAIL Decision      │  │     PASS Decision      │
        │  * Abort pipeline      │  │  * Extract metadata &  │
        │  * Return HTTP 422     │  │    column data types   │
        │  * Output QualityReport│  └──────────┬─────────────┘
        └────────────────────────┘             │
                                               ▼
                                    ┌────────────────────────┐
                                    │  Generate Column Info  │
                                    │    via LLM (Groq)      │
                                    └──────────┬─────────────┘
                                               │
                                               ▼
                                    ┌────────────────────────┐
                                    │    Classify Domain     │
                                    │    via LLM (Groq)      │
                                    └──────────┬─────────────┘
                                               │
                                               ▼
                                    ┌────────────────────────┐
                                    │  Store in PostgreSQL   │
                                    │ (Quality Score/Report) │
                                    └──────────┬─────────────┘
                                               │
                                               ▼
                                    ┌────────────────────────┐
                                    │ Cache in RAM Registry  │
                                    └──────────┬─────────────┘
                                               │
                                               ▼
                                    ┌────────────────────────┐
                                    │  Return HTTP 200 OK    │
                                    │    with full metadata  │
                                    └────────────────────────┘
```

---

## 3. Connecting to Downstream Agents (e.g., Data Profiling Layer)

Once the dataset passes validation and completes classification, the data can be consumed by the next agent (e.g., the **Data Profiling Layer**) in one of three ways:

### Pattern A: RAM Registry Hand-off (Monolith / Shared Process Space)
If the Schema Intelligence Layer and Data Profiling Layer are running inside the same application process, the Profiling agent can load the dataset directly from memory:
```python
from app.datastore.registry import get_dataframe

# Retrieve the cached Pandas DataFrame directly from RAM
df = get_dataframe("DS_001")
if df is not None:
    # Run data profiling statistics (mean, distributions, correlations, etc.)
    profile_data(df)
```

### Pattern B: API JSON Ingestion (Decoupled Microservices)
If the agents are run as separate microservices, the Schema Intelligence Layer can push the validated dataset records directly to the Data Profiling Layer over HTTP:

#### 1. Inside Schema Intelligence Layer (Sender):
```python
import httpx
import pandas as pd

async def forward_to_profiler(dataset_id: str, df: pd.DataFrame):
    profiler_url = "http://data-profiling-service/api/profile"
    payload = {
        "dataset_id": dataset_id,
        "records": df.to_dict(orient="records") # Converts DataFrame to list of row dicts
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(profiler_url, json=payload, timeout=60.0)
        return response.json()
```

#### 2. Inside Data Profiling Layer (Receiver):
```python
import pandas as pd
from fastapi import FastAPI, BaseModel
from typing import List, Dict, Any

app = FastAPI()

class ProfileRequest(BaseModel):
    dataset_id: str
    records: List[Dict[str, Any]]

@app.post("/api/profile")
async def receive_dataset(request: ProfileRequest):
    # Reconstruct the Pandas DataFrame from JSON records list
    df = pd.DataFrame(request.records)
    
    # Run your data profiling metrics
    report = run_profiling_logic(df)
    return {"status": "success", "profile": report}
```

### Pattern C: Shared Storage Reference (Large Files)
For large datasets (e.g., >50MB), sending records over JSON creates significant overhead. Instead, store the validated file on a shared volume or cloud bucket (S3, GCS) and pass the path reference:
```json
// POST payload to Data Profiling Layer
{
  "dataset_id": "DS_001",
  "storage_path": "s3://my-bucket/datasets/DS_001.parquet"
}
```
The Profiling Layer can then download and ingest it using `pd.read_parquet(storage_path)`.

---

## 4. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload-dataset` | Ingests a dataset file. Runs quality checks (stops if FAIL). Generates descriptions/domain details, writes to PostgreSQL, caches in registry, and returns validation metadata. Accepts an optional `force_reclassify` query param to re-run LLM classification on append instead of reusing the original result. |
| `GET` | `/datasets` | Returns a list of all cataloged datasets with high-level summaries and quality scores. |
| `GET` | `/datasets/{dataset_id}` | Retrieves the complete cataloged metadata record (including column descriptions and full `quality_report`). |
| `GET` | `/datasets/{dataset_id}/dataframe` | Returns the raw cached dataset rows from the registry as JSON records (supports `limit`). |
| `GET` | `/health` | API health check. |

---

## 5. Local Setup & Usage

### 1. Environment Configuration
Create a `.env` file in the project root directory and add the following configuration:

```env
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=mva_pipeline
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

#### Parameters:
*   `GROQ_API_KEY`: Your private Groq Cloud developer token (obtain one from the [Groq Console](https://console.groq.com/keys)). This token is utilized to run LLM description and classification tasks.
*   `GROQ_MODEL`: The target inference model (defaults to `llama-3.1-8b-instant`).
*   `POSTGRES_HOST` / `POSTGRES_PORT` / `POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD`: Connection details for the PostgreSQL metadata catalog. This service shares a single Postgres server with the other agents in the pipeline — its tables live in the `agent1` schema of the `mva_pipeline` database, kept isolated from other agents' tables by schema, not by a separate database.

### 2. Start PostgreSQL
Postgres for this whole pipeline (all agents) runs from a shared location, not from this repo:
```powershell
cd ..\Shared-Postgres
docker compose up -d
```
This starts one `postgres:16-alpine` container on `localhost:5433` with a persistent volume, serving every agent's tables under its own schema. (Port `5433` is used instead of the default `5432` to avoid clashing with any existing local PostgreSQL install.) Running `docker compose up` from *this* directory does nothing — the local `docker-compose.yml` here is a comment-only pointer to the shared setup.

### 3. Virtual Environment Setup
This project shares one virtual environment with the rest of the pipeline — set up once from the **repo root**, not from inside this folder:
```powershell
cd ..
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cd Schema-Intelligence-Layer
```
See the [root README](../README.md) for why — one dependency set, no version drift between services.

### 4. CLI Testing Tool
Run the offline testing tool directly from the terminal. It prints validation checklists and logs without launching the server:
```powershell
# Test a single file
..\venv\Scripts\python.exe test_local.py test_data/neutral_payments_variance_trxnid_1000.xlsx

# Test all datasets inside the test_data directory
..\venv\Scripts\python.exe test_local.py
```

### 5. Running the API Server
Launch the FastAPI development server:
```powershell
..\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000 --reload
```
Access the interactive OpenAPI swagger documentation at `http://127.0.0.1:8000/docs`.

---

## 6. Example Ingestion API Outputs

### A. Data Quality Validation Failure (HTTP 422 Payload)
If the uploaded dataset fails to satisfy the configured quality score threshold (e.g. has high null cell counts, duplicate rows, empty columns, or corrupted placeholder symbols), the execution halts immediately and returns the quality report:

```json
{
  "dataset_score": 70.54,
  "passing_score": 75.0,
  "decision": "FAIL",
  "summary": {
    "rows_analyzed": 10,
    "columns": 5
  },
  "checks": {
    "column_count": 15.0,
    "missing_values": 6.8,
    "duplicate_columns": 5.0,
    "duplicate_rows": 4.0,
    "empty_columns": 6.0,
    "datatype_consistency": 15.0,
    "corrupted_values": 5.88,
    "null_heavy_rows": 5.0,
    "cell_length_outliers": 5.0,
    "mixed_formats": 2.86
  },
  "warnings": [
    "Missing values density (66.0%) exceeds threshold of 20.0%.",
    "Duplicate rows (20.0%) exceeds threshold of 10.0%.",
    "Entirely empty columns (40.0%) exceeds threshold of 10.0%.",
    "Corrupted placeholders (41.2%) exceeds threshold of 5.0%.",
    "Null-heavy rows (50.0%) exceeds threshold of 10.0%.",
    "Casing format inconsistency (42.9%) exceeds threshold of 10.0%."
  ]
}
```

### B. Ingestion Success (HTTP 200 OK Response Payload)
If the quality score meets or exceeds the required threshold, the metadata is registered in the DB catalog, Cached in memory, and returned to the client:

```json
{
  "dataset_id": "DS_001",
  "dataset_name": "neutral_payments_variance_trxnid_1000.xlsx",
  "business_domain": "Finance",
  "sub_domain": "Transaction Analysis",
  "dataset_summary": "Contains transaction records, including payment methods, settlement data, and risk assessment metrics for a financial services platform.",
  "row_count": 1000,
  "column_count": 25,
  "column_descriptions": {
    "trx_nid": "Unique transaction identification code",
    "txn_amount_usd": "Transaction volume transaction amount in USD",
    "chargeback_risk_score": "Estimated score indicating transaction chargeback risk"
  },
  "status": "Completed",
  "quality_report": {
    "dataset_score": 99.23,
    "passing_score": 75.0,
    "decision": "PASS",
    "summary": {
      "rows_analyzed": 20,
      "columns": 25
    },
    "checks": {
      "column_count": 15.0,
      "missing_values": 20.0,
      "duplicate_columns": 5.0,
      "duplicate_rows": 5.0,
      "empty_columns": 10.0,
      "datatype_consistency": 15.0,
      "corrupted_values": 10.0,
      "null_heavy_rows": 10.0,
      "cell_length_outliers": 5.0,
      "mixed_formats": 4.23
    },
    "warnings": [
      "Casing format inconsistency (10.3%) exceeds threshold of 10.0%."
    ]
  },
  "dataframe_records": [
    {
      "trx_nid": "TRXN_00001",
      "txn_date": "2024-01-14",
      "txn_amount_usd": 83.35,
      "chargeback_risk_score": 0.139
    }
  ]
}
```

