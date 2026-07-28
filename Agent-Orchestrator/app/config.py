"""
Application configuration using Pydantic Settings.
Reads from environment variables and .env file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

# Shared config (GROQ_API_KEY, POSTGRES_*, LOG_LEVEL) genuinely identical
# across Agent 1/3/Orchestrator lives in the repo-root .env — loaded here as
# a fallback UNDER this service's own .env via pydantic-settings' multi-file
# env_file support below (local always wins on any key present in both).
# The orchestrator itself has no DB/LLM of its own, so in practice this only
# gives it LOG_LEVEL.
_ROOT_ENV = str(Path(__file__).resolve().parent.parent.parent / ".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    AGENT1_BASE_URL: str = "http://127.0.0.1:8000"
    AGENT2_BASE_URL: str = "http://127.0.0.1:8001"
    AGENT2_API_PREFIX: str = "/api/v1"

    REQUEST_TIMEOUT_SECONDS: float = 120.0

    # Agent 3 (Analytics Agent) — now a normal FastAPI service (port 8003),
    # same shape as Agent 1/2, called over httpx like the others. Optional
    # stage: runs for any of Agent 2's 5 supported domains (Finance,
    # Payments, Customer, HR, Insurance), gated only on file type (.csv)
    # and a supplied business_question — see Agent3Capabilities in
    # app/agents/orchestration_agent/nodes/pipeline.py.
    ANALYTICS_AGENT_BASE_URL: str = "http://127.0.0.1:8003"
    ANALYTICS_AGENT_TIMEOUT_SECONDS: float = 120.0

    LOG_LEVEL: str = "INFO"

    # Dataset Registry (Stage 0A) — content-fingerprinted Master Dataset
    # storage, shared with Agent 1/2/3 via the same native Postgres instance
    # (see Shared-Postgres/README.md), own schema "orchestrator".
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5433
    POSTGRES_DB: str = "mva_pipeline"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    MASTER_DATASET_STORAGE_DIR: str = r"C:\MVAData\master-datasets"

    class Config:
        env_file = (_ROOT_ENV, ".env")
        env_file_encoding = "utf-8"
        # The orchestrator has no DB/LLM key of its own, so the shared root
        # .env's GROQ_API_KEY/POSTGRES_* keys are "extra" from its point of
        # view — ignore rather than reject them.
        extra = "ignore"


settings = Settings()
