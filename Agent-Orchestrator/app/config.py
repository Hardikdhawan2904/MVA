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
    # stage: only runs for Insurance-domain CSVs with a business_question.
    ANALYTICS_AGENT_BASE_URL: str = "http://127.0.0.1:8003"
    ANALYTICS_AGENT_TIMEOUT_SECONDS: float = 120.0

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = (_ROOT_ENV, ".env")
        env_file_encoding = "utf-8"
        # The orchestrator has no DB/LLM key of its own, so the shared root
        # .env's GROQ_API_KEY/POSTGRES_* keys are "extra" from its point of
        # view — ignore rather than reject them.
        extra = "ignore"


settings = Settings()
