"""
Application configuration using Pydantic Settings.
Reads from environment variables and .env file.
"""

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    AGENT1_BASE_URL: str = "http://127.0.0.1:8000"
    AGENT2_BASE_URL: str = "http://127.0.0.1:8001"
    AGENT2_API_PREFIX: str = "/api/v1"

    REQUEST_TIMEOUT_SECONDS: float = 120.0

    # Agent 3 (Analytics Agent) — vendored at mva/Analytics-Agent, sharing
    # this project's venv, but it's still a CLI tool rather than an HTTP
    # service like Agent 1/2, so it's invoked as a subprocess instead of
    # over httpx. Optional stage: only runs for Insurance-domain CSVs with
    # a business_question. ANALYTICS_AGENT_PYTHON defaults to sys.executable
    # (this process's own interpreter) when left blank, since it now runs
    # in the same shared venv as the orchestrator itself.
    ANALYTICS_AGENT_PATH: str = r"C:\Users\dhawa\mva\Analytics-Agent"
    ANALYTICS_AGENT_PYTHON: str = ""
    ANALYTICS_AGENT_TIMEOUT_SECONDS: float = 120.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
