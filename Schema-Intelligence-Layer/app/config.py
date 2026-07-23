"""
Application configuration using Pydantic Settings.
Reads from environment variables and .env file.
"""

from pathlib import Path

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# Shared config (GROQ_API_KEY, POSTGRES_*, LOG_LEVEL) genuinely identical
# across Agent 1/3/Orchestrator lives in the repo-root .env — loaded here as
# a fallback UNDER this service's own .env via pydantic-settings' multi-file
# env_file support below (local always wins on any key present in both).
_ROOT_ENV = str(Path(__file__).resolve().parent.parent.parent / ".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Groq API configuration
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # Azure OpenAI (optional, tried before Groq in llm_service.py -- see
    # _chat_complete()). Unset by default, so behavior is byte-identical
    # to today until explicitly configured.
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_DEPLOYMENT: str = ""

    # Database configuration
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5433
    POSTGRES_DB: str = "mva_pipeline"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    # Upload configuration
    MAX_UPLOAD_SIZE_MB: int = 100  # Maximum upload size in MB
    SAMPLE_ROWS: int = 5  # Number of sample rows to extract

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = (_ROOT_ENV, ".env")
        env_file_encoding = "utf-8"


# Singleton settings instance
settings = Settings()
