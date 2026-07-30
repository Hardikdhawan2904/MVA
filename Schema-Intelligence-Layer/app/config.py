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

    # Database configuration
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5433
    POSTGRES_DB: str = "mva_pipeline"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"

    # Upload configuration
    MAX_UPLOAD_SIZE_MB: int = 100  # Maximum upload size in MB
    SAMPLE_ROWS: int = 5  # Number of sample rows to extract
    # Same limits/names as Agent 2's MAX_DATASET_ROWS/MAX_DATASET_COLUMNS —
    # MAX_UPLOAD_SIZE_MB alone doesn't bound row/column count: a highly
    # compressible CSV well under the byte cap can still expand into
    # millions of pandas rows, all of which get serialized into the
    # response's dataframe_records field.
    MAX_DATASET_ROWS: int = 200_000
    MAX_DATASET_COLUMNS: int = 200

    # CORS — comma-separated origin list. Previously hardcoded to "*" with
    # allow_credentials=True, which Starlette's CORSMiddleware handles by
    # echoing the request's Origin header back verbatim (required, since
    # "*" and credentials can't legally combine per the CORS spec) —
    # effectively permitting credentialed cross-origin requests from any
    # site. Defaults to the same local dev frontend origins
    # Agent-Orchestrator's own CORS config already allows; widen via env
    # for a real deployed frontend.
    CORS_ALLOWED_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ALLOWED_ORIGINS.split(",") if origin.strip()]

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = (_ROOT_ENV, ".env")
        env_file_encoding = "utf-8"


# Singleton settings instance
settings = Settings()
