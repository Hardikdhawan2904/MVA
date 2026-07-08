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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
