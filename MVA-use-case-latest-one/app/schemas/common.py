"""Common Pydantic schemas used across the API."""

from pydantic import BaseModel
from typing import Any


class ErrorDetail(BaseModel):
    """Structured error response."""
    code: str
    message: str
    details: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    """Top-level error envelope."""
    error: ErrorDetail
