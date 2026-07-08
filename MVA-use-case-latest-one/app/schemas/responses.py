"""API response schemas."""

from typing import Any
from pydantic import BaseModel


class RunCreatedResponse(BaseModel):
    """Response after creating a profile run."""
    run_id: str
    status: str


class RunSummaryResponse(BaseModel):
    """Profile run summary."""
    run_id: str
    status: str
    primary_domain: str
    secondary_domain: dict[str, Any] | None = None
    source_filename: str = ""
    row_count: int | None = None
    column_count: int | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: dict[str, Any] | None = None
