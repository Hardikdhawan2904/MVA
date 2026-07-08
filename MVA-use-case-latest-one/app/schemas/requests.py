"""API request schemas."""

from typing import Any
from pydantic import BaseModel, Field


class ColumnMetadata(BaseModel):
    """Column metadata from caller."""
    column_name: str
    description: str | None = None
    mandatory: bool = False
    expected_unique: bool = False


class SchemaMetadata(BaseModel):
    """Schema metadata sent with profile run request."""
    columns: list[ColumnMetadata] = []


class RunOptions(BaseModel):
    """Optional run configuration."""
    skip_llm: bool = False
    max_charts: int = 5


class DrillDownRequest(BaseModel):
    """Drill-down request body."""
    selected_path: dict[str, str]


class RuleApprovalRequest(BaseModel):
    """Rule approval request."""
    comment: str | None = None


class RuleRejectionRequest(BaseModel):
    """Rule rejection request."""
    reason: str | None = None
