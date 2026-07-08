"""LLM-based classification fallback — used when deterministic rules have low confidence."""

from typing import Any

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.services.llm.interface import LLMProvider, LLMRequest

logger = get_logger(__name__)


# ---- Structured output models for LLM responses ----

class TypeClassificationResult(BaseModel):
    """LLM response for type classification."""
    refined_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class RoleClassificationResult(BaseModel):
    """LLM response for semantic role classification."""
    column_role: str
    semantic_type: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class CategoryClassificationResult(BaseModel):
    """LLM response for data category classification."""
    primary_category: str
    secondary_categories: list[str] = []
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


class BatchClassificationResponse(BaseModel):
    """Batch LLM response for multiple columns."""
    results: list[dict[str, Any]]


# ---- Prompt templates ----

_TYPE_SYSTEM = """You are a data type classification system. Given a column's name, sample values, 
and statistics, determine the most specific data type.

Rules:
- You MUST select from the allowed types list ONLY.
- Return confidence between 0.0 and 1.0.
- Respond ONLY with valid JSON. No extra text."""

_TYPE_PROMPT = """Classify the data type for this column:

Column name: {column_name}
Sample values: {sample_values}
Statistics: null_ratio={null_ratio}, distinct_count={distinct_count}, cardinality_ratio={cardinality_ratio}
Dominant patterns: {patterns}
Numeric parse ratio: {numeric_ratio}
Datetime parse ratio: {datetime_ratio}

Allowed types: {allowed_types}

Return JSON:
{{
  "refined_type": "type_name",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""

_ROLE_SYSTEM = """You are a semantic column role classification system. Given column metadata,
determine what role this column plays in the dataset.

Rules:
- You MUST select from the allowed roles list ONLY.
- Also provide a more specific semantic_type (e.g., "monetary_amount", "creation_date").
- Return confidence between 0.0 and 1.0.
- Respond ONLY with valid JSON. No extra text."""

_ROLE_PROMPT = """Classify the semantic role for this column:

Column name: {column_name}
Refined data type: {refined_type}
Sample values: {sample_values}
Statistics: cardinality_ratio={cardinality_ratio}, distinct_count={distinct_count}, null_ratio={null_ratio}
Min value: {min_value}, Max value: {max_value}
Primary domain: {primary_domain}

Allowed roles: {allowed_roles}

Return JSON:
{{
  "column_role": "role_name",
  "semantic_type": "specific_type_or_null",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""

_CATEGORY_SYSTEM = """You are a data category classification system. Given column metadata,
classify it into one or more business data categories.

Rules:
- primary_category MUST be from the allowed list.
- secondary_categories must also be from the allowed list (max 3).
- Return confidence between 0.0 and 1.0.
- Respond ONLY with valid JSON. No extra text."""

_CATEGORY_PROMPT = """Classify the data category for this column:

Column name: {column_name}
Semantic type: {semantic_type}
Column role: {column_role}
Refined data type: {refined_type}
Sample values: {sample_values}
Primary domain: {primary_domain}

Allowed categories: {allowed_categories}

Return JSON:
{{
  "primary_category": "category_name",
  "secondary_categories": ["cat1", "cat2"],
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}"""


class LLMClassifier:
    """
    LLM-based fallback classifier for ambiguous columns.

    Used when deterministic YAML rules produce confidence below the threshold.
    Constrains LLM outputs to configured allowed values.
    """

    def __init__(self, llm_provider: LLMProvider, llm_config: dict[str, Any] | None = None):
        self._llm = llm_provider
        self._config = llm_config or {}
        self._temperature = self._config.get("temperature", 0.1)
        self._max_tokens = self._config.get("max_tokens", 2000)

    def classify_type(
        self,
        column_name: str,
        sample_values: list[Any],
        statistics: dict[str, Any],
        allowed_types: list[str],
    ) -> TypeClassificationResult | None:
        """Ask LLM to classify the column data type."""
        prompt = _TYPE_PROMPT.format(
            column_name=column_name,
            sample_values=str(sample_values[:8]),
            null_ratio=statistics.get("null_ratio", 0),
            distinct_count=statistics.get("distinct_count", 0),
            cardinality_ratio=statistics.get("cardinality_ratio", 0),
            patterns=statistics.get("dominant_patterns", []),
            numeric_ratio=statistics.get("numeric_parse_ratio", 0),
            datetime_ratio=statistics.get("datetime_parse_ratio", 0),
            allowed_types=", ".join(allowed_types),
        )

        request = LLMRequest(
            prompt=prompt,
            system_message=_TYPE_SYSTEM,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        parsed, response = self._llm.complete_structured(request, TypeClassificationResult)
        if parsed is None:
            logger.warning("llm_type_classification_failed", column=column_name, error=response.error)
            return None

        # Validate output is in allowed list
        if parsed.refined_type not in allowed_types:
            logger.warning(
                "llm_type_invalid_output",
                column=column_name,
                output=parsed.refined_type,
                allowed=allowed_types,
            )
            return None

        return parsed

    def classify_role(
        self,
        column_name: str,
        refined_type: str,
        sample_values: list[Any],
        statistics: dict[str, Any],
        primary_domain: str,
        allowed_roles: list[str],
    ) -> RoleClassificationResult | None:
        """Ask LLM to classify the column semantic role."""
        prompt = _ROLE_PROMPT.format(
            column_name=column_name,
            refined_type=refined_type,
            sample_values=str(sample_values[:8]),
            cardinality_ratio=statistics.get("cardinality_ratio", 0),
            distinct_count=statistics.get("distinct_count", 0),
            null_ratio=statistics.get("null_ratio", 0),
            min_value=statistics.get("min_value"),
            max_value=statistics.get("max_value"),
            primary_domain=primary_domain,
            allowed_roles=", ".join(allowed_roles),
        )

        request = LLMRequest(
            prompt=prompt,
            system_message=_ROLE_SYSTEM,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        parsed, response = self._llm.complete_structured(request, RoleClassificationResult)
        if parsed is None:
            logger.warning("llm_role_classification_failed", column=column_name, error=response.error)
            return None

        if parsed.column_role not in allowed_roles:
            logger.warning(
                "llm_role_invalid_output",
                column=column_name,
                output=parsed.column_role,
            )
            return None

        return parsed

    def classify_category(
        self,
        column_name: str,
        semantic_type: str | None,
        column_role: str,
        refined_type: str,
        sample_values: list[Any],
        primary_domain: str,
        allowed_categories: list[str],
    ) -> CategoryClassificationResult | None:
        """Ask LLM to classify the data category."""
        prompt = _CATEGORY_PROMPT.format(
            column_name=column_name,
            semantic_type=semantic_type or "unknown",
            column_role=column_role,
            refined_type=refined_type,
            sample_values=str(sample_values[:5]),
            primary_domain=primary_domain,
            allowed_categories=", ".join(allowed_categories),
        )

        request = LLMRequest(
            prompt=prompt,
            system_message=_CATEGORY_SYSTEM,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        parsed, response = self._llm.complete_structured(request, CategoryClassificationResult)
        if parsed is None:
            logger.warning("llm_category_classification_failed", column=column_name, error=response.error)
            return None

        # Validate primary category
        if parsed.primary_category not in allowed_categories:
            logger.warning(
                "llm_category_invalid_output",
                column=column_name,
                output=parsed.primary_category,
            )
            return None

        # Filter secondary to allowed only
        parsed.secondary_categories = [
            c for c in parsed.secondary_categories if c in allowed_categories
        ]

        return parsed
