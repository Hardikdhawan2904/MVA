"""External Schema Intelligence provider — placeholder for future remote service."""

from app.services.schema_intelligence.models import (
    ColumnAnalysisInput,
    ColumnAnalysisResult,
    DomainContext,
    SchemaIntelligenceResult,
)
from app.core.enums import SchemaIntelligenceDecision


class ExternalSchemaIntelligenceProvider:
    """
    Placeholder for a remote Schema Intelligence service.

    Would call an external API endpoint in production.
    Currently falls back to accepting candidates.
    """

    def __init__(self, base_url: str = "", api_key: str = ""):
        self._base_url = base_url
        self._api_key = api_key

    def analyze(
        self,
        columns: list[ColumnAnalysisInput],
        domain_context: DomainContext,
    ) -> SchemaIntelligenceResult:
        """Remote analysis — not implemented, uses fallback."""
        results = []
        for col in columns:
            results.append(ColumnAnalysisResult(
                column_name=col.column_name,
                decision=SchemaIntelligenceDecision.UNRESOLVED,
                confirmed_semantic_type=col.candidate_semantic_type,
                confirmed_column_role=col.candidate_column_role,
                confidence=col.candidate_confidence,
                reasoning="external_provider_not_configured",
            ))

        return SchemaIntelligenceResult(
            column_results=results,
            success=True,
            fallback_used=True,
            error="External provider not configured",
        )
