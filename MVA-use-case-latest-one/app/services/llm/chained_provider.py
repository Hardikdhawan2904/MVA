"""A provider that tries several LLMProvider implementations in order,
returning the first success -- itself satisfies the LLMProvider Protocol,
so anything that already only depends on that Protocol (e.g.
LocalSchemaIntelligenceProvider) needs zero changes to benefit from a
multi-provider chain.
"""

from pydantic import BaseModel

from app.core.logging import get_logger
from app.services.llm.interface import LLMProvider, LLMRequest, LLMResponse

logger = get_logger(__name__)


class ChainedLLMProvider:
    """Tries each provider's complete()/complete_structured() in order,
    returning the first LLMResponse with success=True. If every provider
    fails, returns the last provider's failure response (so the caller's
    existing "success is False -> deterministic fallback" handling sees a
    real error message, not a silent stand-in)."""

    def __init__(self, providers: list[LLMProvider]):
        if not providers:
            raise ValueError("ChainedLLMProvider needs at least one provider")
        self._providers = providers

    def complete(self, request: LLMRequest) -> LLMResponse:
        last_response: LLMResponse | None = None
        for provider in self._providers:
            response = provider.complete(request)
            if response.success:
                return response
            logger.warning(
                "chained_llm_provider_failed",
                provider=type(provider).__name__,
                error=response.error,
            )
            last_response = response
        return last_response

    def complete_structured(
        self, request: LLMRequest, response_model: type[BaseModel]
    ) -> tuple[BaseModel | None, LLMResponse]:
        last_result: tuple[BaseModel | None, LLMResponse] | None = None
        for provider in self._providers:
            parsed, response = provider.complete_structured(request, response_model)
            if response.success and parsed is not None:
                return parsed, response
            logger.warning(
                "chained_llm_provider_structured_failed",
                provider=type(provider).__name__,
                error=response.error,
            )
            last_result = (parsed, response)
        return last_result
