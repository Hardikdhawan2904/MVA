"""LLM provider factory — creates the appropriate provider based on settings."""

from app.core.config import Settings
from app.services.llm.azure_provider import AzureOpenAIProvider
from app.services.llm.chained_provider import ChainedLLMProvider
from app.services.llm.interface import LLMProvider
from app.services.llm.provider import MockLLMProvider, OpenAICompatibleProvider
from app.services.llm.groq_provider import GroqProvider


def _wrap_with_azure_if_configured(settings: Settings, base_provider: LLMProvider) -> LLMProvider:
    """When Azure OpenAI is configured, try it before base_provider --
    otherwise return base_provider unchanged. Applies to every
    provider_type below (not just "groq") so switching llm_provider
    doesn't silently drop the Azure-first behavior."""
    if settings.azure_openai_api_key and settings.azure_openai_endpoint and settings.azure_openai_deployment:
        return ChainedLLMProvider([AzureOpenAIProvider(settings), base_provider])
    return base_provider


def create_llm_provider(settings: Settings) -> LLMProvider:
    """
    Create the appropriate LLM provider based on configuration.

    Providers:
    - groq: Groq API with Llama 3.3 70B Versatile (default)
    - openai: OpenAI-compatible API
    - mock: Mock provider for testing (no API calls)

    When AZURE_OPENAI_API_KEY/ENDPOINT/DEPLOYMENT are all set, the chosen
    provider above is tried *after* Azure OpenAI, not replaced by it --
    fully backward compatible, zero behavior change when Azure isn't
    configured. Never applies to "mock" (tests expect deterministic,
    network-free responses).
    """
    provider_type = settings.llm_provider.lower()

    if provider_type == "mock":
        return MockLLMProvider()
    elif provider_type == "openai":
        return _wrap_with_azure_if_configured(settings, OpenAICompatibleProvider(settings))
    else:
        # "groq" and any unrecognized value both default to Groq, as before.
        return _wrap_with_azure_if_configured(settings, GroqProvider(settings))
