"""LLM provider factory — creates the appropriate provider based on settings."""

from app.core.config import Settings
from app.services.llm.interface import LLMProvider
from app.services.llm.provider import MockLLMProvider, OpenAICompatibleProvider
from app.services.llm.groq_provider import GroqProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    """
    Create the appropriate LLM provider based on configuration.

    Providers:
    - groq: Groq API with Llama 3.3 70B Versatile (default)
    - openai: OpenAI-compatible API
    - mock: Mock provider for testing (no API calls)
    """
    provider_type = settings.llm_provider.lower()

    if provider_type == "groq":
        return GroqProvider(settings)
    elif provider_type == "openai":
        return OpenAICompatibleProvider(settings)
    elif provider_type == "mock":
        return MockLLMProvider()
    else:
        # Default to Groq
        return GroqProvider(settings)
