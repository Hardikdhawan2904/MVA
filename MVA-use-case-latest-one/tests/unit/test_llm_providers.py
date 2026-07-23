"""Tests for AzureOpenAIProvider, ChainedLLMProvider, and the factory's
Azure-first wiring — Azure OpenAI as a second LLM provider, tried before
whichever base provider llm_provider selects, never replacing it.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.services.llm.azure_provider import AzureOpenAIProvider
from app.services.llm.chained_provider import ChainedLLMProvider
from app.services.llm.factory import create_llm_provider
from app.services.llm.groq_provider import GroqProvider
from app.services.llm.interface import LLMRequest, LLMResponse
from app.services.llm.provider import MockLLMProvider


def _settings(**overrides) -> Settings:
    return Settings(DATABASE_URL="postgresql://x:x@localhost/test", **overrides)


def _mock_httpx_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 10}},
        request=httpx.Request("POST", "https://example.openai.azure.com/openai/v1/chat/completions"),
    )


# ── AzureOpenAIProvider ──────────────────────────────────────────────────────

class TestAzureOpenAIProvider:
    def test_complete_success(self):
        settings = _settings(
            AZURE_OPENAI_API_KEY="fake-key",
            AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com",
            AZURE_OPENAI_DEPLOYMENT="gpt-4o",
        )
        provider = AzureOpenAIProvider(settings)
        with patch("httpx.post", return_value=_mock_httpx_response("hello")) as mock_post:
            response = provider.complete(LLMRequest(prompt="hi"))

        assert response.success is True
        assert response.content == "hello"
        assert response.model == "gpt-4o"
        assert mock_post.call_args[1]["headers"]["api-key"] == "fake-key"
        assert "openai/v1/chat/completions" in mock_post.call_args[0][0]

    def test_complete_http_error_returns_failure_not_raise(self):
        settings = _settings(
            AZURE_OPENAI_API_KEY="fake-key",
            AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com",
            AZURE_OPENAI_DEPLOYMENT="gpt-4o",
            LLM_MAX_RETRIES=0,
        )
        provider = AzureOpenAIProvider(settings)
        error_resp = httpx.Response(500, request=httpx.Request("POST", "https://x"))
        with patch("httpx.post", side_effect=httpx.HTTPStatusError("boom", request=error_resp.request, response=error_resp)):
            response = provider.complete(LLMRequest(prompt="hi"))

        assert response.success is False
        assert "500" in response.error


# ── ChainedLLMProvider ───────────────────────────────────────────────────────

class TestChainedLLMProvider:
    def test_returns_first_success(self):
        first = MagicMock()
        first.complete.return_value = LLMResponse(content="from first", success=True)
        second = MagicMock()

        chain = ChainedLLMProvider([first, second])
        response = chain.complete(LLMRequest(prompt="hi"))

        assert response.content == "from first"
        second.complete.assert_not_called()

    def test_falls_through_to_second_when_first_fails(self):
        first = MagicMock()
        first.complete.return_value = LLMResponse(content="", success=False, error="first failed")
        second = MagicMock()
        second.complete.return_value = LLMResponse(content="from second", success=True)

        chain = ChainedLLMProvider([first, second])
        response = chain.complete(LLMRequest(prompt="hi"))

        assert response.content == "from second"
        first.complete.assert_called_once()
        second.complete.assert_called_once()

    def test_returns_last_failure_when_all_fail(self):
        first = MagicMock()
        first.complete.return_value = LLMResponse(content="", success=False, error="first failed")
        second = MagicMock()
        second.complete.return_value = LLMResponse(content="", success=False, error="second failed")

        chain = ChainedLLMProvider([first, second])
        response = chain.complete(LLMRequest(prompt="hi"))

        assert response.success is False
        assert response.error == "second failed"

    def test_complete_structured_same_ordering(self):
        class Schema(BaseModel):
            value: int

        first = MagicMock()
        first.complete_structured.return_value = (None, LLMResponse(content="", success=False, error="bad json"))
        second = MagicMock()
        second.complete_structured.return_value = (Schema(value=1), LLMResponse(content="{}", success=True))

        chain = ChainedLLMProvider([first, second])
        parsed, response = chain.complete_structured(LLMRequest(prompt="hi"), Schema)

        assert parsed.value == 1
        assert response.success is True

    def test_requires_at_least_one_provider(self):
        with pytest.raises(ValueError):
            ChainedLLMProvider([])


# ── factory.create_llm_provider ─────────────────────────────────────────────

class TestFactoryAzureWiring:
    def test_azure_unconfigured_returns_plain_groq_provider_unchanged(self):
        # Explicitly blank -- .env may genuinely have AZURE_OPENAI_* set in
        # this environment, and Settings() picks that up automatically
        # unless overridden here, same as any other pydantic-settings field.
        settings = _settings(
            LLM_PROVIDER="groq", LLM_API_KEY="fake-groq-key",
            AZURE_OPENAI_API_KEY="", AZURE_OPENAI_ENDPOINT="", AZURE_OPENAI_DEPLOYMENT="",
        )
        provider = create_llm_provider(settings)
        assert isinstance(provider, GroqProvider)

    def test_azure_configured_wraps_groq_in_chain(self):
        settings = _settings(
            LLM_PROVIDER="groq", LLM_API_KEY="fake-groq-key",
            AZURE_OPENAI_API_KEY="fake-key",
            AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com",
            AZURE_OPENAI_DEPLOYMENT="gpt-4o",
        )
        provider = create_llm_provider(settings)
        assert isinstance(provider, ChainedLLMProvider)
        assert isinstance(provider._providers[0], AzureOpenAIProvider)
        assert isinstance(provider._providers[1], GroqProvider)

    def test_mock_provider_never_wrapped_even_if_azure_configured(self):
        settings = _settings(
            LLM_PROVIDER="mock",
            AZURE_OPENAI_API_KEY="fake-key",
            AZURE_OPENAI_ENDPOINT="https://example.openai.azure.com",
            AZURE_OPENAI_DEPLOYMENT="gpt-4o",
        )
        provider = create_llm_provider(settings)
        assert isinstance(provider, MockLLMProvider)

    def test_partial_azure_config_does_not_wrap(self):
        """All three fields (key/endpoint/deployment) are required --
        a partially-configured Azure setup falls through to plain Groq,
        not a broken chain that would fail every request."""
        settings = _settings(
            LLM_PROVIDER="groq", LLM_API_KEY="fake-groq-key",
            AZURE_OPENAI_API_KEY="fake-key",  # endpoint/deployment left unset
            AZURE_OPENAI_ENDPOINT="", AZURE_OPENAI_DEPLOYMENT="",
        )
        provider = create_llm_provider(settings)
        assert isinstance(provider, GroqProvider)
