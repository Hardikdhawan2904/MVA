"""tests/test_llm_service.py — Azure OpenAI first, Groq fallback.

app/services/llm_service.py had no automated test coverage before this
(only test_local.py, a manual CLI script) — added alongside the Azure
OpenAI integration to cover the new _chat_complete()/_call_azure_openai()
functions specifically, not a general retrofit of the whole module.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from app.services.llm_service import _call_azure_openai, _chat_complete


def _mock_httpx_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"content": content}}]},
        request=httpx.Request("POST", "https://example.openai.azure.com/openai/v1/chat/completions"),
    )


def test_azure_unconfigured_returns_none():
    with patch("app.services.llm_service.settings") as mock_settings:
        mock_settings.AZURE_OPENAI_API_KEY = ""
        mock_settings.AZURE_OPENAI_ENDPOINT = ""
        mock_settings.AZURE_OPENAI_DEPLOYMENT = ""
        assert _call_azure_openai([{"role": "user", "content": "hi"}], 0.1, 100) is None


def test_azure_configured_and_succeeds_returns_content():
    with patch("app.services.llm_service.settings") as mock_settings, \
         patch("httpx.post", return_value=_mock_httpx_response("hello from azure")) as mock_post:
        mock_settings.AZURE_OPENAI_API_KEY = "fake-key"
        mock_settings.AZURE_OPENAI_ENDPOINT = "https://example.openai.azure.com"
        mock_settings.AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
        result = _call_azure_openai([{"role": "user", "content": "hi"}], 0.1, 100)

    assert result == "hello from azure"
    assert mock_post.call_args[1]["headers"]["api-key"] == "fake-key"
    assert "openai/v1/chat/completions" in mock_post.call_args[0][0]


def test_azure_configured_but_call_fails_returns_none_not_raises():
    with patch("app.services.llm_service.settings") as mock_settings, \
         patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        mock_settings.AZURE_OPENAI_API_KEY = "fake-key"
        mock_settings.AZURE_OPENAI_ENDPOINT = "https://example.openai.azure.com"
        mock_settings.AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
        assert _call_azure_openai([{"role": "user", "content": "hi"}], 0.1, 100) is None


def test_chat_complete_uses_azure_when_available_never_touches_groq_client():
    fake_groq_client = MagicMock()
    with patch("app.services.llm_service.settings") as mock_settings, \
         patch("httpx.post", return_value=_mock_httpx_response("azure content")):
        mock_settings.AZURE_OPENAI_API_KEY = "fake-key"
        mock_settings.AZURE_OPENAI_ENDPOINT = "https://example.openai.azure.com"
        mock_settings.AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
        content = _chat_complete(fake_groq_client, [{"role": "user", "content": "hi"}], 0.1, 100)

    assert content == "azure content"
    fake_groq_client.chat.completions.create.assert_not_called()


def test_chat_complete_falls_back_to_groq_client_when_azure_unconfigured():
    fake_groq_client = MagicMock()
    fake_groq_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="groq content"))
    ]
    with patch("app.services.llm_service.settings") as mock_settings:
        mock_settings.AZURE_OPENAI_API_KEY = ""
        mock_settings.AZURE_OPENAI_ENDPOINT = ""
        mock_settings.AZURE_OPENAI_DEPLOYMENT = ""
        mock_settings.GROQ_MODEL = "llama-3.1-8b-instant"
        content = _chat_complete(fake_groq_client, [{"role": "user", "content": "hi"}], 0.1, 100)

    assert content == "groq content"
    fake_groq_client.chat.completions.create.assert_called_once()


def test_chat_complete_falls_back_to_groq_client_when_azure_call_fails():
    fake_groq_client = MagicMock()
    fake_groq_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="groq content after azure failure"))
    ]
    with patch("app.services.llm_service.settings") as mock_settings, \
         patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        mock_settings.AZURE_OPENAI_API_KEY = "fake-key"
        mock_settings.AZURE_OPENAI_ENDPOINT = "https://example.openai.azure.com"
        mock_settings.AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
        mock_settings.GROQ_MODEL = "llama-3.1-8b-instant"
        content = _chat_complete(fake_groq_client, [{"role": "user", "content": "hi"}], 0.1, 100)

    assert content == "groq content after azure failure"
    fake_groq_client.chat.completions.create.assert_called_once()
