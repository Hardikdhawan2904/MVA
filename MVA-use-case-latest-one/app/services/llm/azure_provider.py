"""LLM provider implementation for Azure OpenAI — same wire shape as
OpenAICompatibleProvider (app/services/llm/provider.py), two differences:
the base URL is the caller's own Azure resource endpoint (not
api.openai.com), and Azure's auth scheme is an `api-key` header, not
`Authorization: Bearer`. `model` is a deployment name here, not a model
name — Azure OpenAI requires deploying a specific model under a
caller-chosen name before it's callable at all; that deployment name is
what goes in the request body's "model" field.
"""

import json
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.llm.interface import LLMRequest, LLMResponse

logger = get_logger(__name__)


class AzureOpenAIProvider:
    """LLM provider that speaks Azure OpenAI's `/openai/v1/chat/completions`
    surface. Confirmed live against a real Azure OpenAI resource: no
    `api-version` query param needed on this endpoint shape (that's only
    required on the older `/openai/deployments/{name}/...` surface)."""

    def __init__(self, settings: Settings):
        self._deployment = settings.azure_openai_deployment
        self._api_key = settings.azure_openai_api_key
        self._timeout = settings.llm_timeout_seconds
        self._max_retries = settings.llm_max_retries
        self._base_url = f"{settings.azure_openai_endpoint.rstrip('/')}/openai/v1"

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request."""
        model = request.model or self._deployment
        messages = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        if request.response_schema:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(self._max_retries + 1):
            try:
                response = httpx.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "api-key": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {}).get("total_tokens", 0)

                # Try to parse as JSON
                parsed = None
                try:
                    parsed = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    pass

                return LLMResponse(
                    content=content,
                    parsed=parsed,
                    model=model,
                    usage_tokens=usage,
                    success=True,
                )
            except httpx.TimeoutException:
                logger.warning("azure_llm_timeout", attempt=attempt, model=model)
                if attempt == self._max_retries:
                    return LLMResponse(
                        content="",
                        model=model,
                        success=False,
                        error="Azure OpenAI request timed out",
                    )
            except httpx.HTTPStatusError as e:
                logger.warning("azure_llm_http_error", status=e.response.status_code, attempt=attempt)
                if attempt == self._max_retries:
                    return LLMResponse(
                        content="",
                        model=model,
                        success=False,
                        error=f"HTTP {e.response.status_code}",
                    )
            except Exception as e:
                logger.error("azure_llm_unexpected_error", error=str(e), attempt=attempt)
                if attempt == self._max_retries:
                    return LLMResponse(
                        content="",
                        model=model,
                        success=False,
                        error=str(e),
                    )

        return LLMResponse(content="", model=model, success=False, error="Max retries exceeded")

    def complete_structured(
        self, request: LLMRequest, response_model: type[BaseModel]
    ) -> tuple[BaseModel | None, LLMResponse]:
        """Send request and validate response against a Pydantic model."""
        request_with_schema = LLMRequest(
            prompt=request.prompt,
            system_message=request.system_message,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            response_schema=response_model.model_json_schema(),
        )

        response = self.complete(request_with_schema)
        if not response.success or not response.parsed:
            return None, response

        try:
            parsed_model = response_model.model_validate(response.parsed)
            return parsed_model, response
        except ValidationError as e:
            logger.warning("azure_llm_validation_error", error=str(e))
            response.error = f"Response validation failed: {str(e)}"
            return None, response
