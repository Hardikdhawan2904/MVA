"""LLM provider implementation — wraps an HTTP-based LLM API."""

import json
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.llm.interface import LLMProvider, LLMRequest, LLMResponse

logger = get_logger(__name__)


class OpenAICompatibleProvider:
    """LLM provider that speaks the OpenAI-compatible API format."""

    def __init__(self, settings: Settings):
        self._model = settings.llm_model
        self._api_key = settings.llm_api_key
        self._timeout = settings.llm_timeout_seconds
        self._max_retries = settings.llm_max_retries
        self._base_url = "https://api.openai.com/v1"

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request."""
        model = request.model or self._model
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
                        "Authorization": f"Bearer {self._api_key}",
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
                logger.warning("llm_timeout", attempt=attempt, model=model)
                if attempt == self._max_retries:
                    return LLMResponse(
                        content="",
                        model=model,
                        success=False,
                        error="LLM request timed out",
                    )
            except httpx.HTTPStatusError as e:
                logger.warning("llm_http_error", status=e.response.status_code, attempt=attempt)
                if attempt == self._max_retries:
                    return LLMResponse(
                        content="",
                        model=model,
                        success=False,
                        error=f"HTTP {e.response.status_code}",
                    )
            except Exception as e:
                logger.error("llm_unexpected_error", error=str(e), attempt=attempt)
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
        # Force JSON response format
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
            logger.warning("llm_validation_error", error=str(e))
            response.error = f"Response validation failed: {str(e)}"
            return None, response


class MockLLMProvider:
    """Mock LLM provider for testing and demo without API keys."""

    def __init__(self):
        self._call_count = 0
        self._responses: list[LLMResponse] = []

    def set_responses(self, responses: list[LLMResponse]) -> None:
        """Pre-configure responses for testing."""
        self._responses = responses
        self._call_count = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return pre-configured or default mock response."""
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp

        self._call_count += 1
        return LLMResponse(
            content="{}",
            parsed={},
            model="mock",
            prompt_version="mock-v1",
            success=True,
        )

    def complete_structured(
        self, request: LLMRequest, response_model: type[BaseModel]
    ) -> tuple[BaseModel | None, LLMResponse]:
        """Return mock structured response."""
        response = self.complete(request)
        if not response.success or not response.parsed:
            return None, response

        try:
            parsed = response_model.model_validate(response.parsed)
            return parsed, response
        except ValidationError:
            return None, response
