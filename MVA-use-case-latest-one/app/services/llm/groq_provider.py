"""Groq LLM provider — uses Llama 3.3 Versatile 70B via Groq API."""

import json
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.llm.interface import LLMProvider, LLMRequest, LLMResponse

logger = get_logger(__name__)


class GroqProvider:
    """
    LLM provider using Groq's API with Llama 3.3 Versatile 70B.

    Groq uses an OpenAI-compatible API format.
    Endpoint: https://api.groq.com/openai/v1/chat/completions
    Model: llama-3.3-70b-versatile
    """

    def __init__(self, settings: Settings):
        self._model = settings.llm_model or "llama-3.3-70b-versatile"
        self._api_key = settings.llm_api_key
        self._timeout = settings.llm_timeout_seconds
        self._max_retries = settings.llm_max_retries
        self._base_url = "https://api.groq.com/openai/v1"

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a completion request to Groq."""
        if not self._api_key:
            return LLMResponse(
                content="",
                model=self._model,
                success=False,
                error="GROQ_API_KEY not configured",
            )

        model = request.model or self._model
        messages: list[dict[str, str]] = []
        if request.system_message:
            messages.append({"role": "system", "content": request.system_message})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }

        # Groq supports JSON mode
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

                # Parse JSON response
                parsed = None
                try:
                    parsed = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    # Try to extract JSON from markdown code blocks
                    parsed = self._extract_json(content)

                return LLMResponse(
                    content=content,
                    parsed=parsed,
                    model=model,
                    prompt_version=request.system_message[:20] if request.system_message else "",
                    usage_tokens=usage,
                    success=True,
                )

            except httpx.TimeoutException:
                logger.warning("groq_timeout", attempt=attempt, model=model)
                if attempt == self._max_retries:
                    return LLMResponse(
                        content="", model=model, success=False,
                        error="Groq request timed out",
                    )

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                logger.warning("groq_http_error", status=status_code, attempt=attempt)

                # Rate limit — don't retry immediately
                if status_code == 429:
                    import time
                    time.sleep(min(2 ** attempt, 10))

                if attempt == self._max_retries:
                    return LLMResponse(
                        content="", model=model, success=False,
                        error=f"Groq HTTP {status_code}",
                    )

            except Exception as e:
                logger.error("groq_unexpected_error", error=str(e), attempt=attempt)
                if attempt == self._max_retries:
                    return LLMResponse(
                        content="", model=model, success=False,
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
            logger.warning("groq_validation_error", error=str(e)[:200])
            response.error = f"Response validation failed: {str(e)[:200]}"
            return None, response

    def _extract_json(self, content: str) -> dict[str, Any] | None:
        """Try to extract JSON from content that may have markdown wrapping."""
        # Try direct parse first
        content = content.strip()

        # Remove markdown code block markers
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return None
