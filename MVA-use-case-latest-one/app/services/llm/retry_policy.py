"""Retry policy configuration for LLM calls."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Configurable retry policy for LLM operations."""
    max_retries: int = 2
    timeout_seconds: int = 30
    retry_on_validation_failure: bool = False  # Never retry validation failures indefinitely

    def should_retry(self, attempt: int, is_validation_error: bool) -> bool:
        """Determine whether to retry based on attempt number and error type."""
        if is_validation_error and not self.retry_on_validation_failure:
            return False
        return attempt < self.max_retries
