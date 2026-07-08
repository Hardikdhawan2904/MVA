"""Chart template loader — loads domain-specific and profiling chart templates."""

from typing import Any

from app.repositories.configuration_repository import ConfigurationRepository


class ChartTemplateLoader:
    """Loads chart templates from domain and chart policy configuration."""

    def __init__(self, config_repo: ConfigurationRepository):
        self._config_repo = config_repo

    def load_domain_templates(self, primary_domain: str) -> list[dict[str, Any]]:
        """Load chart templates for a specific domain."""
        domain_config = self._config_repo.get_domain_config(primary_domain)
        return domain_config.get("chart_templates", [])

    def load_profiling_templates(self) -> list[dict[str, Any]]:
        """Load domain-agnostic profiling chart templates."""
        policy = self._config_repo.get_chart_policy()
        return policy.get("profiling_chart_templates", [])

    def load_all(self, primary_domain: str) -> dict[str, list[dict[str, Any]]]:
        """Load all templates grouped by source."""
        return {
            "domain": self.load_domain_templates(primary_domain),
            "profiling": self.load_profiling_templates(),
        }
