"""Rules loader — reads classification rules from YAML config files."""

from pathlib import Path
from typing import Any

import yaml

from app.core.logging import get_logger

logger = get_logger(__name__)


class ClassificationRulesLoader:
    """
    Loads and caches classification rules from config/rules/ YAML files.

    Provides a unified interface for all classification layers to access
    their rule definitions without hardcoding patterns in Python.
    """

    def __init__(self, config_dir: str | Path = "config/rules"):
        self._config_dir = Path(config_dir)
        self._cache: dict[str, Any] = {}

    def _load(self, filename: str) -> dict[str, Any]:
        """Load and cache a YAML file."""
        if filename in self._cache:
            return self._cache[filename]

        path = self._config_dir / filename
        if not path.exists():
            logger.warning("classification_rules_not_found", file=filename)
            return {}

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        self._cache[filename] = data
        return data

    def get_type_detection_rules(self) -> dict[str, Any]:
        """Load type detection rules (boolean, email, phone, etc.)."""
        return self._load("type_detection_rules.yaml")

    def get_semantic_role_rules(self) -> dict[str, Any]:
        """Load semantic role classification rules (metric, dimension, temporal, etc.)."""
        return self._load("semantic_role_rules.yaml")

    def get_data_category_rules(self) -> dict[str, Any]:
        """Load data category rules (Transaction, Financial, Geographic, etc.)."""
        return self._load("data_category_rules.yaml")

    def get_llm_config(self) -> dict[str, Any]:
        """Load LLM classification fallback configuration."""
        return self._load("llm_classification_config.yaml")

    def get_type_known_values(self, type_name: str) -> set[str]:
        """Get the known values set for a specific type (e.g., currency codes)."""
        rules = self.get_type_detection_rules()
        type_config = rules.get("types", {}).get(type_name, {})
        values = type_config.get("known_values", [])
        return {v.upper() for v in values}

    def get_role_keywords(self, role_name: str) -> list[str]:
        """Get keywords for a specific semantic role."""
        rules = self.get_semantic_role_rules()
        role_config = rules.get("roles", {}).get(role_name, {})
        return role_config.get("keywords", [])

    def get_role_sub_types(self, role_name: str) -> dict[str, list[str]]:
        """Get sub-type keyword mappings for a role."""
        rules = self.get_semantic_role_rules()
        role_config = rules.get("roles", {}).get(role_name, {})
        sub_types = role_config.get("sub_types", {})
        return {name: cfg.get("keywords", []) for name, cfg in sub_types.items()}

    def get_category_keywords(self, category_name: str) -> list[str]:
        """Get keywords for a specific data category."""
        rules = self.get_data_category_rules()
        categories = rules.get("categories", {})
        cat_config = categories.get(category_name, {})
        return cat_config.get("keywords", [])

    def get_all_category_names(self) -> list[str]:
        """Get all configured category names."""
        rules = self.get_data_category_rules()
        return list(rules.get("categories", {}).keys())

    def get_domain_boosts(self, primary_domain: str) -> dict[str, list[str]]:
        """Get category boost config for a primary domain."""
        rules = self.get_data_category_rules()
        boosts = rules.get("domain_boosts", {})
        return boosts.get(primary_domain, {})

    def get_llm_threshold(self, layer: str) -> float:
        """Get the LLM fallback confidence threshold for a classification layer."""
        config = self.get_llm_config()
        layers = config.get("layers", {})
        layer_config = layers.get(layer, {})
        return layer_config.get("confidence_threshold", 0.75)

    def is_llm_enabled(self, layer: str) -> bool:
        """Check if LLM fallback is enabled for a classification layer."""
        config = self.get_llm_config()
        if not config.get("enabled", True):
            return False
        layers = config.get("layers", {})
        layer_config = layers.get(layer, {})
        return layer_config.get("enabled", False)

    def invalidate_cache(self) -> None:
        """Clear all cached rules."""
        self._cache.clear()
