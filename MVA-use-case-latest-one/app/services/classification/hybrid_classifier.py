"""Hybrid classifier — YAML rules first, LLM fallback for low confidence."""

import re
from typing import Any

from app.core.enums import RefinedDataType, ColumnRole, DataCategory
from app.core.logging import get_logger
from app.services.classification.rules_loader import ClassificationRulesLoader
from app.services.classification.llm_classifier import LLMClassifier
from app.services.llm.interface import LLMProvider
from app.services.profiling.column_profiler import ColumnProfileResult

logger = get_logger(__name__)


class HybridTypeResult:
    """Result of hybrid type classification."""

    def __init__(self, refined_type: RefinedDataType, confidence: float,
                 source: str, reasoning: str = ""):
        self.refined_type = refined_type
        self.confidence = confidence
        self.source = source  # "yaml_rules" or "llm_fallback"
        self.reasoning = reasoning


class HybridRoleResult:
    """Result of hybrid role classification."""

    def __init__(self, column_role: ColumnRole, semantic_type: str | None,
                 confidence: float, source: str, reasoning: str = ""):
        self.column_role = column_role
        self.semantic_type = semantic_type
        self.confidence = confidence
        self.source = source
        self.reasoning = reasoning


class HybridCategoryResult:
    """Result of hybrid category classification."""

    def __init__(self, primary_category: DataCategory,
                 secondary_categories: list[DataCategory],
                 confidence: float, source: str, reasoning: str = ""):
        self.primary_category = primary_category
        self.secondary_categories = secondary_categories
        self.confidence = confidence
        self.source = source
        self.reasoning = reasoning


class HybridClassifier:
    """
    Hybrid classification engine.

    Strategy:
    1. Load rules from YAML config files.
    2. Apply deterministic matching (keywords, patterns, statistics).
    3. Compute confidence score for the match.
    4. If confidence >= threshold → return deterministic result.
    5. If confidence < threshold AND LLM is enabled → ask LLM.
    6. If LLM fails or is disabled → return deterministic result anyway.
    """

    def __init__(
        self,
        rules_loader: ClassificationRulesLoader,
        llm_provider: LLMProvider | None = None,
    ):
        self._rules = rules_loader
        self._llm_classifier: LLMClassifier | None = None
        if llm_provider:
            llm_config = rules_loader.get_llm_config()
            self._llm_classifier = LLMClassifier(llm_provider, llm_config)

    # ----------------------------------------------------------------
    # TYPE CLASSIFICATION (Layer 1)
    # ----------------------------------------------------------------

    def classify_type(self, profile: ColumnProfileResult) -> HybridTypeResult:
        """
        Classify column data type using YAML rules + LLM fallback.

        Checks parse ratios, patterns, known value sets from config.
        """
        rules = self._rules.get_type_detection_rules()
        types_config = rules.get("types", {})
        threshold = rules.get("llm_fallback_threshold", 0.75)

        # Run deterministic classification
        result = self._deterministic_type(profile, types_config)

        # If confidence is high enough, return immediately
        if result.confidence >= threshold:
            return result

        # Try LLM fallback
        if self._llm_classifier and self._rules.is_llm_enabled("type_refinement"):
            llm_config = self._rules.get_llm_config()
            allowed = llm_config.get("allowed_types", [])
            llm_result = self._llm_classifier.classify_type(
                column_name=profile.physical_name,
                sample_values=profile.representative_values[:8],
                statistics=profile.to_statistics_dict(),
                allowed_types=allowed,
            )
            if llm_result and llm_result.confidence > result.confidence:
                try:
                    refined = RefinedDataType(llm_result.refined_type)
                    return HybridTypeResult(
                        refined_type=refined,
                        confidence=llm_result.confidence,
                        source="llm_fallback",
                        reasoning=llm_result.reasoning,
                    )
                except ValueError:
                    pass  # Invalid type from LLM, use deterministic

        return result

    def _deterministic_type(
        self, profile: ColumnProfileResult, types_config: dict[str, Any]
    ) -> HybridTypeResult:
        """Apply YAML-driven deterministic type rules."""
        # All null → unknown
        if profile.non_null_count == 0:
            return HybridTypeResult(RefinedDataType.UNKNOWN, 0.0, "yaml_rules")

        # Boolean check
        bool_cfg = types_config.get("boolean", {})
        if profile.boolean_parse_ratio >= bool_cfg.get("min_parse_ratio", 0.95):
            if profile.distinct_count <= bool_cfg.get("max_distinct_count", 3):
                return HybridTypeResult(RefinedDataType.BOOLEAN, 0.92, "yaml_rules")

        # Datetime check (skip if clearly numeric)
        dt_cfg = types_config.get("datetime", {})
        if (profile.datetime_parse_ratio >= dt_cfg.get("min_parse_ratio", 0.90)
                and profile.numeric_parse_ratio < 0.90):
            date_patterns = dt_cfg.get("date_only_patterns", [])
            if any(p in profile.dominant_patterns for p in date_patterns):
                return HybridTypeResult(RefinedDataType.DATE, 0.90, "yaml_rules")
            return HybridTypeResult(RefinedDataType.DATETIME, 0.88, "yaml_rules")

        # Email check
        email_cfg = types_config.get("email", {})
        if "EMAIL" in profile.dominant_patterns:
            return HybridTypeResult(RefinedDataType.EMAIL, 0.93, "yaml_rules")
        email_regex = email_cfg.get("regex", r"^[\w.+-]+@[\w-]+\.[\w.]+$")
        email_matches = sum(
            1 for v in profile.representative_values
            if isinstance(v, str) and re.match(email_regex, v)
        )
        min_ratio = email_cfg.get("min_match_ratio", 0.80)
        if email_matches / max(len(profile.representative_values), 1) >= min_ratio:
            return HybridTypeResult(RefinedDataType.EMAIL, 0.88, "yaml_rules")

        # Phone check
        if "PHONE" in profile.dominant_patterns:
            return HybridTypeResult(RefinedDataType.PHONE, 0.88, "yaml_rules")

        # Currency code
        currency_cfg = types_config.get("currency_code", {})
        if (profile.distinct_count <= currency_cfg.get("max_distinct_count", 50)
                and "THREE_LETTER_CODE" in profile.dominant_patterns):
            known = self._rules.get_type_known_values("currency_code")
            match_count = sum(
                1 for v in profile.representative_values
                if isinstance(v, str) and v.upper() in known
            )
            if match_count / max(len(profile.representative_values), 1) >= currency_cfg.get("min_match_ratio", 0.70):
                return HybridTypeResult(RefinedDataType.CURRENCY_CODE, 0.90, "yaml_rules")

        # Country code
        country_cfg = types_config.get("country_code", {})
        if profile.distinct_count <= country_cfg.get("max_distinct_count", 250):
            for variant in ["two_letter", "three_letter"]:
                var_cfg = country_cfg.get(variant, {})
                if not isinstance(var_cfg, dict):
                    continue
                var_patterns = var_cfg.get("patterns", [])
                if any(p in profile.dominant_patterns for p in var_patterns):
                    known_values = var_cfg.get("known_values", [])
                    known = {str(v).upper() for v in known_values if isinstance(v, str)}
                    match_count = sum(
                        1 for v in profile.representative_values
                        if isinstance(v, str) and v.upper() in known
                    )
                    if match_count / max(len(profile.representative_values), 1) >= country_cfg.get("min_match_ratio", 0.70):
                        return HybridTypeResult(RefinedDataType.COUNTRY_CODE, 0.89, "yaml_rules")

        # Percentage
        pct_cfg = types_config.get("percentage", {})
        if profile.numeric_parse_ratio >= pct_cfg.get("min_numeric_ratio", 0.90):
            value_range = pct_cfg.get("value_range", [0, 100])
            if (profile.min_value is not None and profile.max_value is not None
                    and profile.min_value >= value_range[0]
                    and profile.max_value <= value_range[1]):
                name_hints = pct_cfg.get("name_hints", [])
                if any(h in profile.normalized_key.lower() for h in name_hints):
                    return HybridTypeResult(RefinedDataType.PERCENTAGE, 0.85, "yaml_rules")

        # Numeric
        if profile.numeric_parse_ratio >= 0.90:
            if "INTEGER" in profile.dominant_patterns:
                return HybridTypeResult(RefinedDataType.INTEGER, 0.85, "yaml_rules")
            if "DECIMAL_2DP" in profile.dominant_patterns:
                return HybridTypeResult(RefinedDataType.DECIMAL, 0.85, "yaml_rules")
            # Check if values are whole numbers
            if profile.min_value is not None and profile.max_value is not None:
                try:
                    if float(profile.min_value) == int(profile.min_value) and float(profile.max_value) == int(profile.max_value):
                        return HybridTypeResult(RefinedDataType.INTEGER, 0.78, "yaml_rules")
                except (ValueError, TypeError, OverflowError):
                    pass
            return HybridTypeResult(RefinedDataType.DECIMAL, 0.80, "yaml_rules")

        # Identifier
        id_cfg = types_config.get("identifier", {})
        if (profile.cardinality_ratio >= id_cfg.get("min_cardinality_ratio", 0.98)
                and profile.non_null_count >= id_cfg.get("min_row_count", 10)):
            return HybridTypeResult(RefinedDataType.IDENTIFIER, 0.85, "yaml_rules")
        if "UUID" in profile.dominant_patterns:
            return HybridTypeResult(RefinedDataType.IDENTIFIER, 0.92, "yaml_rules")

        # Categorical
        cat_cfg = types_config.get("categorical", {})
        if (profile.cardinality_ratio <= cat_cfg.get("max_cardinality_ratio", 0.05)
                and profile.distinct_count <= cat_cfg.get("max_distinct_count", 50)):
            return HybridTypeResult(RefinedDataType.CATEGORICAL, 0.80, "yaml_rules")
        alt = cat_cfg.get("alternate", {})
        if (profile.distinct_count <= alt.get("max_distinct_count", 20)
                and profile.non_null_count >= alt.get("min_row_count", 50)):
            return HybridTypeResult(RefinedDataType.CATEGORICAL, 0.75, "yaml_rules")

        # Text fallback
        if profile.non_null_count > 0:
            return HybridTypeResult(RefinedDataType.TEXT, 0.60, "yaml_rules")

        return HybridTypeResult(RefinedDataType.UNKNOWN, 0.30, "yaml_rules")

    # ----------------------------------------------------------------
    # SEMANTIC ROLE CLASSIFICATION (Layer 2)
    # ----------------------------------------------------------------

    def classify_role(
        self, profile: ColumnProfileResult, refined_type: RefinedDataType,
        is_identifier: bool, primary_domain: str = "",
    ) -> HybridRoleResult:
        """
        Classify column semantic role using YAML rules + LLM fallback.

        Checks column name against keyword lists loaded from config.
        """
        rules = self._rules.get_semantic_role_rules()
        threshold = rules.get("llm_fallback_threshold", 0.70)

        # Run deterministic classification
        result = self._deterministic_role(profile, refined_type, is_identifier, rules)

        # If high confidence, return
        if result.confidence >= threshold:
            return result

        # Try LLM fallback
        if self._llm_classifier and self._rules.is_llm_enabled("semantic_role"):
            llm_config = self._rules.get_llm_config()
            allowed = llm_config.get("allowed_roles", [])
            llm_result = self._llm_classifier.classify_role(
                column_name=profile.physical_name,
                refined_type=refined_type.value,
                sample_values=profile.representative_values[:8],
                statistics=profile.to_statistics_dict(),
                primary_domain=primary_domain,
                allowed_roles=allowed,
            )
            if llm_result and llm_result.confidence > result.confidence:
                try:
                    role = ColumnRole(llm_result.column_role)
                    return HybridRoleResult(
                        column_role=role,
                        semantic_type=llm_result.semantic_type,
                        confidence=llm_result.confidence,
                        source="llm_fallback",
                        reasoning=llm_result.reasoning,
                    )
                except ValueError:
                    pass

        return result

    def _deterministic_role(
        self, profile: ColumnProfileResult, refined_type: RefinedDataType,
        is_identifier: bool, rules: dict[str, Any],
    ) -> HybridRoleResult:
        """Apply YAML-driven deterministic role rules."""
        roles_config = rules.get("roles", {})
        name_lower = profile.normalized_key.lower()

        # Identifier (detected by cardinality, not keywords)
        if is_identifier:
            # But don't override if column clearly matches metric keywords
            metric_keywords = roles_config.get("metric", {}).get("keywords", [])
            if not any(self._keyword_matches(kw, name_lower) for kw in metric_keywords):
                return HybridRoleResult(ColumnRole.IDENTIFIER, "identifier", 0.90, "yaml_rules")

        # Temporal — refined type takes precedence
        if refined_type in (RefinedDataType.DATE, RefinedDataType.DATETIME):
            sem_type = self._match_sub_type(name_lower, roles_config.get("temporal_dimension", {}))
            return HybridRoleResult(
                ColumnRole.TEMPORAL_DIMENSION, sem_type or "datetime", 0.88, "yaml_rules"
            )

        # Flag/boolean
        if refined_type == RefinedDataType.BOOLEAN:
            flag_keywords = roles_config.get("flag", {}).get("keywords", [])
            sem_type = "flag"
            for kw in flag_keywords:
                if kw in name_lower:
                    sem_type = f"{kw}_flag" if kw not in ("flag", "is_", "has_") else "flag"
                    break
            return HybridRoleResult(ColumnRole.FLAG, sem_type, 0.85, "yaml_rules")

        # Check metric keywords (must be a word boundary match, not substring)
        metric_keywords = roles_config.get("metric", {}).get("keywords", [])
        if any(self._keyword_matches(kw, name_lower) for kw in metric_keywords):
            sem_type = self._match_sub_type(name_lower, roles_config.get("metric", {}))
            return HybridRoleResult(ColumnRole.METRIC, sem_type or "numeric_measure", 0.85, "yaml_rules")

        # Check temporal keywords (for string columns that look like dates by name)
        temporal_keywords = roles_config.get("temporal_dimension", {}).get("keywords", [])
        suffix_patterns = roles_config.get("temporal_dimension", {}).get("suffix_patterns", [])
        if any(kw in name_lower for kw in temporal_keywords):
            sem_type = self._match_sub_type(name_lower, roles_config.get("temporal_dimension", {}))
            return HybridRoleResult(ColumnRole.TEMPORAL_DIMENSION, sem_type or "date", 0.75, "yaml_rules")
        if any(name_lower.endswith(sp) for sp in suffix_patterns):
            return HybridRoleResult(ColumnRole.TEMPORAL_DIMENSION, "datetime", 0.72, "yaml_rules")

        # Check text field keywords
        text_keywords = roles_config.get("text_field", {}).get("keywords", [])
        if any(kw in name_lower for kw in text_keywords):
            return HybridRoleResult(ColumnRole.TEXT_FIELD, "description", 0.78, "yaml_rules")

        # Check dimension keywords
        dim_keywords = roles_config.get("dimension", {}).get("keywords", [])
        if any(kw in name_lower for kw in dim_keywords):
            sem_type = self._match_sub_type(name_lower, roles_config.get("dimension", {}))
            return HybridRoleResult(ColumnRole.DIMENSION, sem_type or "dimension", 0.80, "yaml_rules")

        # Numeric with high cardinality → metric
        if refined_type in (RefinedDataType.INTEGER, RefinedDataType.DECIMAL):
            if profile.cardinality_ratio > 0.05:
                return HybridRoleResult(ColumnRole.METRIC, "numeric_measure", 0.65, "yaml_rules")
            else:
                return HybridRoleResult(ColumnRole.DIMENSION, "numeric_category", 0.65, "yaml_rules")

        # Categorical → dimension
        if refined_type == RefinedDataType.CATEGORICAL:
            return HybridRoleResult(ColumnRole.DIMENSION, "dimension", 0.70, "yaml_rules")

        # Text → text_field
        if refined_type == RefinedDataType.TEXT:
            return HybridRoleResult(ColumnRole.TEXT_FIELD, "text", 0.50, "yaml_rules")

        # Fallback
        return HybridRoleResult(ColumnRole.UNKNOWN, None, 0.30, "yaml_rules")

    def _match_sub_type(self, name_lower: str, role_config: dict[str, Any]) -> str | None:
        """Find the most specific sub-type from YAML config."""
        sub_types = role_config.get("sub_types", {})
        for sub_name, sub_cfg in sub_types.items():
            keywords = sub_cfg.get("keywords", []) if isinstance(sub_cfg, dict) else sub_cfg
            if any(kw in name_lower for kw in keywords):
                return sub_name
        return None

    # ----------------------------------------------------------------
    # DATA CATEGORY CLASSIFICATION (Layer 3)
    # ----------------------------------------------------------------

    def classify_category(
        self, profile: ColumnProfileResult, semantic_type: str | None,
        column_role: ColumnRole, refined_type: RefinedDataType,
        primary_domain: str,
    ) -> HybridCategoryResult:
        """
        Classify column into a data category using YAML rules + LLM fallback.
        """
        rules = self._rules.get_data_category_rules()
        threshold = rules.get("llm_fallback_threshold", 0.65)

        # Run deterministic
        result = self._deterministic_category(
            profile, semantic_type, column_role, refined_type, primary_domain, rules
        )

        # If high confidence, return
        if result.confidence >= threshold:
            return result

        # Try LLM fallback
        if self._llm_classifier and self._rules.is_llm_enabled("data_category"):
            llm_config = self._rules.get_llm_config()
            allowed = llm_config.get("allowed_categories", [])
            llm_result = self._llm_classifier.classify_category(
                column_name=profile.physical_name,
                semantic_type=semantic_type,
                column_role=column_role.value,
                refined_type=refined_type.value,
                sample_values=profile.representative_values[:5],
                primary_domain=primary_domain,
                allowed_categories=allowed,
            )
            if llm_result and llm_result.confidence > result.confidence:
                try:
                    primary = DataCategory(llm_result.primary_category)
                    secondary = [DataCategory(c) for c in llm_result.secondary_categories
                                 if c in [dc.value for dc in DataCategory]]
                    return HybridCategoryResult(
                        primary_category=primary,
                        secondary_categories=secondary,
                        confidence=llm_result.confidence,
                        source="llm_fallback",
                        reasoning=llm_result.reasoning,
                    )
                except ValueError:
                    pass

        return result

    def _deterministic_category(
        self, profile: ColumnProfileResult, semantic_type: str | None,
        column_role: ColumnRole, refined_type: RefinedDataType,
        primary_domain: str, rules: dict[str, Any],
    ) -> HybridCategoryResult:
        """Apply YAML-driven deterministic category rules."""
        categories_config = rules.get("categories", {})
        name_lower = profile.normalized_key.lower()
        scores: dict[str, float] = {}

        # Score each category by keyword match
        for cat_name, cat_cfg in categories_config.items():
            keywords = cat_cfg.get("keywords", [])
            score = 0.0

            # Name keyword match
            for kw in keywords:
                if kw in name_lower:
                    score += 0.4
                    break

            # Semantic type hint match
            sem_hints = cat_cfg.get("semantic_type_hints", [])
            if semantic_type and any(h in (semantic_type or "").lower() for h in sem_hints):
                score += 0.5

            # Role-based boost
            if column_role == ColumnRole.TEMPORAL_DIMENSION and cat_name == "Time Series Data":
                score += 0.3
            if column_role == ColumnRole.IDENTIFIER and cat_name == "Master Data":
                score += 0.3

            if score > 0:
                scores[cat_name] = score

        # Domain context boosts
        domain_boosts = rules.get("domain_boosts", {}).get(primary_domain, {})
        for cat_name in domain_boosts.get("strong", []):
            scores[cat_name] = scores.get(cat_name, 0) + 0.2
        for cat_name in domain_boosts.get("weak", []):
            scores[cat_name] = scores.get(cat_name, 0) + 0.1

        # Datetime parse ratio boost
        if profile.datetime_parse_ratio >= 0.9:
            scores["Time Series Data"] = scores.get("Time Series Data", 0) + 0.4

        if not scores:
            default = self._default_category(column_role)
            return HybridCategoryResult(default, [], 0.40, "yaml_rules", "no_strong_signals")

        # Sort and select
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary_name = sorted_cats[0][0]
        primary_score = sorted_cats[0][1]

        try:
            primary = DataCategory(primary_name)
        except ValueError:
            primary = DataCategory.OPERATIONAL

        # Secondary categories
        secondary: list[DataCategory] = []
        for cat_name, score in sorted_cats[1:4]:
            if score >= primary_score * 0.5 and score >= 0.3:
                try:
                    secondary.append(DataCategory(cat_name))
                except ValueError:
                    pass

        confidence = min(1.0, primary_score)
        return HybridCategoryResult(primary, secondary, confidence, "yaml_rules")

    def _default_category(self, role: ColumnRole) -> DataCategory:
        """Fallback when no rules match."""
        if role == ColumnRole.METRIC:
            return DataCategory.OPERATIONAL
        if role == ColumnRole.TEMPORAL_DIMENSION:
            return DataCategory.TIME_SERIES
        if role == ColumnRole.IDENTIFIER:
            return DataCategory.MASTER
        return DataCategory.OPERATIONAL

    @staticmethod
    def _keyword_matches(keyword: str, name: str) -> bool:
        """
        Check if a keyword matches in a column name respecting word boundaries.

        Prevents 'count' matching 'country', 'amount' matching 'discount_amount_id', etc.
        Uses underscore and start/end as word boundaries.
        """
        # Split name into parts by underscore
        parts = name.split("_")
        # Check if keyword matches any part exactly, or is a prefix/suffix of the whole name
        if keyword in parts:
            return True
        # Also check if keyword appears as a complete segment (bounded by _ or edges)
        pattern = rf"(^|_){re.escape(keyword)}(_|$)"
        return bool(re.search(pattern, name))
