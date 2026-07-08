"""Pattern detector — detects dominant value patterns in columns."""

import re
from typing import Any

import pandas as pd


# Compiled patterns for efficient matching
PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "YYYY-MM-DD"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"), "ISO_DATETIME"),
    (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "MM/DD/YYYY"),
    (re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$"), "DATE_SLASH"),
    (re.compile(r"^[A-Z]{3}$"), "THREE_LETTER_CODE"),
    (re.compile(r"^[A-Z]{2}$"), "TWO_LETTER_CODE"),
    (re.compile(r"^[a-f0-9\-]{36}$"), "UUID"),
    (re.compile(r"^[\w.+-]+@[\w-]+\.[\w.]+$"), "EMAIL"),
    (re.compile(r"^\+?\d[\d\s\-\(\)]{7,}$"), "PHONE"),
    (re.compile(r"^\d+\.\d{2}$"), "DECIMAL_2DP"),
    (re.compile(r"^\d+$"), "INTEGER"),
    (re.compile(r"^-?\d+\.?\d*$"), "NUMERIC"),
]


class PatternDetector:
    """Detects dominant value patterns in string columns."""

    def __init__(self, sample_size: int = 500, min_ratio: float = 0.5):
        self._sample_size = sample_size
        self._min_ratio = min_ratio

    def detect(self, series: pd.Series) -> list[str]:
        """
        Detect dominant patterns in a series.

        Returns list of pattern labels sorted by frequency.
        Only includes patterns matching > min_ratio of sampled values.
        """
        non_null = series.dropna()
        if len(non_null) == 0:
            return []

        sample = non_null.head(min(self._sample_size, len(non_null))).astype(str)
        sample_size = len(sample)

        pattern_counts: dict[str, int] = {}

        for pattern, label in PATTERNS:
            match_count = int(sample.str.match(pattern, na=False).sum())
            ratio = match_count / sample_size
            if ratio >= self._min_ratio:
                pattern_counts[label] = match_count

        # Sort by count descending
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        return [p[0] for p in sorted_patterns[:5]]

    def detect_custom_pattern(self, series: pd.Series, pattern: str) -> float:
        """
        Test a custom regex pattern against the series.

        Returns the match ratio (0.0 to 1.0).
        """
        non_null = series.dropna()
        if len(non_null) == 0:
            return 0.0

        sample = non_null.head(min(self._sample_size, len(non_null))).astype(str)
        compiled = re.compile(pattern)
        matches = int(sample.str.match(compiled, na=False).sum())
        return matches / len(sample)
