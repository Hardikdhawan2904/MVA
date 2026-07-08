"""Tests for deterministic type refinement."""

import pytest
import pandas as pd

from app.core.config import Settings
from app.core.enums import RefinedDataType
from app.services.profiling.column_profiler import ColumnProfiler
from app.services.profiling.type_refiner import TypeRefiner


@pytest.fixture
def profiler() -> ColumnProfiler:
    settings = Settings(DATABASE_URL="postgresql://x:x@localhost/test", MAX_SAMPLE_VALUES=10)
    return ColumnProfiler(settings)


@pytest.fixture
def refiner() -> TypeRefiner:
    return TypeRefiner()


def _profile(profiler: ColumnProfiler, values: list, name: str = "col"):
    series = pd.Series(values)
    return profiler.profile_column(series, name, name)


class TestTypeRefiner:
    """Test deterministic type refinement logic."""

    def test_integer_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, ["1", "2", "3", "100", "999"])
        result = refiner.refine(profile)
        assert result == RefinedDataType.INTEGER

    def test_decimal_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, ["1.50", "2.75", "3.00", "100.99", "50.25"])
        result = refiner.refine(profile)
        assert result == RefinedDataType.DECIMAL

    def test_date_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, [
            "2024-01-01", "2024-02-15", "2024-03-20", "2024-04-10", "2024-05-05"
        ])
        result = refiner.refine(profile)
        assert result == RefinedDataType.DATE

    def test_datetime_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, [
            "2024-01-01T10:30:00", "2024-02-15T14:45:00",
            "2024-03-20T08:00:00", "2024-04-10T16:20:00",
        ])
        result = refiner.refine(profile)
        assert result in (RefinedDataType.DATE, RefinedDataType.DATETIME)

    def test_boolean_true_false(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, ["true", "false", "true", "false", "true"] * 10)
        result = refiner.refine(profile)
        assert result == RefinedDataType.BOOLEAN

    def test_boolean_zero_one(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, ["0", "1", "1", "0", "1"] * 10)
        result = refiner.refine(profile)
        assert result == RefinedDataType.BOOLEAN

    def test_currency_code_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, ["USD", "EUR", "GBP", "JPY", "INR", "USD", "EUR"] * 5)
        result = refiner.refine(profile)
        assert result == RefinedDataType.CURRENCY_CODE

    def test_country_code_2_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, ["US", "GB", "DE", "FR", "JP", "IN", "AU"] * 5)
        result = refiner.refine(profile)
        assert result == RefinedDataType.COUNTRY_CODE

    def test_email_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, [
            "alice@example.com", "bob@test.org", "charlie@mail.co",
            "dave@company.io", "eve@domain.net",
        ])
        result = refiner.refine(profile)
        assert result == RefinedDataType.EMAIL

    def test_percentage_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, ["10", "25", "50", "75", "100", "0", "33"], "success_pct")
        result = refiner.refine(profile)
        assert result == RefinedDataType.PERCENTAGE

    def test_identifier_uuid(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        import uuid
        values = [str(uuid.uuid4()) for _ in range(50)]
        profile = _profile(profiler, values)
        result = refiner.refine(profile)
        assert result == RefinedDataType.IDENTIFIER

    def test_identifier_high_cardinality(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        values = [f"TXN-{i:06d}" for i in range(100)]
        profile = _profile(profiler, values)
        result = refiner.refine(profile)
        assert result == RefinedDataType.IDENTIFIER

    def test_categorical_low_cardinality(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        values = ["approved", "declined", "pending"] * 50
        profile = _profile(profiler, values)
        result = refiner.refine(profile)
        assert result == RefinedDataType.CATEGORICAL

    def test_text_detection(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        values = [
            "This is a long description of the transaction",
            "Another detailed note about the payment processing",
            "Short note",
            "Yet another explanation of what happened during settlement",
            "Final remark about the authorization process",
        ] * 10 + ["Repeated text for variety"] * 10
        profile = _profile(profiler, values)
        result = refiner.refine(profile)
        assert result in (RefinedDataType.TEXT, RefinedDataType.CATEGORICAL)

    def test_all_null_returns_unknown(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        profile = _profile(profiler, [None, None, None])
        result = refiner.refine(profile)
        assert result == RefinedDataType.UNKNOWN

    def test_low_cardinality_code_not_identifier(self, profiler: ColumnProfiler, refiner: TypeRefiner):
        """A column like department_code with low cardinality should NOT be identifier."""
        values = ["DEPT01", "DEPT02", "DEPT03"] * 50
        profile = _profile(profiler, values, "department_code")
        result = refiner.refine(profile)
        # Should be categorical, not identifier
        assert result == RefinedDataType.CATEGORICAL
