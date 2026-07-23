"""tests/test_llm_service.py — Groq-only LLM service tests.

Replaces the previous Azure-first/Groq-fallback test file (removed
alongside the Azure OpenAI integration) with coverage of the actual
current behavior: _chat_complete()'s Groq call, and the two public
functions' graceful degradation to placeholder/default output when the
LLM response can't be parsed or the call fails outright.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.schemas import DatasetMetadata
from app.services.llm_service import (
    _chat_complete,
    _generate_column_descriptions_batch,
    classify_dataset,
    generate_column_descriptions,
)


def _metadata(**overrides) -> DatasetMetadata:
    defaults = dict(
        dataset_id="DS_1", dataset_name="test.csv", file_type="csv",
        upload_timestamp="2026-01-01T00:00:00Z", row_count=10, column_count=2,
        column_names=["amount", "region"], column_data_types=["float", "string"],
    )
    defaults.update(overrides)
    return DatasetMetadata(**defaults)


def _client_with_content(content: str) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(content=content))]
    return client


def test_chat_complete_calls_groq_and_returns_content():
    client = _client_with_content("hello")
    result = _chat_complete(client, [{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=50)
    assert result == "hello"
    client.chat.completions.create.assert_called_once()
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 0.1
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 50


def test_generate_column_descriptions_batch_parses_valid_json():
    client = _client_with_content('{"amount": "The transaction amount.", "region": "Geographic region."}')
    result = _generate_column_descriptions_batch(client, _metadata(), ["amount", "region"], ["float", "string"])
    assert result == {"amount": "The transaction amount.", "region": "Geographic region."}


def test_generate_column_descriptions_batch_falls_back_on_malformed_json():
    client = _client_with_content("not valid json at all")
    result = _generate_column_descriptions_batch(client, _metadata(), ["amount", "region"], ["float", "string"])
    assert result == {
        "amount": "Column 'amount' of type float",
        "region": "Column 'region' of type string",
    }


def test_generate_column_descriptions_batch_falls_back_when_groq_call_raises():
    client = MagicMock()
    client.chat.completions.create.side_effect = Exception("groq unavailable")
    result = _generate_column_descriptions_batch(client, _metadata(), ["amount"], ["float"])
    assert result == {"amount": "Column 'amount' of type float"}


def test_generate_column_descriptions_end_to_end_with_mocked_client():
    with patch("app.services.llm_service._get_groq_client", return_value=_client_with_content('{"amount": "desc"}')):
        result = generate_column_descriptions(_metadata(column_names=["amount"], column_data_types=["float"], column_count=1))
    assert result == {"amount": "desc"}


def test_classify_dataset_parses_valid_json():
    content = '{"business_domain": "Finance", "sub_domain": "Banking", "dataset_summary": "x", "confidence": 0.9, "reason": "y"}'
    with patch("app.services.llm_service._get_groq_client", return_value=_client_with_content(content)):
        result = classify_dataset(_metadata())
    assert result.business_domain == "Finance"
    assert result.sub_domain == "Banking"
    assert result.confidence == 0.9


def test_classify_dataset_falls_back_to_other_on_malformed_json():
    with patch("app.services.llm_service._get_groq_client", return_value=_client_with_content("not json")):
        result = classify_dataset(_metadata())
    assert result.business_domain == "Other"
    assert result.sub_domain == "General"


def test_classify_dataset_strips_markdown_json_fences():
    content = '```json\n{"business_domain": "HR", "sub_domain": "Payroll"}\n```'
    with patch("app.services.llm_service._get_groq_client", return_value=_client_with_content(content)):
        result = classify_dataset(_metadata())
    assert result.business_domain == "HR"
