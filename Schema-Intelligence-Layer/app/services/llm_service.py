"""
LLM service using Groq API.
Handles dataset classification and column description generation.
Only metadata (column names, types, sample values) is sent to the LLM — never the full dataset.
"""

import json
import logging
from groq import Groq

from app.config import settings
from app.models.schemas import DatasetMetadata, LLMClassification
from app.prompts.llm_service_prompt import (
    get_column_descriptions_prompt,
    get_classify_dataset_prompt,
)

logger = logging.getLogger(__name__)

# Removed static VALID_DOMAINS and SUB_DOMAINS taxonomies.
# Domain and sub-domain classification now runs fully dynamically via the LLM prompts.

# Groq's free-tier cap is 6,000 tokens/minute. A 141-column dataset needed ~6,283 tokens
# for one description prompt and ~7,597 for one classification prompt — both over the
# limit, causing a permanent 413 regardless of retries. 30 columns per LLM call keeps
# either prompt comfortably under that cap.
MAX_COLUMNS_PER_LLM_CALL = 30



def _get_groq_client() -> Groq:
    """Create and return a Groq client instance."""
    return Groq(api_key=settings.GROQ_API_KEY)


def _strip_json_fences(response_text: str) -> str:
    """Strip a markdown code fence (```json ... ``` or ``` ... ```) that Groq
    sometimes wraps its JSON response in, despite being asked for JSON only."""
    if response_text.startswith("```"):
        response_text = response_text.strip("`").strip()
        if response_text.startswith("json"):
            response_text = response_text[4:].strip()
    return response_text


def _build_metadata_context(metadata: DatasetMetadata) -> str:
    """
    Build a text summary of dataset metadata for the LLM prompt.
    """
    lines = [
        f"Dataset Name: {metadata.dataset_name}",
        f"File Type: {metadata.file_type}",
        f"Number of Rows: {metadata.row_count}",
        f"Number of Columns: {metadata.column_count}",
        "",
        "Columns:",
    ]

    # Pre-extract up to 3 distinct non-empty sample values per column from sample_data list
    col_samples = {col: [] for col in metadata.column_names}
    if metadata.sample_data:
        for row in metadata.sample_data:
            for col in metadata.column_names:
                val = row.get(col)
                if val is not None and val != "":
                    val_str = str(val)[:50]  # Limit length of a single sample value
                    if val_str not in col_samples[col] and len(col_samples[col]) < 3:
                        col_samples[col].append(val_str)

    for i, col_name in enumerate(metadata.column_names):
        dtype = metadata.column_data_types[i] if i < len(metadata.column_data_types) else "unknown"
        samples = col_samples.get(col_name, [])
        samples_str = f", samples: {samples}" if samples else ""
        lines.append(f"  - {col_name} (type: {dtype}{samples_str})")

    return "\n".join(lines)


def _generate_column_descriptions_batch(
    client: Groq, batch_metadata: DatasetMetadata, batch_names: list[str], batch_types: list[str]
) -> dict[str, str]:
    """Generate descriptions for a single batch of columns. Falls back to placeholder
    text for just this batch on any failure, so one bad batch can't wipe out others."""
    context = _build_metadata_context(batch_metadata)
    prompt = get_column_descriptions_prompt(context)

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a data analysis assistant. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            max_tokens=2000,
        )

        response_text = _strip_json_fences(response.choices[0].message.content.strip())
        descriptions = json.loads(response_text)

        return {col: descriptions.get(col, f"Column '{col}' in the dataset.") for col in batch_names}

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM column descriptions as JSON: {e}")
        return {col: f"Column '{col}' of type {batch_types[i]}" for i, col in enumerate(batch_names)}
    except Exception as e:
        logger.error(f"Groq API error during column description generation: {e}")
        return {col: f"Column '{col}' of type {batch_types[i]}" for i, col in enumerate(batch_names)}


def generate_column_descriptions(metadata: DatasetMetadata) -> dict[str, str]:
    """
    Generate short descriptions for each column using the Groq LLM.

    Wide datasets are split into batches of MAX_COLUMNS_PER_LLM_CALL columns so a single
    prompt never exceeds Groq's per-request token limit — each batch is its own LLM call,
    merged back into one dict covering every column.
    """
    client = _get_groq_client()
    all_descriptions: dict[str, str] = {}

    for start in range(0, len(metadata.column_names), MAX_COLUMNS_PER_LLM_CALL):
        batch_names = metadata.column_names[start:start + MAX_COLUMNS_PER_LLM_CALL]
        batch_types = metadata.column_data_types[start:start + MAX_COLUMNS_PER_LLM_CALL]
        batch_metadata = metadata.model_copy(update={
            "column_names": batch_names,
            "column_data_types": batch_types,
            "column_count": len(batch_names),
        })
        all_descriptions.update(
            _generate_column_descriptions_batch(client, batch_metadata, batch_names, batch_types)
        )

    return all_descriptions


def classify_dataset(metadata: DatasetMetadata) -> LLMClassification:
    """
    Classify the dataset into a business domain using the Groq LLM.

    Unlike column descriptions, classification produces one result for the whole
    dataset, so it can't be batched-and-merged. Wide datasets instead use only the
    first MAX_COLUMNS_PER_LLM_CALL columns as a representative sample — enough
    signal to classify the domain without needing exhaustive column coverage —
    keeping the prompt under Groq's per-request token limit.
    """
    client = _get_groq_client()

    capped_names = metadata.column_names[:MAX_COLUMNS_PER_LLM_CALL]
    capped_types = metadata.column_data_types[:MAX_COLUMNS_PER_LLM_CALL]
    capped_metadata = metadata.model_copy(update={
        "column_names": capped_names,
        "column_data_types": capped_types,
        "column_count": len(capped_names),
    })

    context = _build_metadata_context(capped_metadata)

    # Include column descriptions if available (capped to the same column subset)
    desc_section = ""
    if metadata.column_descriptions:
        desc_lines = ["", "Column Descriptions:"]
        for col in capped_names:
            desc = metadata.column_descriptions.get(col)
            if desc:
                desc_lines.append(f"  - {col}: {desc}")
        desc_section = "\n".join(desc_lines)

    prompt = get_classify_dataset_prompt(context, desc_section)

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a business data classification expert. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.1,
            max_tokens=500,
        )

        response_text = _strip_json_fences(response.choices[0].message.content.strip())
        result = json.loads(response_text)

        # Retrieve classification from LLM response (fallback to defaults if empty/missing)
        domain = result.get("business_domain", "Other")
        if not isinstance(domain, str) or not domain.strip():
            domain = "Other"

        sub_domain = result.get("sub_domain", "General")
        if not isinstance(sub_domain, str) or not sub_domain.strip():
            sub_domain = "General"

        return LLMClassification(
            business_domain=domain.strip(),
            sub_domain=sub_domain.strip(),
            dataset_summary=result.get("dataset_summary", ""),
            confidence=result.get("confidence"),
            reason=result.get("reason"),
        )

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM classification response as JSON: {e}")
        return LLMClassification(
            business_domain="Other",
            sub_domain="General",
            dataset_summary="Unable to generate summary — LLM response parsing failed.",
            confidence=0.0,
            reason=str(e),
        )
    except Exception as e:
        logger.error(f"Groq API error during dataset classification: {e}")
        return LLMClassification(
            business_domain="Other",
            sub_domain="General",
            dataset_summary="Unable to generate summary — LLM service error.",
            confidence=0.0,
            reason=str(e),
        )

