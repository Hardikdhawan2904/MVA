"""tests/test_explanation_tool.py — Tests for app/services/explanation_tool.py's
deterministic confidence handling.

Confidence is a rule-based function of which evidence keys are present, not
something that should ever be left to the LLM's judgment (the tool's own
docstring: "Must ONLY narrate the evidence passed to it"). These tests cover
_compute_confidence directly, and _enforce_confidence_section's guarantee
that whatever an LLM writes in a "## Confidence" section gets overridden
with the deterministic value — added after Groq was observed self-labeling
a query "HIGH" confidence when only actuals + variance were present (no
root-cause evidence), contradicting agent.yaml's own confidence_rules.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.explanation_tool import ExplanationTool


# ── _compute_confidence ───────────────────────────────────────────────────────

def test_confidence_high_requires_actuals_variance_and_root_cause():
    level, text = ExplanationTool._compute_confidence({
        "actual": 1.0, "variance_amount": 1.0, "primary_driver": {"label": "x"},
    })
    assert level == "HIGH"
    assert "HIGH" in text


def test_confidence_medium_without_root_cause():
    level, text = ExplanationTool._compute_confidence({
        "actual": 1.0, "variance_amount": 1.0,
    })
    assert level == "MEDIUM"
    assert "root cause not analysed" in text


def test_confidence_medium_actuals_only():
    level, text = ExplanationTool._compute_confidence({"actual": 1.0})
    assert level == "MEDIUM"
    assert "comparison data absent" in text


def test_confidence_low_with_no_relevant_evidence():
    level, text = ExplanationTool._compute_confidence({"kpi": "Loss Ratio"})
    assert level == "LOW"


# ── _enforce_confidence_section ───────────────────────────────────────────────

def test_enforce_replaces_wrong_llm_confidence():
    # Reproduces the exact bug: Groq self-labeled HIGH for evidence that
    # only has actuals + variance (no root-cause), contradicting the
    # deterministic rule.
    groq_output = (
        "## Summary\nGWP is up 4.48%.\n\n"
        "## Confidence\nHIGH, as the evidence provides a clear and complete picture."
    )
    _, correct_text = ExplanationTool._compute_confidence({"actual": 1.0, "variance_amount": 1.0})
    fixed = ExplanationTool._enforce_confidence_section(groq_output, correct_text)

    assert "MEDIUM" in fixed
    assert "clear and complete picture" not in fixed
    assert "## Summary\nGWP is up 4.48%." in fixed  # rest of the narrative untouched


def test_enforce_appends_section_if_llm_omitted_it():
    groq_output = "## Summary\nGWP is up 4.48%."
    _, correct_text = ExplanationTool._compute_confidence({"actual": 1.0})
    fixed = ExplanationTool._enforce_confidence_section(groq_output, correct_text)

    assert fixed.startswith(groq_output)
    assert "## Confidence" in fixed
    assert correct_text in fixed


def test_enforce_is_case_insensitive_on_heading():
    groq_output = "## summary\ntext\n\n## confidence\nwrong text here"
    _, correct_text = ExplanationTool._compute_confidence({"actual": 1.0})
    fixed = ExplanationTool._enforce_confidence_section(groq_output, correct_text)

    assert "wrong text here" not in fixed
    assert correct_text in fixed
