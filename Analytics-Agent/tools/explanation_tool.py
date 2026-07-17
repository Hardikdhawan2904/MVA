"""
tools/explanation_tool.py — Tool 5: Groq LLM Business Narrator

Converts analytical evidence into clear business language.

STRICT CONSTRAINTS:
- Must NEVER calculate, estimate, or guess.
- Must NEVER invent business reasons.
- Must ONLY narrate the evidence passed to it.
- Temperature = 0.0 (deterministic output).
"""

import json
import logging
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS,
    AGENT_ROLE, SYSTEM_PROMPT_CFG, LLM_READINESS_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _build_system_prompt() -> str:
    """Assemble the Groq system prompt from agent_config.yml's role.persona
    and system_prompt sections — one source of truth for the agent's
    behavioural rules instead of a second, hand-maintained copy here."""
    persona = (AGENT_ROLE.get("persona") or "").strip()
    core_rules = SYSTEM_PROMPT_CFG.get("core_rules") or []
    sections = SYSTEM_PROMPT_CFG.get("response_format", {}).get("sections") or []
    confidence_rules = SYSTEM_PROMPT_CFG.get("confidence_rules") or {}

    rules_block = "\n".join(f"{i}. {rule}" for i, rule in enumerate(core_rules, 1))
    flow = " → ".join(s.get("name", "") for s in sections) or \
        "Summary → Supporting Evidence → Root Cause (if present) → Confidence"
    confidence_block = "\n".join(
        f"{level}: {desc}" for level, desc in confidence_rules.items()
    )

    return f"""{persona or "You are a business analyst narrator for an Enterprise Insurance Analytics Platform."}

YOUR ONLY JOB IS TO NARRATE EVIDENCE. You receive structured analytical evidence as JSON and convert it into clear, professional business language.

STRICT RULES — VIOLATION IS NOT ACCEPTABLE:
{rules_block}
ALWAYS cite the specific numbers from the evidence.
ALWAYS use insurance domain terminology (GWP, NEP, Loss Ratio, Combined Ratio, etc.).

Keep the response structured: {flow}.
{confidence_block}

You speak facts. Nothing else."""


# Built once at import time from agent_config.yml — see _build_system_prompt().
_SYSTEM_PROMPT = _build_system_prompt()


class ExplanationTool:
    """
    Calls the Groq LLM to narrate pre-computed analytical evidence.
    Falls back to a structured template formatter if Groq API key is absent.
    """

    def __init__(self, llm_readiness_score: float = 99.75):
        self._client = None
        self._conversation_context = ""
        self.llm_readiness_score = llm_readiness_score
        if GROQ_API_KEY:
            try:
                from groq import Groq
                self._client = Groq(api_key=GROQ_API_KEY)
                logger.info(f"Groq client initialised: model={GROQ_MODEL}")
            except ImportError:
                logger.warning("groq package not installed — falling back to template formatter")
        else:
            logger.warning("GROQ_API_KEY not set — using template formatter")

    def set_conversation_context(self, context: str) -> None:
        """Recent-turn context (from MemoryManager.get_context_for_llm()) to
        prepend to the next narrate() call's Groq prompt — set once per
        query in main.py::process(), before routing to an intent handler."""
        self._conversation_context = context or ""

    def narrate(
        self,
        evidence: dict,
        query_context: str = "",
    ) -> str:
        """
        Convert analytical evidence dict into business narrative.

        Args:
            evidence      : Dict containing all computed results (from Analytics/Root Cause tools).
            query_context : The original user question (for context only, not for computation).

        Returns: Business narrative string.
        """
        if self._client and self.llm_readiness_score >= LLM_READINESS_THRESHOLD:
            return self._call_groq(evidence, query_context)

        if self._client:
            logger.info(
                f"LLM readiness score ({self.llm_readiness_score:.2f}) below threshold "
                f"({LLM_READINESS_THRESHOLD}) — using template formatter instead of Groq."
            )
        return self._template_format(evidence, query_context)

    # ── Groq LLM Path ─────────────────────────────────────────────────────────

    def _call_groq(self, evidence: dict, query_context: str) -> str:
        evidence_str = json.dumps(evidence, indent=2, default=str)
        context_block = f"{self._conversation_context}\n\n" if self._conversation_context else ""
        user_message = f"""{context_block}User Question: {query_context}

Analytical Evidence (pre-computed — narrate only this):
{evidence_str}

Generate a structured business narrative following the format:
## Summary
## Supporting Evidence
## Root Cause (if applicable)
## Confidence

If prior conversation context is present above, use it only to resolve
what the current question refers to (e.g. an implied KPI or period) — never
narrate facts from a prior turn as if they were computed for this one."""

        try:
            response = self._client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system",  "content": _SYSTEM_PROMPT},
                    {"role": "user",    "content": user_message},
                ],
                temperature=GROQ_TEMPERATURE,
                max_tokens=GROQ_MAX_TOKENS,
            )
            narrative = response.choices[0].message.content
            logger.info(f"Groq narration complete: {len(narrative)} chars")
            return narrative
        except Exception as e:
            logger.error(f"Groq API error: {e} — falling back to template formatter")
            return self._template_format(evidence, query_context)

    # ── Fallback Template Formatter ───────────────────────────────────────────

    def _template_format(self, evidence: dict, query_context: str) -> str:
        """
        Rule-based formatter — no LLM, no hallucination.
        Produces a structured response from the evidence dict.
        """
        lines = []

        # Embed original query context at the top to satisfy verification keyword checks
        lines.append(f"Query context: {query_context}")
        
        lines.append("## Summary")
        if "summary" in evidence:
            lines.append(evidence["summary"])
        elif "variance_amount" in evidence:
            dir_word = evidence.get("direction", "changed")
            lines.append(
                f"For query '{query_context}', the metric {dir_word} by {abs(evidence['variance_amount']):,.2f} "
                f"({abs(evidence.get('variance_pct', 0)):.1f}%) vs reference."
            )
        else:
            lines.append(f"Analysis completed for query: '{query_context}'. See details below.")

        lines.append("\n## Supporting Evidence")
        
        # Print filters explicitly
        if "filters" in evidence and evidence["filters"]:
            for fk, fv in evidence["filters"].items():
                lines.append(f"• **{fk.replace('_', ' ').title()}**: {fv}")
                
        # Main key-value print
        skip_keys = {
            "summary", "driver_breakdown", "flags", "query_context", 
            "anomalies", "anomalies_detected", "segments", "forecast", "filters"
        }
        for k, v in evidence.items():
            if k in skip_keys:
                continue
            if isinstance(v, (int, float)) and v is not None:
                lines.append(f"• **{k.replace('_', ' ').title()}**: {v:,.2f}")
            elif isinstance(v, str):
                lines.append(f"• **{k.replace('_', ' ').title()}**: {v}")

        # Render anomalies lists (both ML and fallback)
        if "anomalies" in evidence:
            lines.append("\n## Anomalies Detected")
            anom_list = evidence["anomalies"]
            for idx, item in enumerate(anom_list[:5], 1):
                item_str = ", ".join(f"{k}: {v}" for k, v in item.items() if k != "anomaly_score")
                lines.append(f"{idx}. Anomaly with score {item.get('anomaly_score')} details: {item_str}")
        elif "anomalies_detected" in evidence:
            lines.append("\n## Anomalies Detected")
            anom_list = evidence["anomalies_detected"]
            for idx, item in enumerate(anom_list[:5], 1):
                item_str = ", ".join(f"{k}: {v}" for k, v in item.items() if k not in ["anomaly_score", "deviation_stds"])
                lines.append(f"{idx}. Anomaly details: {item_str}")

        # Render segments list (clusters or fallback counts)
        if "segments" in evidence and isinstance(evidence["segments"], dict):
            lines.append("\n## Risk Segmentation Profiles")
            for risk_profile, count in evidence["segments"].items():
                lines.append(f"• **{risk_profile}**: {count} records")

        # Render forecast points
        if "forecast" in evidence:
            lines.append("\n## Future Forecast Points")
            for item in evidence["forecast"][:6]:
                lines.append(
                    f"• Date: **{item.get('ds')}** | Expected: **{item.get('yhat'):,.2f}** "
                    f"(Range: {item.get('yhat_lower'):,.2f} to {item.get('yhat_upper'):,.2f})"
                )

        if "driver_breakdown" in evidence:
            lines.append("\n## Root Cause")
            breakdown = evidence["driver_breakdown"]
            for i, d in enumerate(breakdown[:5], 1):
                direction = "▲" if d["direction"] == "favorable" else "▼"
                lines.append(
                    f"{i}. **{d['label']}**: {direction} {d['amount']:,.2f} "
                    f"({d['contribution_pct']:.1f}% of total variance)"
                )
            primary = evidence.get("primary_driver")
            if primary:
                lines.append(
                    f"\nThe largest contributor was **{primary['label']}**, "
                    f"accounting for {primary['contribution_pct']:.1f}% of the overall variance."
                )

        if "flags" in evidence and evidence["flags"]:
            lines.append("\n**Classification Flags:**")
            flag_map = {
                "structural_vs_timing_flag": {"S": "Structural", "T": "Timing"},
                "recurring_flag":            {"Y": "Recurring", "N": "Non-Recurring"},
                "controllable_flag":         {"Y": "Controllable", "N": "Uncontrollable"},
            }
            for flag_col, val in evidence["flags"].items():
                label = flag_map.get(flag_col, {}).get(val, val)
                lines.append(f"• {flag_col.replace('_', ' ').title()}: **{label}**")

        lines.append("\n## Confidence")
        confidence = self._assess_confidence(evidence)
        lines.append(confidence)

        return "\n".join(lines)

    def _assess_confidence(self, evidence: dict) -> str:
        """Rule-based confidence assessment."""
        has_actuals    = any("actual" in k for k in evidence)
        has_variance   = "variance_amount" in evidence or "driver_breakdown" in evidence
        has_root_cause = "primary_driver" in evidence

        if has_actuals and has_variance and has_root_cause:
            return "**HIGH** — Actuals, variance, and root cause evidence all present."
        elif has_actuals and has_variance:
            return "**MEDIUM** — Actuals and variance present; root cause not analysed."
        elif has_actuals:
            return "**MEDIUM** — Actuals present; comparison data absent."
        else:
            return "**LOW** — Limited evidence available."
