"""
memory.py — Conversation History & Memory Manager

Implements sliding window memory for the Analytics Agent.
Tracks past Q&A turns and provides context to the Explanation Tool.

Config is read from agent_config.yml → memory section:
  - max_turns: 10
  - include_evidence: true
  - summarise_after_turns: 10
  - persistent: false
  - history_file: data/session_history.json
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import MEMORY_CONFIG, BASE_DIR

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Sliding window conversation history manager.

    Each turn stores:
    - turn_id
    - user_query
    - intent_detected
    - kpi_name
    - filters_applied
    - tools_used
    - key_evidence  (numbers only — no DataFrames)
    - agent_response_summary
    - timestamp
    """

    def __init__(self):
        self.max_turns          = MEMORY_CONFIG.get("max_turns", 10)
        self.include_evidence   = MEMORY_CONFIG.get("include_evidence", True)
        self.summarise_after    = MEMORY_CONFIG.get("summarise_after_turns", 10)
        self.persistent         = MEMORY_CONFIG.get("persistent", False)
        self.history_file       = BASE_DIR / MEMORY_CONFIG.get("history_file", "data/session_history.json")
        self._history: list[dict] = []
        self._turn_counter      = 0

        if self.persistent and self.history_file.exists():
            self._load_from_disk()
            logger.info(f"Memory: loaded {len(self._history)} past turns from disk")
        else:
            logger.info(f"Memory: starting fresh session (max_turns={self.max_turns})")

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_turn(
        self,
        user_query: str,
        intent: str,
        kpi_name: str,
        filters: dict,
        tools_used: list[str],
        evidence: dict,
        response_summary: str,
    ) -> int:
        """
        Record a completed Q&A turn.

        Returns the turn_id assigned.
        """
        self._turn_counter += 1

        # Strip non-serialisable items from evidence (e.g. DataFrames)
        safe_evidence = self._sanitise_evidence(evidence)

        turn = {
            "turn_id":               self._turn_counter,
            "user_query":            user_query,
            "intent_detected":       intent,
            "kpi_name":              kpi_name,
            "filters_applied":       filters,
            "tools_used":            tools_used,
            "key_evidence":          safe_evidence if self.include_evidence else {},
            "agent_response_summary":response_summary[:500],  # Truncate for context efficiency
            "timestamp":             datetime.now(timezone.utc).isoformat(),
        }

        self._history.append(turn)
        logger.info(f"Memory: turn {self._turn_counter} recorded — intent='{intent}', kpi='{kpi_name}'")

        # Sliding window — remove oldest if over limit
        if len(self._history) > self.max_turns:
            dropped = self._history.pop(0)
            logger.debug(f"Memory: dropped turn {dropped['turn_id']} (window full)")

        # Auto-summarise if threshold reached
        if len(self._history) >= self.summarise_after:
            self._summarise()

        if self.persistent:
            self._save_to_disk()

        return self._turn_counter

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_history(self) -> list[dict]:
        """Return full conversation history (within window)."""
        return self._history.copy()

    def get_context_for_llm(self) -> str:
        """
        Format history as a compact string for LLM context injection.
        Only includes fields relevant to the LLM narrator.
        """
        if not self._history:
            return "No prior conversation history."

        lines = ["=== Prior Conversation Context ==="]
        for turn in self._history[-5:]:   # Last 5 turns only for LLM context
            lines.append(
                f"\nTurn {turn['turn_id']} | {turn['timestamp'][:10]}"
                f"\nQ: {turn['user_query']}"
                f"\nIntent: {turn['intent_detected']} | KPI: {turn['kpi_name']}"
            )
            if turn.get("key_evidence"):
                ev = turn["key_evidence"]
                ev_str = ", ".join(
                    f"{k}={v:,.2f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in list(ev.items())[:6]
                )
                lines.append(f"Evidence: {ev_str}")
            lines.append(f"A: {turn['agent_response_summary'][:200]}...")

        return "\n".join(lines)

    def get_last_kpi(self) -> str | None:
        """Return the KPI from the most recent turn (for context carryover)."""
        return self._history[-1]["kpi_name"] if self._history else None

    def get_last_filters(self) -> dict:
        """Return filters from the most recent turn."""
        return self._history[-1]["filters_applied"] if self._history else {}

    def find_turns_by_kpi(self, kpi_name: str) -> list[dict]:
        """Find all past turns that analysed a specific KPI."""
        return [t for t in self._history if t["kpi_name"] == kpi_name]

    def find_turns_by_intent(self, intent: str) -> list[dict]:
        """Find all past turns with a specific intent."""
        return [t for t in self._history if t["intent_detected"] == intent]

    def clear(self):
        """Clear conversation history."""
        self._history = []
        self._turn_counter = 0
        if self.persistent and self.history_file.exists():
            self.history_file.unlink()
        logger.info("Memory: history cleared")

    def summary(self) -> dict:
        """Return a metadata summary of the current session."""
        return {
            "total_turns":   self._turn_counter,
            "active_turns":  len(self._history),
            "max_turns":     self.max_turns,
            "kpis_queried":  list({t["kpi_name"] for t in self._history}),
            "intents_used":  list({t["intent_detected"] for t in self._history}),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _sanitise_evidence(evidence: dict) -> dict:
        """Remove non-JSON-serialisable objects from evidence dict."""
        safe = {}
        for k, v in evidence.items():
            if isinstance(v, (str, int, float, bool, type(None))):
                safe[k] = v
            elif isinstance(v, dict):
                safe[k] = MemoryManager._sanitise_evidence(v)
            elif isinstance(v, list) and all(isinstance(i, (str, int, float)) for i in v):
                safe[k] = v
        return safe

    def _summarise(self):
        """Compress old history into a brief summary (placeholder for LLM summarisation)."""
        logger.info(f"Memory: compressing {len(self._history)} turns into summary")
        # Future enhancement: call Groq to summarise old turns into one summary entry
        # For now, just keep the sliding window approach

    def _save_to_disk(self):
        """Persist history to JSON file."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, "w") as f:
                json.dump({"turns": self._history, "turn_counter": self._turn_counter}, f, indent=2)
        except Exception as e:
            logger.warning(f"Memory: could not save to disk: {e}")

    def _load_from_disk(self):
        """Load persisted history from JSON file."""
        try:
            with open(self.history_file, "r") as f:
                data = json.load(f)
            self._history       = data.get("turns", [])
            self._turn_counter  = data.get("turn_counter", 0)
        except Exception as e:
            logger.warning(f"Memory: could not load from disk: {e}")
            self._history = []
