"""
Decision Envelope — Structured Judgment Before Risky Actions
=============================================================
Workers must declare purpose, expectations, and risk assessment
before executing risky tool calls. This embeds judgment into
the action itself without wasting extra LLM turns.

The envelope is parsed from the assistant's text output and logged
alongside the tool call for downstream diagnosis.
"""

import re
import json
from dataclasses import dataclass, asdict
from typing import Literal


@dataclass
class DecisionEnvelope:
    """Structured judgment emitted by worker before a risky action."""
    purpose: str = ""
    expected_evidence: str = ""
    value_type: str = "setup"       # artifact|verification|setup|exploration
    primary_risk: str = "none"      # env|dependency|timeout|assumption|none
    stronger_alternative: str = ""  # what could be done instead (or "none")

    def to_dict(self) -> dict:
        return asdict(self)

    def is_evidence_producing(self) -> bool:
        return self.value_type in ("artifact", "verification")


# Tools that require a decision envelope
RISKY_TOOLS = {"run_python_code", "pip_install"}

# Prompt fragment to inject into workers (keep SHORT)
DECISION_PROMPT = """## Decision Protocol
Before calling run_python_code or pip_install, include this JSON block in your message:
```json
{"decision": {"purpose": "what this action achieves", "expected": "what output/files you expect", "risk": "env|dependency|timeout|assumption|none"}}
```
This takes 1 line. Do NOT skip it. Do NOT write paragraphs — just the JSON block, then the tool call."""


def parse_envelope(text: str) -> DecisionEnvelope | None:
    """Parse a DecisionEnvelope from assistant text. Returns None if not found."""
    if not text:
        return None

    # Try to find {"decision": {...}} pattern
    match = re.search(r'\{"decision"\s*:\s*\{[^}]+\}\}', text)
    if match:
        try:
            data = json.loads(match.group())
            d = data.get("decision", {})
            return DecisionEnvelope(
                purpose=d.get("purpose", ""),
                expected_evidence=d.get("expected", d.get("expected_evidence", "")),
                value_type=_classify_value(d.get("purpose", "")),
                primary_risk=d.get("risk", d.get("primary_risk", "none")),
                stronger_alternative=d.get("alternative", ""),
            )
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: try simpler patterns like [DECISION: purpose=..., risk=...]
    match = re.search(r'\[DECISION:?\s*(.+?)\]', text, re.IGNORECASE)
    if match:
        content = match.group(1)
        purpose = _extract_field(content, "purpose")
        return DecisionEnvelope(
            purpose=purpose,
            expected_evidence=_extract_field(content, "expected"),
            value_type=_classify_value(purpose),
            primary_risk=_extract_field(content, "risk") or "none",
        )

    return None


def _extract_field(text: str, field: str) -> str:
    match = re.search(rf'{field}\s*=\s*["\']?([^,"\]]+)', text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _classify_value(purpose: str) -> str:
    """Auto-classify value type from purpose description."""
    p = purpose.lower()
    if any(w in p for w in ["train", "benchmark", "run experiment", "execute", "test"]):
        return "artifact"
    if any(w in p for w in ["verify", "check", "validate", "compare", "reproduce"]):
        return "verification"
    if any(w in p for w in ["install", "setup", "configure", "prepare"]):
        return "setup"
    return "exploration"


class CoastingDetector:
    """Detect when a worker is doing easy work without producing evidence."""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self._non_evidence_streak = 0

    def record(self, envelope: DecisionEnvelope | None):
        """Record a decision. Returns warning message if coasting detected."""
        if envelope and envelope.is_evidence_producing():
            self._non_evidence_streak = 0
            return None
        else:
            self._non_evidence_streak += 1
            if self._non_evidence_streak >= self.threshold:
                self._non_evidence_streak = 0
                return ("WARNING: Recent actions produced no evidence. "
                        "Next step must create or verify an artifact.")
        return None
