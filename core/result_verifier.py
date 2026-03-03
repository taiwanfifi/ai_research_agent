"""
Result Verifier
================
Cross-checks numerical claims in LLM summaries against actual
run_python_code stdout. Catches fabricated or hallucinated numbers.

Usage:
    verifier = ResultVerifier()
    verifier.capture(cycle=1, worker="coder", stdout="Accuracy: 85.3%\\nLoss: 0.42")
    result = verifier.verify_output("We achieved 85.3% accuracy with loss 0.42")
    # result.score = 1.0, all claims verified
"""

import re
from dataclasses import dataclass, field


@dataclass
class Claim:
    """A numerical claim extracted from text."""
    label: str
    value: float
    raw_text: str
    status: str = "unverified"  # verified, unverified, contradicted


@dataclass
class CapturedNumber:
    """A labeled number from stdout."""
    label: str
    value: float
    cycle: int
    worker: str


@dataclass
class VerificationResult:
    """Result of verifying claims against captured stdout."""
    score: float  # 0-1, fraction of claims verified
    claims: list[Claim] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def verified(self) -> list[Claim]:
        return [c for c in self.claims if c.status == "verified"]

    @property
    def unverified(self) -> list[Claim]:
        return [c for c in self.claims if c.status == "unverified"]

    @property
    def contradicted(self) -> list[Claim]:
        return [c for c in self.claims if c.status == "contradicted"]

    def summary(self) -> str:
        parts = [f"Verification score: {self.score:.0%} ({len(self.verified)}/{len(self.claims)} claims verified)"]
        if self.contradicted:
            parts.append(f"CONTRADICTED: {', '.join(c.raw_text for c in self.contradicted)}")
        if self.warnings:
            parts.extend(f"WARNING: {w}" for w in self.warnings)
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "claims": [
                {"label": c.label, "value": c.value, "raw_text": c.raw_text, "status": c.status}
                for c in self.claims
            ],
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VerificationResult":
        claims = [
            Claim(label=c["label"], value=c["value"], raw_text=c["raw_text"], status=c["status"])
            for c in d.get("claims", [])
        ]
        return cls(score=d.get("score", 0), claims=claims, warnings=d.get("warnings", []))


class ResultVerifier:
    """Cross-checks LLM claims against actual stdout from code execution."""

    def __init__(self):
        self._captured: list[CapturedNumber] = []

    def capture(self, cycle: int, worker: str, stdout: str):
        """Parse and store labeled numbers from code execution stdout.

        Args:
            cycle: Current supervisor cycle
            worker: Worker that produced this output
            stdout: Raw stdout from run_python_code
        """
        numbers = self._extract_labeled_numbers(stdout)
        for label, value in numbers:
            self._captured.append(CapturedNumber(
                label=label, value=value, cycle=cycle, worker=worker,
            ))

    def verify_output(self, text: str) -> VerificationResult:
        """Verify numerical claims in text against captured stdout.

        Three verification strategies:
        1. Exact label + value match → verified
        2. Value exists in stdout (any label) → verified (weak)
        3. Label matches but value differs → contradicted

        Args:
            text: LLM-generated text that may contain numerical claims

        Returns:
            VerificationResult with score and per-claim status
        """
        claims = self._extract_claims(text)
        if not claims:
            return VerificationResult(score=1.0, claims=[], warnings=[])

        if not self._captured:
            return VerificationResult(
                score=0.0, claims=claims,
                warnings=["No stdout captured — cannot verify any claims"],
            )

        # Build lookup structures
        all_values = set(c.value for c in self._captured)
        label_to_values = {}
        for c in self._captured:
            label_to_values.setdefault(c.label.lower(), set()).add(c.value)

        warnings = []
        for claim in claims:
            claim_label = claim.label.lower()
            claim_value = claim.value

            # Strategy 1: Exact label + value match
            matched = False
            for cap_label, cap_values in label_to_values.items():
                if self._labels_match(claim_label, cap_label):
                    if any(self._values_match(claim_value, v) for v in cap_values):
                        claim.status = "verified"
                        matched = True
                        break
                    else:
                        # Label matches but value differs → contradiction
                        claim.status = "contradicted"
                        closest = min(cap_values, key=lambda v: abs(v - claim_value))
                        warnings.append(
                            f"Claim '{claim.raw_text}' contradicts stdout "
                            f"(found {cap_label} = {closest})"
                        )
                        matched = True
                        break

            if matched:
                continue

            # Strategy 2: Value exists in stdout (any label)
            if any(self._values_match(claim_value, v) for v in all_values):
                claim.status = "verified"
                continue

            # No match found
            claim.status = "unverified"

        # Calculate score
        verified_count = sum(1 for c in claims if c.status == "verified")
        score = verified_count / len(claims) if claims else 1.0

        return VerificationResult(score=score, claims=claims, warnings=warnings)

    def get_all_captured(self) -> list[dict]:
        """Return all captured numbers for checkpoint serialization."""
        return [
            {"label": c.label, "value": c.value, "cycle": c.cycle, "worker": c.worker}
            for c in self._captured
        ]

    def restore_captured(self, data: list[dict]):
        """Restore captured numbers from checkpoint data."""
        self._captured = [
            CapturedNumber(
                label=d["label"], value=d["value"],
                cycle=d["cycle"], worker=d["worker"],
            )
            for d in data
        ]

    def to_dict(self) -> dict:
        return {"captured": self.get_all_captured()}

    @classmethod
    def from_dict(cls, d: dict) -> "ResultVerifier":
        v = cls()
        v.restore_captured(d.get("captured", []))
        return v

    # ── Extraction helpers ─────────────────────────────────────────

    def _extract_labeled_numbers(self, stdout: str) -> list[tuple[str, float]]:
        """Extract labeled numbers from stdout.

        Handles patterns like:
        - "Accuracy: 85.3%"
        - "loss = 0.42"
        - "| Method | 85.3 |" (table rows)
        - "F1-score: 0.87"
        - "Perplexity = 15.6"
        - "Epoch 10: loss=0.342, acc=91.2%"
        """
        results = []

        # Pattern: "label: value" or "label = value" or "label value%"
        patterns = [
            r'([\w\-]+)\s*[:=]\s*(\d+\.?\d*)\s*%?',
            r'([\w\-]+)\s+(\d+\.\d+)\s*%?',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, stdout):
                label = match.group(1).strip()
                try:
                    value = float(match.group(2))
                    # Skip epoch/step/iteration numbers
                    if label.lower() in ("epoch", "step", "iteration", "batch", "iter", "len", "size", "count"):
                        continue
                    results.append((label, value))
                except ValueError:
                    continue

        # Pattern: table rows "| label | value |"
        table_pattern = r'\|\s*([\w\s\-]+?)\s*\|\s*(\d+\.?\d*)\s*%?\s*\|'
        for match in re.finditer(table_pattern, stdout):
            label = match.group(1).strip()
            try:
                value = float(match.group(2))
                if label and not label.replace(" ", "").replace("-", "").isdigit():
                    results.append((label, value))
            except ValueError:
                continue

        return results

    def _extract_claims(self, text: str) -> list[Claim]:
        """Extract numerical claims from LLM text.

        Handles patterns like:
        - "achieved 85.3% accuracy"
        - "PPL ~15.6"
        - "loss of 0.42"
        - "accuracy: 85.3%"
        - "F1 score 0.87"
        """
        claims = []
        seen = set()

        patterns = [
            # "metric: value" or "metric = value"
            (r'([\w\-]+)\s*[:=]\s*~?\s*(\d+\.?\d*)\s*%?', lambda m: (m.group(1), m.group(2))),
            # "achieved X% metric"
            (r'achieved?\s+~?\s*(\d+\.?\d*)\s*%?\s*([\w\-]+)', lambda m: (m.group(2), m.group(1))),
            # "metric of X"
            (r'([\w\-]+)\s+(?:of|is|was|reached)\s+~?\s*(\d+\.?\d*)', lambda m: (m.group(1), m.group(2))),
            # "X% metric"
            (r'(\d+\.?\d*)\s*%\s+([\w\-]+)', lambda m: (m.group(2), m.group(1))),
        ]

        for pattern, extractor in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                label, value_str = extractor(match)
                try:
                    value = float(value_str)
                except ValueError:
                    continue

                label = label.strip()
                if label.lower() in ("epoch", "step", "iteration", "batch", "iter", "the", "a", "an", "and", "with"):
                    continue

                key = (label.lower(), round(value, 4))
                if key not in seen:
                    seen.add(key)
                    claims.append(Claim(
                        label=label,
                        value=value,
                        raw_text=match.group(0).strip(),
                    ))

        return claims

    @staticmethod
    def _labels_match(label1: str, label2: str) -> bool:
        """Check if two labels refer to the same metric."""
        l1 = label1.lower().replace("-", "").replace("_", "").replace(" ", "")
        l2 = label2.lower().replace("-", "").replace("_", "").replace(" ", "")

        if l1 == l2:
            return True

        # Common aliases
        aliases = {
            "acc": "accuracy",
            "ppl": "perplexity",
            "f1score": "f1",
            "f1": "f1score",
            "lr": "learningrate",
        }
        return aliases.get(l1, l1) == aliases.get(l2, l2)

    @staticmethod
    def _values_match(v1: float, v2: float, tolerance: float = 0.01) -> bool:
        """Check if two values are close enough (within 1% or 0.01 absolute)."""
        if v1 == v2:
            return True
        if abs(v1 - v2) <= tolerance:
            return True
        if v2 != 0 and abs(v1 - v2) / abs(v2) <= tolerance:
            return True
        return False
