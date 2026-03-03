"""
Sanity Check Rules Engine
==========================
Rule-based validation for research outputs. No LLM calls.

Catches:
- Unrealistic metric values (PPL < 5 = memorization)
- Tiny datasets (10 sentences → meaningless results)
- Suspicious patterns (accuracy > 99.5% = data leakage)
"""

import re
from dataclasses import dataclass, field


@dataclass
class Violation:
    """A single sanity check violation."""
    rule: str
    severity: str  # "error" or "warning"
    message: str
    value: float = None
    expected_range: tuple = None


@dataclass
class SanityCheckResult:
    """Result of running all sanity checks on an output."""
    passed: bool
    violations: list[Violation] = field(default_factory=list)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "error"]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "warning"]

    def summary(self) -> str:
        if self.passed:
            return "All sanity checks passed"
        parts = []
        for v in self.violations:
            parts.append(f"[{v.severity.upper()}] {v.message}")
        return "\n".join(parts)


# ── Built-in metric ranges ──────────────────────────────────────────

METRIC_RANGES = {
    "perplexity": (5.0, 500.0),
    "ppl": (5.0, 500.0),
    "accuracy": (0.0, 100.0),
    "acc": (0.0, 100.0),
    "f1": (0.0, 1.0),
    "f1_score": (0.0, 1.0),
    "f1-score": (0.0, 1.0),
    "precision": (0.0, 1.0),
    "recall": (0.0, 1.0),
    "loss": (0.0, 100.0),
    "bleu": (0.0, 100.0),
    "rouge": (0.0, 1.0),
    "rouge-l": (0.0, 1.0),
    "rouge-1": (0.0, 1.0),
    "auc": (0.0, 1.0),
    "mse": (0.0, 1000.0),
    "rmse": (0.0, 1000.0),
    "mae": (0.0, 1000.0),
}

# ── Suspicion thresholds (values that suggest problems) ─────────────

SUSPICION_THRESHOLDS = {
    "perplexity": {"below": 5.0, "message": "Perplexity < 5 suggests memorization or data leakage"},
    "ppl": {"below": 5.0, "message": "PPL < 5 suggests memorization or data leakage"},
    "accuracy": {"above": 99.5, "message": "Accuracy > 99.5% suggests data leakage or trivial task"},
    "acc": {"above": 99.5, "message": "Accuracy > 99.5% suggests data leakage or trivial task"},
    "loss": {"below": 0.001, "message": "Loss < 0.001 suggests overfitting or degenerate training"},
    "f1": {"above": 0.995, "message": "F1 > 0.995 suggests data leakage or trivial task"},
    "f1_score": {"above": 0.995, "message": "F1 > 0.995 suggests data leakage or trivial task"},
}

# ── Dataset size minimums ───────────────────────────────────────────

DATASET_MINIMUMS = {
    "language_model": 1000,
    "lm": 1000,
    "perplexity": 1000,
    "classification": 500,
    "sentiment": 500,
    "translation": 1000,
    "summarization": 500,
    "qa": 500,
    "question_answering": 500,
    "ner": 500,
    "generation": 1000,
}


class SanityChecker:
    """Rule-based validation engine for research outputs."""

    def __init__(self, extra_rules: dict = None):
        self.metric_ranges = dict(METRIC_RANGES)
        self.suspicion_thresholds = dict(SUSPICION_THRESHOLDS)
        self.dataset_minimums = dict(DATASET_MINIMUMS)
        if extra_rules:
            self.metric_ranges.update(extra_rules.get("metric_ranges", {}))
            self.suspicion_thresholds.update(extra_rules.get("suspicion_thresholds", {}))
            self.dataset_minimums.update(extra_rules.get("dataset_minimums", {}))

    def check_output(self, output: str, task_description: str = "") -> SanityCheckResult:
        """Run all sanity checks on a text output.

        Args:
            output: The text output from a worker (may contain metrics, dataset info, etc.)
            task_description: Optional task context to determine relevant checks

        Returns:
            SanityCheckResult with pass/fail and list of violations
        """
        violations = []
        violations.extend(self._check_metrics(output))
        violations.extend(self._check_dataset_size(output, task_description))
        violations.extend(self._check_suspicious_patterns(output))

        has_errors = any(v.severity == "error" for v in violations)
        return SanityCheckResult(passed=not has_errors, violations=violations)

    def _check_metrics(self, output: str) -> list[Violation]:
        """Check that reported metrics are within valid ranges."""
        violations = []
        # Extract metric-value pairs from text
        # Patterns: "metric: 85.3", "metric = 0.42", "metric 85.3%", "metric ~15.6"
        patterns = [
            r'(\w[\w\-]*)\s*[:=]\s*~?\s*(\d+\.?\d*)\s*%?',
            r'(\w[\w\-]*)\s+(?:of|is|was|reached|achieved)\s+~?\s*(\d+\.?\d*)\s*%?',
        ]

        found_metrics = []
        for pattern in patterns:
            for match in re.finditer(pattern, output, re.IGNORECASE):
                name = match.group(1).lower().strip()
                try:
                    value = float(match.group(2))
                except ValueError:
                    continue
                found_metrics.append((name, value))

        for name, value in found_metrics:
            # Check valid range
            if name in self.metric_ranges:
                lo, hi = self.metric_ranges[name]
                if value < lo or value > hi:
                    violations.append(Violation(
                        rule="metric_range",
                        severity="error",
                        message=f"Metric '{name}' = {value} is outside valid range [{lo}, {hi}]",
                        value=value,
                        expected_range=(lo, hi),
                    ))

            # Check suspicion thresholds
            if name in self.suspicion_thresholds:
                thresh = self.suspicion_thresholds[name]
                if "below" in thresh and value < thresh["below"]:
                    violations.append(Violation(
                        rule="suspicious_value",
                        severity="warning",
                        message=f"{thresh['message']} (got {name} = {value})",
                        value=value,
                    ))
                if "above" in thresh and value > thresh["above"]:
                    violations.append(Violation(
                        rule="suspicious_value",
                        severity="warning",
                        message=f"{thresh['message']} (got {name} = {value})",
                        value=value,
                    ))

        return violations

    def _check_dataset_size(self, output: str, task_description: str = "") -> list[Violation]:
        """Check for tiny datasets that produce meaningless results."""
        violations = []
        combined = f"{task_description} {output}".lower()

        # Extract dataset sizes: "N samples", "N sentences", "N examples", "dataset size: N"
        size_patterns = [
            r'(\d+)\s+(?:samples?|sentences?|examples?|data\s*points?|texts?|documents?|rows?|instances?)',
            r'(?:dataset|data|corpus)\s+(?:size|length|count)[:=\s]+(\d+)',
            r'(?:train|training)\s+(?:set|data|size)[:=\s]+(\d+)',
            r'len\([^)]*\)\s*[:=]\s*(\d+)',
        ]

        found_sizes = []
        for pattern in size_patterns:
            for match in re.finditer(pattern, combined, re.IGNORECASE):
                try:
                    size = int(match.group(1))
                    found_sizes.append(size)
                except ValueError:
                    continue

        if not found_sizes:
            return violations

        min_size = min(found_sizes)

        # Determine task type for minimum threshold
        task_type = None
        for key in self.dataset_minimums:
            if key in combined:
                task_type = key
                break

        if task_type:
            minimum = self.dataset_minimums[task_type]
            if min_size < minimum:
                violations.append(Violation(
                    rule="dataset_too_small",
                    severity="warning",
                    message=f"Dataset size {min_size} is below minimum {minimum} for {task_type} tasks. Results may be meaningless.",
                    value=float(min_size),
                    expected_range=(float(minimum), float("inf")),
                ))
        elif min_size < 50:
            # Generic check: any dataset under 50 is suspicious
            violations.append(Violation(
                rule="dataset_too_small",
                severity="warning",
                message=f"Dataset size {min_size} is very small. Results may not generalize.",
                value=float(min_size),
            ))

        return violations

    def _check_suspicious_patterns(self, output: str) -> list[Violation]:
        """Check for patterns that suggest fabricated or meaningless results."""
        violations = []

        # Perfect results on non-trivial tasks
        perfect_patterns = [
            (r'accuracy\s*[:=]\s*100\.?0*\s*%', "Perfect 100% accuracy is suspicious — check for data leakage or trivial task"),
            (r'loss\s*[:=]\s*0\.0+\b', "Zero loss suggests degenerate training"),
            (r'perplexity\s*[:=]\s*1\.0+\b', "Perplexity = 1.0 means perfect prediction — likely memorization"),
        ]

        for pattern, message in perfect_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                violations.append(Violation(
                    rule="perfect_result",
                    severity="warning",
                    message=message,
                ))

        # Check for "~" approximation markers on critical metrics (suggests literature reference, not measurement)
        approx_pattern = r'(?:perplexity|ppl|accuracy|loss)\s*[:=]?\s*~\s*\d+\.?\d*'
        approx_matches = re.findall(approx_pattern, output, re.IGNORECASE)
        if len(approx_matches) >= 2:
            violations.append(Violation(
                rule="approximate_values",
                severity="warning",
                message=f"Multiple approximate values found ({len(approx_matches)}x '~'). These may be literature references, not actual measurements.",
            ))

        return violations
