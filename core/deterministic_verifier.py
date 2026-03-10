"""
Deterministic Verifier — Statistical Rigor Without LLM
=======================================================
Four-layer verification pipeline that checks research quality
using pure computation. No LLM calls needed.

Layer 1: Sanity Check (results exist, have metrics)
Layer 2: Training Curves (convergence, overfitting)
Layer 3: Statistical Rigor (seeds, p-values, effect size, multi-comparison)
Layer 4: Reserved for claim verification (future)

From: Demšar (JMLR 2006), Cohen (1988), Holm (1979)
"""

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VerificationResult:
    """Result from deterministic verification."""
    total_score: float           # 0-10
    breakdown: dict              # per-layer scores
    issues: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


class DeterministicVerifier:
    """Four-layer deterministic verification pipeline."""

    def verify(self, workspace: str) -> VerificationResult:
        """Run all verification layers on a mission workspace.

        Returns VerificationResult with score 0-10.
        """
        # Layer 1: Sanity (0-2)
        l1_score, l1_issues, json_data = self._sanity_check(workspace)
        if l1_score == 0:
            return VerificationResult(
                total_score=0, breakdown={"sanity": 0, "curves": 0, "stats": 0},
                issues=l1_issues,
            )

        # Layer 2: Training Curves (0-3)
        l2_score, l2_issues = self._verify_training_curves(json_data)

        # Layer 3: Statistical Rigor (0-5)
        l3_score, l3_issues = self._verify_statistical_rigor(json_data)

        # Layer 4: Data Sanity — catches fatal flaws (penalty up to -5)
        l4_penalty, l4_issues = self._verify_data_sanity(json_data)

        total = l1_score + l2_score + l3_score + l4_penalty
        all_issues = l1_issues + l2_issues + l3_issues + l4_issues

        return VerificationResult(
            total_score=max(0.0, min(10.0, total)),
            breakdown={
                "sanity": l1_score, "curves": l2_score,
                "stats": l3_score, "data_sanity": l4_penalty,
            },
            issues=all_issues,
            details={"json_files": len(json_data)},
        )

    # ══════════════════════════════════════════════════════════════
    #  Layer 1: Sanity Check (0-2)
    # ══════════════════════════════════════════════════════════════

    def _sanity_check(self, workspace: str) -> tuple[float, list[str], list[dict]]:
        """Check: results exist and contain metrics.

        Returns (score, issues, list_of_parsed_json_dicts).
        """
        issues = []
        json_data = []

        if not os.path.isdir(workspace):
            return 0.0, ["NO_WORKSPACE: workspace directory not found"], []

        # Find JSON result files
        json_files = [f for f in glob.glob(os.path.join(workspace, "*.json"))
                      if os.path.basename(f) not in
                      ("execution_log.json", "mission_score.json", "dataset_info.json")]

        if not json_files:
            return 0.0, ["NO_OUTPUT: No JSON result files found"], []

        score = 1.0  # Have output files

        # Check for metrics
        metric_keywords = {
            "accuracy", "loss", "f1", "auc", "mse", "rmse", "score",
            "precision", "recall", "perplexity", "bleu", "rouge",
            "mean_accuracy", "test_accuracy", "train_loss",
            "mean", "std", "p_value", "cohens_d", "t_statistic",
            "per_seed", "epoch", "test_acc", "train_acc",
        }
        metrics_found = False
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)
                json_data.append(data)
                data_str = json.dumps(data).lower()
                if any(k in data_str for k in metric_keywords):
                    metrics_found = True
                # Also detect per-seed array format: {"42": [...], "123": [...]}
                if isinstance(data, dict) and not metrics_found:
                    seed_like_keys = [k for k in data.keys() if k.isdigit()]
                    if len(seed_like_keys) >= 3:
                        vals = [v for v in data.values() if isinstance(v, list) and len(v) > 0]
                        if vals:
                            metrics_found = True
            except Exception:
                pass

        if not metrics_found:
            issues.append("NO_METRICS: Result files contain no recognizable metrics")
            return score, issues, json_data

        score += 1.0  # Have metrics
        return score, issues, json_data

    # ══════════════════════════════════════════════════════════════
    #  Layer 2: Training Curves (0-3)
    # ══════════════════════════════════════════════════════════════

    def _verify_training_curves(self, json_data: list[dict]) -> tuple[float, list[str]]:
        """Verify training convergence and check for overfitting.

        Looks for train_loss arrays in JSON data.
        """
        score = 0.0
        issues = []

        # Find training loss sequences
        train_losses = self._extract_loss_sequence(json_data, "train_loss")
        if not train_losses:
            train_losses = self._extract_loss_sequence(json_data, "loss")
        if not train_losses:
            # Try epoch_means (accuracy per epoch) — treat as proxy for progress
            for data in json_data:
                if isinstance(data, dict) and "epoch_means" in data:
                    em = data["epoch_means"]
                    if isinstance(em, dict):
                        for vals in em.values():
                            if isinstance(vals, list) and len(vals) >= 2:
                                train_losses = vals
                                break
                    break
        if not train_losses:
            # Try per-seed epoch arrays: {"42": [val1, val2, ...]}
            train_losses = self._extract_epoch_means_from_seeds(json_data)
        if not train_losses:
            # No loss curves — give partial credit if final metrics exist
            if self._has_final_metrics(json_data):
                return 1.0, ["NO_TRAIN_CURVES: only final metrics, no per-epoch losses"]
            return 0.0, ["NO_TRAIN_LOSS: no training loss data found"]

        losses = train_losses

        # Detect if this is an accuracy curve (increasing = good) or loss curve (decreasing = good)
        is_accuracy = (losses[-1] > losses[0]) if len(losses) >= 2 else False
        # Heuristic: if values are between 0-100 and increasing, likely accuracy
        if is_accuracy and max(losses) <= 100:
            is_accuracy = True

        # 1. Overall trend (+1)
        n = len(losses)
        if n >= 5:
            first_20 = np.mean(losses[:max(1, n // 5)])
            last_20 = np.mean(losses[-max(1, n // 5):])
            if is_accuracy:
                improving = last_20 > first_20  # accuracy increases
            else:
                improving = last_20 < first_20  # loss decreases
            if improving:
                score += 1.0
            else:
                issues.append(f"NO_CONVERGENCE: first_20%={first_20:.4f}, last_20%={last_20:.4f}")
        elif n >= 2:
            if is_accuracy:
                improving = losses[-1] > losses[0]
            else:
                improving = losses[-1] < losses[0]
            if improving:
                score += 1.0
            else:
                issues.append(f"NO_CONVERGENCE: val[0]={losses[0]:.4f}, val[-1]={losses[-1]:.4f}")

        # 2. Convergence: last 10% relative change < 5% (+1)
        if n >= 10:
            last_10 = losses[-max(1, n // 10):]
            if len(last_10) > 1 and abs(last_10[0]) > 1e-8:
                relative_change = abs(last_10[-1] - last_10[0]) / abs(last_10[0])
                if relative_change < 0.05:
                    score += 1.0
                else:
                    issues.append(f"NOT_CONVERGED: last 10% relative change = {relative_change:.3f}")
        elif n >= 2:
            # Short training — just check it didn't diverge
            if losses[-1] <= losses[0] * 1.1:
                score += 0.5

        # 3. Overfitting check (+1)
        val_losses = self._extract_loss_sequence(json_data, "test_loss")
        if not val_losses:
            val_losses = self._extract_loss_sequence(json_data, "val_loss")

        if val_losses and len(val_losses) == len(losses):
            train_final = losses[-1]
            val_final = val_losses[-1]
            if abs(train_final) > 1e-8:
                gap = (val_final - train_final) / abs(train_final)
                if gap < 0.15:
                    score += 1.0
                elif gap < 0.25:
                    score += 0.5
                    issues.append(f"MILD_OVERFIT: gap={gap:.1%}")
                elif gap < 0.50:
                    issues.append(f"MODERATE_OVERFIT: gap={gap:.1%}")
                else:
                    issues.append(f"SEVERE_OVERFIT: gap={gap:.1%}")
        else:
            # No val loss — give partial credit
            if score > 0:
                score += 0.5
                issues.append("NO_VAL_LOSS: cannot check overfitting")

        return min(3.0, score), issues

    def _extract_loss_sequence(self, json_data: list[dict], key: str) -> list[float]:
        """Extract a loss sequence from JSON data."""
        for data in json_data:
            if not isinstance(data, dict):
                continue
            # Direct key
            if key in data and isinstance(data[key], list):
                vals = data[key]
                if vals and all(isinstance(v, (int, float)) for v in vals):
                    return [float(v) for v in vals]
            # Nested in per-epoch or history
            for container_key in ("history", "training_history", "per_epoch", "epochs"):
                if container_key in data and isinstance(data[container_key], (list, dict)):
                    container = data[container_key]
                    if isinstance(container, list):
                        vals = [e.get(key) for e in container
                                if isinstance(e, dict) and key in e]
                        if vals and all(isinstance(v, (int, float)) for v in vals):
                            return [float(v) for v in vals]
                    elif isinstance(container, dict) and key in container:
                        vals = container[key]
                        if isinstance(vals, list):
                            return [float(v) for v in vals if isinstance(v, (int, float))]
        return []

    def _extract_epoch_means_from_seeds(self, json_data: list[dict]) -> list[float]:
        """Extract averaged epoch progression from per-seed data.

        Handles format: {"42": [acc_e1, acc_e2, ...], "123": [...]}
        Returns averaged values across seeds for each epoch.
        """
        for data in json_data:
            if not isinstance(data, dict):
                continue
            # Check for per-seed epoch arrays (keys are seed numbers)
            seed_keys = [k for k in data.keys() if k.isdigit()]
            if len(seed_keys) < 3:
                continue
            epoch_arrays = []
            for sk in seed_keys:
                v = data[sk]
                if isinstance(v, list) and len(v) >= 2 and all(isinstance(x, (int, float)) for x in v):
                    epoch_arrays.append(v)
            if len(epoch_arrays) >= 3:
                # Average across seeds per epoch
                n_epochs = min(len(a) for a in epoch_arrays)
                means = []
                for e in range(n_epochs):
                    vals = [a[e] for a in epoch_arrays]
                    means.append(float(np.mean(vals)))
                return means
        return []

    def _has_final_metrics(self, json_data: list[dict]) -> bool:
        """Check if any JSON has final accuracy/loss metrics."""
        for data in json_data:
            if not isinstance(data, dict):
                continue
            data_str = json.dumps(data).lower()
            if any(k in data_str for k in (
                "test_accuracy", "mean_accuracy", "final_loss", "mean", "std",
                "p_value", "per_seed", "epoch_means", "best_method",
            )):
                return True
            # Per-seed array format: {"42": [...], "123": [...]}
            seed_like = [k for k in data.keys() if k.isdigit()]
            if len(seed_like) >= 3:
                return True
        return False

    # ══════════════════════════════════════════════════════════════
    #  Layer 3: Statistical Rigor (0-5)
    # ══════════════════════════════════════════════════════════════

    def _verify_statistical_rigor(self, json_data: list[dict]) -> tuple[float, list[str]]:
        """Check multi-seed, statistical tests, effect size, multi-comparison."""
        score = 0.0
        issues = []

        all_flat = {}
        for data in json_data:
            if isinstance(data, dict):
                all_flat.update(self._flatten_keys(data))

        # 1. Multi-seed? (+1.5)
        seed_score, seed_issues = self._check_seeds(all_flat, json_data)
        score += seed_score
        issues.extend(seed_issues)

        # 2. Statistical test? (+1.5)
        stat_score, stat_issues = self._check_statistical_test(all_flat)
        score += stat_score
        issues.extend(stat_issues)

        # 3. Effect size? (+1)
        effect_score, effect_issues = self._check_effect_size(all_flat)
        score += effect_score
        issues.extend(effect_issues)

        # 4. Multi-comparison correction? (+1)
        multi_score, multi_issues = self._check_multi_comparison(all_flat, json_data)
        score += multi_score
        issues.extend(multi_issues)

        return min(5.0, score), issues

    def _check_seeds(self, flat: dict, json_data: list[dict]) -> tuple[float, list[str]]:
        """Check for multi-seed experiments."""
        seed_keys = {"seeds", "seed", "random_seeds"}
        seed_count = 0

        for k, v in flat.items():
            k_lower = k.split(".")[-1].lower()
            if k_lower in seed_keys:
                if isinstance(v, list):
                    seed_count = max(seed_count, len(v))
                elif isinstance(v, (int, float)):
                    seed_count = max(seed_count, 1)

        # Also check for per_seed results
        per_seed_keys = {"per_seed_results", "per_seed", "seed_results"}
        for k, v in flat.items():
            k_lower = k.split(".")[-1].lower()
            if k_lower in per_seed_keys and isinstance(v, dict):
                seed_count = max(seed_count, len(v))

        # Count result files with seed in name
        seed_files = sum(1 for d in json_data
                         if isinstance(d, dict) and "seed" in json.dumps(d).lower())
        seed_count = max(seed_count, min(seed_files, 10))

        if seed_count >= 5:
            return 1.5, [f"GOOD_SEEDS: {seed_count} seeds (≥5, statistically adequate)"]
        elif seed_count >= 3:
            return 0.75, [f"FEW_SEEDS: {seed_count} seeds (need ≥5 for reliable stats)"]
        elif seed_count >= 1:
            return 0.25, [f"INSUFFICIENT_SEEDS: only {seed_count} seed(s) (need ≥5)"]
        else:
            return 0.0, ["NO_SEEDS: no multi-seed evidence found"]

    def _check_statistical_test(self, flat: dict) -> tuple[float, list[str]]:
        """Check for statistical tests (p-value, t-test, etc.)."""
        stat_keywords = {"p_value", "p-value", "t_statistic", "t_test",
                         "wilcoxon", "mann_whitney", "friedman", "anova"}

        has_test = False
        p_value = None

        for k, v in flat.items():
            k_lower = k.lower()
            if any(sk in k_lower for sk in stat_keywords):
                has_test = True
                if "p_value" in k_lower or "p-value" in k_lower:
                    if isinstance(v, (int, float)):
                        p_value = float(v)

        if not has_test:
            return 0.0, ["NO_STAT_TEST: no statistical test performed"]

        issues = []
        if p_value is not None:
            if p_value > 0.05:
                issues.append(f"NOT_SIGNIFICANT: p={p_value:.4f} > 0.05 (honest null result is OK)")
            elif p_value < 0.001:
                issues.append(f"HIGHLY_SIGNIFICANT: p={p_value:.6f}")
        return 1.5, issues

    def _check_effect_size(self, flat: dict) -> tuple[float, list[str]]:
        """Check for effect size reporting (Cohen's d, etc.)."""
        effect_keywords = {"cohens_d", "cohen_d", "effect_size", "eta_squared",
                           "glass_delta", "hedges_g"}

        effect_value = None
        for k, v in flat.items():
            k_lower = k.lower()
            if any(ek in k_lower for ek in effect_keywords):
                if isinstance(v, (int, float)):
                    effect_value = abs(float(v))

        if effect_value is None:
            return 0.0, ["NO_EFFECT_SIZE: effect size not reported"]

        if effect_value >= 0.8:
            return 1.0, [f"LARGE_EFFECT: d={effect_value:.3f}"]
        elif effect_value >= 0.5:
            return 1.0, [f"MEDIUM_EFFECT: d={effect_value:.3f}"]
        elif effect_value >= 0.2:
            return 0.5, [f"SMALL_EFFECT: d={effect_value:.3f}"]
        else:
            return 0.25, [f"TRIVIAL_EFFECT: d={effect_value:.3f} (< 0.2, practically meaningless)"]

    def _check_multi_comparison(self, flat: dict, json_data: list[dict]) -> tuple[float, list[str]]:
        """Check for multi-comparison correction when >2 methods compared."""
        # Count distinct methods/conditions
        method_count = self._count_methods(json_data)

        if method_count <= 2:
            # Only 2 methods — no correction needed
            return 1.0, []  # Full credit

        # >2 methods — need correction
        correction_keywords = {"holm", "bonferroni", "fdr", "benjamini",
                               "nemenyi", "corrected_p", "adjusted_p",
                               "holm_bonferroni", "multiple_comparison"}

        has_correction = False
        for k, v in flat.items():
            k_lower = k.lower()
            if any(ck in k_lower for ck in correction_keywords):
                has_correction = True

        # Also check string values
        all_text = json.dumps([d for d in json_data if isinstance(d, dict)]).lower()
        if any(ck in all_text for ck in correction_keywords):
            has_correction = True

        if has_correction:
            return 1.0, [f"MULTI_CORRECTED: {method_count} methods with correction applied"]

        # Count p-values — if many uncorrected, flag it
        p_count = sum(1 for k in flat if "p_value" in k.lower() or "p-value" in k.lower())
        if p_count > 1:
            return 0.0, [f"NO_MULTI_CORRECTION: {p_count} p-values from {method_count} methods, "
                         f"but no multiple comparison correction (Holm-Bonferroni recommended)"]
        return 0.5, [f"PARTIAL: {method_count} methods but only {p_count} p-value(s)"]

    def _count_methods(self, json_data: list[dict]) -> int:
        """Count distinct experimental conditions/methods."""
        methods = set()
        for data in json_data:
            if not isinstance(data, dict):
                continue
            # Common patterns for method names
            for key in ("method", "model", "dropout_rate", "condition",
                        "learning_rate", "optimizer", "architecture"):
                if key in data:
                    methods.add(str(data[key]))
            # Check for methods list
            if "methods" in data and isinstance(data["methods"], list):
                return len(data["methods"])
        # Count result files as proxy
        result_count = sum(1 for d in json_data if isinstance(d, dict)
                           and any(k in json.dumps(d).lower()
                                   for k in ("accuracy", "test_acc")))
        return max(len(methods), min(result_count, 10))

    # ══════════════════════════════════════════════════════════════
    #  Layer 4: Data Sanity — catches fatal experimental flaws
    #  Returns PENALTIES (negative or zero), not positive scores
    # ══════════════════════════════════════════════════════════════

    def _verify_data_sanity(self, json_data: list[dict]) -> tuple[float, list[str]]:
        """Catch fatal flaws that invalidate the entire experiment.

        Returns (penalty, issues) where penalty is ≤ 0.
        These are problems so serious that they should reduce the score
        regardless of how good everything else looks.

        Checks:
        1. Identical results across conditions (IV not manipulated)
        2. Absurd effect sizes (d > 5 = unfair comparison)
        3. Unreasonable accuracy values
        4. Missing baseline/control condition
        """
        penalty = 0.0
        issues = []

        # 1. Identical results across conditions (-3 to -5)
        dup_penalty, dup_issues = self._check_duplicate_results(json_data)
        penalty += dup_penalty
        issues.extend(dup_issues)

        # 2. Absurd effect sizes (-1 to -2)
        effect_penalty, effect_issues = self._check_absurd_effects(json_data)
        penalty += effect_penalty
        issues.extend(effect_issues)

        # 3. Unreasonable accuracy (-1)
        acc_penalty, acc_issues = self._check_reasonable_accuracy(json_data)
        penalty += acc_penalty
        issues.extend(acc_issues)

        return max(-5.0, penalty), issues

    def _check_duplicate_results(self, json_data: list[dict]) -> tuple[float, list[str]]:
        """CRITICAL: Detect when multiple conditions produce identical results.

        This means the independent variable was never actually manipulated.
        The experiment is fundamentally broken.
        """
        # Extract per-seed result arrays from each condition
        condition_results = {}  # condition_name -> sorted list of metric values

        for data in json_data:
            if not isinstance(data, dict):
                continue

            # Handle "methods" list inside a single JSON (common pattern)
            methods_list = data.get("methods", [])
            if isinstance(methods_list, list) and methods_list:
                for method in methods_list:
                    if not isinstance(method, dict):
                        continue
                    cond_name = method.get("name", method.get("method", ""))
                    if not cond_name:
                        continue
                    metrics = self._extract_condition_metrics(method)
                    if metrics and len(metrics) >= 2:
                        condition_results[cond_name] = tuple(sorted(metrics))
                continue  # Don't also process this file as a single condition

            # Single-condition JSON file
            condition = None
            for key in ("method", "model", "dropout_rate", "condition",
                        "dropout_type", "optimizer", "architecture", "warmup_type",
                        "name"):
                if key in data:
                    condition = f"{key}={data[key]}"
                    break

            if not condition:
                for key in data:
                    if "name" in key.lower() or "type" in key.lower():
                        if isinstance(data[key], str):
                            condition = f"{key}={data[key]}"
                            break

            if not condition:
                continue

            metrics = self._extract_condition_metrics(data)
            if metrics and len(metrics) >= 2:
                condition_results[condition] = tuple(sorted(metrics))

        # Compare all pairs of conditions
        if len(condition_results) < 2:
            return 0.0, []

        conditions = list(condition_results.keys())
        duplicate_groups = []
        seen = set()

        for i in range(len(conditions)):
            if conditions[i] in seen:
                continue
            group = [conditions[i]]
            for j in range(i + 1, len(conditions)):
                if conditions[j] in seen:
                    continue
                if condition_results[conditions[i]] == condition_results[conditions[j]]:
                    group.append(conditions[j])
                    seen.add(conditions[j])
            if len(group) > 1:
                duplicate_groups.append(group)
                seen.add(conditions[i])

        if not duplicate_groups:
            return 0.0, []

        # Calculate penalty based on severity
        total_conditions = len(conditions)
        total_duplicated = sum(len(g) for g in duplicate_groups)
        dup_ratio = total_duplicated / total_conditions

        issues = []
        for group in duplicate_groups:
            vals = condition_results[group[0]]
            issues.append(
                f"FATAL_DUPLICATE: {len(group)} conditions produce IDENTICAL results "
                f"{list(vals)}: {', '.join(group[:4])}"
            )

        if dup_ratio >= 0.5:
            # More than half of conditions are duplicated — experiment is broken
            penalty = -5.0
            issues.insert(0, f"EXPERIMENT_INVALID: {total_duplicated}/{total_conditions} "
                            f"conditions have identical results — IV was never manipulated")
        elif dup_ratio >= 0.3:
            penalty = -3.0
        else:
            penalty = -2.0

        return penalty, issues

    @staticmethod
    def _extract_condition_metrics(data: dict) -> list[float]:
        """Extract per-seed metric values from a condition dict."""
        metrics = []
        for key in ("test_acc_per_seed", "test_accuracy", "test_acc",
                     "accuracy", "accuracies", "scores"):
            val = data.get(key)
            if isinstance(val, list) and val:
                metrics = [round(float(v), 6) for v in val if isinstance(v, (int, float))]
                if metrics:
                    return metrics
            elif isinstance(val, (int, float)):
                return [round(float(val), 6)]

        # Check per_seed_results dict
        psr = data.get("per_seed_results", data.get("per_seed", {}))
        if isinstance(psr, dict):
            for seed_data in psr.values():
                if isinstance(seed_data, dict):
                    for mk in ("test_accuracy", "accuracy", "test_acc"):
                        if mk in seed_data:
                            metrics.append(round(float(seed_data[mk]), 6))
                            break
        return metrics

    def _check_absurd_effects(self, json_data: list[dict]) -> tuple[float, list[str]]:
        """Detect absurdly large effect sizes that indicate unfair comparisons.

        Cohen's d > 5 is essentially never seen in legitimate research.
        d > 10 means the comparison itself is flawed.
        """
        all_flat = {}
        for data in json_data:
            if isinstance(data, dict):
                all_flat.update(self._flatten_keys(data))

        issues = []
        penalty = 0.0

        for k, v in all_flat.items():
            k_lower = k.lower()
            if any(ek in k_lower for ek in ("cohens_d", "cohen_d", "effect_size")):
                if isinstance(v, (int, float)):
                    d = abs(float(v))
                    if d > 10:
                        penalty = min(penalty, -2.0)
                        issues.append(
                            f"ABSURD_EFFECT: d={d:.1f} — comparison is fundamentally unfair. "
                            f"Cohen's d > 10 means conditions are not comparable at matched hyperparameters."
                        )
                    elif d > 5:
                        penalty = min(penalty, -1.0)
                        issues.append(
                            f"SUSPICIOUS_EFFECT: d={d:.1f} — unusually large. "
                            f"Check if conditions are fairly comparable."
                        )

        return penalty, issues

    def _check_reasonable_accuracy(self, json_data: list[dict]) -> tuple[float, list[str]]:
        """Check if accuracy values are within reasonable bounds.

        Flags:
        - Accuracy = 0% or 100% (broken model or data leak)
        - Accuracy near random chance with no acknowledgment
        """
        issues = []
        penalty = 0.0

        for data in json_data:
            if not isinstance(data, dict):
                continue

            for key in ("test_accuracy", "accuracy", "mean_accuracy",
                        "mean_test_accuracy", "best_accuracy"):
                val = data.get(key)
                if not isinstance(val, (int, float)):
                    continue
                acc = float(val)

                # Handle both 0-1 and 0-100 scales
                if 0 < acc <= 1.0:
                    acc *= 100

                if acc >= 99.9:
                    penalty = min(penalty, -1.0)
                    issues.append(f"SUSPICIOUS_PERFECT: {key}={acc:.1f}% — possible data leak or overfitting")
                elif acc <= 0.1:
                    penalty = min(penalty, -1.0)
                    issues.append(f"BROKEN_MODEL: {key}={acc:.1f}% — model is not learning")

        return penalty, issues

    # ══════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _flatten_keys(data: dict, prefix: str = "", depth: int = 0) -> dict:
        """Flatten nested dict to dot-separated keys (max depth 4)."""
        flat = {}
        if depth > 4:
            return flat
        if isinstance(data, dict):
            for k, v in data.items():
                key = f"{prefix}.{k}" if prefix else k
                flat[key] = v
                if isinstance(v, (dict, list)):
                    flat.update(DeterministicVerifier._flatten_keys(v, key, depth + 1))
        elif isinstance(data, list):
            for i, item in enumerate(data[:10]):
                if isinstance(item, dict):
                    flat.update(DeterministicVerifier._flatten_keys(
                        item, f"{prefix}[{i}]", depth + 1))
        return flat
