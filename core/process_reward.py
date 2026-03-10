"""
Process Reward — Per-Cycle Progress Signal
============================================
Instead of only scoring at mission end (Grade A/B/D),
compute a reward signal after each supervisor cycle.

This enables:
1. Mid-mission strategy adjustment (supervisor sees reward trend)
2. Credit assignment (which tasks actually contributed)
3. Anti-coasting detection (consecutive low-reward cycles)

Inspired by: Agent-R1 (process+outcome rewards), RAGEN (bi-level GAE),
kael_daemon (friction as counterfactual signal).
"""

from dataclasses import dataclass, field


@dataclass
class CycleReward:
    """Reward signal for a single supervisor cycle."""
    cycle: int
    worker: str
    task: str
    reward: float = 0.0
    components: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "worker": self.worker,
            "task": self.task[:100],
            "reward": round(self.reward, 3),
            "components": self.components,
        }


class ProcessRewardTracker:
    """Track per-cycle rewards across a mission."""

    def __init__(self):
        self.history: list[CycleReward] = []
        self._seen_files: set = set()
        self._seen_metrics: set = set()

    def score_cycle(self, cycle: int, worker: str, task: str,
                    result: dict, workspace_files: list[str] = None) -> CycleReward:
        """Score a completed cycle based on what it produced.

        Args:
            cycle: cycle number
            worker: worker type
            task: task description
            result: worker result dict
            workspace_files: current workspace file list

        Returns:
            CycleReward with breakdown
        """
        cr = CycleReward(cycle=cycle, worker=worker, task=task)
        components = {}

        success = result.get("success", False)
        output = result.get("output", "")
        tool_calls = result.get("tool_calls", [])

        # 1. Task success/failure
        if success:
            components["task_success"] = 0.3
        else:
            components["task_failure"] = -0.2

        # 2. New artifacts produced
        if workspace_files:
            new_files = set(workspace_files) - self._seen_files
            real_new = [f for f in new_files
                        if not f.startswith(("tmp", "."))
                        and "__pycache__" not in f]
            if real_new:
                components["new_artifacts"] = min(0.3, len(real_new) * 0.1)
                self._seen_files.update(workspace_files)

        # 3. New metrics produced (evidence of real execution)
        metrics_found = _extract_metric_names(output)
        new_metrics = metrics_found - self._seen_metrics
        if new_metrics:
            components["new_metrics"] = min(0.3, len(new_metrics) * 0.1)
            self._seen_metrics.update(metrics_found)

        # 4. Repeated failure penalty (same worker failed before)
        recent_failures = [
            h for h in self.history[-3:]
            if h.worker == worker and h.components.get("task_failure")
        ]
        if recent_failures and not success:
            components["repeated_failure"] = -0.15

        # 5. Figure generation bonus
        if any(tc.get("name") == "write_file" and
               any(ext in str(tc.get("args", {}).get("filename", ""))
                   for ext in [".png", ".jpg", ".svg"])
               for tc in tool_calls):
            components["figure_generated"] = 0.2

        # 6. Statistical test evidence
        stat_keywords = ["t-test", "p-value", "p_value", "cohen", "wilcoxon",
                         "significance", "ttest"]
        if any(kw in output.lower() for kw in stat_keywords):
            components["statistical_test"] = 0.2

        # Compute total
        cr.reward = sum(components.values())
        cr.components = components
        self.history.append(cr)

        return cr

    def get_trend(self, window: int = 3) -> str:
        """Get reward trend for supervisor decision-making."""
        if len(self.history) < 2:
            return "insufficient_data"

        recent = self.history[-window:]
        avg = sum(h.reward for h in recent) / len(recent)

        if avg > 0.3:
            return "strong_progress"
        elif avg > 0.1:
            return "moderate_progress"
        elif avg > -0.1:
            return "stagnating"
        else:
            return "declining"

    def get_summary(self) -> dict:
        """Summary for injection into supervisor prompt."""
        if not self.history:
            return {}

        total = sum(h.reward for h in self.history)
        trend = self.get_trend()

        # Credit attribution: which worker contributed most?
        worker_credit = {}
        for h in self.history:
            w = h.worker
            worker_credit[w] = worker_credit.get(w, 0) + h.reward

        # Find best and worst cycles
        best = max(self.history, key=lambda h: h.reward)
        worst = min(self.history, key=lambda h: h.reward)

        return {
            "total_reward": round(total, 2),
            "trend": trend,
            "cycles_scored": len(self.history),
            "worker_credit": {k: round(v, 2) for k, v in worker_credit.items()},
            "best_cycle": {"cycle": best.cycle, "reward": round(best.reward, 2),
                           "task": best.task[:60]},
            "worst_cycle": {"cycle": worst.cycle, "reward": round(worst.reward, 2),
                            "task": worst.task[:60]},
        }

    def format_for_prompt(self) -> str:
        """Format reward summary for supervisor decision prompt."""
        summary = self.get_summary()
        if not summary:
            return ""

        lines = [f"## Mission Progress Signal"]
        lines.append(f"- Trend: **{summary['trend']}** (total reward: {summary['total_reward']})")
        lines.append(f"- Worker credit: {summary['worker_credit']}")

        if summary["trend"] in ("stagnating", "declining"):
            lines.append("- WARNING: Progress is stalling. Consider changing approach or decomposing tasks.")

        return "\n".join(lines)


def _extract_metric_names(output: str) -> set:
    """Extract metric names from output text."""
    import re
    metrics = set()
    # Match patterns like "accuracy: 85.3" or "loss: 0.42"
    for match in re.finditer(r'(\w[\w_]*)\s*[:=]\s*[\d.]+', output):
        name = match.group(1).lower()
        # Skip common non-metrics
        if name not in {"step", "epoch", "batch", "iteration", "version",
                         "seed", "range", "len", "size", "shape", "dim"}:
            metrics.add(name)
    return metrics
