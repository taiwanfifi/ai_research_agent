"""
Flow Monitor (Meta-Supervisor)
================================
Lightweight rule-based heuristic monitor. No LLM calls.
Detects systemic problems in the research loop.

6 heuristic checks:
1. Repeated failures → skip_worker
2. Strategy stagnation → replan
3. Quality regression → replan
4. Worker imbalance → force_coder
5. Resource waste → simplify_task
6. Exploration-exploitation → force_coder
"""

from dataclasses import dataclass, field


@dataclass
class Advisory:
    """A recommendation from the flow monitor."""
    severity: str  # info, warning, critical
    category: str
    message: str
    suggested_action: str  # skip_worker:{name}, replan, force_coder, simplify_task

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "suggested_action": self.suggested_action,
        }


class FlowMonitor:
    """Rule-based meta-supervisor that detects systemic problems."""

    def __init__(self):
        self._history: list[dict] = []  # per-cycle snapshots

    def analyze(self, cycle: int, tasks: list[dict], dag=None,
                failures: dict = None) -> list[Advisory]:
        """Analyze current mission state and return advisories.

        Args:
            cycle: Current cycle number
            tasks: All completed tasks (list of dicts with worker, success, output, elapsed_s, task)
            dag: InsightDAG instance (optional)
            failures: Dict of worker_name → consecutive failure count

        Returns:
            List of Advisory objects sorted by severity (critical first)
        """
        # Record snapshot
        self._history.append({
            "cycle": cycle,
            "tasks": len(tasks),
            "failures": dict(failures or {}),
        })

        advisories = []
        advisories.extend(self._check_repeated_failures(tasks, failures or {}))
        advisories.extend(self._check_strategy_stagnation(tasks))
        advisories.extend(self._check_quality_regression(tasks, dag))
        advisories.extend(self._check_worker_imbalance(tasks, cycle))
        advisories.extend(self._check_resource_waste(tasks))
        advisories.extend(self._check_exploration_exploitation(tasks))

        # Sort: critical > warning > info
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        advisories.sort(key=lambda a: severity_order.get(a.severity, 3))

        return advisories

    # ── Heuristic 1: Repeated failures ────────────────────────────

    def _check_repeated_failures(self, tasks, failures) -> list[Advisory]:
        """Same worker failing 3+ times → suggest skipping."""
        advisories = []
        for worker, count in failures.items():
            if count >= 3:
                # Check if errors are similar (simple word overlap)
                worker_errors = [
                    t.get("error", "")
                    for t in tasks
                    if t.get("worker") == worker and not t.get("success") and t.get("error")
                ]
                if len(worker_errors) >= 2:
                    # Check similarity of last two errors
                    words1 = set(worker_errors[-1].lower().split())
                    words2 = set(worker_errors[-2].lower().split())
                    overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                    if overlap > 0.4:
                        advisories.append(Advisory(
                            severity="critical",
                            category="repeated_failure",
                            message=f"{worker} has failed {count}x with similar errors. "
                                    f"Likely stuck on the same issue.",
                            suggested_action=f"skip_worker:{worker}",
                        ))
                        continue

                advisories.append(Advisory(
                    severity="warning",
                    category="repeated_failure",
                    message=f"{worker} has failed {count}x consecutively.",
                    suggested_action=f"skip_worker:{worker}",
                ))
        return advisories

    # ── Heuristic 2: Strategy stagnation ──────────────────────────

    def _check_strategy_stagnation(self, tasks) -> list[Advisory]:
        """Task descriptions >60% word overlap → suggest replanning."""
        if len(tasks) < 3:
            return []

        recent = tasks[-3:]
        descs = [t.get("task", "") for t in recent]

        # Pairwise word overlap
        overlaps = []
        for i in range(len(descs)):
            for j in range(i + 1, len(descs)):
                words_i = set(descs[i].lower().split())
                words_j = set(descs[j].lower().split())
                if words_i and words_j:
                    overlap = len(words_i & words_j) / max(len(words_i | words_j), 1)
                    overlaps.append(overlap)

        if overlaps and sum(overlaps) / len(overlaps) > 0.6:
            return [Advisory(
                severity="warning",
                category="stagnation",
                message=f"Last {len(recent)} tasks have very similar descriptions "
                        f"(avg {sum(overlaps)/len(overlaps):.0%} word overlap). "
                        f"Strategy may be stuck.",
                suggested_action="replan",
            )]
        return []

    # ── Heuristic 3: Quality regression ───────────────────────────

    def _check_quality_regression(self, tasks, dag) -> list[Advisory]:
        """Insight quality declining → suggest replanning."""
        if not dag or dag.active_count() < 4:
            return []

        # Get active nodes from DAG (filter non-archived, sort by cycle)
        try:
            nodes = [n for n in dag.nodes.values() if not n.archived]
            nodes.sort(key=lambda n: n.cycle)
        except AttributeError:
            return []

        if len(nodes) < 4:
            return []

        mid = len(nodes) // 2
        first_half = nodes[:mid]
        second_half = nodes[mid:]

        # Compare average relevance scores
        avg_first = sum(n.relevance for n in first_half) / len(first_half) if first_half else 0
        avg_second = sum(n.relevance for n in second_half) / len(second_half) if second_half else 0

        if avg_first > 0 and avg_second < avg_first * 0.7:
            return [Advisory(
                severity="warning",
                category="quality_regression",
                message=f"Insight quality declining: first half avg relevance "
                        f"{avg_first:.2f} vs second half {avg_second:.2f}. "
                        f"Current approach may be hitting diminishing returns.",
                suggested_action="replan",
            )]
        return []

    # ── Heuristic 4: Worker imbalance ─────────────────────────────

    def _check_worker_imbalance(self, tasks, cycle) -> list[Advisory]:
        """Explorer >60% of tasks at cycle 4+ → push toward coding."""
        if cycle < 4 or len(tasks) < 3:
            return []

        worker_counts = {}
        for t in tasks:
            w = t.get("worker", "unknown")
            worker_counts[w] = worker_counts.get(w, 0) + 1

        total = sum(worker_counts.values())
        explorer_pct = worker_counts.get("explorer", 0) / total if total > 0 else 0

        if explorer_pct > 0.6:
            return [Advisory(
                severity="warning",
                category="worker_imbalance",
                message=f"Explorer has {worker_counts.get('explorer', 0)}/{total} tasks "
                        f"({explorer_pct:.0%}). Too much exploration, not enough implementation.",
                suggested_action="force_coder",
            )]
        return []

    # ── Heuristic 5: Resource waste ──────────────────────────────

    def _check_resource_waste(self, tasks) -> list[Advisory]:
        """Task >200s with <500 char output → suggest simplifying."""
        if not tasks:
            return []

        last_task = tasks[-1]
        elapsed = last_task.get("elapsed_s", 0)
        output_len = len(last_task.get("output", ""))

        if elapsed > 200 and output_len < 500:
            return [Advisory(
                severity="info",
                category="resource_waste",
                message=f"Last task took {elapsed:.0f}s but produced only {output_len} chars. "
                        f"Consider breaking into smaller sub-tasks.",
                suggested_action="simplify_task",
            )]
        return []

    # ── Heuristic 6: Exploration-exploitation balance ─────────────

    def _check_exploration_exploitation(self, tasks) -> list[Advisory]:
        """3 consecutive explorer tasks → force implementation."""
        if len(tasks) < 3:
            return []

        last_3_workers = [t.get("worker", "") for t in tasks[-3:]]
        if all(w == "explorer" for w in last_3_workers):
            return [Advisory(
                severity="warning",
                category="exploration_exploitation",
                message="3 consecutive explorer tasks. Time to implement "
                        "based on what's been found.",
                suggested_action="force_coder",
            )]
        return []

    # ── Formatting ───────────────────────────────────────────────

    @staticmethod
    def format_for_prompt(advisories: list[Advisory]) -> str:
        """Format advisories for injection into supervisor decision prompt."""
        if not advisories:
            return ""

        parts = ["## Flow Monitor Advisories"]
        for adv in advisories:
            icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(adv.severity, "")
            parts.append(f"{icon} [{adv.severity.upper()}] {adv.message}")
            parts.append(f"   Suggested: {adv.suggested_action}")

        return "\n".join(parts)

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"history": self._history}

    @classmethod
    def from_dict(cls, d: dict) -> "FlowMonitor":
        fm = cls()
        fm._history = d.get("history", [])
        return fm
