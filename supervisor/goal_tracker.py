"""
Goal Completion Tracker
========================
Parses mission goals into measurable sub-goals, tracks completion
objectively. Uses one LLM call to decompose, then pure rule-based
checking thereafter.

Sub-goal types:
- paper_found: Literature survey completed with N+ papers
- code_exists: Specific code file exists in workspace
- code_runs: Code file executed successfully
- metric_achieved: A specific metric was measured
- visualization_generated: Plot/figure file exists
- comparison_done: Multiple methods compared with metrics
"""

import json
import os
import re

from dataclasses import dataclass, field


@dataclass
class SubGoal:
    """A measurable sub-goal derived from the mission goal."""
    type: str
    description: str
    check_params: dict = field(default_factory=dict)
    completed: bool = False
    evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "description": self.description,
            "check_params": self.check_params,
            "completed": self.completed,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SubGoal":
        return cls(
            type=d["type"],
            description=d["description"],
            check_params=d.get("check_params", {}),
            completed=d.get("completed", False),
            evidence=d.get("evidence", ""),
        )


class GoalTracker:
    """Tracks objective mission goal completion."""

    # Valid sub-goal types and their required check_params
    SUBGOAL_TYPES = {
        "paper_found": {"min_papers": 3},
        "code_exists": {"filename_pattern": "*.py"},
        "code_runs": {"filename_pattern": "*.py"},
        "metric_achieved": {"metric_name": "accuracy"},
        "visualization_generated": {"filename_pattern": "*.png"},
        "comparison_done": {"min_methods": 2},
    }

    def __init__(self, workspace_dir: str, llm=None):
        self.workspace_dir = workspace_dir
        self.llm = llm
        self.sub_goals: list[SubGoal] = []
        self._parsed = False

    def parse_goal(self, goal: str):
        """Decompose a high-level goal into 3-6 measurable sub-goals.

        Uses one LLM call. Falls back to generic sub-goals if LLM fails.
        """
        if self.llm:
            try:
                self._parse_with_llm(goal)
                self._parsed = True
                return
            except Exception as e:
                print(f"  [GoalTracker] LLM parse failed ({e}), using fallback")

        self._parse_fallback(goal)
        self._parsed = True

    def _parse_with_llm(self, goal: str):
        """Use LLM to decompose goal into sub-goals."""
        from core.llm import strip_think

        prompt = f"""Decompose this research goal into 3-6 measurable sub-goals.

Goal: {goal}

Available sub-goal types:
- paper_found: Literature survey (check_params: {{"min_papers": N}})
- code_exists: Code file exists (check_params: {{"filename_pattern": "*.py"}})
- code_runs: Code runs successfully (check_params: {{"filename_pattern": "*.py"}})
- metric_achieved: Metric measured (check_params: {{"metric_name": "accuracy"}})
- visualization_generated: Plot exists (check_params: {{"filename_pattern": "*.png"}})
- comparison_done: Methods compared (check_params: {{"min_methods": N}})

Respond with ONLY a JSON array:
[
  {{"type": "paper_found", "description": "Find 5+ papers on X", "check_params": {{"min_papers": 5}}}},
  ...
]

Rules:
- 3-6 sub-goals
- Always include at least: paper_found, code_exists, metric_achieved
- Be specific about what files/metrics to expect
- Order: research → implement → evaluate → visualize"""

        response = self.llm.chat([
            {"role": "system", "content": "Decompose research goals. Respond with JSON only."},
            {"role": "user", "content": prompt},
        ])
        raw = strip_think(response["choices"][0]["message"]["content"])
        json_match = re.search(r'\[[\s\S]*\]', raw)
        if not json_match:
            raise ValueError("No JSON array in LLM response")

        parsed = json.loads(json_match.group())
        for item in parsed:
            sg_type = item.get("type", "")
            if sg_type not in self.SUBGOAL_TYPES:
                continue
            self.sub_goals.append(SubGoal(
                type=sg_type,
                description=item.get("description", ""),
                check_params=item.get("check_params", self.SUBGOAL_TYPES[sg_type]),
            ))

        if not self.sub_goals:
            raise ValueError("No valid sub-goals parsed")

    def _parse_fallback(self, goal: str):
        """Generic sub-goals when LLM fails."""
        self.sub_goals = [
            SubGoal(type="paper_found", description="Literature survey",
                    check_params={"min_papers": 3}),
            SubGoal(type="code_exists", description="Implementation code",
                    check_params={"filename_pattern": "*.py"}),
            SubGoal(type="code_runs", description="Code executes successfully",
                    check_params={"filename_pattern": "*.py"}),
            SubGoal(type="metric_achieved", description="Quantitative evaluation",
                    check_params={"metric_name": ""}),
            SubGoal(type="visualization_generated", description="Result visualization",
                    check_params={"filename_pattern": "*.png"}),
        ]

    def check_completion(self, tasks: list[dict], knowledge_stats: dict = None,
                         dag=None) -> dict:
        """Check which sub-goals are completed. Fully rule-based.

        Args:
            tasks: List of completed task dicts with worker, success, output keys
            knowledge_stats: Knowledge tree stats dict
            dag: InsightDAG instance (optional)

        Returns:
            {all_complete, completion_rate, status[], blocking[]}
        """
        if not self.sub_goals:
            return {"all_complete": False, "completion_rate": 0.0,
                    "status": [], "blocking": ["No sub-goals parsed"]}

        successful_tasks = [t for t in tasks if t.get("success")]
        all_output = " ".join(t.get("output", "") for t in successful_tasks)
        worker_types = set(t.get("worker", "") for t in successful_tasks)

        for sg in self.sub_goals:
            if sg.completed:
                continue
            checker = getattr(self, f"_check_{sg.type}", None)
            if checker:
                completed, evidence = checker(sg, successful_tasks, all_output,
                                              worker_types, knowledge_stats)
                if completed:
                    sg.completed = True
                    sg.evidence = evidence

        completed_count = sum(1 for sg in self.sub_goals if sg.completed)
        all_complete = completed_count == len(self.sub_goals)
        completion_rate = completed_count / len(self.sub_goals) if self.sub_goals else 0.0
        blocking = [sg.description for sg in self.sub_goals if not sg.completed]

        return {
            "all_complete": all_complete,
            "completion_rate": completion_rate,
            "status": [sg.to_dict() for sg in self.sub_goals],
            "blocking": blocking,
        }

    # ── Per-type checkers ──────────────────────────────────────────

    def _check_paper_found(self, sg, tasks, all_output, workers, kstats):
        min_papers = sg.check_params.get("min_papers", 3)
        if "explorer" not in workers:
            return False, ""

        # Count paper references in explorer output
        paper_markers = re.findall(
            r'(?:arXiv|arxiv|et al\.|Title:|paper\s*\d)',
            all_output, re.IGNORECASE,
        )
        # Also check knowledge items
        paper_count = len(paper_markers)
        if kstats:
            paper_count = max(paper_count, kstats.get("by_category", {}).get("papers", 0))

        if paper_count >= min_papers:
            return True, f"Found {paper_count} paper references"
        return False, ""

    def _check_code_exists(self, sg, tasks, all_output, workers, kstats):
        if "coder" not in workers:
            return False, ""

        # ONLY check actual filesystem, never fall back to text patterns
        ws_dir = sg.check_params.get("workspace_dir", "") or self.workspace_dir
        if ws_dir:
            import glob as glob_mod
            patterns = ["*.py", "*.ipynb", "*.json", "*.csv", "*.pkl", "*.pt", "*.pth"]
            for pat in patterns:
                matches = glob_mod.glob(os.path.join(ws_dir, pat))
                real = [m for m in matches if not os.path.basename(m).startswith('tmp_')]
                if real:
                    return True, f"Found: {', '.join(os.path.basename(m) for m in real[:3])}"
        return False, ""

    def _check_code_runs(self, sg, tasks, all_output, workers, kstats):
        # Check for successful code execution in coder/reviewer tasks
        code_tasks = [t for t in tasks if t.get("worker") in ("coder", "reviewer")]
        for t in code_tasks:
            output = t.get("output", "")
            if t.get("success") and any(m in output.lower() for m in [
                "success", "completed", "results", "accuracy", "loss", "output:",
            ]):
                return True, "Code executed successfully"
        return False, ""

    def _check_metric_achieved(self, sg, tasks, all_output, workers, kstats):
        metric_name = sg.check_params.get("metric_name", "")

        # Strategy 1: Check workspace for results JSON files
        ws_dir = sg.check_params.get("workspace_dir", "") or self.workspace_dir
        if ws_dir:
            import json as json_mod
            for fname in ["experiment_results.json", "results.json", "metrics.json"]:
                fpath = os.path.join(ws_dir, fname)
                if os.path.exists(fpath):
                    try:
                        with open(fpath) as f:
                            data = json_mod.load(f)
                        if isinstance(data, dict):
                            for key, val in data.items():
                                if isinstance(val, (int, float)):
                                    return True, f"Found {key}={val} in {fname}"
                    except Exception:
                        pass

        # Strategy 2: Regex patterns in stdout/output (original + expanded)
        metric_pattern = r'(?:accuracy|loss|perplexity|ppl|f1|precision|recall|bleu|rouge|auc|mse|rmse)\s*[:=]\s*\d+\.?\d*'
        if metric_name:
            metric_pattern = rf'{re.escape(metric_name)}\s*[:=]\s*\d+\.?\d*'

        matches = re.findall(metric_pattern, all_output, re.IGNORECASE)
        if matches:
            return True, f"Metrics found: {', '.join(matches[:3])}"

        # Also try JSON key format: "metric_name": value
        json_metric_pattern = r'"(?:accuracy|loss|perplexity|f1|precision|recall|bleu|rouge|auc|mse|rmse)"\s*:\s*\d+\.?\d*'
        json_matches = re.findall(json_metric_pattern, all_output, re.IGNORECASE)
        if json_matches:
            return True, f"Metrics found (JSON): {', '.join(json_matches[:3])}"

        # Also check for table results
        table_pattern = r'\|\s*[\w\s]+\s*\|\s*\d+\.?\d*'
        table_matches = re.findall(table_pattern, all_output)
        if len(table_matches) >= 2:
            return True, f"Results table found ({len(table_matches)} rows)"

        return False, ""

    def _check_visualization_generated(self, sg, tasks, all_output, workers, kstats):
        pattern = sg.check_params.get("filename_pattern", "*.png")
        # ONLY check actual filesystem for figure files
        try:
            import glob as glob_mod
            matches = glob_mod.glob(os.path.join(self.workspace_dir, pattern))
            if matches:
                names = [os.path.basename(m) for m in matches]
                return True, f"Figures: {', '.join(names[:5])}"
        except Exception:
            pass
        return False, ""

    def _check_comparison_done(self, sg, tasks, all_output, workers, kstats):
        min_methods = sg.check_params.get("min_methods", 2)

        # Look for comparison tables or multiple method results
        table_rows = re.findall(r'\|[^|]+\|[^|]+\|', all_output)
        method_markers = re.findall(
            r'(?:baseline|method|model|approach|variant)\s*\d*\s*[:|\-]',
            all_output, re.IGNORECASE,
        )

        compared = max(len(table_rows) - 1, len(method_markers))  # -1 for header
        if compared >= min_methods:
            return True, f"Compared {compared} methods/variants"

        # Check for "vs" or "compared to" language
        if re.search(r'(?:vs\.?|compared?\s+to|versus|against)\s+\w+', all_output, re.IGNORECASE):
            return True, "Comparison language found in output"

        return False, ""

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "sub_goals": [sg.to_dict() for sg in self.sub_goals],
            "parsed": self._parsed,
        }

    @classmethod
    def from_dict(cls, d: dict, workspace_dir: str) -> "GoalTracker":
        tracker = cls(workspace_dir)
        tracker.sub_goals = [SubGoal.from_dict(sg) for sg in d.get("sub_goals", [])]
        tracker._parsed = d.get("parsed", False)
        return tracker

    def format_for_prompt(self) -> str:
        """Format current goal status for injection into supervisor prompt."""
        if not self.sub_goals:
            return ""

        parts = ["## Goal Completion Status"]
        completed = sum(1 for sg in self.sub_goals if sg.completed)
        total = len(self.sub_goals)
        parts.append(f"Progress: {completed}/{total} sub-goals complete ({completed/total:.0%})")

        for sg in self.sub_goals:
            status = "DONE" if sg.completed else "PENDING"
            parts.append(f"  [{status}] {sg.description}")
            if sg.completed and sg.evidence:
                parts.append(f"         Evidence: {sg.evidence}")

        blocking = [sg.description for sg in self.sub_goals if not sg.completed]
        if blocking:
            parts.append(f"\nBlocking: {', '.join(blocking)}")
            parts.append("Focus next action on the first blocking sub-goal.")

        return "\n".join(parts)
