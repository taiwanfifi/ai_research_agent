"""
Mission Scorer — Rule-Based Quality Assessment
================================================
Scores completed missions across 6 dimensions (0-10 each),
producing a weighted overall score and letter grade.

No LLM calls — purely filesystem + checkpoint based.
Works on ANY mission (doesn't require ExecutionLog).

Dimensions:
    Literature (0.15) — papers found, explorer outputs
    Code       (0.20) — .py files, successful code runs
    Results    (0.25) — parsed metrics, distinct runs
    Verification (0.15) — result_verifier score
    Artifacts  (0.10) — .png, .pt, .json files
    Report     (0.15) — report existence, section headers
"""

import json
import os
import glob
import re
from dataclasses import dataclass, field


@dataclass
class DimensionScore:
    """Score for a single dimension."""
    name: str
    score: float      # 0-10
    weight: float
    evidence: list     # human-readable evidence strings


@dataclass
class MissionScore:
    """Complete mission quality assessment."""
    dimensions: list       # list of DimensionScore
    overall: float         # weighted sum, 0-10
    grade: str             # A, B, C, D, F
    mission_id: str
    scored_at: str

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "overall": round(self.overall, 2),
            "grade": self.grade,
            "scored_at": self.scored_at,
            "dimensions": [
                {
                    "name": d.name,
                    "score": round(d.score, 2),
                    "weight": d.weight,
                    "evidence": d.evidence,
                }
                for d in self.dimensions
            ],
        }


def _grade_from_score(score: float) -> str:
    if score >= 8.5:
        return "A"
    elif score >= 7.0:
        return "B"
    elif score >= 5.0:
        return "C"
    elif score >= 3.0:
        return "D"
    else:
        return "F"


class MissionScorer:
    """Rule-based mission quality scorer."""

    def score_mission(self, mission_dir: str) -> MissionScore:
        """
        Score a mission from its directory.

        Args:
            mission_dir: Path to mission_YYYYMMDD_HHMMSS_slug/

        Returns:
            MissionScore with all dimensions + overall grade
        """
        import time

        # Load mission data
        manifest = self._read_json(os.path.join(mission_dir, "mission.json")) or {}
        checkpoint = self._load_checkpoint(mission_dir) or {}
        workspace_dir = os.path.join(mission_dir, "workspace")
        reports_dir = os.path.join(mission_dir, "reports")
        knowledge_dir = os.path.join(mission_dir, "knowledge")
        mission_id = manifest.get("mission_id", os.path.basename(mission_dir))

        completed_tasks = checkpoint.get("completed_tasks", [])
        execution_log = self._read_json(os.path.join(workspace_dir, "execution_log.json"))

        # Score each dimension
        dimensions = [
            self._score_literature(completed_tasks, knowledge_dir),
            self._score_code(workspace_dir, completed_tasks, execution_log),
            self._score_results(completed_tasks, execution_log),
            self._score_verification(checkpoint),
            self._score_artifacts(workspace_dir),
            self._score_report(reports_dir),
        ]

        # Weighted overall
        overall = sum(d.score * d.weight for d in dimensions)
        grade = _grade_from_score(overall)

        score = MissionScore(
            dimensions=dimensions,
            overall=overall,
            grade=grade,
            mission_id=mission_id,
            scored_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        # Save to workspace
        self._save_score(workspace_dir, score)

        return score

    # ── Dimension scorers ─────────────────────────────────────────

    def _score_literature(self, completed_tasks: list, knowledge_dir: str) -> DimensionScore:
        """Literature dimension: papers found, explorer success."""
        evidence = []
        score = 0.0

        # Count explorer tasks
        explorer_tasks = [t for t in completed_tasks if t.get("worker") == "explorer"]
        successful = [t for t in explorer_tasks if t.get("success")]
        evidence.append(f"{len(successful)}/{len(explorer_tasks)} explorer tasks succeeded")

        if successful:
            score += min(4.0, len(successful) * 2.0)  # up to 4 for explorer success

        # Check for paper mentions in outputs
        paper_mentions = 0
        for t in successful:
            output = t.get("output", "")
            # Count arxiv IDs, DOIs, paper-like references
            paper_mentions += len(re.findall(r'arxiv[:\s]*\d{4}\.\d+', output, re.I))
            paper_mentions += len(re.findall(r'10\.\d{4,}/', output))
        if paper_mentions > 0:
            score += min(3.0, paper_mentions * 0.5)
            evidence.append(f"{paper_mentions} paper references found")

        # Check knowledge tree for papers category
        papers_index = self._read_json(os.path.join(knowledge_dir, "papers", "_index.json"))
        if papers_index:
            count = papers_index.get("item_count", 0)
            score += min(3.0, count * 1.0)
            evidence.append(f"{count} items in papers category")

        return DimensionScore(
            name="literature", score=min(10.0, score), weight=0.15, evidence=evidence,
        )

    def _score_code(self, workspace_dir: str, completed_tasks: list,
                    execution_log: dict | None) -> DimensionScore:
        """Code dimension: .py files written, successful code runs."""
        evidence = []
        score = 0.0

        # Count .py files in workspace
        py_files = glob.glob(os.path.join(workspace_dir, "*.py"))
        py_files = [f for f in py_files if not os.path.basename(f).startswith("tmp")]
        evidence.append(f"{len(py_files)} Python files in workspace")
        score += min(3.0, len(py_files) * 1.5)

        # Count successful coder tasks
        coder_tasks = [t for t in completed_tasks if t.get("worker") == "coder"]
        coder_success = [t for t in coder_tasks if t.get("success")]
        evidence.append(f"{len(coder_success)}/{len(coder_tasks)} coder tasks succeeded")
        score += min(3.0, len(coder_success) * 1.5)

        # Check execution_log for successful runs
        if execution_log:
            entries = execution_log.get("entries", [])
            code_runs = [e for e in entries if e.get("tool_name") == "run_python_code"]
            success_runs = [e for e in code_runs if e.get("success")]
            evidence.append(f"{len(success_runs)}/{len(code_runs)} code runs succeeded")
            score += min(4.0, len(success_runs) * 0.5)
        else:
            # Fallback: count tool calls in tasks
            run_count = sum(
                1 for t in completed_tasks
                for tc in (t.get("tool_calls") or [])
                if tc.get("name") == "run_python_code"
            )
            if run_count > 0:
                evidence.append(f"{run_count} code executions in tasks")
                score += min(4.0, run_count * 0.5)

        return DimensionScore(
            name="code", score=min(10.0, score), weight=0.20, evidence=evidence,
        )

    def _score_results(self, completed_tasks: list,
                       execution_log: dict | None) -> DimensionScore:
        """Results dimension: parsed metrics, distinct experimental runs."""
        evidence = []
        score = 0.0

        # Check execution_log for parsed metrics
        if execution_log:
            entries = execution_log.get("entries", [])
            metrics_entries = [e for e in entries
                              if e.get("parsed_metrics") and e.get("success")]
            all_metrics = set()
            for e in metrics_entries:
                all_metrics.update(e["parsed_metrics"].keys())
            evidence.append(f"{len(metrics_entries)} runs with metrics, {len(all_metrics)} distinct metrics")
            score += min(5.0, len(metrics_entries) * 1.0)
            score += min(3.0, len(all_metrics) * 0.5)
        else:
            # Fallback: check task outputs for numeric patterns
            numeric_outputs = 0
            for t in completed_tasks:
                output = t.get("output", "")
                if re.search(r'(?:accuracy|loss|f1|precision|recall|perplexity)\s*[:=]\s*\d', output, re.I):
                    numeric_outputs += 1
            evidence.append(f"{numeric_outputs} tasks with metric-like outputs")
            score += min(5.0, numeric_outputs * 1.5)

        # Check for multiple distinct runs (reproducibility)
        distinct_cycles = set()
        if execution_log:
            for e in execution_log.get("entries", []):
                if e.get("tool_name") == "run_python_code" and e.get("success"):
                    distinct_cycles.add(e.get("cycle", 0))
        if len(distinct_cycles) > 1:
            score += min(2.0, len(distinct_cycles) * 0.5)
            evidence.append(f"{len(distinct_cycles)} distinct cycles with runs")

        return DimensionScore(
            name="results", score=min(10.0, score), weight=0.25, evidence=evidence,
        )

    def _score_verification(self, checkpoint: dict) -> DimensionScore:
        """Verification dimension: result_verifier score from checkpoint."""
        evidence = []
        score = 0.0

        verifier_data = checkpoint.get("result_verifier", {})
        captured = verifier_data.get("captured", [])
        evidence.append(f"{len(captured)} numbers captured from stdout")

        if captured:
            score += min(4.0, len(captured) * 0.3)

        # Check completed tasks for verification scores
        completed = checkpoint.get("completed_tasks", [])
        verified_tasks = [t for t in completed if "verification_score" in t]
        if verified_tasks:
            avg_score = sum(t["verification_score"] for t in verified_tasks) / len(verified_tasks)
            score += avg_score * 6.0  # 0-1 → 0-6
            evidence.append(f"avg verification: {avg_score:.0%} over {len(verified_tasks)} tasks")

        # Penalty for fabrication blocks
        fabrication_blocks = sum(
            1 for t in completed
            if not t.get("success") and "fabrication" in (t.get("error") or "").lower()
        )
        if fabrication_blocks:
            score = max(0, score - fabrication_blocks * 2.0)
            evidence.append(f"{fabrication_blocks} fabrication blocks (penalty)")

        return DimensionScore(
            name="verification", score=min(10.0, score), weight=0.15, evidence=evidence,
        )

    def _score_artifacts(self, workspace_dir: str) -> DimensionScore:
        """Artifacts dimension: .png, .pt, .json, .csv files."""
        evidence = []
        score = 0.0

        if not os.path.isdir(workspace_dir):
            return DimensionScore(name="artifacts", score=0, weight=0.10, evidence=["no workspace"])

        # Count artifact types
        png_files = glob.glob(os.path.join(workspace_dir, "*.png"))
        pt_files = glob.glob(os.path.join(workspace_dir, "*.pt")) + \
                   glob.glob(os.path.join(workspace_dir, "*.pth"))
        json_files = [f for f in glob.glob(os.path.join(workspace_dir, "*.json"))
                      if os.path.basename(f) not in ("execution_log.json", "mission_score.json")]
        csv_files = glob.glob(os.path.join(workspace_dir, "*.csv"))

        if png_files:
            score += min(4.0, len(png_files) * 2.0)
            evidence.append(f"{len(png_files)} PNG figures")
        if pt_files:
            score += 3.0
            evidence.append(f"{len(pt_files)} model checkpoints")
        if json_files:
            score += min(2.0, len(json_files) * 1.0)
            evidence.append(f"{len(json_files)} JSON data files")
        if csv_files:
            score += min(1.0, len(csv_files) * 0.5)
            evidence.append(f"{len(csv_files)} CSV files")

        if not evidence:
            evidence.append("no artifacts found")

        return DimensionScore(
            name="artifacts", score=min(10.0, score), weight=0.10, evidence=evidence,
        )

    def _score_report(self, reports_dir: str) -> DimensionScore:
        """Report dimension: report file existence, section quality."""
        evidence = []
        score = 0.0

        if not os.path.isdir(reports_dir):
            return DimensionScore(name="report", score=0, weight=0.15, evidence=["no reports dir"])

        report_files = sorted(glob.glob(os.path.join(reports_dir, "*.md")))
        evidence.append(f"{len(report_files)} report files")

        if not report_files:
            return DimensionScore(name="report", score=0, weight=0.15, evidence=evidence)

        # Score the latest report
        score += 3.0  # Base score for having a report

        try:
            with open(report_files[-1], "r", encoding="utf-8") as f:
                content = f.read()

            # Check for key sections
            section_patterns = {
                "summary": r"(?:#+ .*(?:summary|overview|abstract))",
                "methods": r"(?:#+ .*(?:method|approach|implementation))",
                "results": r"(?:#+ .*(?:result|finding|experiment|benchmark))",
                "code": r"(?:#+ .*(?:code|file|implementation))",
                "conclusion": r"(?:#+ .*(?:conclusion|future|next))",
            }
            found_sections = []
            for name, pattern in section_patterns.items():
                if re.search(pattern, content, re.I):
                    found_sections.append(name)
                    score += 1.0
            evidence.append(f"sections found: {', '.join(found_sections) or 'none'}")

            # Report length bonus
            word_count = len(content.split())
            if word_count > 200:
                score += 1.0
                evidence.append(f"{word_count} words")
            if word_count > 500:
                score += 1.0

        except Exception:
            evidence.append("could not read latest report")

        return DimensionScore(
            name="report", score=min(10.0, score), weight=0.15, evidence=evidence,
        )

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _read_json(path: str) -> dict | None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _load_checkpoint(self, mission_dir: str) -> dict | None:
        cp = self._read_json(os.path.join(mission_dir, "state", "mission", "latest_checkpoint.json"))
        if cp and isinstance(cp, dict) and "value" in cp:
            return cp["value"]
        return cp

    @staticmethod
    def _save_score(workspace_dir: str, score: MissionScore):
        try:
            os.makedirs(workspace_dir, exist_ok=True)
            path = os.path.join(workspace_dir, "mission_score.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(score.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass
