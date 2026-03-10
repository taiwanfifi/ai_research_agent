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
from core.deterministic_verifier import DeterministicVerifier


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
    """Mission quality scorer — rule-based or LLM Judge (Round 11)."""

    def __init__(self, llm_judge=None):
        """
        Args:
            llm_judge: Optional LLMJudge instance for semantic scoring.
                       Falls back to regex/rule-based scoring when None.
        """
        self.llm_judge = llm_judge

    def score_mission(self, mission_dir: str) -> MissionScore:
        """
        Score a mission from its directory.

        Uses LLM Judge (Call 3) when available, falls back to rule-based scoring.

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

        # Try LLM Judge scoring first (Round 11)
        if self.llm_judge:
            try:
                score = self._score_with_llm_judge(
                    mission_dir, manifest, checkpoint, workspace_dir,
                    reports_dir, completed_tasks, execution_log, mission_id,
                )
                self._save_score(workspace_dir, score)
                return score
            except Exception as e:
                print(f"  [MissionScorer] LLM Judge scoring failed ({e}), falling back to rule-based")

        # Fallback: rule-based scoring
        dimensions = [
            self._score_literature(completed_tasks, knowledge_dir),
            self._score_code(workspace_dir, completed_tasks, execution_log),
            self._score_results(completed_tasks, execution_log, workspace_dir),
            self._score_verification(checkpoint, workspace_dir),
            self._score_artifacts(workspace_dir),
            self._score_report(reports_dir),
        ]

        # Weighted overall
        overall = sum(d.score * d.weight for d in dimensions)

        # Fatal flaw cap: if verification found data sanity issues, cap the grade
        ver_dim = next((d for d in dimensions if d.name == "verification"), None)
        if ver_dim:
            fatal_flaws = [e for e in ver_dim.evidence
                           if "EXPERIMENT_INVALID" in e or "FATAL_DUPLICATE" in e]
            absurd_effects = [e for e in ver_dim.evidence if "ABSURD_EFFECT" in e]
            if fatal_flaws:
                # IV not manipulated — cap at C regardless of other dimensions
                overall = min(overall, 5.5)
                ver_dim.evidence.append("OVERALL_CAPPED: fatal data flaw detected")
            elif absurd_effects:
                # Unfair comparison — cap at B
                overall = min(overall, 7.5)
                ver_dim.evidence.append("OVERALL_CAPPED: absurd effect size detected")

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

    def _score_with_llm_judge(self, mission_dir, manifest, checkpoint,
                               workspace_dir, reports_dir, completed_tasks,
                               execution_log, mission_id) -> MissionScore:
        """Score using LLM Judge Call 3."""
        import time

        goal = manifest.get("goal", "")

        # Gather workspace files
        workspace_files = []
        if os.path.isdir(workspace_dir):
            for root, dirs, fnames in os.walk(workspace_dir):
                for fn in fnames:
                    if '__pycache__' not in root:
                        rel = os.path.relpath(os.path.join(root, fn), workspace_dir)
                        sz = os.path.getsize(os.path.join(root, fn))
                        workspace_files.append(f"{rel} ({sz}B)")

        # Get execution log summary
        exec_summary = ""
        if execution_log:
            from core.execution_log import ExecutionLog
            elog = ExecutionLog.from_dict(execution_log, workspace_dir)
            exec_summary = elog.get_summary_for_prompt()

        # Read latest report
        report_content = ""
        if os.path.isdir(reports_dir):
            report_files = sorted(glob.glob(os.path.join(reports_dir, "*.md")))
            if report_files:
                try:
                    with open(report_files[-1], "r", encoding="utf-8") as f:
                        report_content = f.read()
                except Exception:
                    pass

        # Call LLM Judge
        judge_result = self.llm_judge.score_mission(
            goal=goal,
            workspace_files=workspace_files,
            exec_summary=exec_summary,
            report_content=report_content,
            completed_tasks=completed_tasks,
            workspace_dir=workspace_dir,
        )

        # Convert to MissionScore
        weights = {
            "literature": 0.15, "code": 0.20, "results": 0.25,
            "verification": 0.15, "artifacts": 0.10, "report": 0.15,
        }
        dimensions = []
        for dim_name, weight in weights.items():
            dim_data = judge_result.get(dim_name, {})
            dimensions.append(DimensionScore(
                name=dim_name,
                score=dim_data.get("score", 0),
                weight=weight,
                evidence=[dim_data.get("evidence", "")],
            ))

        return MissionScore(
            dimensions=dimensions,
            overall=judge_result.get("overall", 0),
            grade=judge_result.get("grade", "F"),
            mission_id=mission_id,
            scored_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

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
                       execution_log: dict | None,
                       workspace_dir: str = "") -> DimensionScore:
        """Results dimension: parsed metrics, distinct experimental runs, result files."""
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

        # Check workspace for result JSON files with numeric data
        if workspace_dir and os.path.isdir(workspace_dir):
            result_files = [f for f in glob.glob(os.path.join(workspace_dir, "*.json"))
                           if os.path.basename(f) not in ("execution_log.json", "mission_score.json",
                                                            "dataset_info.json")]
            result_file_metrics = 0
            metric_names = {
                "accuracy", "loss", "f1", "mean", "std", "precision",
                "recall", "perplexity", "auc", "bleu", "rouge",
                "t_statistic", "p_value", "cohens_d",
            }
            for rf in result_files:
                try:
                    with open(rf) as f:
                        data = json.load(f)
                    result_file_metrics += self._count_numeric_fields(data, metric_names)
                except Exception:
                    pass
            if result_file_metrics > 0:
                bonus = min(4.0, result_file_metrics * 1.0)
                score += bonus
                evidence.append(f"{len(result_files)} result JSON files with {result_file_metrics} numeric fields")

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

    def _score_verification(self, checkpoint: dict,
                             workspace_dir: str = "") -> DimensionScore:
        """Verification dimension: ground-truth JSON + result_verifier + task scores."""
        evidence = []
        score = 0.0

        # 1. Ground-truth JSON files (primary signal — up to 6 points)
        if workspace_dir and os.path.isdir(workspace_dir):
            json_files = [f for f in glob.glob(os.path.join(workspace_dir, "*.json"))
                          if os.path.basename(f) not in
                          ("execution_log.json", "mission_score.json", "dataset_info.json")]
            metric_names = {
                "accuracy", "loss", "f1", "mean", "std", "precision",
                "recall", "perplexity", "auc", "bleu", "rouge",
                "t_statistic", "p_value", "cohens_d", "mean_accuracy",
                "std_accuracy", "training_time", "best_accuracy",
            }
            has_analysis = False
            has_multi_seed = False
            has_stats_test = False
            total_metrics = 0

            for jf in json_files:
                try:
                    with open(jf) as f:
                        data = json.load(f)
                    basename = os.path.basename(jf)

                    # analysis_summary.json = gold standard
                    if "analysis" in basename or "summary" in basename or "final" in basename:
                        has_analysis = True

                    # Deep scan for stats and multi-seed (handles nested structures)
                    if isinstance(data, dict):
                        flat = self._flatten_json_keys(data)
                        # Stats test: any key containing p_value, t_statistic, cohens_d
                        stat_keys = {"p_value", "t_statistic", "t_test", "cohens_d",
                                     "statistics", "methods"}
                        if any(sk in k for k in flat for sk in stat_keys):
                            has_stats_test = True
                        # Multi-seed: any key "seeds" with list≥2, or "accuracies" list≥2
                        seed_array_keys = {"seeds", "accuracies", "scores",
                                           "baseline_accuracies", "warmup_accuracies",
                                           "test_accuracies", "eval_accuracies",
                                           "test_acc_per_seed", "per_seed"}
                        for k, v in flat.items():
                            if isinstance(v, list) and len(v) >= 2:
                                k_lower = k.split(".")[-1]
                                if k_lower in seed_array_keys:
                                    has_multi_seed = True
                                elif all(isinstance(x, (int, float)) for x in v) and len(v) >= 3:
                                    has_multi_seed = True

                    total_metrics += self._count_numeric_fields(data, metric_names)
                except Exception:
                    pass

            if has_analysis:
                score += 3.0
                evidence.append("analysis/summary JSON found (ground truth)")
            if has_multi_seed:
                score += 1.5
                evidence.append("multi-seed results in JSON")
            if has_stats_test:
                score += 1.5
                evidence.append("statistical tests in JSON")
            if total_metrics > 0 and not has_analysis:
                score += min(3.0, total_metrics * 0.5)
                evidence.append(f"{total_metrics} metric fields in JSON files")

        # 2. Stdout capture (secondary — up to 2 points)
        verifier_data = checkpoint.get("result_verifier", {})
        captured = verifier_data.get("captured", [])
        if captured:
            score += min(2.0, len(captured) * 0.2)
            evidence.append(f"{len(captured)} numbers captured from stdout")

        # 3. Task-level verification scores (up to 2 points)
        completed = checkpoint.get("completed_tasks", [])
        verified_tasks = [t for t in completed if "verification_score" in t]
        if verified_tasks:
            avg_score = sum(t["verification_score"] for t in verified_tasks) / len(verified_tasks)
            score += avg_score * 2.0  # 0-1 → 0-2
            evidence.append(f"avg verification: {avg_score:.0%} over {len(verified_tasks)} tasks")

        # 4. Fabrication penalty
        fabrication_blocks = sum(
            1 for t in completed
            if not t.get("success") and "fabrication" in (t.get("error") or "").lower()
        )
        if fabrication_blocks:
            score = max(0, score - fabrication_blocks * 0.5)
            evidence.append(f"{fabrication_blocks} fabrication blocks caught")

        if not evidence:
            evidence.append("no verification data found")

        # ── Deterministic Verifier (Layer 2-3 augmentation) ──────
        if workspace_dir and os.path.isdir(workspace_dir):
            try:
                verifier = DeterministicVerifier()
                det_result = verifier.verify(workspace_dir)
                det_score = det_result.total_score

                # Blend: 50% existing + 50% deterministic
                blended = 0.5 * score + 0.5 * det_score
                ds_penalty = det_result.breakdown.get("data_sanity", 0)
                evidence.append(f"det_verifier: {det_score:.1f}/10 "
                                f"(sanity={det_result.breakdown.get('sanity', 0):.0f}, "
                                f"curves={det_result.breakdown.get('curves', 0):.1f}, "
                                f"stats={det_result.breakdown.get('stats', 0):.1f}, "
                                f"data_sanity={ds_penalty:.0f})")
                # Prioritize fatal issues, then others
                fatal = [i for i in det_result.issues
                         if any(k in i for k in ("FATAL", "INVALID", "ABSURD", "BROKEN"))]
                other = [i for i in det_result.issues if i not in fatal]
                for issue in (fatal + other)[:5]:
                    evidence.append(f"  {issue}")
                score = blended
            except Exception as e:
                evidence.append(f"det_verifier error: {e}")

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
    def _flatten_json_keys(data, prefix: str = "", depth: int = 0) -> dict:
        """Flatten nested dict/list to dot-separated keys (max depth 4)."""
        flat = {}
        if depth > 4:
            return flat
        if isinstance(data, dict):
            for k, v in data.items():
                key = f"{prefix}.{k}" if prefix else k
                flat[key] = v
                if isinstance(v, (dict, list)):
                    flat.update(MissionScorer._flatten_json_keys(v, key, depth + 1))
        elif isinstance(data, list):
            for i, item in enumerate(data[:10]):  # Cap at 10 items
                if isinstance(item, dict):
                    flat.update(MissionScorer._flatten_json_keys(
                        item, f"{prefix}[{i}]", depth + 1))
        return flat

    @staticmethod
    def _count_numeric_fields(data, metric_names: set, depth: int = 0) -> int:
        """Recursively count numeric fields matching metric names (max depth 3)."""
        if depth > 3:
            return 0
        count = 0
        if isinstance(data, dict):
            for k, v in data.items():
                k_lower = k.lower()
                if isinstance(v, (int, float)) and any(m in k_lower for m in metric_names):
                    count += 1
                elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
                    count += 1
                elif isinstance(v, (dict, list)):
                    count += MissionScorer._count_numeric_fields(v, metric_names, depth + 1)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    count += MissionScorer._count_numeric_fields(item, metric_names, depth + 1)
        return count

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
