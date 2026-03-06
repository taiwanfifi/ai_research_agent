"""
LLM Judge — Semantic Validation Service
==========================================
Replaces 39 scattered keyword/regex checks with 3 strategic LLM calls.

Handles any language, domain, and format — the LLM understands context
that keywords can never capture.

Three calls:
1. evaluate_worker_output() — post-worker: metrics, verification, quality
2. assess_progress()        — post-cycle: progress, recommendations
3. score_mission()          — post-mission: dimensional scoring

All calls use MiniMaxClient.chat() (single-turn, ~800 tokens, fast).
Falls back to safe defaults on parse failure.
"""

import json
import re
from core.llm import MiniMaxClient, strip_think


class LLMJudge:
    """Semantic validation using LLM instead of keyword matching."""

    def __init__(self, llm: MiniMaxClient):
        self.llm = llm

    # ── Call 1: Post-Worker Evaluation ─────────────────────────────

    def evaluate_worker_output(self, task: str, output: str,
                                stdout_parts: list[str] = None,
                                tool_calls: list[dict] = None,
                                worker_name: str = "") -> dict:
        """
        Evaluate a worker's output semantically.

        Replaces: _validate_output(), _SKIP_LABELS, _extract_claims(),
                  verify_output(), _extract_summary(), SanityChecker

        Args:
            task: The task description given to the worker
            output: The worker's combined text output
            stdout_parts: List of stdout strings from code execution
            tool_calls: List of tool call dicts [{name, args, ...}]

        Returns:
            {
                "is_substantive": bool,      # not just procedural narration
                "task_completed": bool,       # did the worker accomplish the task
                "metrics": [                  # extracted metrics, classified
                    {"name": str, "value": float, "type": "result"|"hyperparameter"}
                ],
                "claims_vs_stdout": [         # verification of claims
                    {"claim": str, "status": "verified"|"contradicted"|"unverified",
                     "actual": str|None}
                ],
                "summary": str,              # extracted summary for knowledge
                "quality_concerns": [str],    # domain-agnostic quality issues
                "has_code_output": bool,      # whether code was executed
                "has_papers": bool,           # whether papers were found
            }
        """
        # Build context — aggressively truncate to keep under ~2000 chars total
        stdout_text = ""
        if stdout_parts:
            # Only keep the LAST stdout (most likely has final results)
            last_stdout = stdout_parts[-1][:1500]
            if len(stdout_parts) > 1:
                stdout_text = f"({len(stdout_parts)} runs total, showing last)\n{last_stdout}"
            else:
                stdout_text = last_stdout

        tool_summary = ""
        if tool_calls:
            tool_names = [tc.get("name", "") for tc in tool_calls]
            tool_summary = f"Tools: {', '.join(tool_names)}"
            files_written = [tc.get("file_written", "") for tc in tool_calls
                            if tc.get("file_written")]
            if files_written:
                tool_summary += f" | Files: {', '.join(files_written)}"

        # Keep output short — the LLM summary section is at the end
        output_truncated = output[-2000:] if len(output) > 2000 else (output or "(empty)")

        system_prompt = "Evaluate research output. Return ONLY valid JSON. Any language/domain."

        # Worker-specific evaluation guidance
        worker_guide = ""
        if worker_name == "explorer":
            worker_guide = "Worker role: EXPLORER (searches papers/repos). Judge by: papers found, relevance, citations. Do NOT penalize for missing code/results — that's not this worker's job."
        elif worker_name == "coder":
            worker_guide = "Worker role: CODER (writes & runs code). Judge by: code quality, execution success, results produced."
        elif worker_name == "reviewer":
            worker_guide = "Worker role: REVIEWER (benchmarks/evaluates). Judge by: metrics measured, reproducibility, rigor."

        user_prompt = f"""Task: {task[:200]}
{worker_guide}

Output (last 2000 chars):
{output_truncated}

stdout:
{stdout_text or "(none)"}

{tool_summary}

Return JSON:
{{"is_substantive":bool,"task_completed":bool,"metrics":[{{"name":"x","value":1.0,"type":"result|hyperparameter"}}],"claims_vs_stdout":[{{"claim":"x","status":"verified|contradicted|unverified","actual":null}}],"summary":"2 sentences","quality_concerns":["..."],"has_code_output":bool,"has_papers":bool}}

Rules: is_substantive=false if only procedural ("Let me.."). metrics type: result=measured output, hyperparameter=config. claims_vs_stdout: compare output claims vs stdout numbers. quality_concerns: missing seeds/baselines/error bars. IMPORTANT: evaluate based on the worker's specific role, not the overall mission goal."""

        result = self._call_llm_json(system_prompt, user_prompt)

        # If LLM call failed, signal it so caller can fall back to keywords
        if not result.get("_parse_ok"):
            return {"_parse_ok": False}

        # Ensure all expected fields exist with safe defaults
        defaults = {
            "is_substantive": True,
            "task_completed": True,
            "metrics": [],
            "claims_vs_stdout": [],
            "summary": output[:500] if output else "",
            "quality_concerns": [],
            "has_code_output": bool(stdout_parts),
            "has_papers": False,
        }
        for key, default in defaults.items():
            if key not in result:
                result[key] = default

        return result

    # ── Call 2: Post-Cycle Progress Assessment ─────────────────────

    def assess_progress(self, goal: str, completed_tasks: list[dict],
                        workspace_files: list[str],
                        knowledge_summary: str = "",
                        working_memory: str = "") -> dict:
        """
        Assess overall mission progress after a cycle.

        Replaces: GoalTracker SUBGOAL_TYPES, paper/code/metric detection,
                  ResearchStandards trigger keywords, FlowMonitor word overlap

        Args:
            goal: Mission goal
            completed_tasks: List of completed task dicts
            workspace_files: List of file paths in workspace
            knowledge_summary: Brief knowledge tree summary
            working_memory: Current distilled working memory

        Returns:
            {
                "progress_pct": int,          # 0-100
                "sub_goals_completed": [str],  # what's done
                "sub_goals_remaining": [str],  # what's left
                "quality_assessment": str,     # quality evaluation
                "recommended_next": [          # suggested next actions
                    {"worker": str, "task": str}
                ],
                "detected_issues": [str],      # stagnation, repetition, etc.
            }
        """
        # Build task summary
        task_lines = []
        for i, t in enumerate(completed_tasks[-10:]):  # last 10 tasks
            status = "OK" if t.get("success") else "FAIL"
            task_lines.append(
                f"{i+1}. [{status}] {t.get('worker','?')}: {t.get('task','')[:80]}"
            )
        tasks_text = "\n".join(task_lines) if task_lines else "(none)"

        files_text = "\n".join(f"  - {f}" for f in workspace_files[:30]) if workspace_files else "(empty)"

        system_prompt = (
            "You are a research progress assessor. Analyze mission state "
            "and return ONLY valid JSON. Works with ANY language and domain."
        )

        user_prompt = f"""Assess progress toward this research goal.

## Goal
{goal}

## Completed Tasks (recent)
{tasks_text}

## Workspace Files
{files_text}

## Knowledge
{knowledge_summary[:1000] if knowledge_summary else "(none)"}

## Working Memory
{working_memory[:1000] if working_memory else "(none)"}

Return ONLY this JSON:
{{
  "progress_pct": 0-100,
  "sub_goals_completed": ["literature review done", "baseline code running"],
  "sub_goals_remaining": ["run comparison experiments", "generate visualizations"],
  "quality_assessment": "Good progress but needs multiple seeds",
  "recommended_next": [
    {{"worker": "coder", "task": "Add 3-seed repetition to experiment"}},
    {{"worker": "reviewer", "task": "Verify baseline results"}}
  ],
  "detected_issues": ["stagnation: last 3 tasks very similar", "no code execution yet"]
}}

Rules:
- "progress_pct": estimate based on typical research workflow (literature → code → experiments → analysis → report)
- "sub_goals_completed": infer from tasks and files what milestones are done
- "sub_goals_remaining": what's still needed for a complete research output
- "quality_assessment": evaluate research rigor (seeds, baselines, dataset size, error bars)
- "recommended_next": 1-3 specific next actions, prioritized
- "detected_issues": flag repetitive tasks, stalled progress, worker imbalance, quality gaps"""

        result = self._call_llm_json(system_prompt, user_prompt)

        if not result.get("_parse_ok"):
            return {"_parse_ok": False}

        defaults = {
            "progress_pct": 0,
            "sub_goals_completed": [],
            "sub_goals_remaining": ["unknown"],
            "quality_assessment": "",
            "recommended_next": [],
            "detected_issues": [],
        }
        for key, default in defaults.items():
            if key not in result:
                result[key] = default

        return result

    # ── Call 3: Post-Mission Scoring ───────────────────────────────

    def score_mission(self, goal: str, workspace_files: list[str],
                      exec_summary: str = "", report_content: str = "",
                      completed_tasks: list[dict] = None) -> dict:
        """
        Score a completed mission across 6 dimensions.

        Replaces: MissionScorer regex patterns, section detection,
                  arXiv/DOI counting

        Args:
            goal: Mission goal
            workspace_files: List of files in workspace
            exec_summary: ExecutionLog summary text
            report_content: Final report markdown content
            completed_tasks: List of completed task dicts

        Returns:
            {
                "literature": {"score": 0-10, "evidence": str},
                "code": {"score": 0-10, "evidence": str},
                "results": {"score": 0-10, "evidence": str},
                "verification": {"score": 0-10, "evidence": str},
                "artifacts": {"score": 0-10, "evidence": str},
                "report": {"score": 0-10, "evidence": str},
                "overall": float,
                "grade": str,
            }
        """
        files_text = "\n".join(f"  - {f}" for f in workspace_files[:40]) if workspace_files else "(empty)"

        # Build task summary
        task_summary = ""
        if completed_tasks:
            parts = []
            for t in completed_tasks:
                status = "OK" if t.get("success") else "FAIL"
                parts.append(f"[{status}] {t.get('worker','?')}: {t.get('task','')[:60]}")
            task_summary = "\n".join(parts[-15:])  # last 15

        system_prompt = (
            "You are a research mission quality scorer. Analyze the mission outputs "
            "and score across 6 dimensions. Return ONLY valid JSON. "
            "Works with ANY language, domain, and research format."
        )

        user_prompt = f"""Score this completed research mission.

## Goal
{goal}

## Workspace Files
{files_text}

## Execution Summary
{exec_summary[:2000] if exec_summary else "(no execution log)"}

## Report
{report_content[:3000] if report_content else "(no report)"}

## Task History
{task_summary or "(none)"}

Return ONLY this JSON:
{{
  "literature": {{"score": 0-10, "evidence": "Found N papers with citations from M sources"}},
  "code": {{"score": 0-10, "evidence": "N Python files, M/K runs succeeded"}},
  "results": {{"score": 0-10, "evidence": "Clear metrics with N-seed averages"}},
  "verification": {{"score": 0-10, "evidence": "Most claims match stdout / claims unverifiable"}},
  "artifacts": {{"score": 0-10, "evidence": "N figures, M data files"}},
  "report": {{"score": 0-10, "evidence": "Well-structured with all sections / missing sections"}},
  "overall": 0.0-10.0,
  "grade": "A/B/C/D/F"
}}

Scoring guide:
- literature: 0=no papers, 5=some papers found, 10=comprehensive review with citations
- code: 0=no code, 5=code exists but errors, 10=clean code, all runs pass
- results: 0=no results, 5=basic results, 10=multi-seed with error bars and comparisons
- verification: 0=claims fabricated, 5=some verified, 10=all claims match execution output
- artifacts: 0=no files, 5=code files only, 10=figures + data + models
- report: 0=no report, 5=basic report, 10=well-structured with all sections
- overall: weighted average (results 25%, code 20%, literature 15%, verification 15%, report 15%, artifacts 10%)
- grade: A>=8.5, B>=7.0, C>=5.0, D>=3.0, F<3.0"""

        result = self._call_llm_json(system_prompt, user_prompt)

        if not result.get("_parse_ok"):
            raise ValueError("LLM Judge JSON parse failed for score_mission")

        # Validate dimension structure
        dimensions = ["literature", "code", "results", "verification", "artifacts", "report"]
        for dim in dimensions:
            if dim not in result or not isinstance(result[dim], dict):
                result[dim] = {"score": 0, "evidence": "not evaluated"}
            else:
                # Clamp score to 0-10
                score = result[dim].get("score", 0)
                if isinstance(score, (int, float)):
                    result[dim]["score"] = max(0, min(10, score))
                else:
                    result[dim]["score"] = 0
                if "evidence" not in result[dim]:
                    result[dim]["evidence"] = ""

        # Calculate overall if not provided or invalid
        weights = {
            "literature": 0.15, "code": 0.20, "results": 0.25,
            "verification": 0.15, "artifacts": 0.10, "report": 0.15,
        }
        if "overall" not in result or not isinstance(result.get("overall"), (int, float)):
            result["overall"] = sum(
                result[dim]["score"] * weights[dim] for dim in dimensions
            )
        result["overall"] = round(max(0, min(10, result["overall"])), 2)

        # Calculate grade if not provided
        if "grade" not in result or result["grade"] not in ("A", "B", "C", "D", "F"):
            s = result["overall"]
            if s >= 8.5:
                result["grade"] = "A"
            elif s >= 7.0:
                result["grade"] = "B"
            elif s >= 5.0:
                result["grade"] = "C"
            elif s >= 3.0:
                result["grade"] = "D"
            else:
                result["grade"] = "F"

        return result

    # ── Internal: LLM call with JSON parsing ───────────────────────

    def _call_llm_json(self, system_prompt: str, user_prompt: str,
                        max_tokens: int = 1500) -> dict:
        """
        Make a single LLM call expecting a JSON response.
        Returns dict with "_parse_ok" key indicating if parsing succeeded.
        Callers MUST check _parse_ok before trusting results.
        """
        try:
            # Use a fresh client config for judge calls — lower max_tokens for speed
            response = self.llm.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            raw = response["choices"][0]["message"]["content"]
            clean = strip_think(raw)

            # Try to find JSON object in response
            json_match = re.search(r'\{[\s\S]*\}', clean)
            if json_match:
                result = json.loads(json_match.group())
                result["_parse_ok"] = True
                return result

            # Try the whole text
            result = json.loads(clean)
            result["_parse_ok"] = True
            return result

        except json.JSONDecodeError as e:
            print(f"  [LLMJudge] JSON parse failed: {e}")
            return {"_parse_ok": False}
        except Exception as e:
            print(f"  [LLMJudge] LLM call failed: {e}")
            return {"_parse_ok": False}
