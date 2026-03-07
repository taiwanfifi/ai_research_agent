"""
Failure Analyzer — Structured Failure Diagnosis
=================================================
Replaces blind retry/skip with causal analysis.
On any task failure, produces a structured diagnosis that
the supervisor uses to decide the next action.

Uses LLM for diagnosis but with a tight schema —
no open-ended generation.
"""

import json
import re
from dataclasses import dataclass, asdict


@dataclass
class FailureAnalysis:
    """Structured diagnosis of a task failure."""
    failure_class: str  # env_bug|task_too_big|wrong_approach|dependency|bad_assumption|unknown
    root_cause: str     # concise diagnosis
    evidence: list      # stderr snippets, error messages
    next_action: str    # retry_modified|decompose|switch_worker|patch_env|abort_branch
    modification: str   # what to change if retry_modified
    subtasks: list      # if decompose, list of smaller tasks

    def to_dict(self) -> dict:
        return asdict(self)


FAILURE_CLASSES = {
    "env_bug", "task_too_big", "wrong_approach",
    "dependency", "bad_assumption", "unknown",
}

NEXT_ACTIONS = {
    "retry_modified", "decompose", "switch_worker",
    "patch_env", "abort_branch",
}


def analyze_failure(llm, task: str, worker: str, error: str,
                    stderr: str = "", stdout: str = "",
                    envelope: dict = None,
                    prior_failures: list = None) -> FailureAnalysis:
    """Analyze a task failure and produce structured diagnosis.

    Args:
        llm: MiniMaxClient instance
        task: the failed task description
        worker: worker type (coder/reviewer/explorer)
        error: error message from worker
        stderr: raw stderr if available
        stdout: raw stdout if available
        envelope: DecisionEnvelope dict if available
        prior_failures: list of prior FailureAnalysis dicts for anti-retry

    Returns:
        FailureAnalysis with diagnosis and recommended action
    """
    # Build context
    parts = [f"Task: {task[:500]}"]
    parts.append(f"Worker: {worker}")
    parts.append(f"Error: {error[:500]}")
    if stderr:
        parts.append(f"Stderr (last 500 chars): {stderr[-500:]}")
    if stdout:
        parts.append(f"Stdout (last 500 chars): {stdout[-500:]}")
    if envelope:
        parts.append(f"Decision envelope: {json.dumps(envelope)}")

    # Check for exact retry (same task failed before)
    is_retry = False
    if prior_failures:
        for pf in prior_failures:
            if pf.get("task", "")[:100] == task[:100]:
                is_retry = True
                parts.append(f"PRIOR FAILURE on same task: {pf.get('root_cause', '?')}")
                parts.append(f"Prior action taken: {pf.get('next_action', '?')}")
                break

    prompt = f"""A research agent task failed. Diagnose the failure and recommend the next action.

{chr(10).join(parts)}

{"CRITICAL: This task already failed before with the same approach. Do NOT recommend retry_modified unless the modification is fundamentally different." if is_retry else ""}

Respond with JSON only:
{{
  "failure_class": "env_bug|task_too_big|wrong_approach|dependency|bad_assumption|unknown",
  "root_cause": "one sentence diagnosis",
  "evidence": ["key error snippet 1", "key error snippet 2"],
  "next_action": "retry_modified|decompose|switch_worker|patch_env|abort_branch",
  "modification": "what to change (if retry_modified, be specific)",
  "subtasks": ["subtask 1", "subtask 2"]
}}

Rules:
- env_bug: API changed, platform incompatibility, version mismatch
- task_too_big: timeout, memory, too many steps for one task
- wrong_approach: fundamentally incorrect method
- dependency: missing file, missing package, missing prior task output
- bad_assumption: assumed something that isn't true
- decompose: split into 2-3 smaller subtasks (provide them)
- abort_branch: this line of work is not viable"""

    try:
        response = llm.chat([
            {"role": "system", "content": "Diagnose task failures. JSON only. Be specific."},
            {"role": "user", "content": prompt},
        ])
        raw = response["choices"][0]["message"]["content"]

        # Strip thinking tags if present
        from core.llm import strip_think
        raw = strip_think(raw)

        # Parse JSON
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            return _fallback_analysis(error, is_retry)

        data = json.loads(json_match.group())

        # Validate fields
        fc = data.get("failure_class", "unknown")
        if fc not in FAILURE_CLASSES:
            fc = "unknown"
        na = data.get("next_action", "retry_modified")
        if na not in NEXT_ACTIONS:
            na = "retry_modified"

        # Block exact retry if this already failed
        if is_retry and na == "retry_modified" and not data.get("modification"):
            na = "decompose"

        return FailureAnalysis(
            failure_class=fc,
            root_cause=data.get("root_cause", error[:200]),
            evidence=data.get("evidence", [])[:5],
            next_action=na,
            modification=data.get("modification", ""),
            subtasks=data.get("subtasks", [])[:3],
        )

    except Exception as e:
        print(f"  [FailureAnalyzer] LLM analysis failed ({e}), using fallback")
        return _fallback_analysis(error, is_retry)


def _fallback_analysis(error: str, is_retry: bool) -> FailureAnalysis:
    """Mechanical fallback when LLM is unavailable."""
    error_lower = (error or "").lower()

    if "timeout" in error_lower or "timed out" in error_lower:
        fc, na = "task_too_big", "decompose"
    elif "import" in error_lower or "module" in error_lower:
        fc, na = "dependency", "patch_env"
    elif "deprecated" in error_lower or "removed" in error_lower:
        fc, na = "env_bug", "retry_modified"
    elif is_retry:
        fc, na = "wrong_approach", "decompose"
    else:
        fc, na = "unknown", "retry_modified"

    return FailureAnalysis(
        failure_class=fc,
        root_cause=error[:200] if error else "unknown error",
        evidence=[],
        next_action=na,
        modification="",
        subtasks=[],
    )
