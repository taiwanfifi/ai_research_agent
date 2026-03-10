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
from core.error_patterns import (
    classify_error, get_escalation_level, compress_failure_history,
    should_abandon_direction,
)


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
    "patch_env", "abort_branch", "simplify",
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
    # ── Layer 1: Deterministic error classification (zero LLM cost) ──
    error_text = stderr or error or ""
    error_pattern = classify_error(error_text)
    consecutive = len(prior_failures) if prior_failures else 0

    if error_pattern:
        escalation = get_escalation_level(consecutive, error_pattern)
        print(f"  [FailureAnalyzer] Deterministic: {error_pattern.name} "
              f"({error_pattern.category}) → {escalation}")

        # Map escalation levels to FailureAnalysis actions
        if escalation == "RETRY_WITH_FIX":
            fc = "env_bug" if error_pattern.category in ("import", "device") else "wrong_approach"
            return FailureAnalysis(
                failure_class=fc,
                root_cause=f"{error_pattern.name}: {error_pattern.fix_hint}",
                evidence=[error_text[-300:]],
                next_action="retry_modified",
                modification=error_pattern.fix_hint,
                subtasks=[],
            )
        elif escalation == "RETRY_WITH_AMNESIA":
            # Compress failure history to break context loops
            if prior_failures:
                history_strs = [pf.get("root_cause", "") for pf in prior_failures]
                compressed = compress_failure_history(history_strs)
                print(f"  [FailureAnalyzer] Amnesia: compressed {len(prior_failures)} failures")
            return FailureAnalysis(
                failure_class="wrong_approach",
                root_cause=f"Repeated {error_pattern.name} — try fresh approach. Hint: {error_pattern.fix_hint}",
                evidence=[error_text[-200:]],
                next_action="simplify",
                modification=f"Fresh approach needed. {error_pattern.fix_hint}",
                subtasks=[],
            )
        elif escalation == "GENERATE_ALTERNATIVES":
            return FailureAnalysis(
                failure_class="wrong_approach",
                root_cause=f"{consecutive}x {error_pattern.name} — fundamental approach issue",
                evidence=[error_text[-200:]],
                next_action="decompose",
                modification="Try 3 fundamentally different approaches",
                subtasks=[
                    f"Alternative 1: Simplest possible version ({error_pattern.fix_hint})",
                    f"Alternative 2: Different architecture/method entirely",
                    f"Alternative 3: Minimal reproduction to isolate the bug",
                ],
            )
        elif escalation == "BACKTRACK_TO_PARENT":
            return FailureAnalysis(
                failure_class="wrong_approach",
                root_cause=f"{consecutive}x failures on {error_pattern.category} — abandon this direction",
                evidence=[error_text[-200:]],
                next_action="abort_branch",
                modification="",
                subtasks=[],
            )

    # Check if we should abandon based on error history pattern
    if prior_failures and len(prior_failures) >= 3:
        history_strs = [pf.get("root_cause", "") or pf.get("error", "") for pf in prior_failures]
        if should_abandon_direction(history_strs):
            print(f"  [FailureAnalyzer] 3+ same-category errors → abort branch")
            return FailureAnalysis(
                failure_class="wrong_approach",
                root_cause="Same error category repeated 3+ times — method is fundamentally broken",
                evidence=history_strs[-2:],
                next_action="abort_branch",
                modification="",
                subtasks=[],
            )

    # ── Layer 2: LLM-based diagnosis (for unclassified errors) ──
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
- wrong_approach: fundamentally incorrect method (e.g. BatchNorm on transformer, wrong model class)
- dependency: missing file, missing package, missing prior task output
- bad_assumption: assumed something that isn't true
- decompose: split into 2-3 smaller subtasks (provide them)
- simplify: current approach is too complex for this environment — use simpler APIs/models/architecture. Provide the simplified approach in "modification".
- abort_branch: this line of work is not viable

When to recommend "simplify":
- HuggingFace Trainer keeps failing → use raw PyTorch training loop
- Complex model architecture incompatible → use simpler model
- Multiple API errors that suggest the framework is fighting the task
- 2+ prior failures on same task suggest complexity is the problem"""

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


def build_reflexion_context(task_desc: str, completed_tasks: list) -> str:
    """Build cumulative failure history for a task that has failed multiple times.

    Implements the ADAS reflexion pattern: instead of only showing the last failure,
    compile ALL past attempts so the next try can learn from every mistake.

    Args:
        task_desc: current task description (first 100 chars used for matching)
        completed_tasks: list of completed task dicts from supervisor

    Returns:
        Formatted string with failure history, or empty string if <2 failures
    """
    # Find all past failures on similar tasks (match first 100 chars)
    task_prefix = task_desc[:100]
    failures = []
    for t in completed_tasks:
        if not t.get("success") and t.get("task", "")[:100] == task_prefix:
            fa = t.get("failure_analysis", {})
            failures.append({
                "task_variant": t.get("task", "")[:300],
                "root_cause": fa.get("root_cause", t.get("error", "unknown")[:200]),
                "failure_class": fa.get("failure_class", "unknown"),
                "modification_tried": fa.get("modification", ""),
                "evidence": fa.get("evidence", [])[:2],
            })

    if len(failures) < 2:
        return ""

    lines = [
        f"## REFLEXION: {len(failures)} PRIOR FAILURES on this task",
        "Learn from ALL past attempts — do NOT repeat any approach that already failed.",
        ""
    ]
    for i, f in enumerate(failures, 1):
        lines.append(f"### Attempt {i}")
        lines.append(f"- **Root cause**: {f['root_cause']}")
        lines.append(f"- **Class**: {f['failure_class']}")
        if f['modification_tried']:
            lines.append(f"- **Fix tried**: {f['modification_tried']}")
        if f['evidence']:
            lines.append(f"- **Error snippets**: {'; '.join(f['evidence'][:2])}")
        lines.append("")

    lines.append("### What to do differently")
    lines.append("- Each past attempt failed for a DIFFERENT reason — the problem is likely more fundamental than any single fix")
    lines.append("- Consider simplifying the approach entirely rather than patching")
    lines.append("- Test each component in isolation before combining")
    lines.append("")

    return "\n".join(lines)


def _fallback_analysis(error: str, is_retry: bool) -> FailureAnalysis:
    """Mechanical fallback when LLM is unavailable.
    Uses deterministic error patterns first, then keyword heuristics."""
    # Try deterministic classification
    pattern = classify_error(error or "")
    if pattern:
        na = "retry_modified" if pattern.auto_fixable else "simplify"
        if is_retry:
            na = "simplify"
        return FailureAnalysis(
            failure_class="env_bug" if pattern.category in ("import", "device") else "wrong_approach",
            root_cause=f"{pattern.name}: {pattern.fix_hint}",
            evidence=[error[-200:]] if error else [],
            next_action=na,
            modification=pattern.fix_hint,
            subtasks=[],
        )

    # Keyword heuristics
    error_lower = (error or "").lower()

    if "timeout" in error_lower or "timed out" in error_lower:
        fc, na = "task_too_big", "decompose"
    elif "import" in error_lower or "module" in error_lower:
        fc, na = "dependency", "patch_env"
    elif "deprecated" in error_lower or "removed" in error_lower:
        fc, na = "env_bug", "retry_modified"
    elif is_retry:
        fc, na = "wrong_approach", "simplify"
    else:
        fc, na = "unknown", "retry_modified"

    mod = ""
    if na == "simplify":
        mod = "Use raw PyTorch instead of HuggingFace Trainer. Reduce model complexity."

    return FailureAnalysis(
        failure_class=fc,
        root_cause=error[:200] if error else "unknown error",
        evidence=[],
        next_action=na,
        modification=mod,
        subtasks=[],
    )
