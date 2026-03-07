#!/usr/bin/env python3
"""
Pre-Check Simulator — Dry-Run A/B Test
=========================================
Replays historical mission checkpoints and simulates what a structured
pre-check would have caught vs what actually happened.

Three pre-check strategies tested:
  A) No pre-check (baseline — current Opus behavior)
  B) Dependency + File Validator (lightweight)
  C) Dependency + File + Resource Estimator + Friction Buffer (daemon-inspired)

Usage:
    python3 tools/precheck_simulator.py
    python3 tools/precheck_simulator.py --mission 092305
"""

import json
import os
import re
import glob
import sys

MISSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "missions")


def load_checkpoint(mission_dir: str) -> dict:
    cp_path = os.path.join(mission_dir, "state", "mission", "latest_checkpoint.json")
    if not os.path.exists(cp_path):
        return {}
    with open(cp_path) as f:
        data = json.load(f)
    return data.get("value", data)


def list_workspace_files(mission_dir: str) -> set:
    ws = os.path.join(mission_dir, "workspace")
    if not os.path.isdir(ws):
        return set()
    files = set()
    for root, _, fnames in os.walk(ws):
        for fn in fnames:
            if "__pycache__" not in root and ".code_store" not in root:
                files.add(fn)
    return files


# ── Pre-Check Strategy B: Dependency + File Validator ──────────────

def precheck_b(task: dict, completed: list, workspace_files: set) -> dict:
    """Lightweight pre-check: verify dependencies met and referenced files exist."""
    issues = []

    task_desc = task.get("task", "").lower()
    worker = task.get("worker", "")
    depends_on = task.get("depends_on", [])

    # 1. Check if depends_on tasks succeeded
    for dep_idx in depends_on:
        if dep_idx < len(completed):
            dep_task = completed[dep_idx]
            if not dep_task.get("success"):
                issues.append(f"DEPENDENCY_FAILED: depends on task {dep_idx} which failed")

    # 2. Check if task references files that should exist
    # Pattern: "load X.json", "read X.py", "import from X.py"
    file_refs = re.findall(r'(?:load|read|import|open|from)\s+["\']?(\w+\.\w+)', task_desc)
    for ref in file_refs:
        if ref not in workspace_files and not ref.startswith("torch") and not ref.startswith("json"):
            issues.append(f"MISSING_FILE: task references '{ref}' but it doesn't exist in workspace")

    # 3. Check if task says "load all result JSON files" but none exist
    if ("load all result" in task_desc or "load all seed" in task_desc) and worker == "reviewer":
        json_files = [f for f in workspace_files if f.endswith(".json")
                      and f not in ("execution_log.json", "mission_score.json", "analysis_summary.json")]
        result_jsons = [f for f in json_files if "result" in f or "seed" in f]
        if not result_jsons:
            issues.append("MISSING_RESULTS: task wants to load result JSONs but none exist")

    # 4. Check if reviewer references metrics from prior tasks that don't exist
    if worker == "reviewer":
        completed_coders = [t for t in completed if t.get("worker") == "coder" and t.get("success")]
        if not completed_coders:
            issues.append("NO_CODER_OUTPUT: reviewer task but no successful coder tasks completed yet")

    return {
        "would_block": len(issues) > 0,
        "issues": issues,
        "strategy": "B",
    }


# ── Pre-Check Strategy C: B + Resource Estimator + Friction ────────

def precheck_c(task: dict, completed: list, workspace_files: set,
               friction_buffer: list) -> dict:
    """Daemon-inspired pre-check: B + resource estimation + friction awareness."""
    # Start with Strategy B
    result_b = precheck_b(task, completed, workspace_files)
    issues = list(result_b["issues"])

    task_desc = task.get("task", "").lower()
    worker = task.get("worker", "")

    # 5. Resource estimation for training tasks
    if worker == "coder" and any(w in task_desc for w in ["train", "fine-tun", "epoch"]):
        # Check for dataset size mentions
        size_match = re.search(r'(\d+)\s*(?:samples|examples|rows)', task_desc)
        epoch_match = re.search(r'(\d+)\s*epoch', task_desc)

        if size_match and epoch_match:
            n_samples = int(size_match.group(1))
            n_epochs = int(epoch_match.group(1))
            # Rough estimate: 0.15s per sample per epoch on CPU for transformer-sized models
            est_time = n_samples * n_epochs * 0.15
            if est_time > 500:
                issues.append(f"TIMEOUT_RISK: estimated {est_time:.0f}s for {n_samples} samples × {n_epochs} epochs (limit: 600s)")
        elif not size_match and "select" not in task_desc and "subset" not in task_desc:
            issues.append("NO_SUBSET: training task without explicit dataset size — risk of full dataset timeout")

    # 6. Friction-aware: check if similar tasks have failed before
    for friction in friction_buffer:
        friction_lower = friction.get("pattern", "").lower()
        # Check word overlap between friction and current task
        friction_words = set(friction_lower.split())
        task_words = set(task_desc.split())
        overlap = len(friction_words & task_words) / max(len(friction_words | task_words), 1)
        if overlap > 0.3:
            issues.append(f"FRICTION_MATCH: similar to known failure: {friction.get('pattern', '')[:80]}")

    # 7. Post-observation judgment: if task is a retry, check what changed
    similar_completed = [t for t in completed
                        if t.get("worker") == worker and not t.get("success")
                        and _task_similarity(t.get("task", ""), task.get("task", "")) > 0.5]
    if similar_completed:
        issues.append(f"RETRY_WITHOUT_CHANGE: similar task already failed ({len(similar_completed)}x). Strategy change needed?")

    return {
        "would_block": len(issues) > 0,
        "issues": issues,
        "strategy": "C",
    }


def _task_similarity(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0
    return len(words_a & words_b) / max(len(words_a | words_b), 1)


# ── Simulator ──────────────────────────────────────────────────────

def simulate_mission(mission_dir: str, verbose: bool = True) -> dict:
    """Replay a mission and test all three strategies."""
    cp = load_checkpoint(mission_dir)
    if not cp:
        return {}

    mission_id = os.path.basename(mission_dir)
    goal = cp.get("goal", "?")
    completed = cp.get("completed_tasks", [])
    queue = cp.get("task_queue", [])
    all_tasks = completed + [t for t in queue if t.get("status") == "pending"]
    ws_files = list_workspace_files(mission_dir)

    # Build friction buffer from failed tasks (Strategy C)
    friction_buffer = []
    for t in completed:
        if not t.get("success"):
            friction_buffer.append({
                "pattern": f"{t.get('worker', '?')} failed: {(t.get('error') or 'unknown')[:100]}",
                "task": t.get("task", "")[:100],
            })

    results = {"A": {"caught": 0, "missed": 0, "false_positive": 0},
               "B": {"caught": 0, "missed": 0, "false_positive": 0},
               "C": {"caught": 0, "missed": 0, "false_positive": 0}}

    if verbose:
        print(f"\n{'='*70}")
        print(f"Mission: {mission_id}")
        print(f"Goal: {goal[:80]}")
        print(f"Tasks: {len(completed)} completed, {len(all_tasks)} total")
        print(f"{'='*70}")

    # Simulate each task execution in order
    simulated_completed = []
    simulated_ws_files = set()
    simulated_friction = []

    for i, task in enumerate(completed):
        actual_success = task.get("success", False)
        worker = task.get("worker", "?")
        task_desc = task.get("task", "")[:80]

        # Strategy A: no pre-check (baseline)
        # Everything goes through, failures happen at runtime

        # Strategy B: dependency + file check
        check_b = precheck_b(task, simulated_completed, simulated_ws_files)

        # Strategy C: B + resource + friction
        check_c = precheck_c(task, simulated_completed, simulated_ws_files, simulated_friction)

        # Evaluate
        for strategy, check in [("B", check_b), ("C", check_c)]:
            if check["would_block"] and not actual_success:
                results[strategy]["caught"] += 1
            elif check["would_block"] and actual_success:
                results[strategy]["false_positive"] += 1
            elif not check["would_block"] and not actual_success:
                results[strategy]["missed"] += 1

        if not actual_success:
            results["A"]["missed"] += 1

        if verbose:
            status = "OK" if actual_success else "FAIL"
            b_flag = " [B:BLOCK]" if check_b["would_block"] else ""
            c_flag = " [C:BLOCK]" if check_c["would_block"] else ""
            print(f"  {i+1}. [{status}] {worker}: {task_desc}{b_flag}{c_flag}")
            if check_c["issues"] and (not actual_success or check_c["would_block"]):
                for issue in check_c["issues"]:
                    print(f"       → {issue}")

        # Update simulation state
        simulated_completed.append(task)
        if actual_success and worker == "coder":
            # Simulate files being created
            file_refs = re.findall(r'(?:save|write|create)\s+(?:as\s+|to\s+)?["\']?(\w+\.\w+)',
                                   task.get("task", "").lower())
            simulated_ws_files.update(file_refs)
        if not actual_success:
            simulated_friction.append({
                "pattern": f"{worker} failed: {(task.get('error') or 'unknown')[:100]}",
                "task": task.get("task", "")[:100],
            })

    # Summary
    total_failures = sum(1 for t in completed if not t.get("success"))
    if verbose:
        print(f"\n  Summary: {total_failures} failures out of {len(completed)} tasks")
        for s in ["A", "B", "C"]:
            r = results[s]
            caught = r["caught"]
            missed = r["missed"]
            fp = r["false_positive"]
            rate = caught / max(caught + missed, 1) * 100
            print(f"  Strategy {s}: caught {caught}/{caught+missed} failures ({rate:.0f}%), "
                  f"false positives: {fp}")

    return {
        "mission_id": mission_id,
        "goal": goal,
        "total_tasks": len(completed),
        "total_failures": total_failures,
        "results": results,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Pre-Check A/B Test Simulator")
    parser.add_argument("--mission", type=str, default=None, help="Partial mission ID match")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Find missions
    mission_dirs = sorted(glob.glob(os.path.join(MISSIONS_DIR, "mission_*")))
    if args.mission:
        mission_dirs = [d for d in mission_dirs if args.mission in os.path.basename(d)]

    if not mission_dirs:
        print("No missions found.")
        return

    all_results = []
    for md in mission_dirs:
        cp = load_checkpoint(md)
        if not cp or not cp.get("completed_tasks"):
            continue
        result = simulate_mission(md, verbose=not args.quiet)
        if result:
            all_results.append(result)

    # Aggregate
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"AGGREGATE RESULTS ({len(all_results)} missions)")
        print(f"{'='*70}")

        totals = {s: {"caught": 0, "missed": 0, "false_positive": 0} for s in ["A", "B", "C"]}
        total_failures = 0
        total_tasks = 0

        for r in all_results:
            total_failures += r["total_failures"]
            total_tasks += r["total_tasks"]
            for s in ["A", "B", "C"]:
                for k in ["caught", "missed", "false_positive"]:
                    totals[s][k] += r["results"][s][k]

        print(f"Total: {total_tasks} tasks, {total_failures} failures across {len(all_results)} missions")
        print()
        print(f"  {'Strategy':<12} {'Caught':<10} {'Missed':<10} {'FalsePos':<10} {'Catch Rate':<12} {'Precision':<10}")
        print(f"  {'-'*62}")
        for s in ["A", "B", "C"]:
            t = totals[s]
            catch_rate = t["caught"] / max(t["caught"] + t["missed"], 1) * 100
            precision = t["caught"] / max(t["caught"] + t["false_positive"], 1) * 100
            print(f"  {s:<12} {t['caught']:<10} {t['missed']:<10} {t['false_positive']:<10} "
                  f"{catch_rate:<12.1f}% {precision:<10.1f}%")

        print()
        print("  A = No pre-check (current Opus)")
        print("  B = Dependency + File Validator")
        print("  C = B + Resource Estimator + Friction Buffer (daemon-inspired)")


if __name__ == "__main__":
    main()
