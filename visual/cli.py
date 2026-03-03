#!/usr/bin/env python3
"""Terminal CLI viewer for Research AI missions.

Usage:
    python3 visual/cli.py                                     # list all missions
    python3 visual/cli.py --mission <id>                      # mission overview
    python3 visual/cli.py --mission <id> --tasks              # completed tasks
    python3 visual/cli.py --mission <id> --insights           # insight DAG
    python3 visual/cli.py --mission <id> --code               # code versions
    python3 visual/cli.py --mission <id> --knowledge          # knowledge tree
    python3 visual/cli.py --mission <id> --reports            # report list
    python3 visual/cli.py --watch                             # auto-refresh every 5s
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
VISUAL_DIR = Path(__file__).resolve().parent
DEFAULT_MISSIONS = VISUAL_DIR.parent / "missions"

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    # Workers
    BLUE    = "\033[38;5;33m"   # Explorer
    GREEN   = "\033[38;5;35m"   # Coder
    AMBER   = "\033[38;5;214m"  # Reviewer
    # Status
    RED     = "\033[38;5;196m"
    CYAN    = "\033[38;5;75m"
    YELLOW  = "\033[38;5;220m"
    MAGENTA = "\033[38;5;141m"
    WHITE   = "\033[38;5;255m"
    GRAY    = "\033[38;5;245m"

WORKER_COLORS = {
    "explorer": C.BLUE,
    "coder": C.GREEN,
    "reviewer": C.AMBER,
}

STATUS_COLORS = {
    "finished": C.GREEN,
    "running": C.YELLOW,
    "error": C.RED,
    "planning": C.MAGENTA,
}


def colored(text, color):
    return f"{color}{text}{C.RESET}"


def worker_badge(worker):
    c = WORKER_COLORS.get(worker, C.GRAY)
    return colored(f"[{worker.upper()}]", c + C.BOLD) if worker else ""


def status_badge(status):
    c = STATUS_COLORS.get(status, C.GRAY)
    return colored(f"[{status.upper()}]", c + C.BOLD) if status else ""


def progress_bar(current, total, width=20):
    if total <= 0:
        return "[" + "░" * width + "] 0/0"
    filled = round(current / total * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = round(current / total * 100)
    return f"[{bar}] {current}/{total} ({pct}%)"


def relevance_bar(value, width=10):
    filled = round(value * width)
    bar = "▓" * filled + "░" * (width - filled)
    return f"{bar} {value:.2f}"


# ---------------------------------------------------------------------------
# Data reading (mirrors server.py logic)
# ---------------------------------------------------------------------------
def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_checkpoint(mission_dir):
    cp = read_json(mission_dir / "state" / "mission" / "latest_checkpoint.json")
    if cp is None:
        return None
    if isinstance(cp, dict) and "value" in cp:
        return cp["value"]
    return cp


def get_missions(missions_dir):
    results = []
    if not missions_dir.is_dir():
        return results
    for entry in sorted(missions_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("mission_"):
            continue
        manifest = read_json(entry / "mission.json") or {"mission_id": entry.name}
        cp = load_checkpoint(entry)
        results.append({
            "id": entry.name,
            "dir": entry,
            "manifest": manifest,
            "checkpoint": cp,
        })
    return results


def synthesize_dag(completed_tasks):
    nodes = {}
    for i, task in enumerate(completed_tasks or []):
        nid = f"i{i:04d}"
        nodes[nid] = {
            "id": nid,
            "cycle": i + 1,
            "worker": task.get("worker", "explorer"),
            "task": (task.get("task", "") or "")[:100],
            "success": task.get("success", True),
            "content": task.get("output", "") or "",
            "references": [f"i{i-1:04d}"] if i > 0 else [],
            "relevance": 0.5,
            "archived": False,
            "code_refs": [],
            "synthetic": True,
        }
    return {"next_id": len(completed_tasks or []), "nodes": nodes}


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------

def print_header(text):
    print(f"\n{C.BOLD}{text}{C.RESET}")
    print("─" * min(len(text) + 4, 72))


def show_mission_list(missions):
    print_header("Research AI Missions")

    if not missions:
        print(colored("  No missions found.", C.DIM))
        return

    # Table header
    fmt = "  {:<50s} {:>8s} {:>10s} {:>6s}"
    print(colored(fmt.format("GOAL", "STATUS", "PROGRESS", "TASKS"), C.DIM))
    print(colored("  " + "─" * 78, C.DIM))

    for m in missions:
        manifest = m["manifest"]
        cp = m["checkpoint"] or {}
        goal = (manifest.get("goal", "") or manifest.get("slug", "") or m["id"])[:48]
        status = manifest.get("status", "?")
        cycle = cp.get("cycle", 0)
        max_c = cp.get("max_cycles", 0)
        tasks = len(cp.get("completed_tasks", []))

        sc = STATUS_COLORS.get(status, C.GRAY)
        prog = f"{cycle}/{max_c}" if max_c else "—"
        print(f"  {goal:<50s} {colored(f'{status:>8s}', sc)} {prog:>10s} {tasks:>6d}")

    print()
    print(colored(f"  {len(missions)} mission(s) total", C.DIM))


def show_mission_overview(mission):
    manifest = mission["manifest"]
    cp = mission["checkpoint"] or {}

    print_header(f"Mission: {manifest.get('slug', mission['id'])}")

    print(f"  {C.BOLD}Goal:{C.RESET}     {manifest.get('goal', '—')}")
    print(f"  {C.BOLD}Status:{C.RESET}   {status_badge(manifest.get('status', '?'))}")
    print(f"  {C.BOLD}Language:{C.RESET} {manifest.get('language', '?')}")
    print(f"  {C.BOLD}Created:{C.RESET}  {manifest.get('created_at', '—')}")
    print()

    cycle = cp.get("cycle", 0)
    max_c = cp.get("max_cycles", 0)
    print(f"  {C.BOLD}Progress:{C.RESET} {progress_bar(cycle, max_c)}")
    print(f"  {C.BOLD}State:{C.RESET}    {cp.get('state', '—')}")
    print(f"  {C.BOLD}Tasks:{C.RESET}    {len(cp.get('completed_tasks', []))} completed, {len(cp.get('task_queue', []))} queued")
    print(f"  {C.BOLD}Errors:{C.RESET}   {len(cp.get('errors', []))}")
    print(f"  {C.BOLD}Reports:{C.RESET}  {cp.get('reports_generated', 0)}")

    # Brief task summary
    completed = cp.get("completed_tasks", [])
    if completed:
        print()
        print(f"  {C.BOLD}Recent tasks:{C.RESET}")
        for t in completed[-5:]:
            w = worker_badge(t.get("worker", ""))
            ok = colored("✓", C.GREEN) if t.get("success") else colored("✗", C.RED)
            task_text = (t.get("task", "") or "")[:60]
            elapsed = t.get("elapsed_s")
            dur = f" ({elapsed:.0f}s)" if elapsed else ""
            print(f"    {ok} {w} {task_text}{colored(dur, C.DIM)}")


def show_tasks(mission):
    cp = mission["checkpoint"] or {}
    completed = cp.get("completed_tasks", [])

    print_header("Completed Tasks")

    if not completed:
        print(colored("  No completed tasks.", C.DIM))
        return

    for i, t in enumerate(completed):
        w = worker_badge(t.get("worker", ""))
        ok = colored("✓", C.GREEN) if t.get("success") else colored("✗", C.RED)
        elapsed = t.get("elapsed_s")
        dur = f" ({elapsed:.0f}s)" if elapsed else ""

        print(f"\n  {C.BOLD}#{i+1}{C.RESET} {ok} {w}{colored(dur, C.DIM)}")
        print(f"  {t.get('task', '—')}")

        output = (t.get("output", "") or "").strip()
        if output:
            # Show first 3 lines of output
            lines = output.split("\n")[:3]
            for line in lines:
                print(colored(f"    │ {line[:100]}", C.DIM))
            if len(output.split("\n")) > 3:
                print(colored(f"    │ ... ({len(output)} chars total)", C.DIM))

        if t.get("error"):
            print(colored(f"    ERROR: {t['error']}", C.RED))


def show_insights(mission):
    cp = mission["checkpoint"] or {}
    dag = cp.get("insight_dag")
    if not dag or not dag.get("nodes"):
        dag = synthesize_dag(cp.get("completed_tasks", []))
        if dag["nodes"]:
            print(colored("  (Synthesized from completed tasks — legacy mission)", C.DIM))

    print_header("Insight DAG")

    nodes = dag.get("nodes", {})
    if not nodes:
        print(colored("  No insights.", C.DIM))
        return

    for nid in sorted(nodes.keys()):
        n = nodes[nid]
        w = worker_badge(n.get("worker", ""))
        ok = colored("✓", C.GREEN) if n.get("success") else colored("✗", C.RED)
        rel = relevance_bar(n.get("relevance", 0))
        archived = colored(" [archived]", C.DIM) if n.get("archived") else ""

        print(f"\n  {C.BOLD}{nid}{C.RESET} {ok} {w} cycle={n.get('cycle', '?')} {colored(rel, C.CYAN)}{archived}")
        print(f"  {(n.get('task', '') or '')[:80]}")

        refs = n.get("references", [])
        if refs:
            print(colored(f"    refs: {', '.join(refs)}", C.DIM))

        code_refs = n.get("code_refs", [])
        for cr in code_refs:
            mods = ", ".join(cr.get("modules_changed", []))
            print(colored(f"    code: {cr.get('filename', '')}@{cr.get('version', '')} ({mods})", C.DIM))


def show_code(mission):
    code_store = mission["dir"] / "workspace" / ".code_store"

    print_header("Code Versions")

    if not code_store.is_dir():
        print(colored("  No tracked files.", C.DIM))
        return

    for stem_dir in sorted(code_store.iterdir()):
        if not stem_dir.is_dir():
            continue
        manifest = read_json(stem_dir / "manifest.json")
        if not manifest:
            continue

        filename = manifest.get("filename", stem_dir.name)
        latest = manifest.get("latest", "?")

        print(f"\n  {C.BOLD}{filename}{C.RESET} (latest: {colored(latest, C.CYAN)})")

        for v in manifest.get("versions", []):
            ver = v.get("version", "?")
            reason = (v.get("reason", "") or "")[:50]
            mods = v.get("modules_changed", [])
            mod_str = " ".join(mods) if mods else ""
            cycle = v.get("cycle", "?")
            print(f"    {colored(ver, C.BOLD)} c{cycle} — {reason} {colored(mod_str, C.DIM)}")

        # Show module map
        module_map = read_json(stem_dir / "module_map.json")
        if module_map:
            print(colored(f"    Module map ({len(module_map)} items):", C.DIM))
            for m in module_map[:8]:
                sig = (m.get("signature", "") or "")[:50]
                print(colored(f"      {m.get('kind', ''):>10s}  {m.get('name', '')}  {sig}", C.DIM))
            if len(module_map) > 8:
                print(colored(f"      ... and {len(module_map) - 8} more", C.DIM))


def show_knowledge(mission):
    knowledge_dir = mission["dir"] / "knowledge"

    print_header("Knowledge Tree")

    if not knowledge_dir.is_dir():
        print(colored("  No knowledge directory.", C.DIM))
        return

    icons = {"papers": "📄", "experiments": "🔬", "methods": "⚙", "code": "💻", "reports": "📊"}

    for cat_dir in sorted(knowledge_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        index = read_json(cat_dir / "_index.json")
        if not index:
            continue

        cat = cat_dir.name
        icon = icons.get(cat, "📁")
        count = index.get("item_count", 0)
        print(f"\n  {icon} {C.BOLD}{cat}{C.RESET} ({count} items)")

        for item_id, item in (index.get("items", {}) or {}).items():
            title = (item.get("title", "") or item_id)[:70]
            keywords = ", ".join(item.get("keywords", []))
            print(f"    • {title}")
            if keywords:
                print(colored(f"      tags: {keywords}", C.DIM))


def show_reports(mission):
    reports_dir = mission["dir"] / "reports"

    print_header("Reports")

    if not reports_dir.is_dir():
        print(colored("  No reports directory.", C.DIM))
        return

    reports = sorted(reports_dir.glob("*.md"))
    if not reports:
        print(colored("  No reports.", C.DIM))
        return

    for r in reports:
        stat = r.stat()
        size = f"{stat.st_size / 1024:.1f} KB" if stat.st_size >= 1024 else f"{stat.st_size} B"
        print(f"  {C.BOLD}{r.name}{C.RESET}  {colored(size, C.DIM)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_mission(missions, query):
    """Find mission by id, slug, or partial match."""
    for m in missions:
        if m["id"] == query:
            return m
        if m["manifest"].get("slug") == query:
            return m
    # Partial match
    for m in missions:
        if query in m["id"] or query in m["manifest"].get("slug", ""):
            return m
    return None


def main():
    parser = argparse.ArgumentParser(description="Research AI CLI Viewer")
    parser.add_argument("--missions", type=str, default=None, help="Path to missions/ directory")
    parser.add_argument("--mission", "-m", type=str, default=None, help="Mission ID or slug")
    parser.add_argument("--tasks", action="store_true", help="Show completed tasks")
    parser.add_argument("--insights", action="store_true", help="Show insight DAG")
    parser.add_argument("--code", action="store_true", help="Show code versions")
    parser.add_argument("--knowledge", action="store_true", help="Show knowledge tree")
    parser.add_argument("--reports", action="store_true", help="Show reports")
    parser.add_argument("--watch", action="store_true", help="Auto-refresh every 5s")
    args = parser.parse_args()

    missions_dir = Path(args.missions) if args.missions else DEFAULT_MISSIONS

    def run():
        missions = get_missions(missions_dir)

        if args.mission:
            m = find_mission(missions, args.mission)
            if not m:
                print(colored(f"Mission not found: {args.mission}", C.RED))
                sys.exit(1)

            if args.tasks:
                show_tasks(m)
            elif args.insights:
                show_insights(m)
            elif args.code:
                show_code(m)
            elif args.knowledge:
                show_knowledge(m)
            elif args.reports:
                show_reports(m)
            else:
                show_mission_overview(m)
        else:
            show_mission_list(missions)

    if args.watch:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                run()
                print(colored(f"\n  Auto-refreshing every 5s... (Ctrl+C to stop)", C.DIM))
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n")
    else:
        run()


if __name__ == "__main__":
    main()
