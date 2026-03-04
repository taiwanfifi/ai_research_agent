#!/usr/bin/env python3
"""
Self-Evolving AI Research Agent — Main Entry Point
=====================================================
Mission-isolated research system with bilingual reports.

Usage:
    python3 main.py "research Flash Attention"
    python3 main.py --zh "研究 Flash Attention 優化方法"
    python3 main.py --cross "基於之前的研究繼續深入"

    python3 main.py --resume flash_attention
    python3 main.py --resume 20260303_19
    python3 main.py --resume                             # most recent
    python3 main.py --resume attention --direction "改成專注 Flash v2"

    python3 main.py --list-missions
    python3 main.py --status
    python3 main.py --interactive
"""

import argparse
import json
import sys
import os

# Ensure we're running from the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import (
    API_KEY, BASE_URL, MODEL, MAX_TURNS, MAX_TOKENS, TEMPERATURE,
    MISSIONS_DIR, MCP_SERVERS_DIR, GENERATED_MCP_DIR, SKILLS_DIR,
)
from core.llm import MiniMaxClient
from core.tool_registry import ToolRegistry
from core.event_bus import EventBus, EventType
from core.state import StateStore
from core.mission import MissionManager, MissionContext
from core.code_store import CodeVersionStore
from core.evolution_store import EvolutionStore
from knowledge.tree import KnowledgeTree
from supervisor.supervisor import Supervisor
from skills.registry import SkillRegistry
from skills.meta_skill import MetaSkill


def _check_system_resources():
    """Warn if system is overloaded before starting a new mission."""
    try:
        load_1, load_5, _ = os.getloadavg()
        cpu_count = os.cpu_count() or 4
        if load_5 > cpu_count * 0.8:
            print(f"  WARNING: System load high ({load_5:.1f}, {cpu_count} cores)")
            print(f"  Consider waiting for existing missions to finish.")
    except OSError:
        pass


def _make_llm() -> MiniMaxClient:
    return MiniMaxClient(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL,
        max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
    )


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    from mcp_servers import paper_search, dataset_fetch, code_runner
    registry.register_module(paper_search)
    registry.register_module(dataset_fetch)
    registry.register_module(code_runner)
    try:
        from mcp_servers import github_search
        registry.register_module(github_search)
    except ImportError:
        pass
    registry.load_builtin_servers(GENERATED_MCP_DIR)
    return registry


def build_system(ctx: MissionContext, manager: MissionManager,
                  pipeline_mode: str = "classic",
                  validation_mode: str = "llm_full") -> dict:
    """Initialize all system components scoped to a mission context."""
    _check_system_resources()

    llm = _make_llm()
    registry = _make_registry()

    # Scope code tools to mission workspace — this is what connects
    # write_file/read_file/run_python_code to the mission directory
    # instead of the global ai_research_agent/workspace/
    from mcp_servers import code_runner
    scoped_tools = code_runner.create_workspace_tools(ctx.workspace_dir)
    for tool_def in code_runner.TOOLS:
        name = tool_def["function"]["name"]
        if name in scoped_tools:
            registry.register(tool_def, scoped_tools[name],
                              source=f"code_runner@{ctx.mission_id}")

    event_bus = EventBus()
    state_store = StateStore(ctx.state_dir)
    knowledge = KnowledgeTree(ctx.knowledge_dir, llm_client=llm)
    code_store = CodeVersionStore(ctx.workspace_dir)
    evolution_store = EvolutionStore(MISSIONS_DIR)

    skill_registry = SkillRegistry(SKILLS_DIR)
    skill_registry.load_builtin()
    skill_registry.load_from_directory()

    meta_skill = MetaSkill(llm, GENERATED_MCP_DIR, tool_registry=registry)

    supervisor = Supervisor(
        llm=llm, registry=registry, event_bus=event_bus,
        state_store=state_store, knowledge=knowledge,
        reports_dir=ctx.reports_dir,
        mission_ctx=ctx, mission_manager=manager,
        code_store=code_store,
        evolution_store=evolution_store,
        pipeline_mode=pipeline_mode,
        validation_mode=validation_mode,
    )

    return {
        "llm": llm,
        "registry": registry,
        "event_bus": event_bus,
        "state_store": state_store,
        "knowledge": knowledge,
        "code_store": code_store,
        "evolution_store": evolution_store,
        "skill_registry": skill_registry,
        "meta_skill": meta_skill,
        "supervisor": supervisor,
    }


# ── Display helpers ──────────────────────────────────────────────────

def print_missions(manager: MissionManager):
    """List all missions."""
    missions = manager.list_missions()
    if not missions:
        print("  No missions found.")
        return
    print(f"\n{'='*60}")
    print(f"  Missions ({len(missions)})")
    print(f"{'='*60}")
    for m in missions:
        status = m.get("status", "?")
        lang = m.get("language", "en")
        cross = " [cross]" if m.get("cross_knowledge") else ""
        print(f"  {m['mission_id']}")
        print(f"    Goal: {m.get('goal', '?')}")
        direction = m.get("direction", "")
        if direction and direction != m.get("goal", ""):
            print(f"    Direction: {direction}")
        print(f"    Status: {status}  Language: {lang}{cross}")
        print()


def print_status(system: dict, ctx: MissionContext):
    """Print system status for the current mission."""
    registry = system["registry"]
    knowledge = system["knowledge"]
    skill_registry = system["skill_registry"]

    print(f"\n{'='*60}")
    print(f"  Self-Evolving AI Research Agent")
    print(f"{'='*60}")
    print(f"  Model: {MODEL}")
    print(f"  Mission: {ctx.mission_id}")
    print(f"  Goal: {ctx.goal}")
    if ctx.direction != ctx.goal:
        print(f"  Direction: {ctx.direction}")
    print(f"  Language: {ctx.language}")
    print(f"  Cross-knowledge: {'on' if ctx.cross_knowledge else 'off'}")
    print(f"  Tools: {len(registry)} ({', '.join(registry.list_names())})")

    print(f"\n  Knowledge Tree:")
    stats = knowledge.stats()
    print(f"    Total items: {stats['total_items']}")
    for cat, count in stats["by_category"].items():
        if count > 0:
            print(f"    - {cat}: {count}")

    skills = skill_registry.list_skills()
    if skills:
        print(f"\n  Skills ({len(skills)}):")
        for s in skills:
            print(f"    - {s.name} v{s.version} ({s.runs} runs, {s.success_rate():.0%} success)")

    print(f"{'='*60}")


# ── Mission selection helpers ────────────────────────────────────────

def _select_mission(manager: MissionManager, partial: str = None) -> MissionContext | None:
    """Find and optionally let user choose a mission."""
    matches = manager.find_mission(partial)
    if not matches:
        print(f"  No mission found matching '{partial or '(latest)'}'" )
        return None
    if len(matches) == 1:
        return matches[0]
    # Multiple matches — let user choose
    print(f"\n  Multiple missions match '{partial}':")
    for i, m in enumerate(matches):
        print(f"    [{i+1}] {m.mission_id}")
        print(f"        Goal: {m.goal}")
    while True:
        try:
            choice = input(f"\n  Select [1-{len(matches)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(matches):
            return matches[int(choice) - 1]
        print("  Invalid choice.")


# ── Interactive mode ─────────────────────────────────────────────────

def interactive_mode(manager: MissionManager):
    """Interactive mode with mission management."""
    llm = _make_llm()
    manager.llm = llm  # ensure slug generation works

    language = "en"
    cross_knowledge = False
    current_system = None
    current_ctx = None

    print(f"\n{'='*60}")
    print(f"  Self-Evolving Research Agent — Interactive Mode")
    print(f"  Commands:")
    print(f"    <query>                   Start new mission")
    print(f"    /resume <pattern>         Resume a mission")
    print(f"    /resume <pattern> <dir>   Resume + change direction")
    print(f"    /missions                 List all missions")
    print(f"    /cross                    Toggle cross-knowledge")
    print(f"    /zh                       Toggle report language")
    print(f"    /status                   Current mission status")
    print(f"    /report                   Generate report")
    print(f"    quit                      Exit")
    print(f"{'='*60}")
    print(f"  Language: {language}  Cross-knowledge: {'on' if cross_knowledge else 'off'}")

    while True:
        try:
            user_input = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        # ── Commands ──────────────────────────────────────────────

        if user_input == "/missions":
            print_missions(manager)

        elif user_input == "/cross":
            cross_knowledge = not cross_knowledge
            print(f"  Cross-knowledge: {'on' if cross_knowledge else 'off'}")
            if current_ctx:
                current_ctx.cross_knowledge = cross_knowledge

        elif user_input == "/zh":
            language = "zh" if language == "en" else "en"
            print(f"  Report language: {language}")
            if current_ctx:
                current_ctx.language = language

        elif user_input == "/status":
            if current_system and current_ctx:
                print_status(current_system, current_ctx)
            else:
                print("  No active mission. Start one or use /resume.")

        elif user_input == "/report":
            if current_system:
                report = current_system["supervisor"]._generate_report()
                print(report)
            else:
                print("  No active mission.")

        elif user_input.startswith("/resume"):
            parts = user_input.split(None, 2)
            pattern = parts[1] if len(parts) > 1 else None
            new_direction = parts[2] if len(parts) > 2 else None

            ctx = _select_mission(manager, pattern)
            if not ctx:
                continue

            if new_direction:
                ctx.direction = new_direction
            ctx.language = language
            ctx.cross_knowledge = cross_knowledge

            print(f"  Resuming: {ctx.mission_id}")
            if ctx.direction != ctx.goal:
                print(f"  New direction: {ctx.direction}")

            current_ctx = ctx
            current_system = build_system(ctx, manager)
            manager.save_mission(ctx)
            report = current_system["supervisor"].resume_mission()
            if report:
                print(f"\n{report}")

        else:
            # New mission
            goal = user_input
            ctx = manager.create_mission(
                goal, language=language, cross_knowledge=cross_knowledge,
            )
            print(f"  Created mission: {ctx.mission_id}")
            current_ctx = ctx
            current_system = build_system(ctx, manager)
            report = current_system["supervisor"].run_mission(goal)
            print(f"\n{report}")


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Self-Evolving AI Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 main.py "research Flash Attention"
    python3 main.py --zh "研究 Flash Attention 優化方法"
    python3 main.py --cross "基於之前的研究繼續深入"
    python3 main.py --resume flash_attention
    python3 main.py --resume 20260303_19
    python3 main.py --resume --direction "改成專注 Flash v2"
    python3 main.py --list-missions
    python3 main.py --interactive
        """,
    )
    parser.add_argument("goal", nargs="*", help="Research goal (starts a new mission)")
    parser.add_argument("--zh", action="store_true", help="Use Traditional Chinese reports")
    parser.add_argument("--cross", action="store_true", help="Enable cross-mission knowledge")
    parser.add_argument("--resume", nargs="?", const="", default=None,
                        help="Resume a mission (optional: partial match pattern)")
    parser.add_argument("--direction", type=str, default=None,
                        help="New direction when resuming")
    parser.add_argument("--list-missions", action="store_true", help="List all missions")
    parser.add_argument("--report", action="store_true", help="Generate report for a mission")
    parser.add_argument("--score", action="store_true", help="Score a mission (use with --resume)")
    parser.add_argument("--compare", action="store_true", help="Run A/B pipeline comparison")
    parser.add_argument("--pipeline-mode", choices=["classic", "structured"], default="classic",
                        help="Pipeline mode: classic (v9.2) or structured (execution log)")
    parser.add_argument("--validation-mode",
                        choices=["keyword", "llm_full", "llm_critical", "exec_first", "hybrid"],
                        default="llm_full",
                        help="Validation mode: keyword (legacy), llm_full (all 3 judge calls, default), "
                             "llm_critical (judge Call 1 only), exec_first (judge Call 3 only), "
                             "hybrid (structured JSON + judge fallback)")
    parser.add_argument("--status", "-s", action="store_true", help="Show system status")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--max-cycles", type=int, default=12, help="Max supervisor cycles (default: 12)")

    args = parser.parse_args()

    llm = _make_llm()
    manager = MissionManager(MISSIONS_DIR, llm=llm)

    language = "zh" if args.zh else "en"

    # ── List missions ─────────────────────────────────────────────
    if args.list_missions:
        print_missions(manager)
        return

    # ── Interactive mode ──────────────────────────────────────────
    if args.interactive:
        interactive_mode(manager)
        return

    # ── A/B Comparison ───────────────────────────────────────────
    if args.compare:
        from tools.pipeline_compare import run_comparison
        goal = " ".join(args.goal) if args.goal else None
        if not goal:
            print("  Error: --compare requires a goal.")
            print("  Usage: python3 main.py --compare 'research PEFT methods' --max-cycles 8")
            sys.exit(1)
        run_comparison(goal, max_cycles=args.max_cycles, language=language)
        return

    # ── Resume ────────────────────────────────────────────────────
    if args.resume is not None:
        partial = args.resume if args.resume else None
        ctx = _select_mission(manager, partial)
        if not ctx:
            sys.exit(1)

        # Apply overrides
        ctx.language = language
        ctx.cross_knowledge = args.cross

        # Direction from --direction flag or positional goal
        new_direction = args.direction
        if not new_direction and args.goal:
            new_direction = " ".join(args.goal)
        if new_direction:
            ctx.direction = new_direction

        print(f"  Resuming: {ctx.mission_id}")
        print(f"  Goal: {ctx.goal}")
        if ctx.direction != ctx.goal:
            print(f"  Direction: {ctx.direction}")

        manager.save_mission(ctx)

        # Score-only mode (no need to build full system)
        if args.score:
            from core.mission_scorer import MissionScorer
            scorer = MissionScorer()
            mission_dir = os.path.dirname(ctx.workspace_dir)
            score = scorer.score_mission(mission_dir)
            print(f"\n  Mission: {ctx.mission_id}")
            print(f"  Grade: {score.grade} ({score.overall:.1f}/10)")
            print(f"\n  {'Dimension':<15} {'Score':<8} {'Weight':<8} Evidence")
            print(f"  {'-'*70}")
            for d in score.dimensions:
                ev = "; ".join(d.evidence[:2])
                print(f"  {d.name:<15} {d.score:<8.1f} {d.weight:<8} {ev}")
            return

        system = build_system(ctx, manager)

        if args.report:
            report = system["supervisor"]._generate_report()
            print(report)
        elif args.status:
            print_status(system, ctx)
        else:
            report = system["supervisor"].resume_mission()
            if report:
                print(f"\n{report}")
            else:
                print("  No checkpoint to resume from.")
        return

    # ── Status (no mission) ───────────────────────────────────────
    if args.status:
        missions = manager.list_missions()
        if missions:
            ctx = manager._manifest_to_ctx(missions[0])
            system = build_system(ctx, manager)
            print_status(system, ctx)
        else:
            print("  No missions found. Run a mission first.")
        return

    # ── New mission ───────────────────────────────────────────────
    if not args.goal:
        interactive_mode(manager)
        return

    goal = " ".join(args.goal)
    ctx = manager.create_mission(goal, language=language, cross_knowledge=args.cross)
    print(f"  Created mission: {ctx.mission_id}")

    pipeline_mode = getattr(args, 'pipeline_mode', 'classic') or 'classic'
    validation_mode = getattr(args, 'validation_mode', 'keyword') or 'keyword'
    system = build_system(ctx, manager, pipeline_mode=pipeline_mode,
                          validation_mode=validation_mode)

    if args.report:
        report = system["supervisor"]._generate_report()
        print(report)
    else:
        report = system["supervisor"].run_mission(goal, max_cycles=args.max_cycles)
        print(f"\n{report}")


if __name__ == "__main__":
    main()
