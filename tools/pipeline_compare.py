#!/usr/bin/env python3
"""
Pipeline A/B Comparison Runner
================================
Runs the same research goal through N pipeline configurations
and compares quality scores.

Supports both pipeline_mode (classic/structured) and validation_mode
(keyword/llm_full/llm_critical/exec_first/hybrid) for Round 11 testing.

Usage:
    python3 -m tools.pipeline_compare "research PEFT methods" --max-cycles 8
    python3 -m tools.pipeline_compare "implement LoRA PEFT" --max-cycles 6
    python3 -m tools.pipeline_compare "implement LoRA PEFT" --max-cycles 8 \
        --configs keyword llm_full llm_critical exec_first hybrid

Or via main.py:
    python3 main.py --compare "research PEFT methods" --max-cycles 8
"""

import argparse
import json
import os
import sys
import time

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    API_KEY, BASE_URL, MODEL, MAX_TURNS, MAX_TOKENS, TEMPERATURE,
    MISSIONS_DIR,
)


# ── Predefined validation mode configs ─────────────────────────────

VALIDATION_CONFIGS = {
    "keyword": {
        "name": "keyword",
        "pipeline_mode": "structured",
        "validation_mode": "keyword",
    },
    "llm_full": {
        "name": "llm_full",
        "pipeline_mode": "structured",
        "validation_mode": "llm_full",
    },
    "llm_critical": {
        "name": "llm_critical",
        "pipeline_mode": "structured",
        "validation_mode": "llm_critical",
    },
    "exec_first": {
        "name": "exec_first",
        "pipeline_mode": "structured",
        "validation_mode": "exec_first",
    },
    "hybrid": {
        "name": "hybrid",
        "pipeline_mode": "structured",
        "validation_mode": "hybrid",
    },
    # Legacy configs
    "classic": {
        "name": "classic",
        "pipeline_mode": "classic",
        "validation_mode": "keyword",
    },
    "structured": {
        "name": "structured",
        "pipeline_mode": "structured",
        "validation_mode": "keyword",
    },
}


def run_comparison(goal: str, max_cycles: int = 8,
                   language: str = "en",
                   configs: list[dict] | None = None) -> dict:
    """
    Run N-way pipeline comparison.

    Args:
        goal: Research goal string
        max_cycles: Max supervisor cycles per run
        language: Report language
        configs: Optional list of config dicts. Default: keyword + llm_full.

    Returns:
        Comparison result dict
    """
    from core.llm import MiniMaxClient
    from core.mission import MissionManager

    if configs is None:
        configs = [
            VALIDATION_CONFIGS["keyword"],
            VALIDATION_CONFIGS["llm_full"],
        ]

    llm = MiniMaxClient(
        api_key=API_KEY, base_url=BASE_URL, model=MODEL,
        max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
    )
    manager = MissionManager(MISSIONS_DIR, llm=llm)

    results = []

    for i, cfg in enumerate(configs):
        config_name = cfg["name"]
        pipeline_mode = cfg.get("pipeline_mode", "structured")
        validation_mode = cfg.get("validation_mode", "keyword")

        print(f"\n{'='*60}")
        print(f"  A/B Comparison — Config {i+1}/{len(configs)}: {config_name}")
        print(f"  Pipeline mode: {pipeline_mode}")
        print(f"  Validation mode: {validation_mode}")
        print(f"  Goal: {goal}")
        print(f"{'='*60}\n")

        # Create mission
        ctx = manager.create_mission(
            goal, language=language, cross_knowledge=False,
        )
        print(f"  Mission: {ctx.mission_id}")

        # Build system with specified modes
        from main import _make_llm, _make_registry, _check_system_resources
        from core.event_bus import EventBus
        from core.state import StateStore
        from core.code_store import CodeVersionStore
        from core.evolution_store import EvolutionStore
        from knowledge.tree import KnowledgeTree
        from supervisor.supervisor import Supervisor
        from mcp_servers import code_runner

        _check_system_resources()

        run_llm = _make_llm()
        registry = _make_registry()

        # Scope code tools to mission workspace
        scoped_tools = code_runner.create_workspace_tools(ctx.workspace_dir)
        for tool_def in code_runner.TOOLS:
            name = tool_def["function"]["name"]
            if name in scoped_tools:
                registry.register(tool_def, scoped_tools[name],
                                  source=f"code_runner@{ctx.mission_id}")

        event_bus = EventBus()
        state_store = StateStore(ctx.state_dir)
        knowledge = KnowledgeTree(ctx.knowledge_dir, llm_client=run_llm)
        code_store = CodeVersionStore(ctx.workspace_dir)
        evolution_store = EvolutionStore(MISSIONS_DIR)

        supervisor = Supervisor(
            llm=run_llm, registry=registry, event_bus=event_bus,
            state_store=state_store, knowledge=knowledge,
            reports_dir=ctx.reports_dir,
            mission_ctx=ctx, mission_manager=manager,
            code_store=code_store,
            evolution_store=evolution_store,
            pipeline_mode=pipeline_mode,
            validation_mode=validation_mode,
        )

        t0 = time.perf_counter()
        try:
            report = supervisor.run_mission(goal, max_cycles=max_cycles)
            elapsed = time.perf_counter() - t0
            success = True
        except Exception as e:
            report = f"Error: {e}"
            elapsed = time.perf_counter() - t0
            success = False
            print(f"  ERROR in {config_name}: {e}")

        # Score the mission (use LLM judge for scoring if the config uses LLM validation)
        from core.mission_scorer import MissionScorer
        judge_for_scoring = None
        if validation_mode in ("llm_full", "exec_first", "hybrid"):
            from core.llm_judge import LLMJudge
            judge_for_scoring = LLMJudge(run_llm)
        scorer = MissionScorer(llm_judge=judge_for_scoring)
        mission_dir = os.path.dirname(ctx.workspace_dir)
        try:
            score = scorer.score_mission(mission_dir)
            score_dict = score.to_dict()
        except Exception as e:
            score_dict = {"overall": 0, "grade": "F", "error": str(e)}

        results.append({
            "config": config_name,
            "pipeline_mode": pipeline_mode,
            "validation_mode": validation_mode,
            "mission_id": ctx.mission_id,
            "success": success,
            "elapsed_s": round(elapsed, 1),
            "score": score_dict,
        })

        manager.save_mission(ctx)

    # Build comparison
    comparison = {
        "goal": goal,
        "max_cycles": max_cycles,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "configs": results,
    }

    # Save comparison
    comparisons_dir = os.path.join(MISSIONS_DIR, "_comparisons")
    os.makedirs(comparisons_dir, exist_ok=True)
    comp_id = time.strftime("%Y%m%d_%H%M%S")
    comp_path = os.path.join(comparisons_dir, f"comparison_{comp_id}.json")
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"  A/B Comparison Results")
    print(f"{'='*60}")
    print(f"  Goal: {goal}")
    print(f"  Max cycles: {max_cycles}")
    print(f"\n  {'Config':<15} {'Validation':<14} {'Grade':<6} {'Score':<8} {'Time':<10} {'Mission'}")
    print(f"  {'-'*80}")
    for r in results:
        s = r["score"]
        grade = s.get("grade", "?")
        overall = s.get("overall", 0)
        print(f"  {r['config']:<15} {r['validation_mode']:<14} {grade:<6} {overall:<8.1f} {r['elapsed_s']:<10.0f}s {r['mission_id']}")

    print(f"\n  Saved: {comp_path}")
    print(f"{'='*60}")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline A/B Comparison Runner (supports N configs)",
    )
    parser.add_argument("goal", help="Research goal")
    parser.add_argument("--max-cycles", type=int, default=8,
                        help="Max supervisor cycles per run (default: 8)")
    parser.add_argument("--language", default="en", choices=["en", "zh"],
                        help="Report language")
    parser.add_argument("--configs", nargs="+",
                        choices=list(VALIDATION_CONFIGS.keys()),
                        default=None,
                        help="Config names to compare (default: keyword + llm_full)")

    args = parser.parse_args()

    # Build config list from names
    configs = None
    if args.configs:
        configs = [VALIDATION_CONFIGS[name] for name in args.configs]

    run_comparison(args.goal, max_cycles=args.max_cycles,
                   language=args.language, configs=configs)


if __name__ == "__main__":
    main()
