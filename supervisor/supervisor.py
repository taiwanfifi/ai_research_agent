"""
Supervisor Agent — Adaptive Research Loop
==========================================
LLM-driven orchestrator that thinks like a researcher:
each cycle it reflects on all results so far, then decides
what to do next — search more, write code, fix bugs, benchmark,
write interim reports, or declare the mission done.

Full-state checkpoint every cycle so resume works from the exact
point of interruption.
"""

import json
import os
import re
import time

from core.llm import MiniMaxClient
from core.tool_registry import ToolRegistry
from core.event_bus import EventBus, EventType
from core.state import StateStore
from knowledge.tree import KnowledgeTree
from supervisor.planner import TaskPlanner
from supervisor.reporter import Reporter
from workers.explorer import ExplorerWorker
from workers.coder import CoderWorker
from workers.reviewer import ReviewerWorker


class AgentState:
    PLANNING = "planning"
    RUNNING = "running"
    AWAITING_INPUT = "awaiting_input"
    REORGANIZING = "reorganizing"
    REPORTING = "reporting"
    FINISHED = "finished"
    ERROR = "error"


# ── Serialization helpers ────────────────────────────────────────────

def _serialize_task(t: dict) -> dict:
    """Strip non-serializable fields (messages list can be huge)."""
    return {
        "worker": t.get("worker", ""),
        "task": t.get("task", ""),
        "priority": t.get("priority", 0),
        "depends_on": t.get("depends_on", []),
        "success": t.get("success"),
        "output": (t.get("output") or "")[:500],
        "elapsed_s": t.get("elapsed_s"),
        "error": t.get("error"),
    }


class Supervisor:
    """
    Adaptive research supervisor.

    Instead of a fixed plan → execute pipeline, every cycle the LLM
    reflects on all progress so far and freely decides what to do next,
    just like a real researcher would.
    """

    def __init__(self, llm: MiniMaxClient, registry: ToolRegistry,
                 event_bus: EventBus, state_store: StateStore,
                 knowledge: KnowledgeTree, reports_dir: str,
                 mission_ctx=None, mission_manager=None):
        self.llm = llm
        self.registry = registry
        self.event_bus = event_bus
        self.state = state_store
        self.knowledge = knowledge
        self.mission_ctx = mission_ctx
        self.mission_manager = mission_manager

        language = mission_ctx.language if mission_ctx else "en"
        self.reporter = Reporter(reports_dir, language=language)
        self.planner = TaskPlanner(llm)

        # Workers
        self.workers = {
            "explorer": ExplorerWorker(llm, registry, event_bus, knowledge),
            "coder": CoderWorker(llm, registry, event_bus, knowledge),
            "reviewer": ReviewerWorker(llm, registry, event_bus, knowledge),
        }
        if mission_ctx:
            for w in self.workers.values():
                w.mission_id = mission_ctx.mission_id

        # Mission state
        self.goal = ""
        self.direction = ""
        self.agent_state = AgentState.PLANNING
        self.task_queue: list[dict] = []
        self.completed_tasks: list[dict] = []
        self.errors: list[str] = []
        self.reports_generated: int = 0
        self.cycle = 0
        self.max_cycles = 30

    # ── Full-state checkpoint ────────────────────────────────────────

    def _save_checkpoint(self):
        """Save everything needed to resume from this exact point."""
        data = {
            "goal": self.goal,
            "direction": self.direction,
            "state": self.agent_state,
            "cycle": self.cycle,
            "max_cycles": self.max_cycles,
            "completed_tasks": [_serialize_task(t) for t in self.completed_tasks],
            "task_queue": [_serialize_task(t) for t in self.task_queue],
            "errors": self.errors,
            "reports_generated": self.reports_generated,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.state.checkpoint("mission", data)

        # Also update mission.json
        if self.mission_ctx and self.mission_manager:
            self.mission_ctx.status = self.agent_state
            self.mission_manager.save_mission(self.mission_ctx)

    def _load_checkpoint(self) -> dict | None:
        """Load last checkpoint to resume."""
        return self.state.load_checkpoint("mission")

    def _restore_from_checkpoint(self, cp: dict):
        """Restore full mission state from checkpoint."""
        self.goal = cp.get("goal", "")
        self.direction = cp.get("direction", self.goal)
        self.agent_state = cp.get("state", AgentState.RUNNING)
        self.cycle = cp.get("cycle", 0)
        self.max_cycles = cp.get("max_cycles", self.max_cycles)
        self.completed_tasks = cp.get("completed_tasks", [])
        self.task_queue = cp.get("task_queue", [])
        self.errors = cp.get("errors", [])
        self.reports_generated = cp.get("reports_generated", 0)

    # ── Main entry points ────────────────────────────────────────────

    def run_mission(self, goal: str, max_cycles: int = None) -> str:
        """Start a new mission from scratch."""
        self.goal = goal
        self.direction = (self.mission_ctx.direction
                          if self.mission_ctx else goal)
        self.max_cycles = max_cycles or self.max_cycles
        self.cycle = 0
        self.completed_tasks = []
        self.task_queue = []
        self.errors = []
        self.reports_generated = 0

        self._print_header()

        # Initial planning — get a starting set of tasks
        self.agent_state = AgentState.PLANNING
        self._initial_plan()
        self._save_checkpoint()

        return self._run_loop()

    def resume_mission(self) -> str:
        """Resume from last checkpoint, continuing from exact point."""
        cp = self._load_checkpoint()
        if not cp:
            print("  [Supervisor] No checkpoint found")
            return ""

        self._restore_from_checkpoint(cp)

        # Apply new direction if mission_ctx has one
        if self.mission_ctx and self.mission_ctx.direction:
            if self.mission_ctx.direction != self.goal:
                self.direction = self.mission_ctx.direction

        print(f"\n{'='*60}")
        print(f"  Supervisor — Resuming Mission")
        print(f"{'='*60}")
        print(f"  Goal: {self.goal}")
        if self.direction != self.goal:
            print(f"  Direction: {self.direction}")
        print(f"  Resuming from cycle: {self.cycle}")
        print(f"  Completed tasks: {len(self.completed_tasks)}")
        print(f"  Pending tasks: {len(self.task_queue)}")
        print(f"  Errors so far: {len(self.errors)}")
        print(f"  Knowledge: {self.knowledge.stats()['total_items']} items")

        return self._run_loop()

    # ── Core adaptive loop ───────────────────────────────────────────

    def _run_loop(self) -> str:
        """
        The adaptive research loop.

        Each cycle: reflect on everything → decide action → execute → checkpoint.
        Continues until the LLM decides the mission is done or max_cycles hit.
        """
        self.agent_state = AgentState.RUNNING

        while self.cycle < self.max_cycles:
            self.cycle += 1

            print(f"\n{'─'*60}")
            print(f"  Cycle {self.cycle}/{self.max_cycles}")
            print(f"  Done: {len(self.completed_tasks)} tasks | "
                  f"Queue: {len(self.task_queue)} | "
                  f"Errors: {len(self.errors)} | "
                  f"Knowledge: {self.knowledge.stats()['total_items']} items")
            print(f"{'─'*60}")

            # ── Reflect & Decide ──────────────────────────────────
            action = self._reflect_and_decide()
            action_type = action.get("action", "done")

            print(f"  [Supervisor] Decision: {action_type}")
            if action.get("reason"):
                print(f"  [Supervisor] Reason: {action['reason']}")

            # ── Execute ───────────────────────────────────────────
            if action_type == "done":
                print(f"\n  [Supervisor] Mission complete!")
                self.agent_state = AgentState.FINISHED
                report = self._generate_report()
                self._save_checkpoint()
                self._print_footer()
                return report

            elif action_type == "report":
                self.agent_state = AgentState.REPORTING
                self._generate_report()
                self.reports_generated += 1
                self.agent_state = AgentState.RUNNING
                print(f"  [Supervisor] Interim report saved")

            elif action_type in ("search_more", "explore"):
                task_desc = action.get("task", self.direction)
                self._dispatch_worker("explorer", task_desc)

            elif action_type in ("implement", "write_code"):
                task_desc = action.get("task", f"Implement: {self.direction}")
                self._dispatch_worker("coder", task_desc)

            elif action_type == "fix_code":
                error_ctx = action.get("error_context", "")
                task_desc = action.get("task", f"Fix the code. Error: {error_ctx}")
                self._dispatch_worker("coder", task_desc)

            elif action_type in ("benchmark", "evaluate", "review"):
                task_desc = action.get("task", f"Evaluate results for: {self.direction}")
                self._dispatch_worker("reviewer", task_desc)

            elif action_type == "improve":
                worker = action.get("worker", "coder")
                task_desc = action.get("task", f"Improve implementation for: {self.direction}")
                self._dispatch_worker(worker, task_desc)

            elif action_type == "replan":
                self.agent_state = AgentState.PLANNING
                self._replan(action.get("reason", ""))
                self.agent_state = AgentState.RUNNING

            elif action_type == "reorganize":
                self._reorganize_knowledge()

            else:
                # Unknown action — treat as dispatch to explorer
                print(f"  [Supervisor] Unknown action '{action_type}', defaulting to explore")
                self._dispatch_worker("explorer", action.get("task", self.direction))

            # ── Checkpoint ────────────────────────────────────────
            self._save_checkpoint()

        # Hit max cycles — write final report
        print(f"\n  [Supervisor] Max cycles ({self.max_cycles}) reached")
        self.agent_state = AgentState.FINISHED
        report = self._generate_report()
        self._save_checkpoint()
        self._print_footer()
        return report

    # ── The brain: reflect & decide ──────────────────────────────────

    def _reflect_and_decide(self) -> dict:
        """
        LLM reviews ALL progress and decides the single best next action.
        This is the core intelligence of the system.
        """
        # Build a complete picture for the LLM
        completed_summary = ""
        if self.completed_tasks:
            parts = []
            for i, t in enumerate(self.completed_tasks):
                status = "OK" if t.get("success") else "FAILED"
                output = (t.get("output") or "")[:200]
                error = t.get("error", "")
                entry = f"  {i+1}. [{t.get('worker','?')}] {t.get('task','')} → {status}"
                if output:
                    entry += f"\n     Output: {output}"
                if error:
                    entry += f"\n     Error: {error}"
                parts.append(entry)
            completed_summary = "\n".join(parts)

        queue_summary = ""
        if self.task_queue:
            queue_summary = "\n".join(
                f"  - [{t.get('worker','?')}] {t.get('task','')}"
                for t in self.task_queue
            )

        error_summary = ""
        if self.errors:
            error_summary = "\n".join(f"  - {e}" for e in self.errors[-5:])

        knowledge_stats = self.knowledge.stats()

        # Cross-knowledge summary if enabled
        cross_info = ""
        if (self.mission_ctx and self.mission_ctx.cross_knowledge
                and self.mission_manager):
            other_dirs = self.mission_manager.get_all_knowledge_dirs(
                exclude_mission_id=self.mission_ctx.mission_id
            )
            if other_dirs:
                cross_parts = []
                for info in other_dirs:
                    other_tree = KnowledgeTree(info["knowledge_dir"], llm_client=None)
                    stats = other_tree.stats()
                    if stats["total_items"] > 0:
                        cross_parts.append(
                            f"  - {info['goal']}: {stats['total_items']} items"
                        )
                if cross_parts:
                    cross_info = "Knowledge from other missions:\n" + "\n".join(cross_parts)

        prompt = f"""You are a research supervisor managing an autonomous research mission.
Think like a real researcher: be adaptive, fix problems, iterate until results are good.

## Mission
- Goal: {self.goal}
- Direction: {self.direction}
- Cycle: {self.cycle}/{self.max_cycles}

## Completed Tasks ({len(self.completed_tasks)})
{completed_summary or "  (none yet)"}

## Pending Tasks
{queue_summary or "  (none)"}

## Errors
{error_summary or "  (none)"}

## Knowledge Acquired
- Total: {knowledge_stats['total_items']} items
- By category: {json.dumps(knowledge_stats['by_category'], default=str)}
{cross_info}

## Available Actions
Choose ONE action (the most important next step):

- "search_more": Need more papers/info/repos. Specify what to search.
- "implement": Ready to write code. Specify what to implement.
- "fix_code": Previous code failed. Include the error to fix.
- "benchmark": Ready to evaluate/benchmark results.
- "improve": Results aren't good enough. Specify what to improve.
- "replan": Current approach isn't working, need a new plan.
- "report": Meaningful progress reached, write an interim report.
- "done": Research goal is FULLY achieved — has code, experiments, AND results.

## Decision Guidelines — IMPORTANT
- The goal usually requires BOTH literature search AND code implementation AND evaluation
- NEVER choose "done" if you only searched papers but haven't written code yet!
- NEVER choose "done" if you wrote code but haven't run benchmarks/tests!
- Typical flow: search → implement → test → improve → benchmark → report → done
- If no papers/info yet → search_more
- After 2-3 rounds of search → move to "implement" even if search isn't perfect
- If code failed → fix_code (include the error)
- If results are poor → improve or search_more for better approaches
- If benchmark shows good results → report, then done
- Write "report" when there's meaningful progress worth saving
- Don't repeat the same failed approach — try something different
- "done" requires: papers found + code written + code runs successfully + results analyzed

Respond with ONLY JSON:
{{"action": "...", "task": "specific description of what to do", "worker": "explorer|coder|reviewer", "reason": "why this action", "error_context": "if fix_code, paste the error"}}
"""

        try:
            response = self.llm.chat([
                {"role": "system", "content": "You are a research supervisor. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ])
            content = response["choices"][0]["message"]["content"]
            json_match = re.search(r'\{[\s\S]*?\}', content)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"  [Supervisor] Reflection failed: {e}")

        # Fallback: if we have no tasks done, search; otherwise done
        if not self.completed_tasks:
            return {"action": "search_more", "task": self.direction, "reason": "fallback"}
        return {"action": "done", "reason": "reflection failed, ending gracefully"}

    # ── Worker dispatch ──────────────────────────────────────────────

    def _dispatch_worker(self, worker_name: str, task_desc: str):
        """Dispatch a task to a worker with full context."""
        worker = self.workers.get(worker_name)
        if not worker:
            self.errors.append(f"Unknown worker: {worker_name}")
            return

        print(f"  [Supervisor] → {worker_name}: {task_desc[:100]}...")

        self.event_bus.emit(EventType.TASK_ASSIGNED, {
            "worker": worker_name, "task": task_desc,
        }, source="supervisor")

        # Build rich context from recent completed tasks
        context_parts = []
        for ct in self.completed_tasks[-5:]:
            status = "OK" if ct.get("success") else "FAILED"
            ctx_entry = f"[{ct.get('worker','')}] {ct.get('task','')}: {status}"
            output = (ct.get("output") or "")[:400]
            if output:
                ctx_entry += f"\n{output}"
            error = ct.get("error", "")
            if error:
                ctx_entry += f"\nError: {error}"
            context_parts.append(ctx_entry)
        context = "\n\n".join(context_parts)

        result = worker.run(task_desc, context=context)

        result_entry = {
            "worker": worker_name,
            "task": task_desc,
            **result,
        }
        self.completed_tasks.append(result_entry)

        if result.get("success"):
            print(f"  [Supervisor] ✓ {worker_name} completed successfully")
            self.event_bus.emit(EventType.TASK_COMPLETED, {
                "worker": worker_name, "task": task_desc,
            }, source="supervisor")
        else:
            error_msg = f"[{worker_name}] {task_desc[:60]}: {result.get('error', 'unknown')}"
            self.errors.append(error_msg)
            print(f"  [Supervisor] ✗ {worker_name} failed: {result.get('error', '')[:100]}")

    # ── Planning ─────────────────────────────────────────────────────

    def _initial_plan(self):
        """Create initial task plan (used only at mission start)."""
        print(f"\n  [Supervisor] Creating initial plan...")
        knowledge_summary = self.knowledge.get_summary(depth=1)

        cross_knowledge = None
        if (self.mission_ctx and self.mission_ctx.cross_knowledge
                and self.mission_manager):
            other_dirs = self.mission_manager.get_all_knowledge_dirs(
                exclude_mission_id=self.mission_ctx.mission_id
            )
            cross_knowledge = []
            for info in other_dirs:
                other_tree = KnowledgeTree(info["knowledge_dir"], llm_client=None)
                cross_knowledge.append({
                    "mission_id": info["mission_id"],
                    "goal": info["goal"],
                    "summary": other_tree.get_summary(depth=1),
                })

        tasks = self.planner.decompose(
            self.direction,
            knowledge_summary=knowledge_summary,
            available_workers=list(self.workers.keys()),
            cross_knowledge=cross_knowledge,
        )
        self.task_queue = tasks
        print(f"  [Supervisor] Initial plan ({len(tasks)} tasks):")
        for i, t in enumerate(tasks):
            print(f"    {i+1}. [{t['worker']}] {t['task']}")

    def _replan(self, reason: str):
        """Re-decompose with knowledge of what's been done."""
        print(f"\n  [Supervisor] Replanning: {reason}")
        completed_descs = [
            f"[{t.get('worker','')}] {t.get('task','')}: {'OK' if t.get('success') else 'FAILED'}"
            for t in self.completed_tasks
        ]
        knowledge_summary = self.knowledge.get_summary(depth=1)

        cross_knowledge = None
        if (self.mission_ctx and self.mission_ctx.cross_knowledge
                and self.mission_manager):
            other_dirs = self.mission_manager.get_all_knowledge_dirs(
                exclude_mission_id=self.mission_ctx.mission_id
            )
            cross_knowledge = []
            for info in other_dirs:
                other_tree = KnowledgeTree(info["knowledge_dir"], llm_client=None)
                cross_knowledge.append({
                    "mission_id": info["mission_id"],
                    "goal": info["goal"],
                    "summary": other_tree.get_summary(depth=1),
                })

        tasks = self.planner.decompose(
            self.direction,
            knowledge_summary=knowledge_summary,
            completed_tasks=completed_descs,
            available_workers=list(self.workers.keys()),
            cross_knowledge=cross_knowledge,
        )
        self.task_queue = tasks
        print(f"  [Supervisor] New plan ({len(tasks)} tasks):")
        for i, t in enumerate(tasks):
            print(f"    {i+1}. [{t['worker']}] {t['task']}")

    # ── Knowledge management ─────────────────────────────────────────

    def _reorganize_knowledge(self):
        """Trigger knowledge tree reorganization if needed."""
        self.agent_state = AgentState.REORGANIZING
        print(f"  [Supervisor] Reorganizing knowledge...")
        for cat in self.knowledge.list_categories():
            stats = self.knowledge.stats()
            if stats["by_category"].get(cat, 0) > 20:
                from knowledge.index import KnowledgeIndex
                cat_dir = os.path.join(self.knowledge.root_dir, cat)
                idx = KnowledgeIndex(cat_dir)
                self.knowledge._auto_reorganize(cat_dir, idx)
        self.agent_state = AgentState.RUNNING

    # ── Reporting ────────────────────────────────────────────────────

    def _generate_report(self) -> str:
        """Generate progress report."""
        return self.reporter.generate(
            goal=self.goal,
            completed_tasks=self.completed_tasks,
            pending_tasks=self.task_queue,
            knowledge_stats=self.knowledge.stats(),
            errors=self.errors,
        )

    # ── Display ──────────────────────────────────────────────────────

    def _print_header(self):
        print(f"\n{'='*60}")
        print(f"  Supervisor — Research Mission")
        print(f"{'='*60}")
        print(f"  Goal: {self.goal}")
        if self.direction != self.goal:
            print(f"  Direction: {self.direction}")
        print(f"  Workers: {list(self.workers.keys())}")
        print(f"  Max cycles: {self.max_cycles}")
        if self.mission_ctx:
            print(f"  Mission: {self.mission_ctx.mission_id}")
            print(f"  Language: {self.mission_ctx.language}")
            if self.mission_ctx.cross_knowledge:
                print(f"  Cross-knowledge: enabled")

    def _print_footer(self):
        success = sum(1 for t in self.completed_tasks if t.get("success"))
        failed = sum(1 for t in self.completed_tasks if not t.get("success"))
        print(f"\n{'='*60}")
        print(f"  Mission Complete — {self.cycle} cycles")
        print(f"  Tasks: {success} succeeded, {failed} failed")
        print(f"  Knowledge: {self.knowledge.stats()['total_items']} items")
        print(f"  Reports: {self.reports_generated + 1} generated")
        print(f"{'='*60}")
