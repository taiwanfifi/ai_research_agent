"""
Supervisor Agent
=================
High-level orchestrator that reviews state, uses LLM to decide actions,
and dispatches workers. Implements the main decision loop.

Now accepts a MissionContext so each mission is fully isolated.
"""

import json
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
    """Explicit state machine for supervisor."""
    PLANNING = "planning"
    RUNNING = "running"
    AWAITING_INPUT = "awaiting_input"
    REORGANIZING = "reorganizing"
    REPORTING = "reporting"
    FINISHED = "finished"
    ERROR = "error"


class Supervisor:
    """
    LLM-driven supervisor that orchestrates workers.

    Main loop: review state → LLM decides action → dispatch worker or reorg
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

        # Language-aware reporter
        language = mission_ctx.language if mission_ctx else "en"
        self.reporter = Reporter(reports_dir, language=language)

        # Planner
        self.planner = TaskPlanner(llm)

        # Workers
        self.workers = {
            "explorer": ExplorerWorker(llm, registry, event_bus, knowledge),
            "coder": CoderWorker(llm, registry, event_bus, knowledge),
            "reviewer": ReviewerWorker(llm, registry, event_bus, knowledge),
        }

        # Set mission_id on all workers
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
        self.max_cycles = 20

    def run_mission(self, goal: str, max_cycles: int = None) -> str:
        """
        Execute a full research mission.

        Uses mission_ctx.direction if available (may differ from goal on resume).
        """
        self.goal = goal
        self.direction = (self.mission_ctx.direction
                          if self.mission_ctx else goal)
        self.max_cycles = max_cycles or self.max_cycles

        print(f"\n{'='*60}")
        print(f"  Supervisor — Research Mission")
        print(f"{'='*60}")
        print(f"  Goal: {goal}")
        if self.direction != goal:
            print(f"  Direction: {self.direction}")
        print(f"  Workers: {list(self.workers.keys())}")
        print(f"  Max cycles: {self.max_cycles}")
        if self.mission_ctx:
            print(f"  Mission: {self.mission_ctx.mission_id}")
            print(f"  Language: {self.mission_ctx.language}")
            if self.mission_ctx.cross_knowledge:
                print(f"  Cross-knowledge: enabled")

        # Save initial state
        self.state.set("mission", "goal", goal)
        self.state.set("mission", "started_at", time.strftime("%Y-%m-%dT%H:%M:%S"))

        # Phase 1: Plan
        self.agent_state = AgentState.PLANNING
        self._plan()

        # Phase 2: Execute
        self.agent_state = AgentState.RUNNING
        for cycle in range(1, self.max_cycles + 1):
            if self.agent_state == AgentState.FINISHED:
                break

            print(f"\n--- Supervisor Cycle {cycle}/{self.max_cycles} ---")
            print(f"  State: {self.agent_state}")
            print(f"  Tasks: {len(self.completed_tasks)} done, {len(self.task_queue)} pending")

            action = self._decide_action()
            self._execute_action(action)

            # Checkpoint
            self.state.checkpoint("mission", {
                "goal": self.goal,
                "direction": self.direction,
                "state": self.agent_state,
                "completed": len(self.completed_tasks),
                "pending": len(self.task_queue),
                "cycle": cycle,
            })

        # Phase 3: Report
        self.agent_state = AgentState.REPORTING
        report = self._generate_report()

        self.agent_state = AgentState.FINISHED
        self.state.set("mission", "finished_at", time.strftime("%Y-%m-%dT%H:%M:%S"))

        # Update mission manifest
        if self.mission_ctx:
            self.mission_ctx.status = "finished"
            if self.mission_manager:
                self.mission_manager.save_mission(self.mission_ctx)

        print(f"\n{'='*60}")
        print(f"  Mission Complete")
        print(f"  Tasks: {len(self.completed_tasks)} completed, {len(self.errors)} errors")
        print(f"{'='*60}")

        return report

    def _plan(self):
        """Use LLM to decompose goal into tasks."""
        print(f"\n  [Supervisor] Planning...")
        knowledge_summary = self.knowledge.get_summary(depth=1)

        # Build cross-knowledge context if enabled
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
            self.direction,  # use direction, not goal
            knowledge_summary=knowledge_summary,
            available_workers=list(self.workers.keys()),
            cross_knowledge=cross_knowledge,
        )
        self.task_queue = tasks
        print(f"  [Supervisor] Planned {len(tasks)} tasks:")
        for i, t in enumerate(tasks):
            print(f"    {i+1}. [{t['worker']}] {t['task']}")

        self.event_bus.emit(EventType.STATE_CHANGED, {
            "state": AgentState.PLANNING, "tasks": len(tasks),
        }, source="supervisor")

    def _decide_action(self) -> dict:
        """LLM decides what to do next."""
        if not self.task_queue:
            return {"action": "mission_complete"}

        # Check dependencies
        next_task = None
        for task in self.task_queue:
            deps = task.get("depends_on", [])
            if all(d < len(self.completed_tasks) for d in deps):
                next_task = task
                break

        if not next_task:
            # All remaining tasks have unmet deps — ask LLM
            knowledge_summary = self.knowledge.get_summary(depth=1)
            prompt = f"""You are a research supervisor. Current state:
- Goal: {self.direction}
- Completed: {len(self.completed_tasks)} tasks
- Pending: {len(self.task_queue)} tasks with unmet dependencies
- Knowledge: {json.dumps(self.knowledge.stats(), default=str)}

What should we do? Respond with JSON:
{{"action": "dispatch_worker|reorganize_knowledge|generate_report|mission_complete", "reason": "..."}}

If tasks are blocked, either rephrase them or declare mission complete."""

            response = self.llm.chat([
                {"role": "system", "content": "You are a research supervisor. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ])
            content = response["choices"][0]["message"]["content"]
            json_match = re.search(r'\{[\s\S]*?\}', content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {"action": "mission_complete", "reason": "Cannot resolve dependencies"}

        return {"action": "dispatch_worker", "task": next_task}

    def _execute_action(self, action: dict):
        """Execute a supervisor action."""
        action_type = action.get("action", "")

        if action_type == "dispatch_worker":
            self._dispatch_worker(action.get("task", {}))

        elif action_type == "reorganize_knowledge":
            self.agent_state = AgentState.REORGANIZING
            print(f"  [Supervisor] Reorganizing knowledge...")
            for cat in self.knowledge.list_categories():
                stats = self.knowledge.stats()
                if stats["by_category"].get(cat, 0) > 20:
                    from knowledge.index import KnowledgeIndex
                    import os
                    cat_dir = os.path.join(self.knowledge.root_dir, cat)
                    idx = KnowledgeIndex(cat_dir)
                    self.knowledge._auto_reorganize(cat_dir, idx)
            self.agent_state = AgentState.RUNNING

        elif action_type == "generate_report":
            self._generate_report()

        elif action_type == "mission_complete":
            self.agent_state = AgentState.FINISHED
            print(f"  [Supervisor] Mission complete: {action.get('reason', '')}")

        elif action_type == "evolve_skill":
            print(f"  [Supervisor] Skill evolution not yet implemented")

    def _dispatch_worker(self, task: dict):
        """Dispatch a task to the appropriate worker."""
        worker_name = task.get("worker", "explorer")
        task_desc = task.get("task", "")

        worker = self.workers.get(worker_name)
        if not worker:
            self.errors.append(f"Unknown worker: {worker_name}")
            self.task_queue.remove(task)
            return

        print(f"  [Supervisor] Dispatching to {worker_name}: {task_desc[:80]}...")

        self.event_bus.emit(EventType.TASK_ASSIGNED, {
            "worker": worker_name, "task": task_desc,
        }, source="supervisor")

        # Build context from completed tasks
        context_parts = []
        for ct in self.completed_tasks[-3:]:
            context_parts.append(f"[{ct['worker']}] {ct['task']}: {ct.get('output', '')[:300]}")
        context = "\n\n".join(context_parts)

        result = worker.run(task_desc, context=context)

        # Move task from queue to completed
        if task in self.task_queue:
            self.task_queue.remove(task)

        result_entry = {**task, **result}
        self.completed_tasks.append(result_entry)

        if result.get("success"):
            self.event_bus.emit(EventType.TASK_COMPLETED, {
                "worker": worker_name, "task": task_desc,
            }, source="supervisor")
        else:
            self.errors.append(f"[{worker_name}] {task_desc}: {result.get('error', 'unknown')}")

    def _generate_report(self) -> str:
        """Generate progress report."""
        return self.reporter.generate(
            goal=self.goal,
            completed_tasks=self.completed_tasks,
            pending_tasks=self.task_queue,
            knowledge_stats=self.knowledge.stats(),
            errors=self.errors,
        )

    def resume_mission(self) -> str:
        """Resume from last checkpoint."""
        checkpoint = self.state.load_checkpoint("mission")
        if not checkpoint:
            print("  [Supervisor] No checkpoint found")
            return ""

        self.goal = checkpoint.get("goal", "")
        self.direction = checkpoint.get("direction", self.goal)
        print(f"  [Supervisor] Resuming mission: {self.goal}")
        if self.direction != self.goal:
            print(f"  [Supervisor] Direction: {self.direction}")
        print(f"  [Supervisor] Previously completed: {checkpoint.get('completed', 0)} tasks")

        return self.run_mission(self.goal)
