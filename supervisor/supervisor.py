"""
Supervisor Agent — Adaptive Research Loop with Memory Distillation
===================================================================
LLM-driven orchestrator that thinks like a researcher:
each cycle it reflects on all results so far, then decides
what to do next — search more, write code, fix bugs, benchmark,
write interim reports, or declare the mission done.

Memory Distillation:
- After each worker completes, LLM extracts an "insight" (心得)
- Before each decision, LLM distills ALL past insights into
  a working memory — keeping what's important, dropping noise
- This is NOT a sliding window — it's LLM-curated knowledge
  that can reach back to any past cycle (like a DAG, not a queue)
- Newest insights appear first but old important ones persist

Full-state checkpoint every cycle so resume works from the exact
point of interruption.
"""

import json
import os
import re
import time

from core.llm import MiniMaxClient, strip_think
from core.tool_registry import ToolRegistry
from core.event_bus import EventBus, EventType
from core.state import StateStore
from core.insight_dag import InsightDAG
from core.result_verifier import ResultVerifier
from knowledge.tree import KnowledgeTree
from supervisor.planner import TaskPlanner
from supervisor.reporter import Reporter
from supervisor.goal_tracker import GoalTracker
from supervisor.flow_monitor import FlowMonitor
from supervisor.research_standards import get_quality_rules
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
    success = t.get("success")
    return {
        "worker": t.get("worker", ""),
        "task": t.get("task", ""),
        "priority": t.get("priority", 0),
        "depends_on": t.get("depends_on", []),
        "success": success,
        "status": "done" if success else ("failed" if success is False else "pending"),
        "output": (t.get("output") or "")[:2000],
        "elapsed_s": t.get("elapsed_s"),
        "error": t.get("error"),
    }


class Supervisor:
    """
    Adaptive research supervisor with memory distillation.

    Instead of a fixed plan → execute pipeline, every cycle the LLM
    reflects on distilled insights from all past work, then freely
    decides what to do next — just like a real researcher would.
    """

    def __init__(self, llm: MiniMaxClient, registry: ToolRegistry,
                 event_bus: EventBus, state_store: StateStore,
                 knowledge: KnowledgeTree, reports_dir: str,
                 mission_ctx=None, mission_manager=None,
                 code_store=None, evolution_store=None):
        self.llm = llm
        self.registry = registry
        self.event_bus = event_bus
        self.state = state_store
        self.knowledge = knowledge
        self.mission_ctx = mission_ctx
        self.mission_manager = mission_manager

        language = mission_ctx.language if mission_ctx else "en"
        workspace_dir = mission_ctx.workspace_dir if mission_ctx else None
        self.reporter = Reporter(reports_dir, language=language,
                                 workspace_dir=workspace_dir)
        self.planner = TaskPlanner(llm)

        # Result verifier: cross-checks claims vs stdout
        self.result_verifier = ResultVerifier()

        # Workers
        self.workers = {
            "explorer": ExplorerWorker(llm, registry, event_bus, knowledge),
            "coder": CoderWorker(llm, registry, event_bus, knowledge,
                                 code_store=code_store),
            "reviewer": ReviewerWorker(llm, registry, event_bus, knowledge),
        }
        if mission_ctx:
            for w in self.workers.values():
                w.mission_id = mission_ctx.mission_id
        # Wire result verifier to all workers
        for w in self.workers.values():
            w.result_verifier = self.result_verifier

        # Code version store (for workspace summary in context)
        self.code_store = code_store

        # Evolution store (cross-mission learning)
        self.evolution_store = evolution_store

        # Goal tracker (objective goal completion)
        self.goal_tracker = GoalTracker(
            workspace_dir or "", llm=llm,
        )

        # Flow monitor (meta-supervision heuristics)
        self.flow_monitor = FlowMonitor()

        # Mission state
        self.goal = ""
        self.direction = ""
        self.agent_state = AgentState.PLANNING
        self.task_queue: list[dict] = []
        self.completed_tasks: list[dict] = []
        self.errors: list[str] = []
        self.reports_generated: int = 0
        self.cycle = 0
        self.max_cycles = 12
        self._last_action: str = ""
        self._repeat_count: int = 0

        # Failure tracking for smarter pivoting
        self._consecutive_failures: dict = {}  # worker_name → count

        # ── Memory Distillation (DAG-based) ──────────────────────
        # Structured insight graph with relevance scoring
        self.insight_dag: InsightDAG = InsightDAG()
        # Working memory: LLM-distilled summary of what matters NOW
        # This is what the supervisor "thinks" — its current understanding
        self.working_memory: str = ""

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
            "last_action": self._last_action,
            "repeat_count": self._repeat_count,
            # Memory distillation state (DAG-based)
            "insight_dag": self.insight_dag.to_dict(),
            "working_memory": self.working_memory,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "consecutive_failures": self._consecutive_failures,
            # Round 9 modules
            "result_verifier": self.result_verifier.to_dict(),
            "goal_tracker": self.goal_tracker.to_dict(),
            "flow_monitor": self.flow_monitor.to_dict(),
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
        self._last_action = cp.get("last_action", "")
        self._repeat_count = cp.get("repeat_count", 0)
        self._consecutive_failures = cp.get("consecutive_failures", {})
        # Restore memory — try DAG format first, fall back to legacy list
        if "insight_dag" in cp:
            self.insight_dag = InsightDAG.from_dict(cp["insight_dag"])
        elif "insights" in cp and cp["insights"]:
            self.insight_dag = InsightDAG.from_legacy_list(cp["insights"])
        else:
            self.insight_dag = InsightDAG()
        self.working_memory = cp.get("working_memory", "")
        # Restore Round 9 modules
        if "result_verifier" in cp:
            self.result_verifier = ResultVerifier.from_dict(cp["result_verifier"])
            for w in self.workers.values():
                w.result_verifier = self.result_verifier
        ws_dir = self.mission_ctx.workspace_dir if self.mission_ctx else ""
        if "goal_tracker" in cp:
            self.goal_tracker = GoalTracker.from_dict(cp["goal_tracker"], ws_dir)
        if "flow_monitor" in cp:
            self.flow_monitor = FlowMonitor.from_dict(cp["flow_monitor"])

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
        self.insight_dag = InsightDAG()
        self.working_memory = ""
        self.result_verifier = ResultVerifier()
        for w in self.workers.values():
            w.result_verifier = self.result_verifier

        self._print_header()

        # Parse goal into measurable sub-goals
        print(f"  [GoalTracker] Parsing goal into sub-goals...")
        self.goal_tracker.parse_goal(goal)
        for sg in self.goal_tracker.sub_goals:
            print(f"    - [{sg.type}] {sg.description}")

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
        print(f"  Insights: {self.insight_dag.active_count()} active / {self.insight_dag.total_count()} total")
        print(f"  Errors so far: {len(self.errors)}")
        print(f"  Knowledge: {self.knowledge.stats()['total_items']} items")

        return self._run_loop()

    # ── Core adaptive loop ───────────────────────────────────────────

    def _run_loop(self) -> str:
        """
        The adaptive research loop.

        Each cycle:
        1. Distill all past insights into working memory
        2. Reflect on working memory → decide action
        3. Execute (dispatch worker)
        4. Extract insight from result
        5. Checkpoint everything
        """
        self.agent_state = AgentState.RUNNING

        while self.cycle < self.max_cycles:
            self.cycle += 1

            print(f"\n{'─'*60}")
            print(f"  Cycle {self.cycle}/{self.max_cycles}")
            print(f"  Done: {len(self.completed_tasks)} tasks | "
                  f"Insights: {self.insight_dag.active_count()}/{self.insight_dag.total_count()} | "
                  f"Errors: {len(self.errors)} | "
                  f"Knowledge: {self.knowledge.stats()['total_items']} items")
            print(f"{'─'*60}")

            # ── Step 1: Distill insights into working memory ───
            if self.insight_dag.active_count() > 0:
                self._distill_insights()

            # ── Step 2: Reflect & Decide ───────────────────────
            action = self._reflect_and_decide()
            action_type = action.get("action", "done")

            # Anti-loop: if same action repeats 2+ times, force progress
            if action_type == self._last_action:
                self._repeat_count += 1
            else:
                self._repeat_count = 0
            self._last_action = action_type

            if self._repeat_count >= 2:
                if action_type == "report":
                    action = {"action": "done", "reason": "Report already written, mission complete"}
                    action_type = "done"
                elif action_type == "search_more":
                    action = {"action": "implement",
                              "task": f"Based on research so far, implement: {self.direction}",
                              "reason": "Moving from search to implementation"}
                    action_type = "implement"
                self._repeat_count = 0

            # ── Hard guard: block premature "done" ────────────
            if action_type == "done":
                workers_used = set(t.get("worker") for t in self.completed_tasks
                                   if t.get("success"))
                has_code = "coder" in workers_used
                has_eval = "reviewer" in workers_used

                if not has_code:
                    print(f"  [Supervisor] BLOCKED 'done' — no code written yet!")
                    action = {"action": "implement",
                              "task": f"Implement: {self.direction}",
                              "reason": "Cannot finish without code"}
                    action_type = "implement"
                elif not has_eval and self.cycle < self.max_cycles - 1:
                    print(f"  [Supervisor] BLOCKED 'done' — no evaluation yet!")
                    action = {"action": "benchmark",
                              "task": f"Evaluate and benchmark: {self.direction}",
                              "reason": "Cannot finish without evaluation"}
                    action_type = "benchmark"

            print(f"  [Supervisor] Decision: {action_type}")
            if action.get("reason"):
                print(f"  [Supervisor] Reason: {action['reason']}")

            # ── Step 3: Execute ────────────────────────────────
            if action_type == "done":
                print(f"\n  [Supervisor] Mission complete!")
                self.agent_state = AgentState.FINISHED
                report = self._generate_report()
                self._save_checkpoint()
                self._cleanup_workspace_processes()
                self._print_footer()

                # Post-mission evolution reflection
                if self.evolution_store:
                    try:
                        print(f"  [Evolution] Reflecting on mission...")
                        self.evolution_store.reflect_on_mission(
                            mission_id=self.mission_ctx.mission_id if self.mission_ctx else "",
                            goal=self.goal,
                            tasks=self.completed_tasks,
                            dag=self.insight_dag,
                            llm=self.llm,
                        )
                        print(f"  [Evolution] Learnings saved ({len(self.evolution_store.learnings)} total)")
                    except Exception as e:
                        print(f"  [Evolution] Reflection failed: {e}")

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
                print(f"  [Supervisor] Unknown action '{action_type}', defaulting to explore")
                self._dispatch_worker("explorer", action.get("task", self.direction))

            # ── Step 4b: Goal completion check ────────────────
            goal_status = self.goal_tracker.check_completion(
                self.completed_tasks,
                knowledge_stats=self.knowledge.stats(),
                dag=self.insight_dag,
            )
            if goal_status["completion_rate"] > 0:
                completed_n = sum(1 for sg in self.goal_tracker.sub_goals if sg.completed)
                total_n = len(self.goal_tracker.sub_goals)
                print(f"  [GoalTracker] {completed_n}/{total_n} sub-goals complete "
                      f"({goal_status['completion_rate']:.0%})")
                if goal_status["blocking"]:
                    print(f"  [GoalTracker] Blocking: {', '.join(goal_status['blocking'][:3])}")

                # Emit events for completed sub-goals
                for sg in self.goal_tracker.sub_goals:
                    if sg.completed and sg.evidence:
                        self.event_bus.emit(EventType.GOAL_SUBGOAL_COMPLETED, {
                            "type": sg.type, "description": sg.description,
                        }, source="goal_tracker")

                if goal_status["all_complete"]:
                    self.event_bus.emit(EventType.GOAL_ALL_COMPLETE, {
                        "completion_rate": 1.0,
                    }, source="goal_tracker")

            # ── Step 4c: Flow monitor analysis ─────────────────
            advisories = self.flow_monitor.analyze(
                cycle=self.cycle,
                tasks=self.completed_tasks,
                dag=self.insight_dag,
                failures=self._consecutive_failures,
            )
            if advisories:
                for adv in advisories:
                    print(f"  [FlowMonitor] {adv.severity}: {adv.message}")
                    self.event_bus.emit(EventType.FLOW_ADVISORY, adv.to_dict(),
                                        source="flow_monitor")

                # Apply critical advisory hard overrides
                for adv in advisories:
                    if adv.severity == "critical" and adv.suggested_action.startswith("skip_worker:"):
                        worker_to_skip = adv.suggested_action.split(":")[1]
                        self._consecutive_failures[worker_to_skip] = 0  # Reset so prompt doesn't keep warning
                        print(f"  [FlowMonitor] Hard override: skipping {worker_to_skip}")

            # ── Step 5: Checkpoint ─────────────────────────────
            self._save_checkpoint()

        # Hit max cycles — write final report
        print(f"\n  [Supervisor] Max cycles ({self.max_cycles}) reached")
        self.agent_state = AgentState.FINISHED
        report = self._generate_report()
        self._save_checkpoint()
        self._cleanup_workspace_processes()
        self._print_footer()

        # Post-mission evolution reflection
        if self.evolution_store:
            try:
                print(f"  [Evolution] Reflecting on mission...")
                self.evolution_store.reflect_on_mission(
                    mission_id=self.mission_ctx.mission_id if self.mission_ctx else "",
                    goal=self.goal,
                    tasks=self.completed_tasks,
                    dag=self.insight_dag,
                    llm=self.llm,
                )
                print(f"  [Evolution] Learnings saved ({len(self.evolution_store.learnings)} total)")
            except Exception as e:
                print(f"  [Evolution] Reflection failed: {e}")

        return report

    # ══════════════════════════════════════════════════════════════════
    #  MEMORY DISTILLATION SYSTEM
    # ══════════════════════════════════════════════════════════════════

    def _extract_insight(self, worker_name: str, task_desc: str, result: dict):
        """
        After a worker completes, LLM reads the full result and extracts
        a meaningful insight — what was learned, what matters, what to
        do next. This is the researcher's "lab notebook entry".
        Added to the InsightDAG with auto-generated tags and references.
        """
        output = (result.get("output") or "")[:3000]
        error = result.get("error", "")
        success = result.get("success", False)
        elapsed = result.get("elapsed_s", 0)

        # Build workspace context: list what files exist after this task
        workspace_files = ""
        if self.mission_ctx:
            ws_dir = self.mission_ctx.workspace_dir
            try:
                import os
                files = []
                for root, dirs, fnames in os.walk(ws_dir):
                    for fn in fnames:
                        if '__pycache__' not in root and not fn.startswith('.') and '.code_store' not in root:
                            rel = os.path.relpath(os.path.join(root, fn), ws_dir)
                            sz = os.path.getsize(os.path.join(root, fn))
                            files.append(f"{rel} ({sz}B)")
                if files:
                    workspace_files = f"\nWorkspace files: {', '.join(files)}"
            except Exception:
                pass

        prompt = f"""A research worker just completed a task. Extract the KEY INSIGHT from this result.

Worker: {worker_name}
Task: {task_desc}
Status: {"SUCCESS" if success else "FAILED"}
Time: {elapsed:.1f}s
Output: {output}
{"Error: " + error if error else ""}
{workspace_files}

Extract a concise insight (3-5 sentences) that MUST include:
1. **Specific results**: exact numbers (accuracy=X%, loss=Y, time=Zs), file names created, paper titles found
2. **What was accomplished**: concrete deliverables, not vague descriptions
3. **Implications**: what this means for the next step
4. If failed: exact error and what to try differently

CRITICAL: Include ALL numbers and file names from the output. Do NOT write vague statements like "the experiment was successful" — instead write "achieved 85.3% accuracy on CIFAR-10 with CLIP zero-shot, saved confusion_matrix.png".

Write the insight directly, no JSON or formatting needed."""

        try:
            response = self.llm.chat([
                {"role": "system", "content": "Extract key research insights concisely."},
                {"role": "user", "content": prompt},
            ])
            raw = response["choices"][0]["message"]["content"]
            insight_text = strip_think(raw)
        except Exception as e:
            # Fallback: mechanical summary
            if success:
                insight_text = f"[{worker_name}] {task_desc[:80]} — completed in {elapsed:.0f}s. {output[:200]}"
            else:
                insight_text = f"[{worker_name}] {task_desc[:80]} — FAILED: {error[:200]}"

        # Find recent related insights to reference
        references = []
        recent_by_worker = self.insight_dag.get_by_worker(worker_name)
        if recent_by_worker:
            references.append(recent_by_worker[-1].id)
        if not success:
            recent_failures = self.insight_dag.get_failures(limit=1)
            for f in recent_failures:
                if f.id not in references:
                    references.append(f.id)

        # Collect code version refs for coder insights (forward link)
        code_refs = []
        if self.code_store and worker_name == "coder":
            code_refs = self.code_store.get_cycle_writes(self.cycle)

        node_id = self.insight_dag.add(
            cycle=self.cycle,
            worker=worker_name,
            task=task_desc,
            success=success,
            content=insight_text,
            references=references,
            code_refs=code_refs,
        )

        # Create reverse links: code version → insight ID
        if code_refs and self.code_store:
            for ref in code_refs:
                self.code_store.link_insight(
                    ref["filename"], ref["version"], node_id,
                )

        code_info = f", code: {len(code_refs)} files" if code_refs else ""
        print(f"  [Memory] Insight {node_id} ({self.insight_dag.active_count()} active{code_info}): {insight_text[:120]}...")

    def _distill_insights(self):
        """
        Review ALL active insights (sorted by relevance) and distill
        into working memory. Uses the InsightDAG's panoramic view
        and updates relevance scores based on LLM feedback.

        The LLM returns structured JSON identifying which insights
        are most important and how they connect, enabling the DAG
        to decay irrelevant insights and promote important ones.
        """
        if self.insight_dag.active_count() == 0:
            self.working_memory = ""
            return

        panoramic = self.insight_dag.get_panoramic_view(max_items=25)
        # Cap to prevent context overflow
        if len(panoramic) > 8000:
            panoramic = panoramic[:8000] + "\n...(lower-relevance insights omitted)"

        prompt = f"""You are a research supervisor reviewing all accumulated insights from this mission.

## Mission Goal
{self.direction}

## All Insights (sorted by relevance)
{panoramic}

## Your Task
Distill these into a **working memory** and identify which insights matter most.

Respond with JSON:
{{
  "top_insights": ["i0001", "i0003", ...],  // IDs of the most important insights (keep 3-8)
  "connections": [{{"from": "i0003", "to": "i0001"}}, ...],  // new connections you see between insights
  "working_memory": "bullet-point summary of current research understanding"
}}

Rules for working_memory:
1. KEEP insights that are still relevant (key findings, important numbers, useful directions)
2. DROP insights that are obsolete (superseded by newer work, failed approaches already retried)
3. MERGE overlapping insights into single stronger statements
4. PROMOTE old insights that connect to recent progress
5. Be SPECIFIC — preserve exact numbers, paper names, method names
6. End with a 1-2 sentence "current status and next priority"

Write the working_memory as bullet points. Be concise but complete."""

        try:
            response = self.llm.chat([
                {"role": "system", "content": "Distill research insights. Respond with JSON."},
                {"role": "user", "content": prompt},
            ])
            raw = response["choices"][0]["message"]["content"]
            clean = strip_think(raw)

            # Try to parse structured JSON
            json_match = re.search(r'\{[\s\S]*\}', clean)
            if json_match:
                parsed = json.loads(json_match.group())
                top_ids = parsed.get("top_insights", [])
                connections = parsed.get("connections", [])
                wm = parsed.get("working_memory", "")

                if top_ids:
                    self.insight_dag.update_from_distillation(
                        top_ids=top_ids,
                        connections=connections,
                    )
                if wm:
                    self.working_memory = wm
                else:
                    self.working_memory = clean
            else:
                # JSON parse failed — use the raw text as working memory
                self.working_memory = clean

            print(f"  [Memory] Working memory distilled ({len(self.working_memory)} chars "
                  f"from {self.insight_dag.active_count()} active insights)")
        except Exception as e:
            print(f"  [Memory] Distillation failed ({e}), using panoramic view")
            # Fallback: use the panoramic view directly
            self.working_memory = panoramic[:2000]

    # ── The brain: reflect & decide ──────────────────────────────────

    def _reflect_and_decide(self) -> dict:
        """
        LLM reviews working memory (distilled insights) and decides
        the single best next action. This replaces the old crude
        output[:200] approach with rich, LLM-curated understanding.
        """
        # Task completion log (brief, for structure)
        task_log = ""
        if self.completed_tasks:
            parts = []
            for i, t in enumerate(self.completed_tasks):
                status = "✓" if t.get("success") else "✗"
                parts.append(f"  {i+1}. {status} [{t.get('worker','?')}] {t.get('task','')[:80]}")
            task_log = "\n".join(parts)

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

        # Build failure escalation warning
        failure_warnings = ""
        for wname, fcount in self._consecutive_failures.items():
            if fcount >= 2:
                failure_warnings += (
                    f"\n⚠️ {wname} has failed {fcount} times in a row. "
                    f"DO NOT dispatch {wname} with the same approach. "
                    f"Either use a different worker or fundamentally change the task."
                )

        # Goal completion status
        goal_status_text = self.goal_tracker.format_for_prompt()

        # Flow monitor advisories (from last cycle)
        flow_advisories = self.flow_monitor.analyze(
            cycle=self.cycle,
            tasks=self.completed_tasks,
            dag=self.insight_dag,
            failures=self._consecutive_failures,
        )
        flow_text = FlowMonitor.format_for_prompt(flow_advisories) if flow_advisories else ""

        prompt = f"""You are a research supervisor. Based on your working memory (distilled insights from all past work), decide the single best next action.

## Mission
- Goal: {self.goal}
- Direction: {self.direction}
- Cycle: {self.cycle}/{self.max_cycles}
{failure_warnings}

{goal_status_text}

## Your Working Memory (distilled insights)
{self.working_memory or "(no insights yet — this is the first cycle)"}

## Task History
{task_log or "  (none yet)"}

## Pending Tasks (from initial plan)
{queue_summary or "  (none)"}

## Recent Errors
{error_summary or "  (none)"}

## Knowledge Base
- Total: {knowledge_stats['total_items']} items
- By category: {json.dumps(knowledge_stats['by_category'], default=str)}
{cross_info}

## Available Actions
Choose ONE action:

- "search_more": Need more papers/info. Specify what.
- "implement": Ready to write code. Specify what.
- "fix_code": Code failed. Include the error.
- "benchmark": Evaluate/benchmark results.
- "improve": Results not good enough. Specify what.
- "replan": Current approach isn't working.
- "report": Meaningful progress — write interim report.
- "done": FULLY achieved — has papers + code + experiments + results.

## Decision Rules
- NEVER "done" without code written AND benchmarks run
- After 2-3 search rounds → "implement"
- If code failed → "fix_code"
- If results poor → "improve" or "search_more"
- "done" = papers + code + tests + results analyzed

## Research Quality Rules (CRITICAL)
When evaluating if the current direction is working, watch for these failure patterns:

**Type A — Ceiling Too Low**: Are we trying to improve a mature system by < 5%? If manual tuning can achieve similar results, the ceiling is too low. PIVOT to a different approach.

**Type B — Artifact/Unfair Comparison**: Are our positive results coming from unfair baselines? If comparing against the weakest baseline instead of the strongest, results are meaningless. ALWAYS compare against the STRONGEST known baseline.

**Type C — Wrong Hypothesis**: Did results contradict our hypothesis? If so, don't retry the same approach — PIVOT fundamentally.

**Success Formula**: Look for Natural Data Structure × Right Representation × Real System Constraint. If any is missing, the direction will likely fail.

If you detect any of these patterns in the working memory or past results, choose "replan" with a clear explanation of which failure type was detected.

{flow_text}

Respond with ONLY JSON:
{{"action": "...", "task": "specific description", "worker": "explorer|coder|reviewer", "reason": "why this action based on your working memory", "error_context": "if fix_code"}}
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

        # Fallback
        if not self.completed_tasks:
            return {"action": "search_more", "task": self.direction, "reason": "fallback"}
        return {"action": "done", "reason": "reflection failed, ending gracefully"}

    # ── Worker dispatch ──────────────────────────────────────────────

    def _dispatch_worker(self, worker_name: str, task_desc: str):
        """Dispatch a task to a worker with distilled context."""
        worker = self.workers.get(worker_name)
        if not worker:
            self.errors.append(f"Unknown worker: {worker_name}")
            return

        print(f"  [Supervisor] → {worker_name}: {task_desc[:100]}...")

        # Set cycle on code store so tracked writes know which cycle they belong to
        if self.code_store:
            self.code_store.set_current_cycle(self.cycle)

        # Set cycle on worker for result verification
        worker._current_cycle = self.cycle

        self.event_bus.emit(EventType.TASK_ASSIGNED, {
            "worker": worker_name, "task": task_desc,
        }, source="supervisor")

        # Build context from working memory (distilled insights)
        # instead of crude output[:400] from recent tasks
        context_parts = []

        # Working memory gives the big picture
        if self.working_memory:
            context_parts.append(
                f"## Current Research Status (distilled from all prior work)\n"
                f"{self.working_memory}"
            )

        # Add specific recent insights relevant to this worker from the DAG
        worker_insights = self.insight_dag.get_by_worker(worker_name)[-3:]
        failure_insights = self.insight_dag.get_failures(limit=2)
        relevant = {n.id: n for n in worker_insights + failure_insights}
        if relevant:
            parts = []
            for node in relevant.values():
                marker = "\u2713" if node.success else "\u2717"
                parts.append(f"- [{marker} cycle {node.cycle}] {node.content[:200]}")
            context_parts.append(
                f"## Recent relevant insights\n" + "\n".join(parts)
            )

        # Add code store workspace summary for coder tasks
        if self.code_store and worker_name == "coder":
            ws_summary = self.code_store.get_workspace_summary()
            if ws_summary:
                context_parts.append(ws_summary)

        context = "\n\n".join(context_parts)

        result = worker.run(task_desc, context=context)

        result_entry = {
            "worker": worker_name,
            "task": task_desc,
            **result,
        }
        self.completed_tasks.append(result_entry)

        # ── Step 4: Extract insight from result ────────────────
        if result.get("success"):
            print(f"  [Supervisor] ✓ {worker_name} completed successfully")
            self.event_bus.emit(EventType.TASK_COMPLETED, {
                "worker": worker_name, "task": task_desc,
            }, source="supervisor")
        else:
            error_msg = f"[{worker_name}] {task_desc[:60]}: {result.get('error', 'unknown')}"
            self.errors.append(error_msg)
            print(f"  [Supervisor] ✗ {worker_name} failed: {result.get('error', '')[:100]}")

        # Track consecutive failures for smarter pivoting
        if result.get("success"):
            self._consecutive_failures[worker_name] = 0
        else:
            self._consecutive_failures[worker_name] = \
                self._consecutive_failures.get(worker_name, 0) + 1
            fail_count = self._consecutive_failures[worker_name]
            if fail_count >= 2:
                print(f"  [Supervisor] WARNING: {worker_name} has failed {fail_count} times consecutively!")
                print(f"  [Supervisor] Will escalate on next cycle — trying different approach")

        # Always extract insight, whether success or failure
        self._extract_insight(worker_name, task_desc, result)

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

        # Get evolution guidance and quality rules
        evolution_guidance = ""
        if self.evolution_store:
            evolution_guidance = self.evolution_store.get_planner_guidance(self.direction)
            if evolution_guidance:
                print(f"  [Evolution] Injecting {len(self.evolution_store.get_relevant_learnings(self.direction))} learnings into planner")
        quality_rules = get_quality_rules(self.evolution_store, self.direction)

        tasks = self.planner.decompose(
            self.direction,
            knowledge_summary=knowledge_summary,
            available_workers=list(self.workers.keys()),
            cross_knowledge=cross_knowledge,
            evolution_guidance=evolution_guidance,
            quality_rules=quality_rules,
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

        evolution_guidance = ""
        if self.evolution_store:
            evolution_guidance = self.evolution_store.get_planner_guidance(self.direction)
        quality_rules = get_quality_rules(self.evolution_store, self.direction)

        tasks = self.planner.decompose(
            self.direction,
            knowledge_summary=knowledge_summary,
            completed_tasks=completed_descs,
            available_workers=list(self.workers.keys()),
            cross_knowledge=cross_knowledge,
            evolution_guidance=evolution_guidance,
            quality_rules=quality_rules,
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
        """Generate progress report with working memory included."""
        return self.reporter.generate(
            goal=self.goal,
            completed_tasks=self.completed_tasks,
            pending_tasks=self.task_queue,
            knowledge_stats=self.knowledge.stats(),
            errors=self.errors,
            working_memory=self.working_memory,
        )

    # ── Process cleanup ────────────────────────────────────────────

    def _cleanup_workspace_processes(self):
        """Kill any orphan processes from this mission's workspace."""
        workspace = self.mission_ctx.workspace_dir if self.mission_ctx else None
        if not workspace:
            return

        # Method 1: Clean up tracked PIDs from code_runner
        try:
            from mcp_servers.code_runner import _active_pids
            import signal as _sig
            for pid in list(_active_pids):
                try:
                    os.killpg(os.getpgid(pid), _sig.SIGTERM)
                    print(f"  [Cleanup] Killed process group for PID {pid}")
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            _active_pids.clear()
        except Exception:
            pass

        # Method 2: Find Python processes in workspace dir (best-effort)
        try:
            import subprocess as _sp
            result = _sp.run(
                ["pgrep", "-f", workspace],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                my_pid = str(os.getpid())
                for pid_str in pids:
                    pid_str = pid_str.strip()
                    if pid_str and pid_str != my_pid:
                        try:
                            os.kill(int(pid_str), 15)  # SIGTERM
                            print(f"  [Cleanup] Terminated orphan PID {pid_str}")
                        except (ProcessLookupError, PermissionError, ValueError):
                            pass
        except Exception:
            pass

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
