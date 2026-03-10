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
from core.execution_log import ExecutionLog
from core.llm_judge import LLMJudge
from core.hypothesis_generator import HypothesisGenerator
from core.research_validator import ResearchValidator
from core.research_tree import ResearchTree
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
    d = {
        "worker": t.get("worker", ""),
        "task": t.get("task", ""),
        "priority": t.get("priority", 0),
        "depends_on": t.get("depends_on", []),
        "success": success,
        "status": "done" if success else ("failed" if success is False else "pending"),
        "output": (t.get("output") or "")[:4000 if t.get("worker") == "explorer" else 2000],
        "elapsed_s": t.get("elapsed_s"),
        "error": t.get("error"),
    }
    # Preserve verification metadata
    if "verification_score" in t:
        d["verification_score"] = t["verification_score"]
    if "low_verification" in t:
        d["low_verification"] = t["low_verification"]
    # Preserve tool call names (not args — too large) for scoring
    if "tool_calls" in t and t["tool_calls"]:
        d["tool_calls"] = [{"name": tc.get("name", "")} for tc in t["tool_calls"]]
    return d


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
                 code_store=None, evolution_store=None,
                 pipeline_mode: str = "classic",
                 validation_mode: str = "llm_full"):
        self.llm = llm
        self.registry = registry
        self.event_bus = event_bus
        self.state = state_store
        self.knowledge = knowledge
        self.mission_ctx = mission_ctx
        self.mission_manager = mission_manager
        self.pipeline_mode = pipeline_mode  # "classic" or "structured"
        self.validation_mode = validation_mode  # "keyword", "llm_full", "llm_critical", "exec_first", "hybrid"

        language = mission_ctx.language if mission_ctx else "en"
        workspace_dir = mission_ctx.workspace_dir if mission_ctx else None
        self.reporter = Reporter(reports_dir, language=language,
                                 workspace_dir=workspace_dir)
        self.planner = TaskPlanner(llm)

        # LLM Judge: semantic validation (Round 11)
        self.llm_judge = None
        if validation_mode != "keyword":
            self.llm_judge = LLMJudge(llm)

        # Result verifier: cross-checks claims vs stdout
        self.result_verifier = ResultVerifier()

        # Execution log: always enabled when workspace exists (metric capture + done guard)
        self.execution_log = None
        if workspace_dir:
            self.execution_log = ExecutionLog(workspace_dir)

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
        # Wire result verifier to all workers (skip for llm_full — judge replaces it)
        if validation_mode not in ("llm_full", "hybrid"):
            for w in self.workers.values():
                w.result_verifier = self.result_verifier
        # Wire LLM judge to all workers
        if self.llm_judge and validation_mode in ("llm_full", "llm_critical", "hybrid"):
            for w in self.workers.values():
                w.llm_judge = self.llm_judge
                w.validation_mode = validation_mode
        # Inner monologue disabled by default — A/B test showed it hurts
        # (wastes turns on reflection instead of execution, D vs B grade)
        # Can be enabled per-worker if needed via w.enable_monologue = True
        # Wire execution log to all workers (structured pipeline)
        if self.execution_log:
            for w in self.workers.values():
                w.execution_log = self.execution_log

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
        self.max_cycles = 15  # Soft default — auto-extends if progress is strong
        self._last_action: str = ""
        self._repeat_count: int = 0
        self.literature_only: bool = False  # Literature-only mode: explorer tasks only

        # Failure tracking for smarter pivoting
        self._consecutive_failures: dict = {}  # worker_name → count
        # Flow monitor advisories from last cycle (avoids double-call)
        self._last_advisories: list = []
        # Friction buffer: accumulated failure diagnoses (Round 13)
        self._friction_buffer: list = []
        # Process reward: per-cycle progress signal (Round 13.1)
        from core.process_reward import ProcessRewardTracker
        self.process_reward = ProcessRewardTracker()
        # Hypothesis generator: research iteration (Round 13.2)
        self.hypothesis_gen = HypothesisGenerator(llm)
        self._pending_hypotheses = None  # Last generated hypotheses
        # Research design validator (Round 15.3)
        self.research_validator = ResearchValidator(llm)
        self._design_verdict = None
        # Thread lock for parallel dispatch (Round 16)
        import threading
        self._dispatch_lock = threading.Lock()
        # Live terminal message bus (None in batch mode)
        self.message_bus = None
        # Research tree: progressive branching search (Round 17)
        self.research_tree: ResearchTree | None = None
        self._active_branch_id: str = ""  # Currently exploring branch

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
            # Round 10: pipeline mode (execution_log lives on disk, not in checkpoint)
            "pipeline_mode": self.pipeline_mode,
            "design_validation": self._design_verdict.to_dict() if self._design_verdict else None,
            "has_execution_log": self.execution_log is not None,
            # Round 11: validation mode
            "validation_mode": self.validation_mode,
            # Round 16.2: hypothesis chain
            "hypothesis_chain": [r.to_dict() for r in self.hypothesis_gen.history],
            # Round 17: research tree
            "research_tree": self.research_tree.to_dict() if self.research_tree else None,
            "active_branch_id": self._active_branch_id,
        }
        # Flush execution log to disk before checkpoint
        if self.execution_log:
            self.execution_log.flush()
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
        self._initial_max_cycles = self.max_cycles  # For auto-extend hard limit
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
        # Restore validation mode (Round 11)
        self.validation_mode = cp.get("validation_mode", self.validation_mode)
        if self.validation_mode != "keyword" and not self.llm_judge:
            self.llm_judge = LLMJudge(self.llm)
        if self.llm_judge and self.validation_mode in ("llm_full", "llm_critical", "hybrid"):
            for w in self.workers.values():
                w.llm_judge = self.llm_judge
                w.validation_mode = self.validation_mode
        # Restore execution log (Round 10) — load from disk, not checkpoint
        self.pipeline_mode = cp.get("pipeline_mode", self.pipeline_mode)
        if (cp.get("has_execution_log") or cp.get("execution_log")) and ws_dir:
            # Prefer loading from disk (execution_log.json) — avoids checkpoint bloat
            if cp.get("execution_log"):
                # Legacy: old checkpoint with embedded execution_log
                self.execution_log = ExecutionLog.from_dict(cp["execution_log"], ws_dir)
            else:
                # New: load from workspace/execution_log.json
                self.execution_log = ExecutionLog(ws_dir)
            for w in self.workers.values():
                w.execution_log = self.execution_log
        # Restore research tree (Round 17)
        if "research_tree" in cp and cp["research_tree"]:
            self.research_tree = ResearchTree.from_dict(cp["research_tree"])
            self._active_branch_id = cp.get("active_branch_id", self.research_tree.root_id)
        # Restore hypothesis chain (Round 16.2)
        if "hypothesis_chain" in cp:
            from core.hypothesis_generator import HypothesisRecord, Hypothesis
            self.hypothesis_gen.history = []
            for rec in cp["hypothesis_chain"]:
                self.hypothesis_gen.history.append(HypothesisRecord(
                    hypothesis=Hypothesis(
                        claim=rec.get("claim", ""),
                        reasoning="", experiment="", expected_outcome="",
                    ),
                    outcome=rec.get("outcome", "untested"),
                    evidence=rec.get("evidence", ""),
                    cycle=rec.get("cycle", 0),
                ))

    # ── Main entry points ────────────────────────────────────────────

    def run_mission(self, goal: str, max_cycles: int = None) -> str:
        """Start a new mission from scratch."""
        self.goal = goal
        self.direction = (self.mission_ctx.direction
                          if self.mission_ctx else goal)
        self.max_cycles = max_cycles or self.max_cycles
        self._initial_max_cycles = self.max_cycles  # Hard cap for auto-extend
        self.cycle = 0
        self.completed_tasks = []
        self.task_queue = []
        self.errors = []
        self.reports_generated = 0
        self.insight_dag = InsightDAG()
        self.working_memory = ""
        self.result_verifier = ResultVerifier()
        if self.validation_mode not in ("llm_full", "hybrid"):
            for w in self.workers.values():
                w.result_verifier = self.result_verifier

        # Reset execution log for new mission
        if self.pipeline_mode == "structured" and self.mission_ctx:
            self.execution_log = ExecutionLog(self.mission_ctx.workspace_dir)
            for w in self.workers.values():
                w.execution_log = self.execution_log

        self._print_header()

        # ── Research design validation (Round 15.3) ──
        if not self.literature_only:
            print(f"  [Validator] Checking experiment design...")
            verdict = self.research_validator.validate(
                goal=self.direction,
                evolution_store=self.evolution_store,
            )
            self._design_verdict = verdict

            if not verdict.viable:
                print(f"  [Validator] REJECTED: {verdict.reject_reason}")
                for iss in verdict.issues:
                    print(f"    [{iss.severity}] {iss.description}")
                # Only modify direction for FATAL rejections
                self.direction = f"{self.direction}\n\nWARNING: {verdict.reject_reason}"

            elif verdict.modified_goal:
                print(f"  [Validator] Modified design: {verdict.modifications_summary}")
                for iss in verdict.issues:
                    print(f"    [{iss.severity}] {iss.description} → {iss.fix}")
                # Use modified goal as direction (this is a genuine redesign)
                self.direction = verdict.modified_goal

            elif verdict.issues:
                # Minor/major issues: log them but do NOT modify direction
                # These are noted in _design_verdict for the planner to consider
                for iss in verdict.issues:
                    print(f"  [Validator] [{iss.severity}] {iss.description}")

        # Parse goal into measurable sub-goals
        print(f"  [GoalTracker] Parsing goal into sub-goals...")
        self.goal_tracker.parse_goal(self.direction)
        for sg in self.goal_tracker.sub_goals:
            print(f"    - [{sg.type}] {sg.description}")

        # Initialize research tree (progressive branching)
        self.research_tree = ResearchTree(self.direction)
        self._active_branch_id = self.research_tree.root_id

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
            # ── Adaptive cycle management (inspired by kael_daemon) ──
            # Auto-extend: if tasks remain and progress is strong, add cycles
            if self.cycle == self.max_cycles - 1 and self.task_queue:
                trend = self.process_reward.get_trend()
                hard_limit = self._initial_max_cycles * 2  # Never exceed 2x original budget
                if trend in ("strong_progress", "moderate_progress") and self.max_cycles < hard_limit:
                    extension = min(3, len(self.task_queue), hard_limit - self.max_cycles)
                    if extension > 0:
                        self.max_cycles += extension
                        print(f"  [Supervisor] Auto-extending {extension} cycles "
                              f"(tasks remain, progress={trend}) → max={self.max_cycles}")

            # Early completion: if research question is answered, skip remaining cycles
            if (not self.task_queue and len(self.completed_tasks) >= 3
                    and self.cycle > 5):
                workers_used = set(t.get("worker") for t in self.completed_tasks
                                   if t.get("success"))
                has_all = ("coder" in workers_used and "reviewer" in workers_used)
                if has_all and not self.literature_only:
                    # Check for result files
                    if hasattr(self, 'mission_ctx') and self.mission_ctx:
                        import glob as _glob
                        ws = self.mission_ctx.workspace_dir
                        has_results = bool(_glob.glob(os.path.join(ws, "results*.json")) or
                                          _glob.glob(os.path.join(ws, "analysis*.json")))
                        has_figures = bool(_glob.glob(os.path.join(ws, "*.png")))
                        if has_results and has_figures:
                            print(f"  [Supervisor] Early completion: all tasks done, "
                                  f"results + figures present (cycle {self.cycle}/{self.max_cycles})")
                            break

            # R20.3: "Good enough" detector — stop if analysis is statistically complete
            if (self.cycle > 7 and not self.literature_only
                    and hasattr(self, 'mission_ctx') and self.mission_ctx):
                if self._is_research_complete():
                    print(f"  [Supervisor] Research complete: statistical analysis found "
                          f"(cycle {self.cycle}/{self.max_cycles})")
                    break
                # Also stop if stagnating/declining with partial results
                trend = self.process_reward.get_trend()
                if trend in ("stagnating", "declining") and self.cycle > 10:
                    import glob as _glob
                    ws = self.mission_ctx.workspace_dir
                    has_any_results = bool(_glob.glob(os.path.join(ws, "results*.json")))
                    if has_any_results:
                        print(f"  [Supervisor] Stopping: {trend} trend with results present "
                              f"(cycle {self.cycle}/{self.max_cycles})")
                        break

            self.cycle += 1

            # ── Live terminal: check for user messages ──
            self._check_live_messages()

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

            # ── Stagnation detection: consecutive coder failures → pivot ──
            recent_tasks = self.completed_tasks[-3:] if len(self.completed_tasks) >= 3 else []
            consecutive_coder_fails = 0
            for t in reversed(recent_tasks):
                if t.get("worker") == "coder" and not t.get("success"):
                    consecutive_coder_fails += 1
                else:
                    break
            if consecutive_coder_fails >= 2 and action_type in ("implement", "write_code", "fix_code"):
                print(f"  [Supervisor] Stagnation: {consecutive_coder_fails} consecutive coder failures → pivoting to reviewer")
                action = {"action": "benchmark",
                          "task": f"Evaluate whatever results exist so far for: {self.direction}. "
                                  "Load any result JSON files, compute statistics, create figures. "
                                  "If no results exist, report that and summarize literature findings.",
                          "reason": "Consecutive coder failures — pivot to evaluation"}
                action_type = "benchmark"

            # R20 fix: prevent re-dispatching tasks that already failed 2+ times
            if action_type in ("implement", "write_code", "fix_code"):
                proposed_task = action.get("task", "")
                if proposed_task:
                    task_prefix = proposed_task[:100]
                    past_fails = sum(
                        1 for ct in self.completed_tasks
                        if not ct.get("success")
                        and ct.get("task", "")[:100] == task_prefix
                    )
                    if past_fails >= 2:
                        print(f"  [Supervisor] SKIP repeated failure ({past_fails}x): {proposed_task[:60]}...")
                        # Try the next queued task instead, or skip to reviewer
                        alt_task = self._pop_queue_task("coder")
                        if alt_task and alt_task[:100] != task_prefix:
                            action = {"action": "implement", "task": alt_task,
                                      "reason": "Skipped repeated failure, trying next task"}
                        else:
                            action = {"action": "benchmark",
                                      "task": f"Evaluate whatever results exist so far for: {self.direction}. "
                                              "Load any result JSON files, compute statistics, create figures.",
                                      "reason": "All coder tasks failing — pivot to evaluation"}
                            action_type = "benchmark"

            # ── Hard guard: block premature "done" ────────────
            if action_type == "done":
                workers_used = set(t.get("worker") for t in self.completed_tasks
                                   if t.get("success"))
                has_code = "coder" in workers_used
                has_eval = "reviewer" in workers_used

                # Check if execution log has real metrics (not just file writes)
                has_real_metrics = False
                if self.execution_log:
                    recent = self.execution_log.get_latest_metrics(n=20)
                    # Look for result metrics (accuracy, loss, f1, etc.)
                    result_keywords = {"accuracy", "loss", "f1", "precision", "recall", "bleu", "rouge", "perplexity"}
                    for entry in recent:
                        metric_names = {k.lower() for k in entry.get("metrics", {}).keys()}
                        if metric_names & result_keywords:
                            has_real_metrics = True
                            break

                # Fallback: check for result JSON files in workspace
                if not has_real_metrics and hasattr(self, 'mission_ctx') and self.mission_ctx:
                    import glob
                    ws = self.mission_ctx.workspace_dir
                    result_files = glob.glob(os.path.join(ws, "results*.json")) + \
                                   glob.glob(os.path.join(ws, "**/results*.json"), recursive=True)
                    for rf in result_files:
                        try:
                            with open(rf) as f:
                                data = json.load(f)
                            keys = {k.lower() for k in data.keys()} if isinstance(data, dict) else set()
                            if keys & result_keywords:
                                has_real_metrics = True
                                break
                        except Exception:
                            continue

                if self.literature_only:
                    pass  # Literature-only mode: no code/eval requirements
                elif not has_code:
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
                elif not has_real_metrics and self.cycle < self.max_cycles - 1:
                    print(f"  [Supervisor] BLOCKED 'done' — no real metrics (accuracy/loss/f1) in execution log!")
                    action = {"action": "fix_code",
                              "task": f"Training did not produce metrics. Re-run or fix: {self.direction}",
                              "worker": "coder",
                              "reason": "No result metrics found — training may have crashed"}
                    action_type = "fix_code"

            print(f"  [Supervisor] Decision: {action_type}")
            if action.get("reason"):
                print(f"  [Supervisor] Reason: {action['reason']}")

            # ── Step 3: Execute ────────────────────────────────
            if action_type == "done":
                print(f"\n  [Supervisor] Mission complete!")
                self.agent_state = AgentState.FINISHED
                report = self._generate_report()
                self._auto_score()
                self._save_checkpoint()
                self._cleanup_workspace_processes()
                self._print_footer()

                # Post-mission evolution reflection
                if self.evolution_store:
                    try:
                        print(f"  [Evolution] Reflecting on mission...")
                        mid = self.mission_ctx.mission_id if self.mission_ctx else ""
                        self.evolution_store.reflect_on_mission(
                            mission_id=mid,
                            goal=self.goal,
                            tasks=self.completed_tasks,
                            dag=self.insight_dag,
                            llm=self.llm,
                        )
                        # Extract research findings from workspace
                        ws = self.mission_ctx.workspace_dir if self.mission_ctx else ""
                        if ws:
                            n = self.evolution_store.extract_research_findings(mid, self.goal, ws)
                            if n:
                                print(f"  [Evolution] Extracted {n} research findings")
                        # Extract hypothesis chain as cross-mission knowledge
                        if self.hypothesis_gen.history:
                            chain_dicts = [r.to_dict() for r in self.hypothesis_gen.history]
                            self.evolution_store.extract_hypothesis_chain(mid, self.goal, chain_dicts)
                            evaluated = sum(1 for r in self.hypothesis_gen.history if r.outcome != "untested")
                            if evaluated:
                                print(f"  [Evolution] Stored {evaluated} evaluated hypotheses")
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
                task_desc = self._pop_queue_task("explorer") or action.get("task", self.direction)
                self._dispatch_worker("explorer", task_desc)

            elif action_type in ("implement", "write_code"):
                if self.literature_only:
                    # Redirect to explorer in literature-only mode
                    task_desc = self._pop_queue_task("explorer") or f"Search for more papers on: {self.direction}"
                    self._dispatch_worker("explorer", task_desc)
                else:
                    task_desc = self._pop_queue_task("coder") or action.get("task", f"Implement: {self.direction}")
                    self._try_parallel_dispatch("coder", task_desc)

            elif action_type == "fix_code":
                if self.literature_only:
                    task_desc = self._pop_queue_task("explorer") or f"Search for more papers on: {self.direction}"
                    self._dispatch_worker("explorer", task_desc)
                else:
                    error_ctx = action.get("error_context", "")
                    task_desc = action.get("task", f"Fix the code. Error: {error_ctx}")
                    self._dispatch_worker("coder", task_desc)

            elif action_type in ("benchmark", "evaluate", "review"):
                if self.literature_only:
                    task_desc = self._pop_queue_task("explorer") or f"Summarize all findings for: {self.direction}"
                    self._dispatch_worker("explorer", task_desc)
                else:
                    task_desc = self._pop_queue_task("reviewer") or action.get("task", f"Evaluate results for: {self.direction}")
                    self._try_parallel_dispatch("reviewer", task_desc)

            elif action_type == "improve":
                worker = action.get("worker", "coder")
                task_desc = self._pop_queue_task(worker) or action.get("task", f"Improve implementation for: {self.direction}")
                self._dispatch_worker(worker, task_desc)

            elif action_type == "backtrack":
                self._tree_backtrack(action.get("reason", ""))

            elif action_type == "replan":
                self.agent_state = AgentState.PLANNING
                self._replan(action.get("reason", ""))
                self.agent_state = AgentState.RUNNING

            elif action_type == "reorganize":
                self._reorganize_knowledge()

            else:
                print(f"  [Supervisor] Unknown action '{action_type}', defaulting to explore")
                self._dispatch_worker("explorer", action.get("task", self.direction))

            # ── Step 4b: Goal/progress check ──────────────────
            if self.llm_judge and self.validation_mode in ("llm_full", "hybrid"):
                # LLM Judge Call 2: assess_progress replaces GoalTracker + FlowMonitor
                self._assess_progress_with_judge()
            else:
                # Original keyword-based goal tracking
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

            # ── Step 4c: Flow monitor analysis (single call per cycle) ─
            # In llm_full/hybrid mode, flow monitor is replaced by judge assess_progress
            if self.validation_mode not in ("llm_full", "hybrid"):
                self._last_advisories = self.flow_monitor.analyze(
                    cycle=self.cycle,
                    tasks=self.completed_tasks,
                    dag=self.insight_dag,
                    failures=self._consecutive_failures,
                )
                if self._last_advisories:
                    for adv in self._last_advisories:
                        print(f"  [FlowMonitor] {adv.severity}: {adv.message}")
                        self.event_bus.emit(EventType.FLOW_ADVISORY, adv.to_dict(),
                                            source="flow_monitor")

                    # Apply critical advisory hard overrides
                    for adv in self._last_advisories:
                        if adv.severity == "critical" and adv.suggested_action.startswith("skip_worker:"):
                            worker_to_skip = adv.suggested_action.split(":")[1]
                            self._consecutive_failures[worker_to_skip] = 0
                            print(f"  [FlowMonitor] Hard override: skipping {worker_to_skip}")

            # ── Step 5: Checkpoint ─────────────────────────────
            self._save_checkpoint()

        # Hit max cycles — write final report
        print(f"\n  [Supervisor] Max cycles ({self.max_cycles}) reached")
        self.agent_state = AgentState.FINISHED
        report = self._generate_report()
        self._auto_score()
        self._save_checkpoint()
        self._cleanup_workspace_processes()
        self._print_footer()

        # Close evolution store feedback loop (R20.2)
        if self.evolution_store:
            try:
                # Determine if mission was successful (grade A/B = helpful)
                success_tasks = sum(1 for t in self.completed_tasks if t.get("success"))
                total_tasks = max(len(self.completed_tasks), 1)
                was_helpful = (success_tasks / total_tasks) >= 0.5
                self.evolution_store.record_applied_learnings(was_helpful)
            except Exception as e:
                print(f"  [Evolution] Feedback recording failed: {e}")

        # Post-mission evolution reflection
        if self.evolution_store:
            try:
                print(f"  [Evolution] Reflecting on mission...")
                mid = self.mission_ctx.mission_id if self.mission_ctx else ""
                self.evolution_store.reflect_on_mission(
                    mission_id=mid,
                    goal=self.goal,
                    tasks=self.completed_tasks,
                    dag=self.insight_dag,
                    llm=self.llm,
                )
                # Extract research findings from workspace
                ws = self.mission_ctx.workspace_dir if self.mission_ctx else ""
                if ws:
                    n = self.evolution_store.extract_research_findings(mid, self.goal, ws)
                    if n:
                        print(f"  [Evolution] Extracted {n} research findings")
                # Extract hypothesis chain as cross-mission knowledge
                if self.hypothesis_gen.history:
                    chain_dicts = [r.to_dict() for r in self.hypothesis_gen.history]
                    self.evolution_store.extract_hypothesis_chain(mid, self.goal, chain_dicts)
                    evaluated = sum(1 for r in self.hypothesis_gen.history if r.outcome != "untested")
                    if evaluated:
                        print(f"  [Evolution] Stored {evaluated} evaluated hypotheses")
                print(f"  [Evolution] Learnings saved ({len(self.evolution_store.learnings)} total)")
            except Exception as e:
                print(f"  [Evolution] Reflection failed: {e}")

        return report

    # ── LLM Judge progress assessment (Round 11) ────────────────────

    def _assess_progress_with_judge(self):
        """Use LLM Judge Call 2 to assess progress (replaces GoalTracker + FlowMonitor)."""
        try:
            # Gather workspace files
            workspace_files = []
            if self.mission_ctx:
                ws_dir = self.mission_ctx.workspace_dir
                try:
                    for root, dirs, fnames in os.walk(ws_dir):
                        for fn in fnames:
                            if '__pycache__' not in root and not fn.startswith('.'):
                                rel = os.path.relpath(os.path.join(root, fn), ws_dir)
                                workspace_files.append(rel)
                except Exception:
                    pass

            # get_summary returns a dict — convert to string for the judge
            knowledge_raw = self.knowledge.get_summary(depth=1)
            if isinstance(knowledge_raw, dict):
                import json as _json
                knowledge_str = _json.dumps(knowledge_raw, ensure_ascii=False, default=str)
            else:
                knowledge_str = str(knowledge_raw)

            assessment = self.llm_judge.assess_progress(
                goal=self.goal,
                completed_tasks=self.completed_tasks,
                workspace_files=workspace_files,
                knowledge_summary=knowledge_str,
                working_memory=self.working_memory,
            )

            progress = assessment.get("progress_pct", 0)
            completed = assessment.get("sub_goals_completed", [])
            remaining = assessment.get("sub_goals_remaining", [])
            issues = assessment.get("detected_issues", [])

            print(f"  [LLMJudge] Progress: {progress}%")
            if completed:
                print(f"  [LLMJudge] Completed: {', '.join(completed[:3])}")
            if remaining:
                print(f"  [LLMJudge] Remaining: {', '.join(remaining[:3])}")
            if assessment.get("quality_assessment"):
                print(f"  [LLMJudge] Quality: {assessment['quality_assessment'][:100]}")

            # Convert detected issues to flow-monitor-style advisories for the decision prompt
            self._last_advisories = []
            for issue in issues:
                from supervisor.flow_monitor import Advisory
                self._last_advisories.append(Advisory(
                    severity="warning",
                    category="llm_judge",
                    message=issue,
                    suggested_action="replan" if "stagnation" in issue.lower() else "continue",
                ))

            if self._last_advisories:
                for adv in self._last_advisories:
                    print(f"  [LLMJudge] Issue: {adv.message}")

            # Emit progress event
            if progress >= 100:
                self.event_bus.emit(EventType.GOAL_ALL_COMPLETE, {
                    "completion_rate": 1.0,
                }, source="llm_judge")

            # Store assessment for checkpoint
            self._last_judge_assessment = assessment

        except Exception as e:
            print(f"  [LLMJudge] Progress assessment failed ({e}), falling back to keyword")
            # Fallback to keyword-based checks
            goal_status = self.goal_tracker.check_completion(
                self.completed_tasks,
                knowledge_stats=self.knowledge.stats(),
                dag=self.insight_dag,
            )
            self._last_advisories = self.flow_monitor.analyze(
                cycle=self.cycle,
                tasks=self.completed_tasks,
                dag=self.insight_dag,
                failures=self._consecutive_failures,
            )

    def _format_friction_buffer(self) -> str:
        """Format recent failure diagnoses for supervisor prompt. Max 3 items."""
        if not hasattr(self, '_friction_buffer') or not self._friction_buffer:
            return "  (none)"
        # Show last 3 friction items
        items = self._friction_buffer[-3:]
        lines = []
        for f in items:
            lines.append(f"- [{f['trigger']}] {f['root_cause'][:80]}")
            if f.get('better_action'):
                lines.append(f"  → Better: {f['better_action'][:80]}")
        return "\n".join(lines)

    def _format_hypotheses_for_prompt(self) -> str:
        """Format pending hypotheses for supervisor decision prompt."""
        if not self._pending_hypotheses or not self._pending_hypotheses.hypotheses:
            return ""
        return self.hypothesis_gen.format_for_supervisor(self._pending_hypotheses)

    # ══════════════════════════════════════════════════════════════════
    #  RESEARCH TREE — Progressive Branch Search (Round 17)
    # ══════════════════════════════════════════════════════════════════

    def _format_tree_for_prompt(self) -> str:
        """Format research tree state for supervisor decision prompt."""
        if not self.research_tree or len(self.research_tree.nodes) <= 1:
            return ""
        tree_summary = self.research_tree.get_tree_summary()
        branch_ctx = ""
        if self._active_branch_id:
            branch_ctx = self.research_tree.get_branch_context(self._active_branch_id)
        parts = [tree_summary]
        if branch_ctx:
            parts.append(branch_ctx)
        # Backtracking advisory
        if (self._active_branch_id and
                self.research_tree.should_backtrack(self._active_branch_id)):
            parts.append('\n⚠️ Current branch scores below parent — consider "backtrack".')
        return "\n".join(parts)

    def _tree_backtrack(self, reason: str = ""):
        """Backtrack: prune current branch, select next via UCB1."""
        if not self.research_tree or not self._active_branch_id:
            print(f"  [Tree] No active branch to backtrack from")
            return

        current = self.research_tree.nodes.get(self._active_branch_id)
        if current:
            print(f"  [Tree] Pruning branch: {current.hypothesis[:60]} "
                  f"(score: {current.score:.2f})")
            self.research_tree.prune(self._active_branch_id)

        # Select next branch via UCB1
        next_branch = self.research_tree.select_next()
        if next_branch:
            next_branch.status = "exploring"
            self._active_branch_id = next_branch.id
            print(f"  [Tree] Switching to: {next_branch.hypothesis[:60]}")
            # Queue an exploration task for the new branch
            self.task_queue.insert(0, {
                "worker": "coder",
                "task": (f"Test hypothesis: {next_branch.hypothesis}. "
                         f"Approach: {next_branch.approach or 'implement and evaluate'}"),
                "priority": 0,
                "depends_on": [],
                "source": "tree_backtrack",
            })
        else:
            print(f"  [Tree] No more branches to explore")

    def _tree_update_after_result(self, worker_name: str, result: dict):
        """Update research tree after worker completes."""
        if not self.research_tree or not self._active_branch_id:
            return
        if worker_name not in ("coder", "reviewer"):
            return

        # Compute score from result
        score = 0.0
        if result.get("success"):
            score = 0.5  # Base success score
            # Boost from verification
            vs = result.get("verification_score")
            if vs is not None:
                score = max(score, vs * 0.7)
            # Boost from metrics in output
            output = result.get("output", "")
            if any(k in output.lower() for k in ("accuracy", "f1", "loss", "p_value")):
                score += 0.2
        else:
            score = 0.1  # Failure gets minimal score

        self.research_tree.update_score(
            self._active_branch_id, min(1.0, score),
            cycle=self.cycle,
        )

        # Track debug depth: reset on success, increment on failure
        if result.get("success"):
            self.research_tree.reset_debug_depth(self._active_branch_id)
        else:
            still_alive = self.research_tree.increment_debug_depth(self._active_branch_id)
            if not still_alive:
                # Debug depth cap reached — select next branch
                print(f"  [Tree] Debug cap reached, selecting next branch")
                next_node = self.research_tree.select_next()
                if next_node:
                    next_node.status = "exploring"
                    self._active_branch_id = next_node.id
                    print(f"  [Tree] Switched to: {next_node.hypothesis[:60]}")

        node = self.research_tree.nodes.get(self._active_branch_id)
        if node:
            print(f"  [Tree] Branch score: {node.score:.2f} "
                  f"(visits: {node.visits}, depth: {node.depth}, "
                  f"debug: {node.debug_depth})")

    def _tree_expand_from_hypotheses(self, hypotheses: list):
        """Expand tree when new hypotheses are generated."""
        if not self.research_tree or not self._active_branch_id:
            return

        children = []
        for h in hypotheses[:3]:  # Max 3 children
            children.append({
                "claim": h.claim,
                "approach": h.experiment if hasattr(h, 'experiment') else "",
            })

        if children:
            new_ids = self.research_tree.expand(self._active_branch_id, children)
            if new_ids:
                # Mark current branch as completed (it produced children)
                self.research_tree.complete(self._active_branch_id)
                # Switch to first child (highest priority hypothesis)
                first_child = self.research_tree.nodes[new_ids[0]]
                first_child.status = "exploring"
                self._active_branch_id = new_ids[0]
                print(f"  [Tree] Expanded {len(new_ids)} branches from current. "
                      f"Now exploring: {first_child.hypothesis[:60]}")

    # ══════════════════════════════════════════════════════════════════
    #  FAILURE ANALYSIS (Round 13)
    # ══════════════════════════════════════════════════════════════════

    def _analyze_and_adapt_failure(self, worker_name: str, task_desc: str,
                                    result: dict, result_entry: dict):
        """Diagnose failure and inject corrective action into task queue."""
        from core.failure_analyzer import analyze_failure

        # Gather prior failures for anti-retry
        prior = [
            t.get("failure_analysis", {}) for t in self.completed_tasks
            if not t.get("success") and t.get("failure_analysis")
        ]

        # Extract stderr/stdout from result
        stderr = ""
        stdout = ""
        output = result.get("output", "")
        error = result.get("error", "")
        # Try to get stderr from tool calls
        for tc in (result.get("tool_calls") or []):
            if tc.get("name") == "run_python_code":
                stderr = tc.get("stderr", "")
                break

        # Get envelope if available
        envelopes = result.get("decision_envelopes", [])
        envelope = envelopes[-1] if envelopes else None

        try:
            analysis = analyze_failure(
                llm=self.llm,
                task=task_desc,
                worker=worker_name,
                error=error or output[:500],
                stderr=stderr,
                stdout=stdout,
                envelope=envelope,
                prior_failures=prior[-3:],  # last 3 failures
            )
        except Exception as e:
            print(f"  [FailureAnalyzer] Analysis failed: {e}")
            return

        # Store analysis on the result entry
        result_entry["failure_analysis"] = analysis.to_dict()
        print(f"  [FailureAnalyzer] {analysis.failure_class}: {analysis.root_cause[:80]}")
        print(f"  [FailureAnalyzer] → {analysis.next_action}"
              + (f": {analysis.modification[:80]}" if analysis.modification else ""))

        # ── Build reflexion context if multiple failures ─────
        from core.failure_analyzer import build_reflexion_context
        reflexion = build_reflexion_context(task_desc, self.completed_tasks)

        # ── Inject corrective action into queue ──────────────
        if analysis.next_action == "retry_modified" and analysis.modification:
            modified_task = f"{task_desc}\n\nIMPORTANT FIX: {analysis.modification}"
            if reflexion:
                modified_task = f"{task_desc}\n\n{reflexion}\n\nLATEST FIX: {analysis.modification}"
            self.task_queue.insert(0, {
                "worker": worker_name,
                "task": modified_task,
                "priority": 0,  # highest priority
                "depends_on": [],
                "status": "pending",
                "source": "failure_analyzer",
            })
            print(f"  [FailureAnalyzer] Queued modified retry"
                  + (f" (with {len(reflexion.splitlines())}-line reflexion)" if reflexion else ""))

        elif analysis.next_action == "decompose":
            if analysis.subtasks:
                for i, subtask in enumerate(analysis.subtasks):
                    # Inject reflexion into first subtask so context carries over
                    enriched = f"{subtask}\n\n{reflexion}" if (i == 0 and reflexion) else subtask
                    self.task_queue.insert(i, {
                        "worker": worker_name,
                        "task": enriched,
                        "priority": i,
                        "depends_on": [],
                        "status": "pending",
                        "source": "failure_analyzer",
                    })
                print(f"  [FailureAnalyzer] Decomposed into {len(analysis.subtasks)} subtasks")
            elif "timeout" in (error or "").lower() or "timeout" in analysis.root_cause.lower():
                # Timeout with no subtasks — auto-modify task to be smaller
                timeout_fix = (
                    f"{task_desc}\n\n"
                    "CRITICAL: Previous attempt TIMED OUT (600s limit). You MUST:\n"
                    "- Reduce to 1 epoch (not 2-3)\n"
                    "- Use only 1000 samples (not 2000)\n"
                    "- Use smallest model variant available\n"
                    "- Save results to JSON after EACH seed (not all at end)"
                )
                self.task_queue.insert(0, {
                    "worker": worker_name,
                    "task": timeout_fix,
                    "priority": 0,
                    "depends_on": [],
                    "status": "pending",
                    "source": "failure_analyzer_timeout",
                })
                print(f"  [FailureAnalyzer] Timeout → queued reduced-scope retry")

        elif analysis.next_action == "switch_worker":
            alt_worker = "coder" if worker_name != "coder" else "reviewer"
            self.task_queue.insert(0, {
                "worker": alt_worker,
                "task": task_desc,
                "priority": 0,
                "depends_on": [],
                "status": "pending",
                "source": "failure_analyzer",
            })
            print(f"  [FailureAnalyzer] Switched to {alt_worker}")

        elif analysis.next_action == "patch_env":
            if analysis.modification:
                self.task_queue.insert(0, {
                    "worker": "coder",
                    "task": f"Fix environment: {analysis.modification}",
                    "priority": 0,
                    "depends_on": [],
                    "status": "pending",
                    "source": "failure_analyzer",
                })
                print(f"  [FailureAnalyzer] Queued env patch")

        elif analysis.next_action == "simplify":
            simplify_directive = (
                f"{task_desc}\n\n"
                "⚠️ SIMPLIFICATION REQUIRED — previous approaches failed repeatedly.\n"
                f"Diagnosis: {analysis.root_cause}\n"
                "You MUST simplify:\n"
                "- Use raw PyTorch training loop instead of HuggingFace Trainer\n"
                "- Use the simplest model that works (e.g. nn.Linear, small CNN, or basic LSTM)\n"
                "- If an architecture is fundamentally incompatible, use an alternative\n"
                "- Reduce to bare minimum: 1 epoch, 500 samples, basic metrics\n"
                "- Get SOMETHING working first, then improve\n"
            )
            if analysis.modification:
                simplify_directive += f"- Specific fix: {analysis.modification}\n"
            self.task_queue.insert(0, {
                "worker": worker_name,
                "task": simplify_directive,
                "priority": 0,
                "depends_on": [],
                "status": "pending",
                "source": "failure_analyzer_simplify",
            })
            print(f"  [FailureAnalyzer] Simplification directive queued")

        elif analysis.next_action == "abort_branch":
            print(f"  [FailureAnalyzer] Branch aborted — will not retry this task")

        # Add to mission friction buffer
        if not hasattr(self, '_friction_buffer'):
            self._friction_buffer = []
        self._friction_buffer.append({
            "trigger": analysis.failure_class,
            "task": task_desc[:100],
            "root_cause": analysis.root_cause,
            "better_action": analysis.modification or analysis.next_action,
        })

    # ══════════════════════════════════════════════════════════════════
    #  LIVE TERMINAL — USER MESSAGE HANDLING
    # ══════════════════════════════════════════════════════════════════

    def _check_live_messages(self):
        """Check for user messages from the live terminal (called each cycle)."""
        if not self.message_bus:
            return

        messages = self.message_bus.check_user_messages()
        for msg in messages:
            if msg.msg_type == "abort":
                print(f"  [Supervisor] User requested abort — finishing up")
                self.agent_state = AgentState.FINISHED
                self.max_cycles = self.cycle  # Stop after this cycle

            elif msg.msg_type == "direction":
                print(f"  [Supervisor] User direction: {msg.text[:100]}")
                self.direction = msg.text
                # Add to working memory so the LLM sees it
                self.knowledge.update(
                    key="user_direction",
                    category="context",
                    data={"direction": msg.text, "cycle": self.cycle},
                )
                # Trigger replan if tasks remain
                if self.task_queue:
                    self._replan(reason=f"User changed direction: {msg.text}")

            elif msg.msg_type == "command":
                self._handle_user_command(msg)

            elif msg.msg_type == "chat":
                self._handle_user_chat(msg)

    def _handle_user_command(self, msg):
        """Handle /commands from the user."""
        if "/status" in msg.text:
            status = (
                f"Cycle {self.cycle}/{self.max_cycles} | "
                f"Queue: {len(self.task_queue)} tasks | "
                f"Done: {len(self.completed_tasks)} | "
                f"Errors: {len(self.errors)}"
            )
            if self.task_queue:
                next_task = self.task_queue[0]
                status += f"\nNext: [{next_task.get('worker')}] {str(next_task.get('task', ''))[:80]}"

            if self.message_bus:
                from terminal.message_bus import DisplayEvent
                self.message_bus.emit(DisplayEvent(
                    source="opus", event_type="status",
                    content=status,
                ))

    def _handle_user_chat(self, msg):
        """Handle free-text chat from the user. Answer from working memory."""
        # Add user message as context for this cycle
        self.knowledge.update(
            key="user_message",
            category="context",
            data={"message": msg.text, "cycle": self.cycle},
        )

        # Quick LLM response using working memory
        try:
            wm = self.knowledge.to_report()
            context = wm[:3000] if wm else "No research data yet."

            response = self.llm.chat([
                {"role": "system", "content":
                    "You are Opus, a research agent mid-mission. "
                    "Answer the user's question based on current research progress. "
                    "Be concise (1-3 sentences). Use data from your current findings."},
                {"role": "user", "content":
                    f"Current research state:\n{context}\n\n"
                    f"User asks: {msg.text}"},
            ])
            answer = response["choices"][0]["message"]["content"]

            # Strip thinking tags
            from core.llm import strip_think
            answer = strip_think(answer)

            if self.message_bus:
                from terminal.message_bus import DisplayEvent
                self.message_bus.emit(DisplayEvent(
                    source="opus", event_type="status",
                    content=answer,
                ))
        except Exception as e:
            print(f"  [Supervisor] Failed to answer user: {e}")

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

        # Skip LLM extraction for low-verification results to prevent amplifying fabricated numbers
        verification_score = result.get("verification_score")
        tool_calls = result.get("tool_calls", [])
        if success and verification_score is not None and verification_score < 0.3 and not tool_calls:
            # Low verification + no tool calls = likely fabricated
            insight_text = f"[{worker_name}] {task_desc[:80]} — completed but results UNVERIFIED (no tool execution detected)"
            node_id = self.insight_dag.add(
                cycle=self.cycle,
                worker=worker_name,
                task=task_desc,
                success=success,
                content=insight_text,
                references=[],
                code_refs=[],
            )
            print(f"  [Memory] Insight {node_id} (UNVERIFIED, skipping LLM extraction): {insight_text[:120]}...")
            return

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

        # Flow monitor advisories (from previous cycle's post-dispatch analysis)
        flow_text = FlowMonitor.format_for_prompt(self._last_advisories) if self._last_advisories else ""

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

## Active Friction (failure diagnoses — adapt your strategy)
{self._format_friction_buffer()}

{self.process_reward.format_for_prompt()}

{self._format_hypotheses_for_prompt()}

{self._format_tree_for_prompt()}

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
- "backtrack": Current research branch is a dead end. Return to parent and try a sibling.
- "report": Meaningful progress — write interim report.
- "done": FULLY achieved — has papers + code + experiments + results.

## Decision Rules
- NEVER "done" without code written AND benchmarks run
- After 2-3 search rounds → "implement"
- If code failed ONCE → "fix_code" (with specific error context)
- If same code failed TWICE → "replan" (the approach is wrong, not just a bug)
- If results poor → "improve" or "search_more"
- "done" = papers + code + tests + results analyzed with 5 seeds
- Prefer executing queued tasks over generating new ones
- Each condition MUST have 5 seeds (42, 123, 456, 789, 1024) — 3 seeds is NOT enough
- If analysis_summary.json exists with p_value + effect size → you can choose "done"

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

        # Fallback: if reflection fails, execute next queued task instead of giving up
        if not self.completed_tasks:
            return {"action": "search_more", "task": self.direction, "reason": "fallback — no tasks done yet"}
        if self.task_queue:
            next_task = self.task_queue[0]
            worker = next_task.get("worker", "coder")
            action_map = {"explorer": "search_more", "coder": "implement", "reviewer": "benchmark"}
            return {
                "action": action_map.get(worker, "implement"),
                "task": next_task.get("task", self.direction),
                "worker": worker,
                "reason": f"fallback — reflection failed, executing next queued task",
            }
        return {"action": "done", "reason": "reflection failed and no tasks queued"}

    def _pop_queue_task(self, worker_name: str) -> str:
        """Pop the next matching task from the planner's queue for this worker.
        Respects priority ordering. Skips tasks attempted 3+ times (anti-stagnation).
        Returns the task description, or empty string if no match."""
        best_idx = None
        best_priority = float('inf')
        for i, t in enumerate(self.task_queue):
            if t.get("worker") == worker_name:
                pri = t.get("priority", 999)
                if pri < best_priority:
                    best_priority = pri
                    best_idx = i
        if best_idx is not None:
            task = self.task_queue.pop(best_idx)
            desc = task.get("task", "")

            # Anti-stagnation: skip if same task prefix attempted 3+ times
            if desc:
                task_prefix = desc[:100]
                attempts = sum(
                    1 for ct in self.completed_tasks
                    if ct.get("task", "")[:100] == task_prefix
                )
                if attempts >= 3:
                    print(f"  [Supervisor] SKIP stale task (attempted {attempts}x): {desc[:80]}...")
                    return self._pop_queue_task(worker_name)  # try next task
                print(f"  [Supervisor] Using planned task from queue (priority {best_priority})")
            return desc
        return ""

    # ── Parallel dispatch ─────────────────────────────────────────────

    def _try_parallel_dispatch(self, primary_worker: str, primary_task: str):
        """Run primary task + an independent explorer task in parallel.

        Only parallelizes explorer with coder/reviewer (never coder+reviewer,
        as both write workspace files). Explorer is safe because it only
        reads external APIs.

        Returns after both tasks complete. The cycle count increments by 1
        but two tasks are done — effectively 2x throughput for that cycle.
        """
        # Only parallelize if primary is NOT explorer (avoid 2 explorers)
        if primary_worker == "explorer":
            self._dispatch_worker(primary_worker, primary_task)
            return

        # Check queue for an independent explorer task
        explorer_task = None
        for i, t in enumerate(self.task_queue):
            if t.get("worker") == "explorer":
                explorer_task = self.task_queue.pop(i)
                break

        if not explorer_task:
            # No parallel opportunity — dispatch normally
            self._dispatch_worker(primary_worker, primary_task)
            return

        explorer_desc = explorer_task.get("task", self.direction)
        print(f"  [Supervisor] ⚡ PARALLEL: {primary_worker} + explorer")

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading

        # Each worker needs its own context built on the main thread
        # (reading shared state is safe, writing happens inside _dispatch_worker)
        results = {}

        def run_primary():
            self._dispatch_worker(primary_worker, primary_task)

        def run_explorer():
            self._dispatch_worker("explorer", explorer_desc)

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(run_primary): primary_worker,
                executor.submit(run_explorer): "explorer",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  [Supervisor] Parallel {name} error: {e}")

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

        # Give all workers access to workspace_dir (for inner monologue context + file checks)
        if self.mission_ctx:
            worker._workspace_dir = self.mission_ctx.workspace_dir

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

        # Inject task-specific policies from catalog (Round 13)
        from core.policy_selector import get_policy_prompt
        friction = self._friction_buffer if hasattr(self, '_friction_buffer') else []
        policy_prompt = get_policy_prompt(task_desc, worker_name, friction)
        if policy_prompt:
            context_parts.append(policy_prompt)

        context = "\n\n".join(context_parts)

        result = worker.run(task_desc, context=context)

        # ── Thread-safe state mutations (for parallel dispatch) ──
        with self._dispatch_lock:
            self._process_worker_result(worker_name, worker, task_desc, result)

    def _process_worker_result(self, worker_name: str, worker, task_desc: str, result: dict):
        """Process worker result — must be called under _dispatch_lock."""
        # ── Handle pushback: worker refused the task ──────────
        if result.get("pushback"):
            reasoning = result.get("pushback_reasoning", "")
            print(f"  [Supervisor] ⚡ {worker_name} pushed back: {reasoning[:150]}")
            # Store as insight — pushbacks are valuable signal
            self.insight_dag.add(
                cycle=self.cycle, worker=worker_name,
                task=task_desc, success=False,
                content=f"PUSHBACK from {worker_name}: {reasoning}",
            )
            result_entry = {
                "worker": worker_name, "task": task_desc,
                "success": False, "pushback": True,
                "pushback_reasoning": reasoning,
            }
            self.completed_tasks.append(result_entry)
            return

        # Transfer tool call log and decision envelopes for downstream checks
        if hasattr(worker, '_last_tool_calls'):
            result["tool_calls"] = worker._last_tool_calls
        if hasattr(worker, '_last_envelopes'):
            result["decision_envelopes"] = [e.to_dict() for e in worker._last_envelopes]

        result_entry = {
            "worker": worker_name,
            "task": task_desc,
            **result,
        }
        self.completed_tasks.append(result_entry)

        # ── Verification quality gate ────────────────────────
        verification_score = result.get("verification_score")
        if result.get("success") and verification_score is not None:
            if verification_score < 0.3 and worker_name in ("coder", "reviewer"):
                print(f"  [Supervisor] LOW VERIFICATION: {worker_name} score={verification_score:.0%}")
                result_entry["low_verification"] = True

        # ── Step 4: Handle success or failure ────────────────
        if result.get("success"):
            print(f"  [Supervisor] ✓ {worker_name} completed successfully")
            self.event_bus.emit(EventType.TASK_COMPLETED, {
                "worker": worker_name, "task": task_desc,
            }, source="supervisor")
            self._consecutive_failures[worker_name] = 0
        else:
            error_msg = result.get('error', 'unknown')
            self.errors.append(f"[{worker_name}] {task_desc[:60]}: {error_msg}")
            print(f"  [Supervisor] ✗ {worker_name} failed: {(error_msg or '')[:100]}")

            # ── FailureAnalyzer: diagnose and adapt ──────────
            self._analyze_and_adapt_failure(
                worker_name, task_desc, result, result_entry
            )

            self._consecutive_failures[worker_name] = \
                self._consecutive_failures.get(worker_name, 0) + 1
            fail_count = self._consecutive_failures[worker_name]
            if fail_count >= 3:
                # Hard skip: remove all queued tasks from same worker to avoid infinite retry
                before = len(self.task_queue)
                self.task_queue = [t for t in self.task_queue
                                   if t.get("worker") != worker_name
                                   or t.get("source") != "failure_analyzer_timeout"]
                skipped = before - len(self.task_queue)
                print(f"  [Supervisor] SKIP: {worker_name} failed {fail_count}x — removing {skipped} retry tasks, moving on")
                self._consecutive_failures[worker_name] = 0
            elif fail_count >= 2:
                print(f"  [Supervisor] WARNING: {worker_name} has failed {fail_count}x — will try different approach")

        # ── Research Tree: update branch score ─────────────
        self._tree_update_after_result(worker_name, result)

        # ── Process Reward: score this cycle ────────────────
        try:
            import os, glob
            ws = self.mission_ctx.workspace_dir if self.mission_ctx else ""
            ws_files = []
            if ws and os.path.isdir(ws):
                ws_files = [os.path.basename(f) for f in glob.glob(os.path.join(ws, '*'))
                            if '__pycache__' not in f]
            cr = self.process_reward.score_cycle(
                self.cycle, worker_name, task_desc, result, ws_files)
            if cr.reward > 0.3:
                print(f"  [Reward] +{cr.reward:.2f} ({', '.join(cr.components.keys())})")
            elif cr.reward < -0.1:
                print(f"  [Reward] {cr.reward:.2f} — progress stalling")
        except Exception as e:
            print(f"  [Reward] scoring failed: {e}")

        # ── Explorer depth check: if too few papers, auto-queue more search ──
        if result.get("success") and worker_name == "explorer":
            self._check_explorer_depth(result, task_desc)

        # ── Hypothesis Generation: after reviewer produces results ──
        if result.get("success") and worker_name == "reviewer":
            self._generate_hypotheses(result)

        # Always extract insight, whether success or failure
        self._extract_insight(worker_name, task_desc, result)

    # ── Explorer Depth Check ─────────────────────────────────────────

    def _check_explorer_depth(self, result: dict, task_desc: str):
        """After explorer finishes, check if enough papers were found.
        If <5 unique papers and cycles remain, auto-queue ONE deeper search.
        Max 1 expansion (to avoid wasting cycles on repeated explorer tasks)."""
        output = result.get("output", "")
        if not output:
            return

        # Count unique papers using multiple heuristics
        import re
        paper_markers = re.findall(r'###\s*\d+\.', output)
        et_al_count = output.count("et al")
        title_count = len(re.findall(r'\*\*Title\*\*|\btitle\b.*:', output, re.IGNORECASE))
        # Also detect table rows with arXiv IDs or year numbers (common in paper tables)
        arxiv_ids = len(re.findall(r'\d{4}\.\d{4,5}', output))
        # Bold titles in table rows: | **Title Here** |
        bold_in_table = len(re.findall(r'\|\s*\*\*[^|]+\*\*\s*\|', output))
        paper_count = max(len(paper_markers), et_al_count, title_count, arxiv_ids, bold_in_table)

        # Track how many depth expansions have already been done (max 1)
        if not hasattr(self, '_explorer_depth_expansions'):
            self._explorer_depth_expansions = 0

        # Check if we already have a queued explorer deepening task
        has_queued_explorer = any(
            t.get("worker") == "explorer" and t.get("source") == "depth_check"
            for t in self.task_queue
        )

        if (paper_count < 5 and not has_queued_explorer
                and self._explorer_depth_expansions < 1
                and self.cycle < self.max_cycles - 2):
            self._explorer_depth_expansions += 1
            print(f"  [Supervisor] Explorer found only ~{paper_count} papers, queueing deeper search (expansion {self._explorer_depth_expansions}/1)")
            self.task_queue.insert(0, {
                "worker": "explorer",
                "task": (f"EXPAND literature: Previous search found only {paper_count} papers. "
                         f"Use get_citation_graph on the best paper found to discover more related work. "
                         f"Try different search queries and synonyms for: {self.direction}. "
                         f"Use web_search for recent blog posts and discussions. "
                         f"Target: find at least {8 - paper_count} MORE unique papers."),
                "priority": 0,
                "depends_on": [],
                "status": "pending",
                "source": "depth_check",
            })

    # ── Hypothesis Generation ────────────────────────────────────────

    def _generate_hypotheses(self, reviewer_result: dict):
        """Generate research hypotheses after reviewer produces results.

        Key feature: hypothesis chaining — new hypotheses build on previous
        confirmed/refuted ones, creating a research narrative.
        """
        try:
            output = reviewer_result.get("output", "")
            if len(output) < 100:
                return  # Too little to reason about

            # ── Step 1: Auto-evaluate previous hypotheses against new results ──
            self._evaluate_pending_hypotheses(output)

            # Build results summary from execution log + reviewer output
            results_summary = output[:3000]
            if self.execution_log:
                exec_summary = self.execution_log.get_summary_for_prompt()
                if exec_summary:
                    results_summary = f"{exec_summary}\n\n## Reviewer Analysis\n{output[:2000]}"

            # Literature context from knowledge tree
            lit_context = ""
            try:
                lit_raw = self.knowledge.get_summary(depth=1)
                if isinstance(lit_raw, dict):
                    lit_context = json.dumps(lit_raw, ensure_ascii=False, default=str)[:1500]
                else:
                    lit_context = str(lit_raw)[:1500]
            except Exception:
                pass

            print(f"  [HypothesisGen] Analyzing results for follow-up hypotheses "
                  f"(chain depth: {len(self.hypothesis_gen.history)})...")
            hyp_result = self.hypothesis_gen.generate(
                goal=self.goal,
                results_summary=results_summary,
                literature_context=lit_context,
                working_memory=self.working_memory,
            )

            if not hyp_result.hypotheses:
                print(f"  [HypothesisGen] No hypotheses generated")
                return

            self._pending_hypotheses = hyp_result

            # ── Step 2: Record new hypotheses as untested ──
            for h in hyp_result.hypotheses[:3]:
                from core.hypothesis_generator import HypothesisRecord
                self.hypothesis_gen.history.append(HypothesisRecord(
                    hypothesis=h, outcome="untested", cycle=self.cycle,
                ))

            # Print hypotheses
            chain_depth = len(self.hypothesis_gen.history)
            for i, h in enumerate(hyp_result.hypotheses[:3]):
                tag = " [not testable]" if not h.testable else ""
                print(f"  [HypothesisGen] H{chain_depth-2+i}: {h.claim[:120]}{tag}")

            if hyp_result.validity_concerns:
                print(f"  [HypothesisGen] Concerns: {'; '.join(hyp_result.validity_concerns[:2])}")

            # Queue the recommended follow-up as a high-priority task
            # But only if enough cycles remain (need ≥3: follow-up + reviewer + report)
            remaining_cycles = self.max_cycles - self.cycle
            if hyp_result.recommended_next and remaining_cycles >= 4:
                follow_up_task = {
                    "worker": "coder",
                    "task": hyp_result.recommended_next,
                    "priority": 1,  # High priority
                    "depends_on": [],
                    "source": "hypothesis_generator",
                }
                self.task_queue.insert(0, follow_up_task)
                print(f"  [HypothesisGen] Queued follow-up: {hyp_result.recommended_next[:100]}")

            # ── Expand research tree with new hypotheses ──
            if hyp_result.hypotheses:
                self._tree_expand_from_hypotheses(hyp_result.hypotheses)

        except Exception as e:
            print(f"  [HypothesisGen] Failed: {e}")

    def _evaluate_pending_hypotheses(self, new_results: str):
        """Auto-evaluate untested hypotheses against new results.

        Uses a quick LLM call to check: did the new results confirm, refute,
        or leave inconclusive any pending hypothesis?
        """
        untested = [r for r in self.hypothesis_gen.history if r.outcome == "untested"]
        if not untested:
            return

        # Build evaluation prompt
        hyp_list = "\n".join(
            f"{i+1}. {r.hypothesis.claim} (expected: {r.hypothesis.expected_outcome[:80]})"
            for i, r in enumerate(untested[-3:])  # Only evaluate recent 3
        )

        prompt = f"""Given these new experimental results, evaluate each hypothesis:

## Pending Hypotheses
{hyp_list}

## New Results
{new_results[:2000]}

For each hypothesis, respond with JSON:
```json
[
  {{"index": 1, "outcome": "confirmed|refuted|inconclusive", "evidence": "brief explanation referencing specific numbers"}}
]
```

Rules:
- "confirmed" = results support the claim (metrics match expected direction AND threshold)
- "refuted" = results contradict the claim
- "inconclusive" = results are ambiguous or don't test this hypothesis"""

        try:
            resp = self.llm.chat([
                {"role": "system", "content": "Evaluate hypotheses against results. JSON only."},
                {"role": "user", "content": prompt},
            ])
            from core.llm import strip_think
            text = strip_think(resp["choices"][0]["message"]["content"])

            # Parse
            import re
            json_match = re.search(r'\[[\s\S]*?\]', text)
            if json_match:
                evaluations = json.loads(json_match.group())
                recent_untested = untested[-3:]
                for ev in evaluations:
                    idx = ev.get("index", 0) - 1
                    if 0 <= idx < len(recent_untested):
                        rec = recent_untested[idx]
                        outcome = ev.get("outcome", "inconclusive")
                        if outcome in ("confirmed", "refuted", "inconclusive"):
                            rec.outcome = outcome
                            rec.evidence = ev.get("evidence", "")[:200]
                            rec.cycle = self.cycle
                            icon = {"confirmed": "✓", "refuted": "✗", "inconclusive": "~"}[outcome]
                            print(f"  [HypothesisGen] {icon} H evaluated: {rec.hypothesis.claim[:80]} → {outcome}")
        except Exception as e:
            print(f"  [HypothesisGen] Evaluation failed: {e}")

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

        # Get evolution guidance, research context, and quality rules
        evolution_guidance = ""
        if self.evolution_store:
            evolution_guidance = self.evolution_store.get_planner_guidance(self.direction)
            research_ctx = self.evolution_store.get_research_context(self.direction)
            if research_ctx:
                evolution_guidance = (evolution_guidance + "\n\n" + research_ctx) if evolution_guidance else research_ctx
            if evolution_guidance:
                print(f"  [Evolution] Injecting {len(self.evolution_store.get_relevant_learnings(self.direction))} learnings into planner")
        quality_rules = get_quality_rules(self.evolution_store, self.direction)

        # Inject validator notes (one-time, into planner only, not into direction)
        validator_notes = ""
        if self._design_verdict and self._design_verdict.issues:
            notes = []
            for iss in self._design_verdict.issues:
                notes.append(f"[{iss.severity}] {iss.description} → {iss.fix}")
            validator_notes = "\n## Design Validator Notes (consider but do not over-react to minor issues):\n" + "\n".join(notes)
            if evolution_guidance:
                evolution_guidance += "\n" + validator_notes
            else:
                evolution_guidance = validator_notes

        if self.literature_only:
            tasks = self._literature_only_plan()
        else:
            tasks = self.planner.decompose(
                self.direction,
                knowledge_summary=knowledge_summary,
                available_workers=list(self.workers.keys()),
                cross_knowledge=cross_knowledge,
                evolution_guidance=evolution_guidance,
                quality_rules=quality_rules,
                max_cycles=self.max_cycles,
            )
        self.task_queue = tasks
        print(f"  [Supervisor] Initial plan ({len(tasks)} tasks):")
        for i, t in enumerate(tasks):
            print(f"    {i+1}. [{t['worker']}] {t['task']}")

    def _literature_only_plan(self) -> list[dict]:
        """Generate explorer-only tasks for deep literature review mode.

        Multiple iterative search rounds + hypothesis generation.
        Each round searches different angles of the topic.
        """
        goal = self.direction
        budget = min(self.max_cycles - 1, 6)  # Reserve 1 cycle for summary

        tasks = [
            {
                "worker": "explorer",
                "task": (f"ROUND 1 — Broad search: Search for academic papers on: {goal}. "
                         "Search arxiv, semantic scholar, openalex, papers_with_code in parallel. "
                         "Use get_citation_graph on the top paper to expand results. "
                         "Find at least 8 unique papers with titles, authors, years, citation counts, key contributions."),
                "priority": 1, "depends_on": [], "status": "pending",
            },
            {
                "worker": "explorer",
                "task": (f"ROUND 2 — Deep dive: Read the top 2-3 most cited papers from Round 1 using read_paper. "
                         "Extract: methodology details, experimental setup, baselines compared, key results, limitations. "
                         "Also use web_search to find blog posts, tutorials, or benchmark comparisons about: {goal}"),
                "priority": 2, "depends_on": [1], "status": "pending",
            },
            {
                "worker": "explorer",
                "task": (f"ROUND 3 — Citation expansion: Use get_citation_graph on 2-3 papers from Round 1 "
                         "to discover additional related work. Focus on: (a) papers that cite the core work (newer approaches), "
                         "(b) papers referenced by the core work (foundational methods). "
                         "Target: 15+ total unique papers across all rounds. "
                         "Search for open-source implementations on GitHub."),
                "priority": 3, "depends_on": [2], "status": "pending",
            },
        ]

        if budget >= 4:
            tasks.append({
                "worker": "explorer",
                "task": (f"ROUND 4 — Alternative angles: Search for related but different approaches to: {goal}. "
                         "Try broader queries, synonyms, and adjacent research areas. "
                         "Use web_search for recent developments not yet in academic databases. "
                         "Read 1-2 more papers in depth. Target: 20+ total unique papers."),
                "priority": 4, "depends_on": [3], "status": "pending",
            })

        if budget >= 5:
            tasks.append({
                "worker": "explorer",
                "task": (f"ROUND 5 — Hypothesis generation: Based on ALL papers found across rounds, provide:\n"
                         "1. COMPREHENSIVE PAPER TABLE: All papers found (20+), sorted by relevance, with citation counts\n"
                         "2. TAXONOMY: Categorize approaches into 3-5 main families\n"
                         "3. TIMELINE: How the field evolved (key milestones)\n"
                         "4. RESEARCH GAPS: What hasn't been tried? What are the limitations?\n"
                         "5. HYPOTHESES: Propose 3-5 concrete, testable research directions with expected impact\n"
                         "6. RECOMMENDED EXPERIMENTS: For each hypothesis, what experiment would validate it?"),
                "priority": 5, "depends_on": [4] if budget >= 4 else [3], "status": "pending",
            })

        return tasks

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
            research_ctx = self.evolution_store.get_research_context(self.direction)
            if research_ctx:
                evolution_guidance = (evolution_guidance + "\n\n" + research_ctx) if evolution_guidance else research_ctx
        quality_rules = get_quality_rules(self.evolution_store, self.direction)

        # Inject friction buffer into quality rules so planner avoids known pitfalls
        friction = getattr(self, '_friction_buffer', [])
        if friction:
            friction_lines = ["## Known Failures This Mission (avoid repeating)"]
            for f in friction[-5:]:
                friction_lines.append(f"- {f.get('trigger','?')}: {f.get('root_cause','')} → {f.get('better_action','')}")
            quality_rules = quality_rules + "\n\n" + "\n".join(friction_lines)

        remaining_cycles = self.max_cycles - self.cycle
        tasks = self.planner.decompose(
            self.direction,
            knowledge_summary=knowledge_summary,
            completed_tasks=completed_descs,
            available_workers=list(self.workers.keys()),
            cross_knowledge=cross_knowledge,
            evolution_guidance=evolution_guidance,
            quality_rules=quality_rules,
            max_cycles=remaining_cycles,
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
        """Generate progress report with working memory + hypothesis chain + data sanity."""
        # Run data sanity check before report
        wm = self.working_memory or ""
        try:
            from core.deterministic_verifier import DeterministicVerifier
            verifier = DeterministicVerifier()
            det_result = verifier.verify(self.workspace_dir)
            fatal_issues = [i for i in det_result.issues
                            if any(k in i for k in ("FATAL", "INVALID", "ABSURD", "BROKEN"))]
            if fatal_issues:
                warning = "\n\n## ⚠ DATA SANITY WARNINGS (from deterministic verifier)\n"
                for issue in fatal_issues:
                    warning += f"- **{issue}**\n"
                warning += ("\nThese issues indicate fundamental experimental flaws. "
                            "The report MUST acknowledge these problems honestly.\n")
                wm = warning + wm
                print(f"  [DataSanity] {len(fatal_issues)} fatal issues injected into report context")
            elif det_result.issues:
                # Non-fatal but noteworthy
                minor = [i for i in det_result.issues if "SUSPICIOUS" in i or "NO_" in i]
                if minor:
                    note = "\n\n## Data Quality Notes\n"
                    for issue in minor[:3]:
                        note += f"- {issue}\n"
                    wm = note + wm
        except Exception as e:
            print(f"  [DataSanity] Pre-report check failed: {e}")

        # Inject research tree summary + hypothesis chain into working memory for report
        if self.research_tree and len(self.research_tree.nodes) > 1:
            wm = wm + "\n" + self.research_tree.get_tree_summary()
        if self.hypothesis_gen.history:
            chain_lines = ["\n## Research Hypothesis Chain"]
            for rec in self.hypothesis_gen.history:
                icon = {"confirmed": "✓", "refuted": "✗",
                        "inconclusive": "~", "untested": "?"}.get(rec.outcome, "?")
                chain_lines.append(f"- [{icon}] {rec.hypothesis.claim}")
                if rec.evidence:
                    chain_lines.append(f"  Evidence: {rec.evidence[:150]}")
            wm = wm + "\n".join(chain_lines)

        return self.reporter.generate(
            goal=self.goal,
            completed_tasks=self.completed_tasks,
            pending_tasks=self.task_queue,
            knowledge_stats=self.knowledge.stats(),
            errors=self.errors,
            working_memory=wm,
        )

    # ── Auto-scoring ────────────────────────────────────────────────

    def _is_research_complete(self) -> bool:
        """R20.3: Check if research is statistically complete (good enough to stop).

        Returns True if any JSON file has p_value with multiple result files.
        This prevents wasting cycles on follow-up hypotheses when the core question is answered.
        """
        import glob as _glob
        ws = self.mission_ctx.workspace_dir

        # Must have result files with per-seed data
        result_files = _glob.glob(os.path.join(ws, "results*.json"))
        if len(result_files) < 2:
            return False

        # Check any JSON file for statistical content
        all_json = _glob.glob(os.path.join(ws, "*.json"))
        has_stats = False
        for jf in all_json:
            if "execution_log" in jf:
                continue
            try:
                with open(jf) as f:
                    data = json.load(f)
                flat_str = json.dumps(data).lower()
                has_pvalue = "p_value" in flat_str or "p-value" in flat_str
                has_effect = "cohen" in flat_str or "effect_size" in flat_str
                has_ttest = "t_statistic" in flat_str or "t_stat" in flat_str
                if has_pvalue and (has_effect or has_ttest):
                    has_stats = True
                    break
            except (json.JSONDecodeError, IOError):
                continue

        if not has_stats:
            return False

        # Check reviewer has run at least once
        reviewer_done = any(
            t.get("worker") == "reviewer" and t.get("success")
            for t in self.completed_tasks
        )
        return reviewer_done

        return False

    def _auto_score(self):
        """Run MissionScorer at mission end. Uses LLM Judge when available."""
        if not self.mission_ctx:
            return
        try:
            from core.mission_scorer import MissionScorer
            # Pass LLM judge for semantic scoring in llm_full/exec_first/hybrid modes
            judge = self.llm_judge if self.validation_mode in ("llm_full", "exec_first", "hybrid") else None
            scorer = MissionScorer(llm_judge=judge)
            mission_dir = os.path.dirname(self.mission_ctx.workspace_dir)
            score = scorer.score_mission(mission_dir)
            print(f"  [Scorer] Mission grade: {score.grade} ({score.overall:.1f}/10)")
            for d in score.dimensions:
                print(f"    {d.name}: {d.score:.1f}/10 (weight {d.weight})")
        except Exception as e:
            print(f"  [Scorer] Auto-scoring failed: {e}")

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
