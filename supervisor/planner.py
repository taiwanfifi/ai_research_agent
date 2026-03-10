"""
Task Planner
==============
Decomposes high-level goals into concrete worker tasks using LLM.
Supports research protocols (mission-level skills) for structured experiments.
"""

import json
import re
from core.llm import MiniMaxClient


# Keywords that suggest a controlled comparison goal
_COMPARISON_KEYWORDS = [
    "compare", "comparison", "vs", "versus", "benchmark",
    "ablation", "evaluate", "which is better",
]


class TaskPlanner:
    """LLM-driven goal → task decomposition."""

    def __init__(self, llm: MiniMaxClient):
        self.llm = llm

    def _detect_protocol(self, goal: str) -> str | None:
        """Detect which research protocol matches the goal.
        Returns the protocol prompt or None."""
        goal_lower = goal.lower()

        # Check for controlled comparison pattern
        if sum(1 for kw in _COMPARISON_KEYWORDS if kw in goal_lower) >= 1:
            try:
                from skills.builtin.controlled_comparison import SKILL
                return SKILL.prompt
            except ImportError:
                pass

        return None

    def decompose(self, goal: str, knowledge_summary: dict = None,
                  completed_tasks: list = None, available_workers: list = None,
                  cross_knowledge: list[dict] = None,
                  evolution_guidance: str = "",
                  quality_rules: str = "",
                  max_cycles: int = 12) -> list[dict]:
        """
        Decompose a goal into a list of worker tasks.

        Args:
            cross_knowledge: Optional list of summaries from other missions
                             [{"mission_id": str, "goal": str, "summary": dict}]
            evolution_guidance: Learnings from previous missions (from EvolutionStore)
            quality_rules: Research quality rules to follow

        Returns:
            List of dicts with keys: worker, task, priority, depends_on
        """
        context_parts = []
        if knowledge_summary:
            context_parts.append(f"Current knowledge:\n{json.dumps(knowledge_summary, indent=2, default=str)}")
        if completed_tasks:
            context_parts.append(f"Already completed:\n" + "\n".join(f"- {t}" for t in completed_tasks))
        if available_workers:
            context_parts.append(f"Available workers: {', '.join(available_workers)}")
        else:
            context_parts.append("Available workers: explorer (paper search), coder (implementation), reviewer (benchmarks)")
        if cross_knowledge:
            cross_parts = []
            for ck in cross_knowledge:
                cross_parts.append(
                    f"- Mission: {ck.get('goal', ck['mission_id'])}\n"
                    f"  Knowledge: {json.dumps(ck.get('summary', {}), indent=2, default=str)}"
                )
            context_parts.append(
                "Knowledge from other missions (for reference only, do not duplicate work):\n"
                + "\n".join(cross_parts)
            )

        if evolution_guidance:
            context_parts.append(evolution_guidance)
        if quality_rules:
            context_parts.append(quality_rules)

        context = "\n\n".join(context_parts)

        # Check for matching research protocol
        protocol = self._detect_protocol(goal)
        protocol_section = ""
        if protocol:
            protocol_section = f"""
## Research Protocol (FOLLOW THIS STRUCTURE)
{protocol}

Generate tasks that follow this protocol's phases. Each phase maps to one or more worker tasks.
"""

        # Extract architecture from goal for spec compliance
        arch_hint = ""
        goal_lower = goal.lower()
        if "simple cnn" in goal_lower or "simple conv" in goal_lower:
            arch_hint = "\n⚠️ ARCHITECTURE: Goal specifies 'simple CNN' — use a 3-4 layer CNN (~100K params). Do NOT use ResNet, VGG, or any heavy architecture.\n"
        elif "mlp" in goal_lower:
            arch_hint = "\n⚠️ ARCHITECTURE: Goal specifies 'MLP' — use fully connected layers only.\n"

        prompt = f"""Decompose this research goal into concrete tasks for specialized workers.

Goal: {goal}
{arch_hint}

{context}
{protocol_section}

Respond with ONLY a JSON array of tasks:
[
  {{"worker": "explorer|coder|reviewer", "task": "specific task description", "priority": 1, "depends_on": []}},
  ...
]

## Worker Capabilities
- **explorer**: Searches arxiv, semantic scholar, openalex, GitHub repos. MUST output: paper titles, authors, venues, citations, key contributions, arXiv IDs, relevant GitHub repos with star counts.
- **coder**: Writes and runs Python code. Has pip_install, write_file, read_file, run_python_code. MUST save all code to workspace with write_file. MUST produce runnable .py files.
- **reviewer**: Runs benchmarks, evaluates code. Has run_python_code, write_file, read_file. MUST produce quantitative metrics (accuracy, loss, time, etc.) and save results to workspace.

## CRITICAL: Architecture Fidelity
- Use EXACTLY the model/architecture specified in the goal. "simple CNN" = 3-4 conv layers (NOT ResNet, NOT VGG). "MLP" = fully connected layers only. "ResNet-18" = ResNet-18. NEVER upgrade or substitute architectures.
- ResNet-18 has 11M params and is SLOW on CPU (~5min/seed). A simple CNN has ~100K params (~10s/seed). Choose wisely.

## Planning Rules
1. **CYCLE BUDGET: You have {max_cycles} cycles total.** Generate AT MOST {max_cycles - 1} tasks. Reserve at least 1 cycle for the reviewer.
2. **Max 1-2 explorer tasks** — one broad search, one deep dive.
3. **Coder tasks MUST be atomic** — ONE file, ONE responsibility. If budget tight (≤8), combine seed runs.
4. **Coder tasks MUST specify the output filename**: "Save as X.py" or "Edit function Y in X.py"
5. **Each coder task MUST produce output** — write_file + run_python_code in the SAME task. Print results as 'metric_name: value'.
6. **Reviewer tasks MUST specify metrics and figures**: "Load results, create comparison_chart.png (bar chart with error bars), training_loss.png (loss curves), and save analysis_summary.json". Always request at least 2 figures.
7. **Task distribution**: 1-2 explorer, N coder, 1-2 reviewer. The LAST task MUST be a reviewer task.
8. **Order by dependency** — explorer first, then coder, then reviewer. For comparison tasks: add a SECOND explorer task AFTER the coder tasks to find known baselines/benchmarks for validation (e.g. "Search for reported accuracy of [method] on [dataset] for comparison"). This validates our results against published numbers.
9. **Each task description must be self-contained** — a worker should understand what to do without seeing other tasks. **Include ALL numerical parameters AND architecture names from the goal** (epochs, samples, seeds, learning rate, batch size, model type) — the coder uses EXACTLY what's in the task description. If the goal says "simple CNN", do NOT upgrade to ResNet/VGG — use a simple 3-4 layer CNN.
10. **Coder training tasks MUST use small subsets first** — e.g. "use first 2000 samples, 1-2 epochs"
11. **Time budget**: Each code execution has a 600s timeout. Estimate per condition per seed: Simple CNN ~30s (10 epochs), ~15s (5 epochs). For ≥3 conditions: split into SEPARATE coder tasks (one per condition, each runs 5 seeds). Do NOT combine ≥3 conditions in one script — they will timeout.
12. **Separate training from evaluation**: Training tasks save metrics to JSON. Reviewer tasks load and analyze.

## Statistical Protocol (MANDATORY for reviewer tasks)
- Reviewer MUST specify: paired t-test for within-subject, independent t-test for between-subject
- Reviewer MUST compute: effect sizes (Cohen's d), confidence intervals
- Reviewer MUST save analysis_summary.json containing ALL numerical claims
- All code tasks MUST run 5 seeds (42, 123, 456, 789, 1024) and save per-seed results to JSON
"""

        response = self.llm.chat([
            {"role": "system", "content": "You are a research project planner. Respond with JSON only."},
            {"role": "user", "content": prompt},
        ])

        content = response["choices"][0]["message"]["content"]
        json_match = re.search(r'\[[\s\S]*\]', content)
        if not json_match:
            return self._default_plan(goal)

        try:
            tasks = json.loads(json_match.group())
            # Guard: if planner returns too few tasks, expand with default structure
            if len(tasks) < 3:
                return self._default_plan(goal)
            # Guard: enforce architecture compliance
            tasks = self._enforce_architecture(tasks, goal)
            # Guard: if planner returns more tasks than cycles, trim
            tasks = self._fit_to_budget(tasks, max_cycles)
            return tasks
        except json.JSONDecodeError:
            return self._default_plan(goal)

    def _enforce_architecture(self, tasks: list[dict], goal: str) -> list[dict]:
        """Deterministic post-processing: replace architecture mismatches in task descriptions.
        The MiniMax LLM has strong priors (CIFAR→ResNet) that override prompt instructions.
        This mechanically fixes the most common mismatches."""
        goal_lower = goal.lower()

        # Detect if goal specifies a simple architecture
        wants_simple = any(k in goal_lower for k in [
            "simple cnn", "simple conv", "4-layer cnn", "3-layer cnn",
            "not resnet", "not vgg",
        ])
        wants_mlp = "mlp" in goal_lower and "resnet" not in goal_lower

        if not wants_simple and not wants_mlp:
            return tasks

        replacement = "simple 4-layer CNN (Conv2d 32→64→128→256, NOT ResNet)"
        if wants_mlp:
            replacement = "MLP (fully connected layers only, NOT CNN)"

        heavy_patterns = [
            r"ResNet-?\d*", r"VGG-?\d*", r"DenseNet-?\d*",
            r"resnet_?\d*", r"vgg_?\d*", r"densenet_?\d*",
        ]

        for task in tasks:
            desc = task.get("task", "")
            for pattern in heavy_patterns:
                desc = re.sub(pattern, replacement, desc, flags=re.IGNORECASE)
            task["task"] = desc

        return tasks

    def _fit_to_budget(self, tasks: list[dict], max_cycles: int) -> list[dict]:
        """Ensure task count fits within cycle budget.
        Preserves reviewer tasks (they tend to be last and most valuable).
        Merges excess coder tasks if over budget."""
        budget = max_cycles - 1  # Reserve 1 cycle for supervisor flexibility
        if len(tasks) <= budget:
            return tasks

        # Separate by worker type
        explorers = [t for t in tasks if t.get("worker") == "explorer"]
        coders = [t for t in tasks if t.get("worker") == "coder"]
        reviewers = [t for t in tasks if t.get("worker") == "reviewer"]

        # Ensure at least 1 reviewer
        if not reviewers:
            reviewers = [{
                "worker": "reviewer",
                "task": "Load all result JSON files, compute mean ± std, paired t-test, "
                        "create comparison_chart.png and training_curves.png, "
                        "save analysis_summary.json.",
                "priority": 99, "depends_on": [],
            }]

        # Budget: 1 explorer + N coder + 1 reviewer = budget
        max_explorers = min(len(explorers), 1)
        max_reviewers = min(len(reviewers), max(1, budget - max_explorers - 1))
        max_coders = budget - max_explorers - max_reviewers

        trimmed = explorers[:max_explorers] + coders[:max_coders] + reviewers[:max_reviewers]

        if len(trimmed) < len(tasks):
            print(f"  [Planner] Trimmed {len(tasks)} → {len(trimmed)} tasks "
                  f"(budget: {max_cycles} cycles)")

        # Re-assign priorities
        for i, t in enumerate(trimmed):
            t["priority"] = i + 1

        return trimmed

    def _default_plan(self, goal: str) -> list[dict]:
        """Fallback plan when LLM planner fails or returns too few tasks."""
        return [
            {"worker": "explorer", "task": f"Search for 3+ academic papers relevant to: {goal}. "
             "Output paper titles, authors, citations, arXiv IDs, key contributions.",
             "priority": 1, "depends_on": []},
            {"worker": "coder", "task": f"Write AND run the core implementation for: {goal}. "
             "Save code as experiment.py, then EXECUTE it with run_python_code. "
             "Print results as 'metric_name: value' (e.g. accuracy: 85.3).",
             "priority": 2, "depends_on": []},
            {"worker": "coder", "task": f"Run additional experiments/configurations for: {goal}. "
             "If errors occur, fix AND re-run. Must produce real metrics.",
             "priority": 3, "depends_on": []},
            {"worker": "explorer", "task": f"Search for published baselines and benchmark results for: {goal}. "
             "Find reported accuracy/metrics on the same dataset. This validates our experimental results.",
             "priority": 4, "depends_on": []},
            {"worker": "reviewer", "task": f"Load all result JSON files, compute statistics (mean, std, t-test), "
             "create comparison_chart.png (bar chart with error bars) and training_loss.png (loss curves). "
             "Save analysis_summary.json with all computed statistics.",
             "priority": 5, "depends_on": []},
        ]
