"""
Task Planner
==============
Decomposes high-level goals into concrete worker tasks using LLM.
"""

import json
import re
from core.llm import MiniMaxClient


class TaskPlanner:
    """LLM-driven goal → task decomposition."""

    def __init__(self, llm: MiniMaxClient):
        self.llm = llm

    def decompose(self, goal: str, knowledge_summary: dict = None,
                  completed_tasks: list = None, available_workers: list = None,
                  cross_knowledge: list[dict] = None,
                  evolution_guidance: str = "",
                  quality_rules: str = "") -> list[dict]:
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

        prompt = f"""Decompose this research goal into concrete tasks for specialized workers.

Goal: {goal}

{context}

Respond with ONLY a JSON array of tasks:
[
  {{"worker": "explorer|coder|reviewer", "task": "specific task description", "priority": 1, "depends_on": []}},
  ...
]

## Worker Capabilities
- **explorer**: Searches arxiv, semantic scholar, openalex, GitHub repos. MUST output: paper titles, authors, venues, citations, key contributions, arXiv IDs, relevant GitHub repos with star counts.
- **coder**: Writes and runs Python code. Has pip_install, write_file, read_file, run_python_code. MUST save all code to workspace with write_file. MUST produce runnable .py files.
- **reviewer**: Runs benchmarks, evaluates code. Has run_python_code, write_file, read_file. MUST produce quantitative metrics (accuracy, loss, time, etc.) and save results to workspace.

## Task Type Awareness
- **Classification tasks** (sentiment, NLI, topic): Coder MUST use `AutoModelForSequenceClassification`, NOT `AutoModelForCausalLM`. Specify this in the task description!
- **Generation tasks** (summarization, translation): Coder uses `AutoModelForCausalLM`
- When writing coder tasks for classification, explicitly state: "Use AutoModelForSequenceClassification with num_labels=N"
- **ALL training/fine-tuning tasks**: Add "Force CPU (os.environ['CUDA_VISIBLE_DEVICES']=''), do NOT use MPS" to the task description

## Planning Rules
1. **Max 2 explorer tasks** — one broad search, one deep dive. Do NOT waste cycles on repeated searches.
2. **Coder tasks MUST be atomic** — ONE file, ONE responsibility per task. Examples:
   - GOOD: "Write model definition in lora_model.py with LoRA layer class"
   - GOOD: "Write training script train.py that imports lora_model and trains 3 epochs on SST-2"
   - GOOD: "Run train.py with rank=8,16,32 and save results to results.json"
   - BAD: "Implement LoRA, train it, benchmark it, and plot results" (too many things!)
3. **Coder tasks MUST specify the output filename**: "Save as X.py" or "Edit function Y in X.py"
4. **Each coder task MUST produce output** — "Write training script AND run it, print accuracy/loss results". Never create a task that only writes a file without executing it. The coder should write_file + run_python_code in the SAME task.
5. **Reviewer tasks MUST specify metrics**: "Measure accuracy, F1, inference time; save results as results.json and comparison_chart.png"
6. **6-10 tasks is ideal** — distribute as: 1-2 explorer, 3-5 coder, 1-2 reviewer
7. **Order by dependency** — explorer first, then coder (write → run → plot), then reviewer
8. **Each task description must be self-contained** — a worker should understand what to do without seeing other tasks
9. The adaptive supervisor will add more tasks as needed, so don't over-plan
10. **Coder training tasks MUST use small subsets first** — e.g. "use first 2000 samples, 1-2 epochs" to verify code works, then scale up
11. **Time budget**: Each code execution has a 600s timeout. ONE training run on 2000 samples ≈ 200-400s CPU. NEVER put multiple training runs (e.g. 3 seeds or multiple configs) in one task — split them into separate tasks.
12. **Separate training from evaluation**: Training tasks MUST run the training AND save result metrics (accuracy/loss). Evaluation tasks load saved metrics and compute statistics/plots. Each training task should end with real printed metrics.
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
            return tasks
        except json.JSONDecodeError:
            return self._default_plan(goal)

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
            {"worker": "coder", "task": "Generate comparison plots from results. "
             "Save as comparison_plot.png with English labels.",
             "priority": 4, "depends_on": []},
            {"worker": "reviewer", "task": f"Evaluate the results for: {goal}. "
             "Run the code independently, verify metrics, check reproducibility.",
             "priority": 5, "depends_on": []},
        ]
