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
                  cross_knowledge: list[dict] = None) -> list[dict]:
        """
        Decompose a goal into a list of worker tasks.

        Args:
            cross_knowledge: Optional list of summaries from other missions
                             [{"mission_id": str, "goal": str, "summary": dict}]

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

        context = "\n\n".join(context_parts)

        prompt = f"""Decompose this research goal into concrete tasks for specialized workers.

Goal: {goal}

{context}

Respond with ONLY a JSON array of tasks:
[
  {{"worker": "explorer|coder|reviewer", "task": "specific task description", "priority": 1, "depends_on": []}},
  ...
]

Rules:
- Use "explorer" for searching papers, datasets, repos
- Use "coder" for implementing algorithms, writing code, running experiments
- Use "reviewer" for evaluating results, running benchmarks
- Order by dependency (earlier tasks first)
- depends_on is a list of task indices (0-based) that must complete first
- Keep tasks focused and actionable (1 specific thing per task)
- Usually 3-7 tasks is appropriate
"""

        response = self.llm.chat([
            {"role": "system", "content": "You are a research project planner. Respond with JSON only."},
            {"role": "user", "content": prompt},
        ])

        content = response["choices"][0]["message"]["content"]
        json_match = re.search(r'\[[\s\S]*\]', content)
        if not json_match:
            return [{"worker": "explorer", "task": goal, "priority": 1, "depends_on": []}]

        try:
            tasks = json.loads(json_match.group())
            return tasks
        except json.JSONDecodeError:
            return [{"worker": "explorer", "task": goal, "priority": 1, "depends_on": []}]
