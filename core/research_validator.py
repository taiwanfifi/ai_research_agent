"""
Research Design Validator — Pre-Planning Sanity Check
======================================================
Before the planner decomposes a goal into tasks, this component
evaluates whether the experiment design is fundamentally sound.

Catches:
- Architectural impossibilities (BatchNorm on Transformer)
- Dataset too simple for the comparison (MNIST for optimizer study)
- Trivial/well-known comparisons that lack a hypothesis
- Missing baselines or controls

One LLM call, structured output. A gate, not an agent.
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from core.llm import MiniMaxClient, strip_think


PITFALLS_PATH = Path(__file__).parent / "design_pitfalls.json"


@dataclass
class DesignIssue:
    category: str       # architecture, dataset, comparison, methodology, scope
    severity: str       # fatal, major, minor
    description: str
    fix: str


@dataclass
class ValidationVerdict:
    viable: bool
    confidence: float
    issues: list[DesignIssue]
    modified_goal: str | None
    modifications_summary: str
    reject_reason: str

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class ResearchValidator:
    """Validates experiment design before planning."""

    def __init__(self, llm: MiniMaxClient):
        self.llm = llm
        self._pitfalls = self._load_pitfalls()

    @staticmethod
    def _load_pitfalls() -> list[dict]:
        if not PITFALLS_PATH.exists():
            return []
        try:
            with open(PITFALLS_PATH) as f:
                return json.load(f)
        except Exception:
            return []

    def _get_relevant_pitfalls(self, goal: str) -> list[dict]:
        """Select pitfalls whose trigger keywords appear in the goal."""
        goal_lower = goal.lower()
        relevant = []
        for p in self._pitfalls:
            triggers = p.get("triggers", [])
            if any(t.lower() in goal_lower for t in triggers):
                relevant.append(p)
        return relevant

    def validate(self, goal: str, evolution_store=None) -> ValidationVerdict:
        """Evaluate a research goal for design flaws.

        Args:
            goal: The research goal string
            evolution_store: Optional EvolutionStore for past learnings

        Returns:
            ValidationVerdict with issues and possible modified goal
        """
        # Get relevant pitfalls
        pitfalls = self._get_relevant_pitfalls(goal)
        pitfall_text = ""
        if pitfalls:
            lines = ["## Known Pitfalls (relevant to this goal)"]
            for p in pitfalls:
                lines.append(f"- [{p['severity']}] {p['pitfall']}")
            pitfall_text = "\n".join(lines)

        # Get evolution context
        evolution_text = ""
        if evolution_store:
            try:
                learnings = evolution_store.get_relevant_learnings(goal)
                if learnings:
                    evolution_text = f"\n## Lessons from Past Missions\n{learnings}"
            except Exception:
                pass

        prompt = f"""Evaluate this ML/NLP experiment design for fundamental flaws.

Research goal: {goal}

{pitfall_text}
{evolution_text}

Check for:
1. **Architecture**: Are the methods compatible with the model/task? Any structural impossibilities?
2. **Dataset**: Is the dataset appropriate for showing meaningful differences? Too simple? Too complex?
3. **Comparison**: Is this trivial or well-known? Does it need a specific hypothesis?
4. **Methodology**: Are baselines and controls adequate? Parameter budget fair?
5. **Scope**: Is the goal specific enough to execute?

Respond with ONLY a JSON object:
{{
  "viable": true,
  "confidence": 0.8,
  "issues": [
    {{"category": "dataset", "severity": "major", "description": "...", "fix": "..."}}
  ],
  "modified_goal": "rewritten goal with fixes, or null if original is fine",
  "modifications_summary": "what was changed and why, or empty string",
  "reject_reason": ""
}}

Rules:
- viable=false ONLY if there's a truly impossible flaw (e.g., BatchNorm on Transformer)
- If issues exist but are fixable, set viable=true AND provide modified_goal
- If the goal is fine, set modified_goal=null and issues=[]
- Be conservative: most goals should pass. Only flag clear, structural problems.
- A null result ("no significant difference found") is VALID research — do NOT reject experiments that might show small effects
- Keep modified_goal close to the original intent — don't redesign the study
- IMPORTANT: If you set viable=false, you MUST also provide a modified_goal that fixes the fatal issue
- DO NOT change numerical parameters (epochs, sample count, batch size, learning rate) in modified_goal — these are the user's design choices. Only flag them as minor issues with suggested values. The planner will adjust if needed.
- modified_goal should ONLY change structural issues (wrong model class, incompatible architecture, impossible comparison)"""

        try:
            response = self.llm.chat([
                {"role": "system",
                 "content": "You are a research design reviewer. Evaluate experiment proposals for fundamental flaws. JSON only."},
                {"role": "user", "content": prompt},
            ])
            raw = response["choices"][0]["message"]["content"]
            raw = strip_think(raw)

            # Parse JSON
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if not json_match:
                return self._pass_verdict()

            data = json.loads(json_match.group())

            issues = []
            for iss in data.get("issues", []):
                issues.append(DesignIssue(
                    category=iss.get("category", "methodology"),
                    severity=iss.get("severity", "minor"),
                    description=iss.get("description", ""),
                    fix=iss.get("fix", ""),
                ))

            viable = data.get("viable", True)
            modified_goal = data.get("modified_goal")

            # Safety: if viable=False but no modified_goal, override to viable=True
            # (a rejection without alternative is worse than running a flawed experiment)
            if not viable and not modified_goal:
                print(f"  [Validator] Overriding: viable=False with no alternative → viable=True with warnings")
                viable = True

            return ValidationVerdict(
                viable=viable,
                confidence=data.get("confidence", 0.5),
                issues=issues,
                modified_goal=modified_goal,
                modifications_summary=data.get("modifications_summary", ""),
                reject_reason=data.get("reject_reason", ""),
            )

        except Exception as e:
            print(f"  [Validator] Failed ({e}), passing through")
            return self._pass_verdict()

    @staticmethod
    def _pass_verdict() -> ValidationVerdict:
        """Default: let everything through."""
        return ValidationVerdict(
            viable=True, confidence=0.5, issues=[],
            modified_goal=None, modifications_summary="",
            reject_reason="",
        )
