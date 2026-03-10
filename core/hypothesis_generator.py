"""
Hypothesis Generator — From Results to Research Questions
=========================================================
After a mission produces results, generate structured hypotheses
that drive follow-up experiments. This is what separates a code
executor from a researcher: the ability to look at results and ask
"why?" and "what if?"

Uses a single LLM call with structured prompting — no keyword
matching. The LLM reasons about results in context of the goal,
literature, and methodology to produce testable hypotheses.
"""

import json
from dataclasses import dataclass, field


@dataclass
class Hypothesis:
    """A testable research hypothesis with experimental design."""
    claim: str              # The hypothesis statement
    reasoning: str          # Why this hypothesis is worth testing
    experiment: str         # Concrete experiment to test it
    expected_outcome: str   # What would confirm/refute it
    priority: int = 0       # 0=highest priority
    testable: bool = True   # Can we actually test this with our setup?

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "reasoning": self.reasoning,
            "experiment": self.experiment,
            "expected_outcome": self.expected_outcome,
            "priority": self.priority,
            "testable": self.testable,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hypothesis":
        return cls(
            claim=d.get("claim", ""),
            reasoning=d.get("reasoning", ""),
            experiment=d.get("experiment", ""),
            expected_outcome=d.get("expected_outcome", ""),
            priority=d.get("priority", 0),
            testable=d.get("testable", True),
        )


@dataclass
class HypothesisResult:
    """Output of hypothesis generation."""
    hypotheses: list[Hypothesis] = field(default_factory=list)
    validity_concerns: list[str] = field(default_factory=list)
    expected_vs_actual: str = ""
    recommended_next: str = ""  # Task description for the best follow-up

    def to_dict(self) -> dict:
        return {
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "validity_concerns": self.validity_concerns,
            "expected_vs_actual": self.expected_vs_actual,
            "recommended_next": self.recommended_next,
        }


@dataclass
class HypothesisRecord:
    """A hypothesis with its experimental outcome — for chaining."""
    hypothesis: Hypothesis
    outcome: str = ""        # "confirmed", "refuted", "inconclusive", "untested"
    evidence: str = ""       # Key result that confirmed/refuted it
    cycle: int = 0

    def to_dict(self) -> dict:
        return {
            "claim": self.hypothesis.claim,
            "outcome": self.outcome,
            "evidence": self.evidence,
            "cycle": self.cycle,
        }


class HypothesisGenerator:
    """Generate research hypotheses from experimental results."""

    def __init__(self, llm):
        self.llm = llm
        self.history: list[HypothesisRecord] = []  # Hypothesis chain

    def record_outcome(self, claim: str, outcome: str, evidence: str, cycle: int = 0):
        """Record the outcome of a tested hypothesis."""
        # Find matching hypothesis in history
        for rec in self.history:
            if rec.hypothesis.claim == claim and rec.outcome == "untested":
                rec.outcome = outcome
                rec.evidence = evidence
                rec.cycle = cycle
                return
        # If not found, create a new record
        self.history.append(HypothesisRecord(
            hypothesis=Hypothesis(claim=claim, reasoning="", experiment="", expected_outcome=""),
            outcome=outcome, evidence=evidence, cycle=cycle,
        ))

    def _format_history(self) -> str:
        """Format hypothesis chain for prompt injection."""
        if not self.history:
            return ""
        lines = ["## Hypothesis Chain (previous iterations)"]
        for i, rec in enumerate(self.history[-5:], 1):
            icon = {"confirmed": "✓", "refuted": "✗", "inconclusive": "~"}.get(rec.outcome, "?")
            lines.append(f"{i}. [{icon} {rec.outcome}] {rec.hypothesis.claim}")
            if rec.evidence:
                lines.append(f"   Evidence: {rec.evidence[:150]}")
        lines.append("\nBuild on these findings. Do NOT re-test confirmed/refuted hypotheses.")
        lines.append("Each new hypothesis must REFERENCE a previous finding as motivation.")
        return "\n".join(lines)

    def generate(self, goal: str, results_summary: str,
                 literature_context: str = "",
                 methodology_notes: str = "",
                 working_memory: str = "") -> HypothesisResult:
        """Generate hypotheses from completed experiment results.

        Args:
            goal: The research goal/question
            results_summary: What the experiment actually found (metrics, observations)
            literature_context: Relevant findings from literature search
            methodology_notes: How the experiment was conducted
            working_memory: Current research state (distilled insights)

        Returns:
            HypothesisResult with ranked hypotheses and follow-up experiments
        """
        prompt = self._build_prompt(
            goal, results_summary, literature_context,
            methodology_notes, working_memory
        )

        messages = [
            {"role": "system", "content": (
                "You are a research scientist analyzing experimental results. "
                "Think critically about what the results mean, what's surprising, "
                "and what follow-up experiments would advance understanding. "
                "Be specific and falsifiable — vague hypotheses are worthless."
            )},
            {"role": "user", "content": prompt},
        ]

        try:
            resp = self.llm.chat(messages)
            text = resp["choices"][0]["message"]["content"]
            from core.llm import strip_think
            text = strip_think(text)
            return self._parse_response(text)
        except Exception as e:
            print(f"  [HypothesisGen] LLM call failed: {e}")
            return HypothesisResult()

    def _build_prompt(self, goal: str, results_summary: str,
                      literature_context: str, methodology_notes: str,
                      working_memory: str) -> str:
        parts = [f"## Research Goal\n{goal}"]

        # Inject hypothesis chain history (for chaining)
        chain_text = self._format_history()
        if chain_text:
            parts.append(chain_text)

        if working_memory:
            parts.append(f"## Current Research State\n{working_memory[:2000]}")

        if literature_context:
            parts.append(f"## Literature Context\n{literature_context[:1500]}")

        if methodology_notes:
            parts.append(f"## Methodology\n{methodology_notes[:1000]}")

        parts.append(f"## Experimental Results\n{results_summary}")

        parts.append("""## Your Task

Analyze these results as a researcher would. Respond in this exact JSON format:

```json
{
  "expected_vs_actual": "Brief analysis: what did you expect vs what happened? What's surprising?",
  "hypotheses": [
    {
      "claim": "Specific, falsifiable hypothesis statement",
      "reasoning": "Why this hypothesis explains the observed results — cite which mechanism or prior work motivates it",
      "experiment": "Exact experiment: dataset (name, N samples), model config, seeds (42,123,456,789,1024), epochs, what to measure",
      "expected_outcome": "What result would confirm (e.g. accuracy > 85%, p < 0.05, Cohen's d > 0.5) vs refute (accuracy < baseline)",
      "priority": 0,
      "testable": true
    }
  ],
  "validity_concerns": [
    "Specific methodological issue (e.g. 'possible data leakage between train/test', 'confounding: learning rate differs between conditions')"
  ],
  "recommended_next": "One-sentence task description for the highest-priority follow-up experiment"
}
```

Requirements:
- Generate 2-4 hypotheses, ranked by priority (0 = most important)
- Each hypothesis must be FALSIFIABLE — state the specific threshold that would disprove it (e.g. "p > 0.05", "accuracy < 80%")
- **Theory-grounded**: Each reasoning MUST explain WHICH mechanism or prior finding motivates the hypothesis (not just "it might work better")
- Experiments must be OVER-SPECIFIED — exact dataset size, exact seeds, exact metric + statistical test (paired t-test, Wilcoxon, bootstrap CI)
- Flag hypotheses as testable=false if they require resources we don't have
- Validity concerns must be TESTABLE themselves (e.g. "if data leakage, cross-validate instead of train/test split")
- Do NOT repeat experiments already done — check the results summary for what was already tested
- recommended_next should be a specific task, not a vague direction""")

        return "\n\n".join(parts)

    def _parse_response(self, text: str) -> HypothesisResult:
        """Parse LLM response into structured HypothesisResult."""
        # Extract JSON from response (may be wrapped in ```json blocks)
        json_str = text
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            json_str = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            json_str = text[start:end].strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to find any JSON object in the text
            import re
            match = re.search(r'\{[\s\S]*\}', text)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    print(f"  [HypothesisGen] Failed to parse JSON response")
                    return HypothesisResult()
            else:
                return HypothesisResult()

        result = HypothesisResult()
        result.expected_vs_actual = data.get("expected_vs_actual", "")
        result.validity_concerns = data.get("validity_concerns", [])
        result.recommended_next = data.get("recommended_next", "")

        for h_data in data.get("hypotheses", []):
            result.hypotheses.append(Hypothesis.from_dict(h_data))

        # Sort by priority
        result.hypotheses.sort(key=lambda h: h.priority)

        return result

    def format_for_supervisor(self, result: HypothesisResult) -> str:
        """Format hypothesis result for injection into supervisor decision prompt."""
        if not result.hypotheses:
            return ""

        lines = ["## Research Hypotheses (generated from results)"]
        lines.append(f"Analysis: {result.expected_vs_actual[:200]}")
        lines.append("")

        for i, h in enumerate(result.hypotheses[:3]):
            testable_tag = "" if h.testable else " [NOT TESTABLE WITH CURRENT SETUP]"
            lines.append(f"**H{i+1}**: {h.claim}{testable_tag}")
            lines.append(f"  - Test: {h.experiment[:150]}")
            lines.append(f"  - Expected: {h.expected_outcome[:100]}")

        if result.validity_concerns:
            lines.append(f"\nMethodology concerns: {'; '.join(result.validity_concerns[:2])}")

        if result.recommended_next:
            lines.append(f"\n**Recommended next experiment**: {result.recommended_next}")

        return "\n".join(lines)
