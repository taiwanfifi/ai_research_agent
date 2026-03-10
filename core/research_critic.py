"""
Research Critic — Adversarial Debate Before Execution
======================================================
Unlike ResearchValidator (rule-based gate), the Critic engages in
genuine adversarial reasoning about research decisions.

Two personas debate:
- Advocate: argues FOR the current plan/decision
- Critic: argues AGAINST, probing for weaknesses

The debate produces:
- Scientific value assessment (is this worth doing?)
- Experimental design challenges (will this actually answer the question?)
- Knowledge gap identification (what do we need to know first?)
- Suggested improvements (how to make this meaningful)

Inspired by Constitutional AI debate and Kael's discrepancy monitor.
"""

import json
import re
from dataclasses import dataclass, asdict
from core.llm import MiniMaxClient, strip_think


@dataclass
class CritiqueResult:
    """Result of adversarial debate about a research decision."""
    worth_doing: bool           # Critic's verdict: is this scientifically valuable?
    confidence: float           # 0-1, how confident the critic is
    scientific_value: str       # Why this matters (or doesn't)
    design_flaws: list[str]     # Specific issues with experimental design
    missing_knowledge: list[str]  # What we'd need to know first
    improvements: list[str]     # How to make this better
    debate_summary: str         # Key points from advocate vs critic
    revised_goal: str | None    # Improved goal if critic has suggestions


def critique_research_goal(llm: MiniMaxClient, goal: str,
                           domain_context: str = "",
                           past_findings: str = "") -> CritiqueResult:
    """Run adversarial debate on a research goal before planning.

    This is NOT validation (pass/fail). It's a genuine intellectual
    challenge that may improve the research design.

    Args:
        llm: LLM client
        goal: The proposed research goal
        domain_context: Knowledge from domain brain (principles, past experiments)
        past_findings: Relevant findings from previous missions
    """
    prompt = f"""You are two researchers debating whether this experiment is worth doing.

## Proposed Research
{goal}

{f"## Domain Knowledge (from previous research){chr(10)}{domain_context}" if domain_context else ""}
{f"## Past Findings{chr(10)}{past_findings}" if past_findings else ""}

## Debate Format

**ADVOCATE** argues FOR doing this experiment. Why is it valuable? What could we learn?

**CRITIC** challenges the experiment. Key questions:
1. **Scientific value**: Is the answer already known? Is this just a replication?
2. **Experimental design**: Will this setup actually reveal meaningful differences?
   - Are the conditions different enough to produce detectable effects?
   - Is the dataset/model/scale appropriate?
   - Are there confounding variables?
3. **Knowledge prerequisites**: Do we understand the mechanisms well enough to interpret results?
4. **Opportunity cost**: Could we ask a more interesting question with the same effort?

**ADVOCATE** responds to the criticism with either:
- Counterarguments (why the criticism is wrong)
- Improvements (how to fix the issues raised)
- Concessions (the critic is right, here's a better experiment)

## Output
After the debate, provide a verdict as JSON:
{{
  "worth_doing": true/false,
  "confidence": 0.0-1.0,
  "scientific_value": "1-2 sentence assessment",
  "design_flaws": ["specific flaw 1", "specific flaw 2"],
  "missing_knowledge": ["what we need to know first"],
  "improvements": ["concrete improvement 1", "concrete improvement 2"],
  "debate_summary": "2-3 sentence summary of key debate points",
  "revised_goal": "improved goal incorporating critic's suggestions, or null if original is fine"
}}

Rules:
- Be genuinely critical, not rubber-stamp approval
- A null result IS valuable if the experiment is well-designed
- "Already well-known" is a valid reason to say not worth doing
- If the experiment CAN be improved, always provide revised_goal
- Focus on SCIENTIFIC merit, not engineering feasibility"""

    try:
        response = llm.chat([
            {"role": "system", "content": (
                "You are a research debate moderator. Present both sides "
                "of the argument, then deliver a verdict. Be intellectually "
                "honest — challenge weak experiments, support strong ones."
            )},
            {"role": "user", "content": prompt},
        ])

        raw = strip_think(response["choices"][0]["message"]["content"])

        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            return _default_pass(goal)

        data = json.loads(json_match.group())

        return CritiqueResult(
            worth_doing=data.get("worth_doing", True),
            confidence=data.get("confidence", 0.5),
            scientific_value=data.get("scientific_value", ""),
            design_flaws=data.get("design_flaws", []),
            missing_knowledge=data.get("missing_knowledge", []),
            improvements=data.get("improvements", []),
            debate_summary=data.get("debate_summary", ""),
            revised_goal=data.get("revised_goal"),
        )

    except Exception as e:
        print(f"  [Critic] Debate failed ({e}), passing through")
        return _default_pass(goal)


def critique_mid_mission(llm: MiniMaxClient, goal: str,
                         completed_tasks: list[dict],
                         current_results: dict,
                         domain_context: str = "") -> CritiqueResult:
    """Mid-mission critique: are we on track? Should we pivot?

    Called after results are available but before deciding next steps.
    This is where the Critic can say "these results suggest we should
    change our approach" or "we're wasting cycles on a dead end."
    """
    task_summary = "\n".join(
        f"- [{t.get('worker', '?')}] {'✓' if t.get('success') else '✗'} {t.get('task', '')[:80]}"
        for t in completed_tasks[-5:]  # Last 5 tasks
    )

    results_str = json.dumps(current_results, indent=2, default=str)[:2000]

    prompt = f"""You are a research advisor reviewing an experiment mid-way through.

## Original Goal
{goal}

## Completed Tasks (recent)
{task_summary}

## Current Results
{results_str}

{f"## Domain Knowledge{chr(10)}{domain_context}" if domain_context else ""}

## Questions to Answer

1. **Are we learning anything?** Do the results so far tell us something meaningful,
   or are we just generating numbers?

2. **Should we pivot?** Based on what we've seen, should we:
   - Continue as planned
   - Modify the experiment (different parameters, conditions)
   - Investigate an unexpected finding
   - Stop early (we already have our answer)

3. **What's the most valuable next step?** Not "what's next in the plan" but
   "what would teach us the most right now?"

Respond with JSON:
{{
  "worth_doing": true,
  "confidence": 0.7,
  "scientific_value": "assessment of what we've learned so far",
  "design_flaws": ["issues revealed by the results"],
  "missing_knowledge": ["what the results make us want to know"],
  "improvements": ["suggested pivots or modifications"],
  "debate_summary": "should we continue, pivot, or stop?",
  "revised_goal": "modified goal if pivot recommended, null if continue"
}}"""

    try:
        response = llm.chat([
            {"role": "system", "content": (
                "You are a senior research advisor. Be direct. "
                "If the experiment is going nowhere, say so. "
                "If results suggest a more interesting question, propose it."
            )},
            {"role": "user", "content": prompt},
        ])

        raw = strip_think(response["choices"][0]["message"]["content"])
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if not json_match:
            return _default_pass(goal)

        data = json.loads(json_match.group())
        return CritiqueResult(
            worth_doing=data.get("worth_doing", True),
            confidence=data.get("confidence", 0.5),
            scientific_value=data.get("scientific_value", ""),
            design_flaws=data.get("design_flaws", []),
            missing_knowledge=data.get("missing_knowledge", []),
            improvements=data.get("improvements", []),
            debate_summary=data.get("debate_summary", ""),
            revised_goal=data.get("revised_goal"),
        )

    except Exception as e:
        print(f"  [Critic] Mid-mission critique failed ({e})")
        return _default_pass(goal)


def _default_pass(goal: str) -> CritiqueResult:
    """Default: let it through (fail-open)."""
    return CritiqueResult(
        worth_doing=True, confidence=0.3,
        scientific_value="Unable to evaluate",
        design_flaws=[], missing_knowledge=[],
        improvements=[], debate_summary="Critique unavailable",
        revised_goal=None,
    )
