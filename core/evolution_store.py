"""
Evolution Store — Cross-Mission Learning
==========================================
Persistent structured learning store across missions.
Stored at {missions_dir}/_evolution/learnings.json.

Learning types:
- strategy_success: What worked well
- strategy_failure: What failed and why
- tool_preference: Which tools/APIs work best for what
- parameter_guidance: Hyperparameters, dataset sizes, etc.
- pitfall: Common mistakes to avoid
- research_finding: Scientific results and conclusions from experiments
"""

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict


@dataclass
class Learning:
    """A single learning extracted from a mission."""
    id: str
    type: str  # strategy_success, strategy_failure, tool_preference, parameter_guidance, pitfall
    category: str  # e.g., "training", "search", "evaluation", "data"
    pattern: str  # The core learning
    context: str  # When this applies
    confidence: float = 0.5  # 0-1, increases with reinforcement
    mission_ids: list = field(default_factory=list)
    times_applied: int = 0
    times_helpful: int = 0
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Learning":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


VALID_TYPES = {"strategy_success", "strategy_failure", "tool_preference",
               "parameter_guidance", "pitfall", "research_finding"}


class EvolutionStore:
    """Persistent cross-mission learning store."""

    def __init__(self, missions_dir: str):
        self.store_dir = os.path.join(missions_dir, "_evolution")
        self.store_path = os.path.join(self.store_dir, "learnings.json")
        self.learnings: list[Learning] = []
        self._last_injected_ids: list[str] = []  # Track which learnings were injected
        os.makedirs(self.store_dir, exist_ok=True)
        self._load()

    def _load(self):
        """Load learnings from disk."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path) as f:
                    data = json.load(f)
                self.learnings = [Learning.from_dict(d) for d in data.get("learnings", [])]
            except (json.JSONDecodeError, KeyError):
                self.learnings = []

    def _save(self):
        """Save learnings to disk."""
        data = {"learnings": [l.to_dict() for l in self.learnings],
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(self.store_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_learning(self, type: str, category: str, pattern: str,
                     context: str, mission_id: str = "",
                     confidence: float = 0.5) -> str:
        """Add a learning, auto-deduplicating by word overlap.

        If >60% word overlap with existing learning, reinforces it instead.

        Returns:
            Learning ID (new or existing)
        """
        if type not in VALID_TYPES:
            type = "pitfall"

        # Check for duplicates by word overlap
        pattern_words = set(pattern.lower().split())
        for existing in self.learnings:
            existing_words = set(existing.pattern.lower().split())
            if not pattern_words or not existing_words:
                continue
            overlap = len(pattern_words & existing_words) / max(len(pattern_words | existing_words), 1)
            if overlap > 0.6:
                # Reinforce existing learning
                existing.confidence = min(1.0, existing.confidence + 0.15)
                if mission_id and mission_id not in existing.mission_ids:
                    existing.mission_ids.append(mission_id)
                self._save()
                return existing.id

        # New learning
        learning = Learning(
            id=f"L{uuid.uuid4().hex[:8]}",
            type=type,
            category=category,
            pattern=pattern,
            context=context,
            confidence=confidence,
            mission_ids=[mission_id] if mission_id else [],
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self.learnings.append(learning)

        # Auto-prune: keep max 30 learnings, drop lowest confidence
        MAX_LEARNINGS = 30
        if len(self.learnings) > MAX_LEARNINGS:
            # Sort by confidence (descending), keep top MAX_LEARNINGS
            self.learnings.sort(key=lambda l: l.confidence, reverse=True)
            self.learnings = self.learnings[:MAX_LEARNINGS]

        self._save()
        return learning.id

    def get_relevant_learnings(self, goal: str, limit: int = 10) -> list[Learning]:
        """Get learnings most relevant to a goal, sorted by relevance score.

        Score = goal-word overlap + confidence + times_helpful bonus.
        """
        goal_words = set(goal.lower().split())
        scored = []
        for l in self.learnings:
            # Word overlap with goal
            pattern_words = set(l.pattern.lower().split())
            context_words = set(l.context.lower().split())
            all_words = pattern_words | context_words
            overlap = len(goal_words & all_words) / max(len(goal_words | all_words), 1) if all_words else 0

            # Boost by confidence and helpfulness
            score = overlap * 0.5 + l.confidence * 0.3
            if l.times_applied > 0:
                helpfulness = l.times_helpful / l.times_applied
                score += helpfulness * 0.2

            scored.append((score, l))

        scored.sort(key=lambda x: -x[0])
        return [l for _, l in scored[:limit] if scored]

    def get_planner_guidance(self, goal: str) -> str:
        """Format relevant learnings for planner prompt injection.

        Returns formatted text or empty string if no relevant learnings.
        Also tracks which learnings were injected for later feedback.
        """
        relevant = self.get_relevant_learnings(goal, limit=8)
        if not relevant:
            self._last_injected_ids = []
            return ""

        # Track injected IDs for feedback loop
        self._last_injected_ids = [l.id for l in relevant]

        parts = ["## Learnings from Previous Missions"]
        for l in relevant:
            icon = {"strategy_success": "✅", "strategy_failure": "❌",
                    "tool_preference": "🔧", "parameter_guidance": "📊",
                    "pitfall": "⚠️"}.get(l.type, "•")
            conf = f"(confidence: {l.confidence:.0%})" if l.confidence > 0.5 else ""
            parts.append(f"{icon} [{l.category}] {l.pattern} {conf}")
            if l.context:
                parts.append(f"   Context: {l.context}")

        parts.append("\nApply these learnings to avoid known pitfalls and use proven strategies.")
        return "\n".join(parts)

    def record_applied_learnings(self, was_helpful: bool):
        """Record that the last-injected learnings were applied.
        Call after mission tasks complete to close the feedback loop."""
        for lid in self._last_injected_ids:
            self.record_application(lid, was_helpful)
        if self._last_injected_ids:
            n = len(self._last_injected_ids)
            status = "helpful" if was_helpful else "not helpful"
            print(f"  [Evolution] Recorded {n} learnings as {status}")
        self._last_injected_ids = []

    def reflect_on_mission(self, mission_id: str, goal: str, tasks: list[dict],
                           dag=None, llm=None):
        """Post-mission reflection: extract learnings from completed mission.

        Uses LLM if available, falls back to mechanical extraction.
        """
        if llm:
            try:
                self._reflect_with_llm(mission_id, goal, tasks, dag, llm)
                return
            except Exception as e:
                print(f"  [Evolution] LLM reflection failed ({e}), using fallback")

        self._reflect_mechanical(mission_id, goal, tasks)

    def _reflect_with_llm(self, mission_id, goal, tasks, dag, llm):
        """Use LLM to extract learnings from mission."""
        from core.llm import strip_think

        task_summary = []
        for t in tasks:
            status = "OK" if t.get("success") else "FAILED"
            worker = t.get("worker", "?")
            desc = t.get("task", "")[:100]
            output_snippet = (t.get("output") or "")[:200]
            task_summary.append(f"- [{status}] {worker}: {desc}\n  Output: {output_snippet}")

        prompt = f"""A research mission just completed. Extract 2-5 key learnings for future missions.

Goal: {goal}
Tasks:
{chr(10).join(task_summary[:10])}

Extract learnings as JSON array:
[
  {{
    "type": "strategy_success|strategy_failure|tool_preference|parameter_guidance|pitfall",
    "category": "training|search|evaluation|data|architecture|tools",
    "pattern": "Concise learning (1-2 sentences)",
    "context": "When this applies"
  }}
]

Focus on:
- What worked well vs what failed
- Dataset sizes that were adequate vs too small
- Tools/APIs that worked vs broke
- Hyperparameter choices that mattered
- Common errors and how to avoid them"""

        response = llm.chat([
            {"role": "system", "content": "Extract research learnings. Respond with JSON only."},
            {"role": "user", "content": prompt},
        ])
        raw = strip_think(response["choices"][0]["message"]["content"])
        json_match = re.search(r'\[[\s\S]*\]', raw)
        if not json_match:
            raise ValueError("No JSON in response")

        items = json.loads(json_match.group())
        for item in items:
            self.add_learning(
                type=item.get("type", "pitfall"),
                category=item.get("category", "general"),
                pattern=item.get("pattern", ""),
                context=item.get("context", ""),
                mission_id=mission_id,
            )

    def _reflect_mechanical(self, mission_id, goal, tasks):
        """Mechanical fallback: extract basic learnings from task results."""
        successes = [t for t in tasks if t.get("success")]
        failures = [t for t in tasks if not t.get("success")]

        # Learning from failures
        for t in failures[:3]:
            error = t.get("error", "unknown error")
            worker = t.get("worker", "")
            self.add_learning(
                type="strategy_failure",
                category="tools" if "timeout" in error.lower() else "general",
                pattern=f"{worker} failed: {error[:100]}",
                context=f"During mission: {goal[:80]}",
                mission_id=mission_id,
                confidence=0.4,
            )

        # Learning from successes
        if successes:
            workers_used = set(t.get("worker", "") for t in successes)
            self.add_learning(
                type="strategy_success",
                category="general",
                pattern=f"Successful workflow used workers: {', '.join(workers_used)}",
                context=f"For goal type: {goal[:80]}",
                mission_id=mission_id,
                confidence=0.4,
            )

    def extract_research_findings(self, mission_id: str, goal: str,
                                   workspace_dir: str, llm=None):
        """Extract research findings (metrics, conclusions) from mission workspace.

        Reads analysis_summary.json and result JSONs to store reproducible findings.
        These persist as semantic memory — future missions can reference them.
        """
        import glob

        findings = []

        # 1. Read analysis_summary.json (primary source of truth)
        summary_path = os.path.join(workspace_dir, "analysis_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path) as f:
                    summary = json.load(f)

                # Extract methods + metrics
                methods = summary.get("methods", {})
                stats = summary.get("statistics", {})
                conclusion = summary.get("conclusion", "")
                best_method = summary.get("best_method", "")

                if methods:
                    method_strs = []
                    for name, data in methods.items():
                        mean = data.get("mean", data.get("accuracy", ""))
                        std = data.get("std", "")
                        if mean:
                            method_strs.append(f"{name}: {mean}" + (f"±{std}" if std else ""))
                    if method_strs:
                        finding = f"Results: {'; '.join(method_strs)}"
                        if best_method:
                            finding += f". Best: {best_method}"
                        findings.append({
                            "pattern": finding,
                            "context": goal[:120],
                        })

                # Statistical significance
                if stats:
                    p_val = stats.get("paired_t_test", {}).get("p_value",
                            stats.get("p_value"))
                    cohen_d = stats.get("cohens_d")
                    if p_val is not None:
                        sig = "significant" if p_val < 0.05 else "not significant"
                        stat_str = f"p={p_val:.4f} ({sig})"
                        if cohen_d:
                            stat_str += f", Cohen's d={cohen_d:.2f}"
                        findings.append({
                            "pattern": f"Statistical test: {stat_str}",
                            "context": f"{goal[:80]} — {best_method or 'comparison'}",
                        })

                if conclusion and isinstance(conclusion, str):
                    findings.append({
                        "pattern": conclusion[:200],
                        "context": goal[:120],
                    })

            except (json.JSONDecodeError, Exception) as e:
                pass

        # 2. Scan result JSONs for key metrics
        result_files = glob.glob(os.path.join(workspace_dir, "results_*.json"))
        result_files += glob.glob(os.path.join(workspace_dir, "*_results.json"))
        for rpath in result_files[:3]:
            try:
                with open(rpath) as f:
                    rdata = json.load(f)
                # Extract if it has accuracy/loss/f1
                for key in ("accuracy", "eval_accuracy", "test_accuracy", "f1", "eval_loss"):
                    if key in rdata:
                        method = rdata.get("method", rdata.get("model", os.path.basename(rpath)))
                        findings.append({
                            "pattern": f"{method}: {key}={rdata[key]}",
                            "context": goal[:120],
                        })
                        break
            except Exception:
                pass

        # 3. Store findings
        for f in findings[:5]:  # Max 5 findings per mission
            self.add_learning(
                type="research_finding",
                category="results",
                pattern=f["pattern"],
                context=f["context"],
                mission_id=mission_id,
                confidence=0.7,  # Findings start higher (they're from data)
            )

        return len(findings)

    def extract_hypothesis_chain(self, mission_id: str, goal: str,
                                  hypothesis_chain: list[dict]):
        """Extract confirmed/refuted hypotheses as cross-mission learnings.

        Hypothesis chains are the most valuable scientific knowledge —
        they encode what was TESTED and what the outcome was.
        """
        for rec in hypothesis_chain:
            outcome = rec.get("outcome", "untested")
            if outcome == "untested":
                continue  # Only store evaluated hypotheses

            claim = rec.get("claim", "")
            evidence = rec.get("evidence", "")
            if not claim:
                continue

            icon = {"confirmed": "CONFIRMED", "refuted": "REFUTED",
                    "inconclusive": "INCONCLUSIVE"}.get(outcome, outcome)
            pattern = f"[{icon}] {claim}"
            if evidence:
                pattern += f" — {evidence[:100]}"

            self.add_learning(
                type="research_finding",
                category="hypothesis",
                pattern=pattern,
                context=goal[:120],
                mission_id=mission_id,
                confidence=0.8 if outcome in ("confirmed", "refuted") else 0.5,
            )

    def get_research_context(self, goal: str) -> str:
        """Get relevant past research findings for a new mission.

        Returns formatted text with prior results that might inform this research.
        """
        findings = [l for l in self.learnings if l.type == "research_finding"]
        if not findings:
            return ""

        goal_words = set(goal.lower().split())
        scored = []
        for f in findings:
            words = set(f.pattern.lower().split()) | set(f.context.lower().split())
            overlap = len(goal_words & words) / max(len(goal_words | words), 1)
            if overlap > 0.05:  # Very low threshold — any relevance
                scored.append((overlap + f.confidence * 0.3, f))

        scored.sort(key=lambda x: -x[0])
        relevant = scored[:5]

        if not relevant:
            return ""

        parts = ["## Prior Research Findings (from previous missions)"]
        for _, f in relevant:
            missions = ", ".join(f.mission_ids[:2]) if f.mission_ids else "?"
            parts.append(f"- {f.pattern}")
            parts.append(f"  (Source: {f.context[:60]}, mission: {missions})")
        parts.append("\nBuild on these findings — don't repeat experiments that are already done.")
        return "\n".join(parts)

    def record_application(self, learning_id: str, was_helpful: bool):
        """Record whether applying a learning was helpful.

        Adjusts confidence: +0.1 if helpful, -0.05 if not.
        """
        for l in self.learnings:
            if l.id == learning_id:
                l.times_applied += 1
                if was_helpful:
                    l.times_helpful += 1
                    l.confidence = min(1.0, l.confidence + 0.1)
                else:
                    l.confidence = max(0.1, l.confidence - 0.05)
                self._save()
                return
