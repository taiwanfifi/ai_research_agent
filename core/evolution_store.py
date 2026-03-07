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
               "parameter_guidance", "pitfall"}


class EvolutionStore:
    """Persistent cross-mission learning store."""

    def __init__(self, missions_dir: str):
        self.store_dir = os.path.join(missions_dir, "_evolution")
        self.store_path = os.path.join(self.store_dir, "learnings.json")
        self.learnings: list[Learning] = []
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
        """
        relevant = self.get_relevant_learnings(goal, limit=8)
        if not relevant:
            return ""

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
