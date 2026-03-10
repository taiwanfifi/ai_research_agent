"""
Watcher — Metacognition Engine (Opus v3)
=========================================
Separate LLM call that monitors ONLY the execution trace,
NOT code or papers. Detects when the agent is stuck,
looping, or making no progress.

Expert consensus: "Don't give it code/papers, only the execution trace."
Input: [Search → Code → Bug → Fix → Bug → Fix → Bug]
Output: {"status": "stuck", "directive": "abandon and rewrite"}

Implements:
- Frustration metric (multi-signal composite)
- Trajectory hashing (cosine similarity of recent states → loop detection)
- Structured directives with escalation levels
- Decision journal for systematic error tracking

References:
- Cox (2005) Metacognition in Computation
- Shimizu (2023) Reflexion
- Hayes-Roth (1985) Blackboard Architecture
"""

import json
import math
from dataclasses import dataclass, field, asdict


# ── Data structures ──────────────────────────────────────────────

@dataclass
class WatcherVerdict:
    """Structured output from the Watcher."""
    status: str             # "progressing" | "slowing" | "stuck" | "looping" | "thrashing"
    frustration: float      # 0.0 (calm) → 1.0 (critical)
    directive: str          # What to do: "" (continue), "pivot", "simplify", "abandon", "report_now"
    reasoning: str          # Why this verdict
    signals: dict = field(default_factory=dict)  # Individual signal values

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WatcherVerdict":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TraceEvent:
    """Minimal representation of one cycle's outcome."""
    cycle: int
    worker: str         # "explorer", "coder", "reviewer"
    action: str         # "search_more", "implement", "fix_code", etc.
    success: bool
    elapsed_s: float
    output_snippet: str  # First 200 chars of output (for trajectory hashing)
    error_snippet: str   # First 200 chars of error (if any)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TraceEvent":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Frustration signals ─────────────────────────────────────────

def _consecutive_failures(trace: list[TraceEvent]) -> float:
    """Count consecutive failures from the end. Returns 0-1 scaled."""
    count = 0
    for evt in reversed(trace):
        if not evt.success:
            count += 1
        else:
            break
    # Scale: 0 failures → 0, 3+ → 1.0
    return min(count / 3.0, 1.0)


def _same_worker_ratio(trace: list[TraceEvent], window: int = 5) -> float:
    """How much of recent work is the same worker? High = stuck on one thing."""
    if len(trace) < 2:
        return 0.0
    recent = trace[-window:]
    workers = [e.worker for e in recent]
    most_common = max(set(workers), key=workers.count)
    return workers.count(most_common) / len(workers)


def _action_repetition(trace: list[TraceEvent], window: int = 5) -> float:
    """Are we doing the same action repeatedly?"""
    if len(trace) < 2:
        return 0.0
    recent = trace[-window:]
    actions = [e.action for e in recent]
    most_common = max(set(actions), key=actions.count)
    return actions.count(most_common) / len(actions)


def _trajectory_similarity(trace: list[TraceEvent], window: int = 3) -> float:
    """Cosine similarity of recent output snippets. High = looping.

    Uses word-bag representation for simplicity (no external deps).
    """
    if len(trace) < window * 2:
        return 0.0

    def _word_bag(events: list[TraceEvent]) -> dict[str, int]:
        bag: dict[str, int] = {}
        for e in events:
            text = f"{e.worker} {e.action} {e.output_snippet} {e.error_snippet}"
            for word in text.lower().split():
                bag[word] = bag.get(word, 0) + 1
        return bag

    recent = trace[-window:]
    previous = trace[-window * 2:-window]

    bag_a = _word_bag(recent)
    bag_b = _word_bag(previous)

    # Cosine similarity
    all_words = set(bag_a) | set(bag_b)
    dot = sum(bag_a.get(w, 0) * bag_b.get(w, 0) for w in all_words)
    mag_a = math.sqrt(sum(v * v for v in bag_a.values())) or 1.0
    mag_b = math.sqrt(sum(v * v for v in bag_b.values())) or 1.0

    return dot / (mag_a * mag_b)


def _time_waste_ratio(trace: list[TraceEvent], window: int = 5) -> float:
    """What fraction of recent time was wasted on failures?"""
    if len(trace) < 2:
        return 0.0
    recent = trace[-window:]
    total_time = sum(e.elapsed_s for e in recent) or 1.0
    wasted_time = sum(e.elapsed_s for e in recent if not e.success)
    return wasted_time / total_time


def _progress_stall(trace: list[TraceEvent]) -> float:
    """Are we getting new information? Compare unique output words in halves."""
    if len(trace) < 6:
        return 0.0
    mid = len(trace) // 2
    first_half = trace[:mid]
    second_half = trace[mid:]

    def _unique_words(events: list[TraceEvent]) -> set[str]:
        words = set()
        for e in events:
            if e.success:
                words.update(e.output_snippet.lower().split())
        return words

    first_words = _unique_words(first_half)
    second_words = _unique_words(second_half)

    if not second_words:
        return 1.0  # No successful outputs in second half

    # New information ratio: what fraction of second half words are novel?
    novel = second_words - first_words
    novelty_ratio = len(novel) / len(second_words) if second_words else 0
    # Invert: high novelty → low stall, low novelty → high stall
    return max(0.0, 1.0 - novelty_ratio)


# ── Watcher Engine ──────────────────────────────────────────────

class Watcher:
    """Metacognition engine that monitors the execution trace.

    Does NOT see code, papers, or detailed results.
    Only sees: worker, action, success/fail, timing, snippets.

    Produces a WatcherVerdict with frustration score and directive.
    """

    # Signal weights (sum to 1.0)
    WEIGHTS = {
        "consecutive_failures": 0.25,
        "trajectory_similarity": 0.20,
        "action_repetition": 0.15,
        "same_worker_ratio": 0.10,
        "time_waste_ratio": 0.15,
        "progress_stall": 0.15,
    }

    # Frustration thresholds
    SLOWING_THRESHOLD = 0.30
    STUCK_THRESHOLD = 0.55
    CRITICAL_THRESHOLD = 0.75

    def __init__(self, llm=None):
        """Initialize watcher.

        Args:
            llm: Optional LLM client for meta-reflection at high frustration.
                 If None, uses purely deterministic signals.
        """
        self.llm = llm
        self.trace: list[TraceEvent] = []
        self.verdicts: list[WatcherVerdict] = []
        self._decision_journal: list[dict] = []  # Record reasoning for post-hoc analysis

    def record(self, cycle: int, worker: str, action: str,
               success: bool, elapsed_s: float,
               output: str = "", error: str = "") -> None:
        """Record a cycle outcome into the trace."""
        self.trace.append(TraceEvent(
            cycle=cycle,
            worker=worker,
            action=action,
            success=success,
            elapsed_s=elapsed_s,
            output_snippet=output[:200] if output else "",
            error_snippet=error[:200] if error else "",
        ))

    def evaluate(self) -> WatcherVerdict:
        """Evaluate current state and produce a verdict.

        Called once per cycle AFTER recording the outcome.
        """
        if len(self.trace) < 3:
            verdict = WatcherVerdict(
                status="progressing",
                frustration=0.0,
                directive="",
                reasoning="Too early to evaluate (< 3 cycles)",
            )
            self.verdicts.append(verdict)
            return verdict

        # Compute all signals
        signals = {
            "consecutive_failures": _consecutive_failures(self.trace),
            "trajectory_similarity": _trajectory_similarity(self.trace),
            "action_repetition": _action_repetition(self.trace),
            "same_worker_ratio": _same_worker_ratio(self.trace),
            "time_waste_ratio": _time_waste_ratio(self.trace),
            "progress_stall": _progress_stall(self.trace),
        }

        # Weighted frustration score
        frustration = sum(
            signals[name] * weight
            for name, weight in self.WEIGHTS.items()
        )
        frustration = min(frustration, 1.0)

        # Determine status
        if frustration >= self.CRITICAL_THRESHOLD:
            status = "thrashing"
        elif frustration >= self.STUCK_THRESHOLD:
            status = "stuck"
        elif frustration >= self.SLOWING_THRESHOLD:
            status = "slowing"
        else:
            status = "progressing"

        # Determine directive based on status and specific signals
        directive = ""
        reasoning = ""

        if status == "thrashing":
            if signals["trajectory_similarity"] > 0.85:
                directive = "abandon"
                reasoning = (
                    f"Looping detected: trajectory similarity {signals['trajectory_similarity']:.2f}. "
                    f"Last {len(self.trace)} cycles show repeated patterns. Abandon this approach entirely."
                )
            elif signals["consecutive_failures"] >= 0.9:
                directive = "simplify"
                reasoning = (
                    f"3+ consecutive failures. The current approach is too complex. "
                    f"Simplify radically or switch methods."
                )
            else:
                directive = "pivot"
                reasoning = (
                    f"Frustration critical ({frustration:.2f}). Multiple signals elevated: "
                    + ", ".join(f"{k}={v:.2f}" for k, v in signals.items() if v > 0.4)
                )

        elif status == "stuck":
            if signals["action_repetition"] > 0.7:
                directive = "pivot"
                reasoning = (
                    f"Same action repeated: {signals['action_repetition']:.0%} of recent cycles. "
                    f"Need a fundamentally different approach."
                )
            elif signals["progress_stall"] > 0.6:
                directive = "report_now"
                reasoning = (
                    f"Progress stalled: no new information in recent cycles. "
                    f"Consider writing report with current results."
                )
            else:
                directive = "simplify"
                reasoning = (
                    f"Frustration elevated ({frustration:.2f}). "
                    f"Time waste: {signals['time_waste_ratio']:.0%}. Simplify approach."
                )

        elif status == "slowing":
            directive = ""  # Advisory only, no forced action
            reasoning = (
                f"Progress slowing ({frustration:.2f}). Watch for: "
                + ", ".join(f"{k}={v:.2f}" for k, v in signals.items() if v > 0.3)
            )

        else:
            reasoning = f"On track (frustration={frustration:.2f})"

        # Use LLM for meta-reflection at high frustration
        if self.llm and frustration >= self.STUCK_THRESHOLD:
            llm_verdict = self._llm_meta_reflect(signals, frustration, status)
            if llm_verdict:
                # LLM can override directive but not lower frustration
                if llm_verdict.get("directive"):
                    directive = llm_verdict["directive"]
                if llm_verdict.get("reasoning"):
                    reasoning = llm_verdict["reasoning"]

        verdict = WatcherVerdict(
            status=status,
            frustration=frustration,
            directive=directive,
            reasoning=reasoning,
            signals=signals,
        )
        self.verdicts.append(verdict)

        # Decision journal entry
        self._decision_journal.append({
            "cycle": self.trace[-1].cycle if self.trace else 0,
            "frustration": frustration,
            "status": status,
            "directive": directive,
            "signals": signals,
        })

        return verdict

    def _llm_meta_reflect(self, signals: dict, frustration: float,
                          status: str) -> dict | None:
        """LLM meta-reflection: look at the trace and decide what's wrong.

        This is a SEPARATE LLM call that only sees the execution trace,
        not code or papers. Per expert advice.
        """
        if not self.llm:
            return None

        # Format trace as simple timeline
        trace_lines = []
        for evt in self.trace[-10:]:  # Last 10 events only
            icon = "✓" if evt.success else "✗"
            trace_lines.append(
                f"  C{evt.cycle}: {icon} [{evt.worker}] {evt.action} "
                f"({evt.elapsed_s:.0f}s)"
                + (f" ERR: {evt.error_snippet[:80]}" if evt.error_snippet else "")
            )

        prompt = f"""You are a metacognition monitor. You see ONLY the execution trace of a research agent (not its code or papers).

## Execution Trace (most recent 10 cycles)
{chr(10).join(trace_lines)}

## Current Signals
- Frustration: {frustration:.2f} (0=calm, 1=critical)
- Status: {status}
- Consecutive failures: {signals['consecutive_failures']:.2f}
- Trajectory similarity: {signals['trajectory_similarity']:.2f} (>0.85 = looping)
- Action repetition: {signals['action_repetition']:.2f}
- Progress stall: {signals['progress_stall']:.2f}
- Time waste: {signals['time_waste_ratio']:.2f}

## Your Task
Analyze the execution pattern. What is the agent doing wrong?

Respond with JSON only:
{{"directive": "pivot|simplify|abandon|report_now|continue", "reasoning": "one sentence: what pattern do you see and why this directive"}}

Rules:
- "continue" = the agent is making progress despite some friction
- "simplify" = approach is too complex, reduce scope
- "pivot" = change method entirely (not just fix a bug)
- "abandon" = this direction is fundamentally unworkable
- "report_now" = enough data collected, write report with what we have"""

        try:
            response = self.llm.chat([
                {"role": "system", "content": "Metacognition monitor. JSON only. One sentence reasoning."},
                {"role": "user", "content": prompt},
            ])
            content = response["choices"][0]["message"]["content"]

            # Strip thinking tags
            try:
                from core.llm import strip_think
                content = strip_think(content)
            except ImportError:
                pass

            # Parse JSON
            import re
            match = re.search(r'\{[^{}]*\}', content)
            if match:
                data = json.loads(match.group())
                valid_directives = {"pivot", "simplify", "abandon", "report_now", "continue"}
                if data.get("directive") in valid_directives:
                    return data
        except Exception as e:
            print(f"  [Watcher] LLM meta-reflection failed: {e}")

        return None

    def format_for_prompt(self) -> str:
        """Format the latest verdict for injection into supervisor's decision prompt."""
        if not self.verdicts:
            return ""

        v = self.verdicts[-1]
        if v.status == "progressing":
            return ""  # Don't clutter prompt when things are fine

        parts = [f"## Watcher (Metacognition Monitor)"]
        icon = {
            "slowing": "⚡",
            "stuck": "⚠️",
            "looping": "🔄",
            "thrashing": "🚨",
        }.get(v.status, "ℹ️")

        parts.append(f"{icon} Status: **{v.status.upper()}** (frustration={v.frustration:.2f})")
        parts.append(f"Analysis: {v.reasoning}")

        if v.directive:
            directive_map = {
                "pivot": "CHANGE your approach entirely. The current method is not working.",
                "simplify": "REDUCE scope/complexity. Use simpler models, fewer conditions.",
                "abandon": "STOP this direction. It is fundamentally unworkable.",
                "report_now": "WRITE REPORT with current results. Enough data collected.",
                "continue": "Continue but be alert to the slowing signals.",
            }
            parts.append(f"Directive: **{v.directive.upper()}** — {directive_map.get(v.directive, v.directive)}")

        return "\n".join(parts)

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "trace": [e.to_dict() for e in self.trace],
            "verdicts": [v.to_dict() for v in self.verdicts],
            "decision_journal": self._decision_journal,
        }

    @classmethod
    def from_dict(cls, d: dict, llm=None) -> "Watcher":
        w = cls(llm=llm)
        w.trace = [TraceEvent.from_dict(e) for e in d.get("trace", [])]
        w.verdicts = [WatcherVerdict.from_dict(v) for v in d.get("verdicts", [])]
        w._decision_journal = d.get("decision_journal", [])
        return w
