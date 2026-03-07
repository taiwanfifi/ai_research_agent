#!/usr/bin/env python3
"""
Peer Consultation — Ask another LLM for architectural feedback.
================================================================
Multi-turn conversation with GPT-5.4 about Opus's design problems.
Saves full transcript to tools/consultations/

Usage:
    python3 tools/consult_peer.py
    python3 tools/consult_peer.py --model gpt-5.2 --turns 3
"""

import json
import os
import sys
import time
from pathlib import Path

# Load API key from Tools/.env
TOOLS_DIR = Path(__file__).parent.parent.parent / "Tools"
ENV_PATH = TOOLS_DIR / ".env"

def load_env():
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

load_env()

from openai import OpenAI

CONSULT_DIR = Path(__file__).parent / "consultations"
CONSULT_DIR.mkdir(exist_ok=True)


def build_context() -> str:
    """Build a concise but complete description of Opus for the peer."""
    return """I'm building an autonomous AI research agent called "Opus". It decomposes research goals into tasks, dispatches them to specialized workers (explorer, coder, reviewer), and produces research reports with real experiments.

## Current Architecture
- **Supervisor**: LLM-driven adaptive loop. Decomposes goal → task queue → dispatches workers → checkpoints. Max 10-12 cycles.
- **Workers**: Each is an LLM agent with tools. Explorer searches papers. Coder writes+runs Python. Reviewer benchmarks+visualizes.
- **LLM**: MiniMax-M2.5-highspeed (204K context). Each worker gets ~10 tool-calling turns.
- **Validation**: LLM Judge evaluates worker output quality. ResultVerifier cross-checks claims vs stdout.
- **Learning**: EvolutionStore persists learnings across missions (JSON, word-overlap dedup).
- **Scoring**: Rule-based MissionScorer (6 dimensions: literature/code/results/verification/artifacts/report).

## Concrete Problems (with data)

### 1. Prompt Bloat — Rules accumulate, LLM ignores them
The coder prompt is 160+ lines. Every environment bug (eval_strategy deprecated, bitsandbytes crashes on Mac, subprocess bypasses timeout) becomes a new rule. The LLM increasingly ignores rules deep in the prompt. We need a meta-prompt or layered architecture.

### 2. No Error Analysis — Supervisor just retries or skips
When a task fails, supervisor either retries the same task or skips it. It doesn't analyze stderr, diagnose root cause, or adapt strategy. From 36 missions (263 tasks, 69 failures):
- 67% of failures had diagnostic info in stderr that an LLM could have used
- 33% were environment bugs, 33% task-too-big, 22% wrong approach, 11% dependency failures

### 3. No Pre-Execution Judgment
Workers execute tasks blindly. They don't check if dependencies are met, if referenced files exist, or if the task is feasible within timeout. A dry-run A/B test showed:
- Strategy B (dependency+file check): catches 4.3% of failures
- Strategy C (B + resource estimation + friction buffer): catches 53.6% but 31.6% precision (too many false positives)

### 4. No Post-Observation Checkpoint
After each tool result, the worker just continues to the next step. It doesn't pause to ask "does this match my expectation?" — unlike how a human developer would check output before proceeding.

### 5. Evolution Store is Surface-Level
Stores string patterns like "use eval_strategy not evaluation_strategy". No causal understanding. Dedup is word-overlap based. The LLM may or may not read these learnings.

## What I've Studied
I analyzed a related system (kael_daemon) that has:
- **Two-layer cognition**: THINK → QUESTION → ACT (structured question before execution)
- **Friction buffer**: Accumulates "[FRICTION: what failed | what would be better]" — injected into next cycle
- **Anti-coasting detection**: Monitors if agent is just doing easy work, injects provocative questions
- **Capability decay**: Unused skills decay, pushing exploration of neglected areas
- **Goal verification with tool evidence**: Can't declare done without proof

## My Questions
I want to make Opus genuinely better, not just add more patches. The core issue: it executes blindly without judgment."""


def consult(model: str = "gpt-5.4", max_turns: int = 5):
    """Run a multi-turn consultation."""
    client = OpenAI()

    system_msg = """You are a senior AI systems architect consulting on an autonomous research agent.
Be direct, specific, and opinionated. Don't hedge or give generic advice.
When you suggest a change, describe the exact mechanism (not just "add error handling").
Challenge assumptions. Point out things the developer might not want to hear.
Keep responses focused — max 500 words per turn."""

    context = build_context()

    questions = [
        # Turn 1: Overall diagnosis
        context + """

Given this architecture, what are the 3 highest-leverage changes I should make? Not incremental patches — structural changes that would address the root causes. Be specific about mechanisms.""",

        # Turn 2: Meta-prompt architecture
        """You mentioned [reference previous answer]. Let me zoom in on the prompt architecture problem.

Currently the coder has a 160-line system prompt that grows with each bug fix. I'm considering:
A) Layered prompts: base (tools/workflow) + mission-context (goal, workspace state) + friction (recent failures + evolution learnings), each managed separately
B) A meta-prompt that selects which rules are relevant to the current task
C) Moving rules out of prompts entirely — into tool-gated checks (the system blocks bad patterns before the LLM sees them)

Which approach, or what combination? How would you implement the meta-prompt specifically?""",

        # Turn 3: The "judgment voice"
        """The daemon system I studied has a structured QUESTION phase between thinking and acting:
- "Am I choosing this because it's easy or because it matters?"
- "What's ONE harder thing I could do instead?"

I tried "inner monologue" in Opus (letting workers reflect before executing). A/B test showed it HURTS — Grade D with monologue vs Grade B without. It wasted tool-calling turns on reflection instead of execution.

But the daemon's approach works because its questions are STRUCTURED and SHORT, not open-ended reflection.

How do I add judgment to Opus workers without wasting their limited turns (10 tool calls max)? The key constraint: every turn spent on reflection is a turn not spent on actual work.""",

        # Turn 4: Friction-driven learning vs current evolution store
        """Current learning system: EvolutionStore saves {"type": "pitfall", "pattern": "use eval_strategy not evaluation_strategy", "confidence": 0.8}. Word-overlap dedup. Injected into planner prompt.

Daemon's friction system: [FRICTION: what failed | what would be better]. Accumulated per-session, injected into next cycle. Friction includes IMAGINED improvements, not just error logs.

The fundamental difference: EvolutionStore records WHAT happened. Friction records WHAT SHOULD CHANGE.

How should I redesign the learning system? Should friction be per-mission (resets each run) or persistent? How do I prevent the friction buffer from becoming another bloated prompt?""",

        # Turn 5: Synthesis — what would you build?
        """Final question. If you were rebuilding this from the current codebase (not from scratch), what's the minimal set of changes that would take it from "sometimes Grade B, often Grade D" to "consistently Grade B+, occasionally Grade A"?

Give me a prioritized implementation plan — what to build first, second, third. Each item should be a concrete code change, not a principle.""",
    ]

    messages = [{"role": "system", "content": system_msg}]
    transcript = []
    total_tokens = 0

    print(f"\n{'='*70}")
    print(f"  Peer Consultation with {model}")
    print(f"  Max turns: {max_turns}")
    print(f"{'='*70}")

    for i, question in enumerate(questions[:max_turns]):
        print(f"\n{'─'*70}")
        print(f"  Turn {i+1}/{min(max_turns, len(questions))}")
        print(f"{'─'*70}")

        # If turn > 1, allow referencing previous answer
        if i > 0 and "[reference previous answer]" in question:
            # Replace placeholder with brief reference
            question = question.replace("[reference previous answer]",
                                       "(from your previous response)")

        print(f"\n  Q: {question[:200]}...")

        messages.append({"role": "user", "content": question})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=2048,
            )
        except Exception as e:
            print(f"\n  ERROR: {e}")
            break

        answer = response.choices[0].message.content
        usage = response.usage
        total_tokens += (usage.prompt_tokens + usage.completion_tokens)

        messages.append({"role": "assistant", "content": answer})
        transcript.append({
            "turn": i + 1,
            "question": question,
            "answer": answer,
            "tokens": {"prompt": usage.prompt_tokens, "completion": usage.completion_tokens},
        })

        print(f"\n  A ({usage.completion_tokens} tokens):\n")
        # Print with indent
        for line in answer.split("\n"):
            print(f"    {line}")

    # Save transcript
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = CONSULT_DIR / f"consult_{model}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": model,
            "timestamp": ts,
            "total_tokens": total_tokens,
            "estimated_cost_usd": total_tokens * 0.00003,  # rough estimate
            "turns": transcript,
        }, f, indent=2, ensure_ascii=False)

    # Also save readable markdown
    md_path = CONSULT_DIR / f"consult_{model}_{ts}.md"
    with open(md_path, "w") as f:
        f.write(f"# Peer Consultation: {model}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Total tokens: {total_tokens} (~${total_tokens * 0.00003:.2f})\n\n")
        for t in transcript:
            f.write(f"---\n## Turn {t['turn']}\n\n")
            f.write(f"### Question\n{t['question']}\n\n")
            f.write(f"### Answer\n{t['answer']}\n\n")

    print(f"\n{'='*70}")
    print(f"  Consultation complete. {len(transcript)} turns, {total_tokens} tokens")
    print(f"  Saved: {out_path}")
    print(f"  Readable: {md_path}")
    print(f"{'='*70}")

    return transcript


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4", help="Model to consult")
    parser.add_argument("--turns", type=int, default=5, help="Max conversation turns")
    args = parser.parse_args()
    consult(model=args.model, max_turns=args.turns)
