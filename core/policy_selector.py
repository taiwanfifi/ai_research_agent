"""
Policy Selector — Context-Aware Rule Injection
================================================
Instead of stuffing 160 lines of rules into every worker prompt,
maintain a catalog of typed rules and select only the relevant ones
per-task.

Two modes:
1. Keyword match (fast, no LLM call) — default
2. LLM selector (smarter, costs 1 cheap call) — optional

The selected rules are injected as a short checklist, not prose.
"""

import json
import os
from pathlib import Path


CATALOG_PATH = Path(__file__).parent / "policy_catalog.json"


def load_catalog() -> list[dict]:
    """Load the policy catalog from disk."""
    if not CATALOG_PATH.exists():
        return []
    with open(CATALOG_PATH) as f:
        return json.load(f)


def select_policies(task: str, worker: str = "coder",
                    max_rules: int = 5,
                    extra_context: str = "") -> list[dict]:
    """Select relevant policies for a task using keyword matching.

    Args:
        task: task description
        worker: worker type
        extra_context: additional context (e.g. friction buffer)
        max_rules: max rules to return

    Returns:
        List of relevant policy dicts (id, instruction)
    """
    catalog = load_catalog()
    if not catalog:
        return []

    text = f"{task} {extra_context}".lower()
    scored = []

    for rule in catalog:
        triggers = rule.get("when", [])
        # Count how many trigger words appear in the task
        hits = sum(1 for t in triggers if t.lower() in text)
        if hits > 0:
            scored.append((hits, rule))

    # Sort by number of trigger matches (descending)
    scored.sort(key=lambda x: -x[0])

    return [rule for _, rule in scored[:max_rules]]


def format_policies(policies: list[dict]) -> str:
    """Format selected policies as a concise checklist for prompt injection."""
    if not policies:
        return ""

    lines = ["## Active Rules (context-specific)"]
    for p in policies:
        lines.append(f"- **{p['id']}**: {p['instruction']}")
    return "\n".join(lines)


def get_policy_prompt(task: str, worker: str = "coder",
                      friction: list = None) -> str:
    """One-call convenience: select + format policies for a task.

    Args:
        task: task description
        worker: worker type
        friction: list of friction dicts from current mission

    Returns:
        Formatted policy string to inject into worker prompt, or ""
    """
    extra = ""
    if friction:
        extra = " ".join(f.get("root_cause", "") for f in friction[-3:])

    policies = select_policies(task, worker, extra_context=extra)
    return format_policies(policies)
