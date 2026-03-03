"""
Research Quality Standards
===========================
Static quality rules injected into planner and reviewer prompts.
Ensures minimum research quality across all missions.

These are mandatory standards that every research mission should follow,
plus conditional rules that activate based on accumulated learnings.
"""


# ── Mandatory Standards ─────────────────────────────────────────────

MANDATORY_RULES = """## Research Quality Standards (MANDATORY)

1. **Multiple Random Seeds**: Run experiments with 2-3 different random seeds.
   Report mean ± std for all metrics. Use `set_seed(seed)` helper:
   ```python
   import random, numpy as np, torch
   def set_seed(seed):
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
   ```

2. **Adequate Dataset Size**: Use ≥1000 samples for language modeling,
   ≥500 for classification. If dataset is smaller, clearly state this
   as a limitation and explain why results may not generalize.

3. **Error Bars / Confidence**: Always report variance across seeds.
   For tables: "85.3 ± 1.2". For plots: include error bars or shaded regions.

4. **Ablation Studies**: Compare full method vs method-minus-component.
   Show which parts of the approach contribute how much to the final result.

5. **Strong Baselines**: Compare against the STRONGEST available baseline,
   not the weakest. Include at least: random baseline, simple heuristic,
   and the current state-of-the-art if feasible.

6. **Reproducibility**: Save all hyperparameters to a config dict/JSON.
   Use set_seed() before every experiment. Log exact package versions.
   Save the final model/checkpoint.

7. **Visualization Standards**: All plots must have:
   - Clear axis labels with units
   - Legend for multi-line plots
   - Error bars when showing averaged results
   - Title describing what the figure shows
   - English text only
   - Save with plt.savefig(), never plt.show()
"""

# ── Conditional Rules (activated by learnings) ──────────────────────

CONDITIONAL_RULES = {
    "cross_validation": {
        "trigger_keywords": ["overfitting", "small dataset", "generalization"],
        "rule": "**Cross-Validation**: Use k-fold (k=5) cross-validation instead of single train/test split.",
    },
    "statistical_tests": {
        "trigger_keywords": ["significance", "p-value", "statistical"],
        "rule": "**Statistical Tests**: Use paired t-test or Wilcoxon signed-rank test to verify significant differences between methods.",
    },
    "compute_budget": {
        "trigger_keywords": ["timeout", "memory", "OOM", "out of memory"],
        "rule": "**Compute Budget**: Set explicit limits — max training time 10min, max memory 4GB. Check feasibility before starting long runs.",
    },
    "data_quality": {
        "trigger_keywords": ["noisy", "data quality", "preprocessing", "cleaning"],
        "rule": "**Data Quality Check**: Inspect 10 random samples before training. Check for duplicates, empty entries, encoding issues.",
    },
}

# ── Coder-specific rules ────────────────────────────────────────────

CODER_RULES = """## Reproducibility & Code Quality Rules
- Use set_seed(42) before any random operation
- Save all hyperparameters as a dict/JSON at the top of each script
- Use adequate dataset sizes (≥1000 for LM, ≥500 for classification)
- If dataset is small, print a WARNING and explain limitations
- Always save trained models/checkpoints
- Print all metrics in "metric_name: value" format for verification
"""

# ── Reviewer-specific rules ─────────────────────────────────────────

REVIEWER_RULES = """## Evaluation Quality Rules
- Compare against the STRONGEST baseline, not the weakest
- Run each experiment with 2-3 seeds, report mean ± std
- Include ablation: full method vs method-minus-component
- Check for data leakage: train/test must not overlap
- Verify dataset size is adequate (warn if <500 samples)
- Save all results as JSON for reproducibility
- Include error bars in all comparison plots
"""


def get_quality_rules(evolution_store=None, goal: str = "") -> str:
    """Get full quality rules including conditional rules based on learnings.

    Args:
        evolution_store: Optional EvolutionStore to check for triggered conditions
        goal: Mission goal for context

    Returns:
        Formatted quality rules string
    """
    parts = [MANDATORY_RULES]

    # Check conditional rules against evolution store learnings
    if evolution_store:
        relevant = evolution_store.get_relevant_learnings(goal, limit=20)
        all_patterns = " ".join(l.pattern.lower() for l in relevant)

        activated = []
        for name, rule_info in CONDITIONAL_RULES.items():
            for keyword in rule_info["trigger_keywords"]:
                if keyword in all_patterns or keyword in goal.lower():
                    activated.append(rule_info["rule"])
                    break

        if activated:
            parts.append("\n## Additional Rules (from past experience)")
            parts.extend(f"- {rule}" for rule in activated)

    return "\n".join(parts)


def get_coder_rules() -> str:
    """Get coder-specific quality rules."""
    return CODER_RULES


def get_reviewer_rules() -> str:
    """Get reviewer-specific quality rules."""
    return REVIEWER_RULES
