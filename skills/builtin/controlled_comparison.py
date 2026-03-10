"""
Built-in Research Protocol: Controlled Comparison
===================================================
A structured protocol for comparing two methods/approaches.
Defines hypothesis, variables, minimum requirements, and evaluation criteria.

This is a MISSION-LEVEL protocol — the planner uses it to generate
the full task queue, ensuring research quality by design.
"""
from skills.base_skill import Skill

SKILL = Skill(
    name="controlled_comparison",
    description="Compare two methods on a task with controlled variables, multiple seeds, and statistical testing",
    prompt="""You are planning a controlled comparison experiment.

## Research Protocol

### Phase 1: Literature (1 explorer task)
Search for prior work comparing the two methods. Find:
- At least 2 papers with relevant comparisons
- Known results/baselines for context
- Key insights about when each method works better

### Phase 2: Implementation (2-3 coder tasks, sequential)
Task A: Write shared infrastructure
- Model definition, data loading, evaluation function
- Save as reusable modules (model.py, data_utils.py)
- Run a 1-batch smoke test to verify everything works

Task B: Write training script for Method 1
- Import shared modules
- Train with dataset.select(range(2000)), 2-5 epochs
- 5 seeds (42, 123, 456, 789, 1024), ONE seed per run_python_code call
- Print metrics as 'metric_name: value'
- Save results to results_method1.json

Task C: Write training script for Method 2
- Same architecture, same data, same hyperparameters except the method difference
- Same 5 seeds, same format
- Save results to results_method2.json

### Phase 3: Analysis (1 reviewer task)
- Load both result JSONs
- Compute: mean ± std for each method
- Paired t-test (scipy.stats.ttest_rel)
- Effect size (Cohen's d)
- Create comparison_chart.png (bar chart with error bars)
- Create training_curves.png (if loss data available)
- Save analysis_summary.json

### Quality Gates
- Each method must have results from all 5 seeds
- Statistical test must be computed (not just eyeballed)
- At least 1 figure with error bars
- Results must be verified against stdout (not self-reported)

### Evaluation Criteria
- Significant difference (p < 0.05): report which method wins and by how much
- Non-significant (p >= 0.05): report "no significant difference" — this is a valid result
- Effect size: small (d<0.2), medium (0.2-0.8), large (d>0.8)""",
    tools=[],  # Mission-level, not worker-level
    workflow_steps=[
        "Literature search for prior comparisons",
        "Implement shared model + data infrastructure",
        "Train Method 1 with 3 seeds",
        "Train Method 2 with 3 seeds",
        "Statistical analysis + visualization",
    ],
    success_criteria="Both methods trained with 3 seeds, statistical test computed, comparison figure generated, results verified against stdout",
    worker_type="supervisor",  # This is a mission-level protocol
)
