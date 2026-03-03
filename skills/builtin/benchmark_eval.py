"""Built-in Skill: Benchmark Evaluation"""
from skills.base_skill import Skill

SKILL = Skill(
    name="benchmark_eval",
    description="Evaluate code against standard benchmarks",
    prompt="""You are running a benchmark evaluation. Your goal is to test an implementation against standard benchmarks and report detailed results.

Steps:
1. Understand the implementation to evaluate
2. Find or create appropriate test cases
3. Run benchmarks (correctness, performance, edge cases)
4. Collect metrics: accuracy, speed, memory
5. Compare against baselines if available
6. Write evaluation report

Guidelines:
- Use multiple test cases (at least 5)
- Test edge cases and boundary conditions
- Measure execution time
- Report both qualitative and quantitative results
- Be honest about limitations

Output a detailed evaluation report with metrics.""",
    tools=["run_python_code", "write_file", "read_file", "fetch_humaneval_sample", "fetch_leetcode_problem"],
    workflow_steps=[
        "Understand what to evaluate",
        "Design test suite",
        "Execute benchmarks",
        "Collect and analyze metrics",
        "Write evaluation report",
    ],
    success_criteria="Ran 5+ test cases, collected metrics, produced evaluation report",
    worker_type="reviewer",
)
