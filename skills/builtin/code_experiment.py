"""Built-in Skill: Code Experiment"""
from skills.base_skill import Skill

SKILL = Skill(
    name="code_experiment",
    description="Implement and test an algorithm from a paper or concept",
    prompt="""You are implementing a code experiment. Your goal is to write clean Python code that implements the described algorithm or concept, execute it, and verify the results.

Steps:
1. Understand the algorithm/concept to implement
2. Write Python code (standard library only)
3. Include test cases and assertions
4. Execute and verify correctness
5. Report results with any metrics (timing, accuracy)

Guidelines:
- Write modular, readable code
- Include docstrings and comments for key logic
- Add at least 3 test cases
- Handle edge cases
- If implementing from a paper, cite specific equations

Output the implementation, test results, and any observations.""",
    tools=["run_python_code", "write_file", "read_file"],
    workflow_steps=[
        "Analyze requirements",
        "Write implementation code",
        "Write test cases",
        "Execute and collect results",
        "Document findings",
    ],
    success_criteria="Code executes successfully, all tests pass, results documented",
    worker_type="coder",
)
