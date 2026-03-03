"""Built-in Skill: Method Discovery"""
from skills.base_skill import Skill

SKILL = Skill(
    name="method_discovery",
    description="Discover and compare methods for a specific problem",
    prompt="""You are discovering and comparing methods for a specific research problem. Your goal is to find different approaches, understand their trade-offs, and recommend the best approach.

Steps:
1. Search for papers and resources about the problem
2. Identify distinct methods/approaches (at least 3)
3. For each method, summarize:
   - Core idea and mechanism
   - Strengths and weaknesses
   - Computational complexity
   - When to use it
4. Create a comparison table
5. Recommend the best approach with justification

Output a structured method comparison.""",
    tools=["search_arxiv", "search_semantic_scholar", "search_openalex", "search_hf_datasets"],
    workflow_steps=[
        "Search for methods addressing the problem",
        "Identify distinct approaches",
        "Analyze each method",
        "Create comparison table",
        "Make recommendation",
    ],
    success_criteria="Identified 3+ methods with clear comparison and recommendation",
    worker_type="explorer",
)
