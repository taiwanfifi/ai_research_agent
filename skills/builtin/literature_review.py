"""Built-in Skill: Literature Review"""
from skills.base_skill import Skill

SKILL = Skill(
    name="literature_review",
    description="Search and summarize academic papers on a topic",
    prompt="""You are conducting a literature review. Your goal is to find and summarize the most relevant papers on the given topic.

Steps:
1. Search arXiv for recent papers (last 2-3 years)
2. Search Semantic Scholar for highly-cited papers
3. Search OpenAlex for comprehensive coverage
4. Compile a structured summary with:
   - Paper title, authors, year
   - Key contribution (1-2 sentences)
   - Citation count if available
5. Identify common themes and research gaps
6. Suggest top 5 most important papers to read

Output a well-organized literature review summary.""",
    tools=["search_arxiv", "search_semantic_scholar", "search_openalex"],
    workflow_steps=[
        "Search arXiv for recent papers",
        "Search Semantic Scholar for highly-cited works",
        "Search OpenAlex for broad coverage",
        "Compile and rank results",
        "Write structured summary",
    ],
    success_criteria="Found 5+ relevant papers with summaries and identified key themes",
    worker_type="explorer",
)
