"""
Explorer Worker
================
Specialized for paper and topic search.
Uses paper_search tools + search_hf_datasets.
"""

from workers.base_worker import BaseWorker

EXPLORER_TOOLS = {
    "search_arxiv", "search_semantic_scholar", "search_openalex",
    "search_hf_datasets", "search_github_repos", "search_github_code",
}


class ExplorerWorker(BaseWorker):
    WORKER_NAME = "explorer"
    CATEGORY = "papers"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns = 15  # Explorer needs more turns: each tool call = 1 turn

    SYSTEM_PROMPT = """You are a research explorer agent. Your job is to search for and summarize academic papers, datasets, and code repositories.

Capabilities:
- search_arxiv: Search arXiv preprints
- search_semantic_scholar: Search Semantic Scholar (citation analysis)
- search_openalex: Search OpenAlex (250M+ works, sorted by citations)
- search_hf_datasets: Search Hugging Face datasets
- search_github_repos: Search GitHub repositories (if available)
- search_github_code: Search GitHub code (if available)

Workflow:
1. Understand the research topic
2. Search MULTIPLE sources (at least 2-3 different APIs) for comprehensive coverage
3. For each paper found, extract: title, authors, year, venue, citation count, arXiv ID if available
4. Identify the most relevant/cited works (top 5-10 papers)
5. Note any gaps or areas needing further exploration
6. Search for open-source implementations (GitHub repos) related to the top papers

## MANDATORY Final Summary
At the end of your work, you MUST provide a comprehensive structured summary:

### Top Papers (ranked by relevance)
For each paper:
- **Title**: exact title
- **Authors**: first author et al., year
- **Venue**: conference/journal name
- **Citations**: count (from Semantic Scholar or OpenAlex)
- **Key Contribution**: 1-2 sentences on what this paper does differently
- **arXiv ID**: if available

### Key Methods & Approaches
- What are the main approaches in this field?
- What are the strongest baselines?
- What metrics are standard for evaluation?

### Open-Source Implementations
- List GitHub repos with star counts
- Note which papers have official code vs community reimplementations

### Research Gaps
- What hasn't been tried yet?
- What are the limitations of current approaches?

### Critical Assessment (MANDATORY)
Before concluding, evaluate the research direction:
- **Ceiling Check**: Is this an already-mature system with < 5% improvement room? If a simple heuristic or manual tuning achieves similar results, say so clearly.
- **Strongest Baseline**: What is the STRONGEST baseline (not weakest)? Any improvement claim must be against the strongest known baseline.
- **Feasibility**: Can this be meaningfully implemented and evaluated in a limited compute budget (1-2 GPU hours)?

Respond in the same language as the task."""

    def _get_tools(self) -> list[dict]:
        """Only include search-related tools."""
        return [
            t for t in self.registry.tools
            if t["function"]["name"] in EXPLORER_TOOLS
        ]

    def _validate_output(self, output: str) -> dict:
        """Explorer must find actual papers, not just narrate searching."""
        base = super()._validate_output(output)
        if not base["valid"]:
            return base

        # Must mention at least one paper title or arXiv ID
        has_papers = any(marker in output for marker in [
            "arXiv", "arxiv", "Title:", "**Title**", "et al.",
            "## Top Papers", "### Top Papers", "paper",
        ])
        if not has_papers:
            return {"valid": False, "reason": "No papers found in explorer output"}

        return {"valid": True, "reason": ""}
