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
2. Search multiple sources for comprehensive coverage
3. Summarize key findings: titles, years, citation counts, key contributions
4. Identify the most relevant/cited works
5. Note any gaps or areas needing further exploration

Output format:
- Provide a structured summary with paper details
- Highlight top 3-5 most relevant papers
- Note trends and common themes
- Suggest follow-up queries if needed

Respond in the same language as the task."""

    def _get_tools(self) -> list[dict]:
        """Only include search-related tools."""
        return [
            t for t in self.registry.tools
            if t["function"]["name"] in EXPLORER_TOOLS
        ]
