"""
GitHub Search MCP Server
==========================
Provides GitHub code and repository search tools.
Uses GitHub REST API (no auth for search, rate-limited to 10 req/min).
"""

import json
import time
from urllib.request import urlopen, Request
from urllib.parse import quote

_last_request_time = 0.0
_MIN_INTERVAL = 6.0  # 10 requests/min → 6s between requests


def _rate_limit():
    """Enforce rate limiting for GitHub API."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_request_time = time.time()


def _github_get(url: str) -> dict:
    """Make a rate-limited GET request to GitHub API."""
    _rate_limit()
    req = Request(url, headers={
        "User-Agent": "ResearchAgent/1.0",
        "Accept": "application/vnd.github.v3+json",
    })
    with urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def search_github_repos(query: str, sort: str = "stars", limit: int = 5) -> dict:
    """Search GitHub repositories."""
    q = quote(query)
    url = f"https://api.github.com/search/repositories?q={q}&sort={sort}&order=desc&per_page={limit}"
    try:
        data = _github_get(url)
        repos = []
        for r in data.get("items", [])[:limit]:
            repos.append({
                "name": r.get("full_name", ""),
                "description": (r.get("description") or "")[:200],
                "stars": r.get("stargazers_count", 0),
                "language": r.get("language", ""),
                "url": r.get("html_url", ""),
                "updated": r.get("updated_at", "")[:10],
                "topics": r.get("topics", [])[:5],
            })
        return {
            "source": "GitHub",
            "total": data.get("total_count", 0),
            "repos": repos,
        }
    except Exception as e:
        return {"source": "GitHub", "error": str(e), "repos": []}


def search_github_code(query: str, language: str = "python", limit: int = 5) -> dict:
    """Search GitHub code."""
    q = quote(f"{query} language:{language}")
    url = f"https://api.github.com/search/code?q={q}&per_page={limit}"
    try:
        data = _github_get(url)
        results = []
        for item in data.get("items", [])[:limit]:
            results.append({
                "name": item.get("name", ""),
                "path": item.get("path", ""),
                "repo": item.get("repository", {}).get("full_name", ""),
                "url": item.get("html_url", ""),
                "score": item.get("score", 0),
            })
        return {
            "source": "GitHub Code",
            "total": data.get("total_count", 0),
            "results": results,
        }
    except Exception as e:
        return {"source": "GitHub Code", "error": str(e), "results": []}


# ── Tool Definitions ─────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_github_repos",
            "description": "Search GitHub repositories by topic. Returns repo names, descriptions, stars, and URLs. Rate-limited to 10 req/min.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g., 'flash attention pytorch')"},
                    "sort": {"type": "string", "description": "Sort by: stars, forks, updated", "default": "stars"},
                    "limit": {"type": "integer", "description": "Max results (1-10)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_github_code",
            "description": "Search GitHub code for specific patterns. Returns file names, paths, and repo info. Rate-limited to 10 req/min.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Code search query"},
                    "language": {"type": "string", "description": "Programming language filter", "default": "python"},
                    "limit": {"type": "integer", "description": "Max results (1-10)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_github_repos": search_github_repos,
    "search_github_code": search_github_code,
}
