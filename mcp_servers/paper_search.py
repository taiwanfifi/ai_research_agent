"""
論文搜尋 MCP Server (模擬)
===========================
提供以下 tools 給 Agent 呼叫：
- search_arxiv: 搜尋 arXiv 預印本
- search_semantic_scholar: 搜尋 Semantic Scholar (引用分析)
- search_openalex: 搜尋 OpenAlex (最全學術目錄)
- get_paper_pdf_url: 取得論文 PDF 連結

這些函數可以被 Agent 直接呼叫，也可以包裝成正式 MCP Server。
"""

import json
import time
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
from urllib.parse import quote

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"


def search_arxiv(query: str, max_results: int = 5, sort_by: str = "submittedDate") -> dict:
    """搜尋 arXiv 預印本"""
    q = quote(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={max_results}&sortBy={sort_by}&sortOrder=descending"
    req = Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
    with urlopen(req, timeout=15) as resp:
        root = ET.fromstring(resp.read())

    total = int(root.findtext(f"{OPENSEARCH_NS}totalResults", "0"))
    papers = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        title = entry.findtext(f"{ATOM_NS}title", "").strip().replace("\n", " ")
        summary = entry.findtext(f"{ATOM_NS}summary", "").strip().replace("\n", " ")
        arxiv_id = entry.findtext(f"{ATOM_NS}id", "").split("/abs/")[-1]
        published = entry.findtext(f"{ATOM_NS}published", "")[:10]
        authors = [a.findtext(f"{ATOM_NS}name", "") for a in entry.findall(f"{ATOM_NS}author")]
        pdf_url = ""
        for link in entry.findall(f"{ATOM_NS}link"):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
        papers.append({
            "title": title, "authors": authors[:5], "summary": summary[:300],
            "arxiv_id": arxiv_id, "published": published, "pdf_url": pdf_url,
        })
    return {"source": "arXiv", "total": total, "papers": papers}


def search_semantic_scholar(query: str, limit: int = 5) -> dict:
    """搜尋 Semantic Scholar (含引用數)"""
    q = quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search/bulk?query={q}&limit={limit}&fields=title,year,citationCount,abstract"
    req = Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    papers = []
    for p in data.get("data", []):
        papers.append({
            "title": p.get("title", ""),
            "year": p.get("year"),
            "citations": p.get("citationCount", 0),
            "abstract": (p.get("abstract") or "")[:300],
        })
    return {"source": "Semantic Scholar", "total": data.get("total", 0), "papers": papers}


def search_openalex(query: str, per_page: int = 5) -> dict:
    """搜尋 OpenAlex (依引用數排序)"""
    q = quote(query)
    url = f"https://api.openalex.org/works?search={q}&per_page={per_page}&sort=cited_by_count:desc&mailto=test@example.com"
    req = Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    papers = []
    for w in data.get("results", []):
        papers.append({
            "title": w.get("title", ""),
            "year": w.get("publication_year"),
            "citations": w.get("cited_by_count", 0),
            "doi": w.get("doi", ""),
        })
    return {"source": "OpenAlex", "total": data.get("meta", {}).get("count", 0), "papers": papers}


# ── Tool 定義 (供 Agent function calling 使用) ──────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "Search arXiv for preprints. Returns titles, authors, abstracts, arxiv IDs, and PDF URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g., 'transformer attention mechanism')"},
                    "max_results": {"type": "integer", "description": "Max results (1-20)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_semantic_scholar",
            "description": "Search Semantic Scholar for papers with citation counts and impact metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (1-20)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_openalex",
            "description": "Search OpenAlex (250M+ works) sorted by citation count. Good for finding seminal papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "per_page": {"type": "integer", "description": "Max results (1-20)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_arxiv": search_arxiv,
    "search_semantic_scholar": search_semantic_scholar,
    "search_openalex": search_openalex,
}
