"""
論文搜尋 MCP Server (模擬)
===========================
提供以下 tools 給 Agent 呼叫：
- search_arxiv: 搜尋 arXiv 預印本
- search_semantic_scholar: 搜尋 Semantic Scholar (引用分析)
- search_openalex: 搜尋 OpenAlex (最全學術目錄)
- fetch_arxiv_by_id: 直接以 arXiv ID 查詢論文
- search_papers_with_code: 搜尋 Papers With Code

這些函數可以被 Agent 直接呼叫，也可以包裝成正式 MCP Server。
"""

import json
import re
import time
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
from urllib.parse import quote
from urllib.error import URLError, HTTPError

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH_NS = "{http://a9.com/-/spec/opensearch/1.1/}"


def search_arxiv(query: str, max_results: int = 5, sort_by: str = "relevance") -> dict:
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
    """搜尋 Semantic Scholar (含引用數) — uses /graph/v1/paper/search with retry"""
    q = quote(query)
    url = (f"https://api.semanticscholar.org/graph/v1/paper/search"
           f"?query={q}&limit={limit}&fields=title,year,citationCount,abstract,externalIds")
    req = Request(url, headers={"User-Agent": "ResearchAgent/1.0"})

    # Retry with exponential backoff (1s, 2s, 4s)
    last_error = None
    for attempt, delay in enumerate([1, 2, 4]):
        try:
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            break
        except (URLError, HTTPError, TimeoutError) as e:
            last_error = e
            if attempt < 2:
                time.sleep(delay)
    else:
        return {"source": "Semantic Scholar", "total": 0, "papers": [],
                "error": f"Failed after 3 retries: {last_error}"}

    papers = []
    for p in data.get("data", []):
        ext_ids = p.get("externalIds") or {}
        papers.append({
            "title": p.get("title", ""),
            "year": p.get("year"),
            "citations": p.get("citationCount", 0),
            "abstract": (p.get("abstract") or "")[:300],
            "arxiv_id": ext_ids.get("ArXiv", ""),
            "doi": ext_ids.get("DOI", ""),
        })
    return {"source": "Semantic Scholar", "total": data.get("total", 0), "papers": papers}


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted abstract index."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort()
    return " ".join(w for _, w in word_positions)


def search_openalex(query: str, per_page: int = 5) -> dict:
    """搜尋 OpenAlex (依引用數排序, 含摘要與作者)"""
    q = quote(query)
    url = (f"https://api.openalex.org/works?search={q}&per_page={per_page}"
           f"&sort=cited_by_count:desc&mailto=test@example.com"
           f"&select=id,title,publication_year,cited_by_count,doi,abstract_inverted_index,authorships")
    req = Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    papers = []
    for w in data.get("results", []):
        abstract = _reconstruct_abstract(w.get("abstract_inverted_index"))
        authors = []
        for a in (w.get("authorships") or [])[:5]:
            author_info = a.get("author", {})
            if author_info.get("display_name"):
                authors.append(author_info["display_name"])
        papers.append({
            "title": w.get("title", ""),
            "year": w.get("publication_year"),
            "citations": w.get("cited_by_count", 0),
            "doi": w.get("doi", ""),
            "abstract": abstract[:300],
            "authors": authors,
        })
    return {"source": "OpenAlex", "total": data.get("meta", {}).get("count", 0), "papers": papers}


def fetch_arxiv_by_id(arxiv_id: str) -> dict:
    """直接以 arXiv ID 查詢論文 (e.g. '2101.03961')"""
    # Normalize: strip version suffix and 'arxiv:' prefix
    clean_id = arxiv_id.strip().replace("arxiv:", "").replace("arXiv:", "")
    clean_id = re.sub(r'v\d+$', '', clean_id) if re.search(r'v\d+$', clean_id) else clean_id

    url = f"http://export.arxiv.org/api/query?id_list={quote(clean_id)}&max_results=1"
    req = Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
    with urlopen(req, timeout=15) as resp:
        root = ET.fromstring(resp.read())

    entry = root.find(f"{ATOM_NS}entry")
    if entry is None:
        return {"success": False, "error": f"No paper found for ID: {arxiv_id}"}

    title = entry.findtext(f"{ATOM_NS}title", "").strip().replace("\n", " ")
    summary = entry.findtext(f"{ATOM_NS}summary", "").strip().replace("\n", " ")
    published = entry.findtext(f"{ATOM_NS}published", "")[:10]
    authors = [a.findtext(f"{ATOM_NS}name", "") for a in entry.findall(f"{ATOM_NS}author")]

    categories = []
    for cat in entry.findall(f"{ARXIV_NS}primary_category") + entry.findall(f"{ATOM_NS}category"):
        term = cat.get("term", "")
        if term and term not in categories:
            categories.append(term)

    pdf_url = ""
    for link in entry.findall(f"{ATOM_NS}link"):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")

    return {
        "success": True,
        "title": title,
        "authors": authors,
        "summary": summary,
        "arxiv_id": clean_id,
        "published": published,
        "categories": categories,
        "pdf_url": pdf_url,
    }


def fetch_paper_fulltext(arxiv_id: str, max_chars: int = 30000) -> dict:
    """下載 arXiv PDF 並提取全文。需要 PyMuPDF (pip install pymupdf)。

    Returns: {"success": bool, "arxiv_id": str, "title": str, "fulltext": str,
              "pages": int, "chars": int}
    """
    import tempfile
    from pathlib import Path

    try:
        import fitz  # PyMuPDF
    except ImportError:
        return {"success": False, "error": "PyMuPDF not installed. Run: pip install pymupdf"}

    clean_id = arxiv_id.strip().replace("arxiv:", "").replace("arXiv:", "")
    clean_id = re.sub(r'v\d+$', '', clean_id) if re.search(r'v\d+$', clean_id) else clean_id

    pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
    try:
        req = Request(pdf_url, headers={"User-Agent": "ResearchAgent/1.0"})
        with urlopen(req, timeout=30) as resp:
            pdf_bytes = resp.read()
    except Exception as e:
        return {"success": False, "arxiv_id": clean_id, "error": f"Download failed: {e}"}

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(pdf_bytes)
    tmp.close()

    try:
        doc = fitz.open(tmp.name)
        pages = doc.page_count
        fulltext = ""
        for page in doc:
            fulltext += page.get_text()
        doc.close()

        # Get title from first few lines
        first_lines = fulltext[:500].strip().split("\n")
        title_guess = first_lines[0] if first_lines else ""

        if len(fulltext) > max_chars:
            fulltext = fulltext[:max_chars] + f"\n\n[... truncated at {max_chars} chars, total {len(fulltext)}]"

        return {
            "success": True,
            "arxiv_id": clean_id,
            "title": title_guess,
            "fulltext": fulltext,
            "pages": pages,
            "chars": len(fulltext),
        }
    except Exception as e:
        return {"success": False, "arxiv_id": clean_id, "error": f"PDF parse failed: {e}"}
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def get_citation_graph(paper_id: str, direction: str = "both", limit: int = 10) -> dict:
    """Follow citation chains from a known paper via Semantic Scholar.

    paper_id can be: arXiv ID (e.g. '2106.09685'), DOI, or S2 paper ID.
    direction: 'citations' (papers citing this), 'references' (papers this cites), 'both'.
    Returns related papers with titles, years, citation counts.
    """
    # Normalize paper_id for S2 API
    pid = paper_id.strip()
    if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', pid):
        pid = f"ArXiv:{pid}"
    elif pid.startswith("10."):
        pid = f"DOI:{pid}"
    # else assume it's already an S2 paper ID

    result = {"paper_id": paper_id, "citations": [], "references": []}
    fields = "title,year,citationCount,externalIds,abstract"

    def _fetch_related(endpoint: str, key: str):
        url = (f"https://api.semanticscholar.org/graph/v1/paper/{quote(pid, safe=':')}"
               f"/{endpoint}?fields={fields}&limit={limit}")
        req = Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
        for attempt, delay in enumerate([1, 2, 4]):
            try:
                with urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())
                papers = []
                for item in data.get("data", []):
                    p = item.get("citingPaper" if endpoint == "citations" else "citedPaper", {})
                    if not p or not p.get("title"):
                        continue
                    ext_ids = p.get("externalIds") or {}
                    papers.append({
                        "title": p.get("title", ""),
                        "year": p.get("year"),
                        "citations": p.get("citationCount", 0),
                        "arxiv_id": ext_ids.get("ArXiv", ""),
                        "abstract": (p.get("abstract") or "")[:200],
                    })
                result[key] = papers
                return
            except (URLError, HTTPError, TimeoutError) as e:
                if attempt < 2:
                    time.sleep(delay)
        result[key] = []
        result[f"{key}_error"] = f"Failed after 3 retries"

    if direction in ("citations", "both"):
        _fetch_related("citations", "citations")
    if direction in ("references", "both"):
        _fetch_related("references", "references")

    total = len(result.get("citations", [])) + len(result.get("references", []))
    result["total_found"] = total
    return result


def search_papers_with_code(query: str, max_results: int = 5) -> dict:
    """搜尋 Papers With Code — 返回論文與相關 GitHub 倉庫
    Uses urllib with redirect handling and browser-like headers."""
    from urllib.request import build_opener, HTTPRedirectHandler
    q = quote(query)
    url = f"https://paperswithcode.com/api/v1/papers/?q={q}&items_per_page={max_results}"

    opener = build_opener(HTTPRedirectHandler)
    req = Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Accept": "application/json",
    })

    try:
        with opener.open(req, timeout=15) as resp:
            raw = resp.read()
            content_type = resp.headers.get("Content-Type", "")
            if "json" not in content_type and "text/html" in content_type:
                return {"source": "Papers With Code", "total": 0, "papers": [],
                        "error": "API returned HTML (likely Cloudflare protected)"}
            if not raw:
                return {"source": "Papers With Code", "total": 0, "papers": [],
                        "error": "Empty response from API"}
            data = json.loads(raw)
    except (URLError, HTTPError, json.JSONDecodeError, TimeoutError) as e:
        return {"source": "Papers With Code", "total": 0, "papers": [],
                "error": f"API error: {e}"}

    papers = []
    for p in data.get("results", []):
        paper_id = p.get("id", "")
        paper_info = {
            "title": p.get("title", ""),
            "abstract": (p.get("abstract") or "")[:300],
            "arxiv_id": p.get("arxiv_id", ""),
            "url_abs": p.get("url_abs", ""),
            "url_pdf": p.get("url_pdf", ""),
            "proceeding": p.get("proceeding", ""),
        }

        # Fetch linked repos for this paper
        if paper_id:
            try:
                repo_url = f"https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/"
                repo_req = Request(repo_url, headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                    "Accept": "application/json",
                })
                with opener.open(repo_req, timeout=10) as resp:
                    repo_raw = resp.read()
                    if repo_raw and "json" in resp.headers.get("Content-Type", ""):
                        repo_data = json.loads(repo_raw)
                        repos = []
                        for r in repo_data.get("results", [])[:3]:
                            repos.append({
                                "url": r.get("url", ""),
                                "stars": r.get("stars", 0),
                                "is_official": r.get("is_official", False),
                                "framework": r.get("framework", ""),
                            })
                        paper_info["repos"] = repos
                    else:
                        paper_info["repos"] = []
            except Exception:
                paper_info["repos"] = []

        papers.append(paper_info)

    return {
        "source": "Papers With Code",
        "total": data.get("count", 0),
        "papers": papers,
    }


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
    {
        "type": "function",
        "function": {
            "name": "fetch_arxiv_by_id",
            "description": "Fetch a specific arXiv paper by its ID (e.g. '2101.03961'). Returns full abstract, authors, categories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arXiv paper ID (e.g. '2101.03961' or '2106.09685')"},
                },
                "required": ["arxiv_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_paper_fulltext",
            "description": "Download an arXiv paper PDF and extract full text. Use this to read the complete paper, not just the abstract.",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arXiv paper ID (e.g. '1802.07042')"},
                    "max_chars": {"type": "integer", "description": "Max characters to return (default 30000)", "default": 30000},
                },
                "required": ["arxiv_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_citation_graph",
            "description": "Follow citation chains from a known paper. Given an arXiv ID, DOI, or S2 paper ID, returns papers that cite it AND papers it references. Use this to expand from 3-4 found papers to 8-10+ by following the citation graph of the most relevant paper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "Paper identifier: arXiv ID (e.g. '2106.09685'), DOI, or Semantic Scholar ID"},
                    "direction": {"type": "string", "enum": ["citations", "references", "both"], "description": "Which direction to traverse", "default": "both"},
                    "limit": {"type": "integer", "description": "Max results per direction (1-20)", "default": 10},
                },
                "required": ["paper_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_papers_with_code",
            "description": "Search Papers With Code for papers with linked GitHub repositories. Returns papers + repos with star counts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g. 'LoRA low rank adaptation')"},
                    "max_results": {"type": "integer", "description": "Max results (1-10)", "default": 5},
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
    "fetch_arxiv_by_id": fetch_arxiv_by_id,
    "fetch_paper_fulltext": fetch_paper_fulltext,
    "get_citation_graph": get_citation_graph,
    "search_papers_with_code": search_papers_with_code,
}
