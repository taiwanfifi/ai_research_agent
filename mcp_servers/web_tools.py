"""
Web Tools — Web search and URL fetching for research
=====================================================
Provides:
- web_search: Search the web via DuckDuckGo (no API key needed)
- web_fetch: Fetch and extract readable text from any URL

Adapted from Environment/external_tools.py for Opus.
Dependencies: httpx, beautifulsoup4 (optional: curl_cffi for anti-bot bypass)
"""

import json
from urllib.parse import quote_plus, urljoin, urlparse, parse_qs

try:
    import httpx
except ImportError:
    httpx = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
}


def web_search(query: str, max_results: int = 8) -> dict:
    """Search the web using DuckDuckGo HTML (no API key needed).

    Returns: {"success": bool, "query": str, "results": [{"title", "url", "snippet"}]}
    """
    if not BeautifulSoup:
        return {"success": False, "query": query, "error": "beautifulsoup4 not installed"}

    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        if HAS_CURL_CFFI:
            resp = cffi_requests.get(search_url, headers=_BROWSER_HEADERS,
                                     impersonate="chrome", timeout=15)
            html = resp.text
        elif httpx:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                resp = client.get(search_url, headers=_BROWSER_HEADERS)
                html = resp.text
        else:
            from urllib.request import urlopen, Request
            req = Request(search_url, headers=_BROWSER_HEADERS)
            with urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")

        soup = BeautifulSoup(html, "html.parser")
        results = []
        for item in soup.select(".result")[:max_results]:
            title_tag = item.select_one(".result__a")
            snippet_tag = item.select_one(".result__snippet")
            if title_tag:
                href = title_tag.get("href", "")
                if "uddg=" in href:
                    parsed = urlparse(href)
                    qs = parse_qs(parsed.query)
                    href = qs.get("uddg", [href])[0]
                results.append({
                    "title": title_tag.get_text(strip=True),
                    "url": href,
                    "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "",
                })
        return {"success": True, "query": query, "results": results}
    except Exception as e:
        return {"success": False, "query": query, "error": str(e)}


def search_google_scholar(query: str, max_results: int = 10) -> dict:
    """Search Google Scholar for academic papers (citation-ranked).

    Returns: {"success": bool, "query": str, "papers": [{"title", "url", "snippet", "cited_by", "year"}]}
    """
    if not BeautifulSoup:
        return {"success": False, "query": query, "error": "beautifulsoup4 not installed"}

    search_url = f"https://scholar.google.com/scholar?q={quote_plus(query)}&hl=en&num={max_results}"
    try:
        headers = {**_BROWSER_HEADERS, "Referer": "https://scholar.google.com/"}
        if HAS_CURL_CFFI:
            resp = cffi_requests.get(search_url, headers=headers,
                                     impersonate="chrome", timeout=15)
            html = resp.text
            status = resp.status_code
        elif httpx:
            with httpx.Client(timeout=15, follow_redirects=True) as client:
                resp = client.get(search_url, headers=headers)
                html = resp.text
                status = resp.status_code
        else:
            from urllib.request import urlopen, Request
            req = Request(search_url, headers=headers)
            with urlopen(req, timeout=15) as resp_obj:
                html = resp_obj.read().decode("utf-8", errors="replace")
                status = resp_obj.status

        if status >= 400:
            return {"success": False, "query": query, "error": f"HTTP {status} — Scholar may be rate-limiting"}

        soup = BeautifulSoup(html, "html.parser")
        papers = []

        for item in soup.select(".gs_r.gs_or.gs_scl")[:max_results]:
            title_tag = item.select_one(".gs_rt a")
            if not title_tag:
                title_tag = item.select_one(".gs_rt")
            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            url = title_tag.get("href", "") if title_tag.name == "a" else ""

            # Snippet / abstract
            snippet_tag = item.select_one(".gs_rs")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            # Author/venue/year line
            info_tag = item.select_one(".gs_a")
            info_text = info_tag.get_text(strip=True) if info_tag else ""

            # Extract year from info
            import re as _re
            year_match = _re.search(r'\b(19|20)\d{2}\b', info_text)
            year = year_match.group() if year_match else ""

            # Citation count
            cited_by = 0
            cite_tag = item.select_one(".gs_fl a")
            if cite_tag:
                for link in item.select(".gs_fl a"):
                    link_text = link.get_text(strip=True)
                    if "Cited by" in link_text:
                        cite_match = _re.search(r'Cited by (\d+)', link_text)
                        if cite_match:
                            cited_by = int(cite_match.group(1))
                        break

            papers.append({
                "title": title,
                "url": url,
                "snippet": snippet,
                "authors_venue": info_text,
                "year": year,
                "cited_by": cited_by,
            })

        if not papers and "unusual traffic" in html.lower():
            return {"success": False, "query": query,
                    "error": "Google Scholar CAPTCHA — too many requests. Use web_search as fallback."}

        return {"success": True, "query": query, "papers": papers}
    except Exception as e:
        return {"success": False, "query": query, "error": str(e)}


def web_fetch(url: str, max_chars: int = 8000) -> dict:
    """Fetch a URL and extract readable text content.

    Returns: {"success": bool, "url": str, "title": str, "text": str, "length": int}
    """
    if not BeautifulSoup:
        return {"success": False, "url": url, "error": "beautifulsoup4 not installed"}

    try:
        if HAS_CURL_CFFI:
            resp = cffi_requests.get(url, headers=_BROWSER_HEADERS,
                                     impersonate="chrome", timeout=20,
                                     allow_redirects=True)
            html = resp.text
            status = resp.status_code
        elif httpx:
            with httpx.Client(timeout=20, follow_redirects=True) as client:
                resp = client.get(url, headers=_BROWSER_HEADERS)
                html = resp.text
                status = resp.status_code
        else:
            from urllib.request import urlopen, Request
            req = Request(url, headers=_BROWSER_HEADERS)
            with urlopen(req, timeout=20) as resp:
                html = resp.read().decode("utf-8", errors="replace")
                status = resp.status

        if status >= 400:
            return {"success": False, "url": url, "error": f"HTTP {status}"}

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        main = soup.find("article") or soup.find("main") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated, {len(text)} total chars]"

        return {"success": True, "url": url, "title": title, "text": text, "length": len(text)}
    except Exception as e:
        return {"success": False, "url": url, "error": str(e)}


# ── Tool definitions for registry ────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_google_scholar",
            "description": "Search Google Scholar for academic papers. Returns citation-ranked results with titles, snippets, citation counts, and years. Use this FIRST before arxiv for broader academic coverage. Falls back gracefully if rate-limited.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Academic search query (e.g. 'batch normalization transformer training stability')"},
                    "max_results": {"type": "integer", "description": "Max results (1-15)", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Use this to find blog posts, tutorials, discussions, and resources not available through academic paper databases. Good for finding practical implementations, community benchmarks, and recent developments.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g. 'DropConnect implementation PyTorch benchmark')"},
                    "max_results": {"type": "integer", "description": "Max results (1-15)", "default": 8},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch and extract readable text from any URL. Use this to read blog posts, documentation pages, or web articles found via web_search. Returns clean text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Max characters to return", "default": 8000},
                },
                "required": ["url"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_google_scholar": search_google_scholar,
    "web_search": web_search,
    "web_fetch": web_fetch,
}
