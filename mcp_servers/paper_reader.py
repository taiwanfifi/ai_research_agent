"""
Paper Reader — Deep Reading for Research Papers
=================================================
Fetches and extracts structured content from academic papers.
Goes beyond abstracts: reads methodology, experiments, results.

Approach:
1. arXiv papers: Use ar5iv.labs.arxiv.org (HTML version, no PDF needed)
2. Extract by section: abstract, methodology, experiments, results, conclusion
3. Compress with LLM for workers that need deep understanding

This is what transforms Opus from "reads abstracts" to "reads papers."
"""

import re
import httpx
from bs4 import BeautifulSoup


# ── Tool definitions for ToolRegistry ──────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_paper",
            "description": (
                "Read a full academic paper from arXiv. Extracts structured sections "
                "(abstract, methodology, experiments, results, conclusion). "
                "Use this when you need to deeply understand a paper's approach, "
                "not just its abstract. Input: arXiv ID (e.g. '1802.07042') or full URL."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv paper ID (e.g. '1802.07042') or full URL",
                    },
                    "sections": {
                        "type": "string",
                        "description": "Comma-separated sections to extract: all, abstract, methodology, experiments, results, conclusion. Default: all",
                    },
                },
                "required": ["arxiv_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_paper_details",
            "description": (
                "Extract specific details from a paper: experimental setup, "
                "hyperparameters, datasets used, baselines compared, key findings. "
                "Use this after read_paper to get structured research-relevant info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_id": {
                        "type": "string",
                        "description": "arXiv paper ID",
                    },
                    "focus": {
                        "type": "string",
                        "description": "What to extract: 'setup' (hyperparams, datasets), 'baselines' (what they compared against), 'findings' (key results + numbers), 'limitations' (acknowledged weaknesses)",
                    },
                },
                "required": ["arxiv_id", "focus"],
            },
        },
    },
]


# ── Implementation ─────────────────────────────────────────────────

def _normalize_arxiv_id(arxiv_id: str) -> str:
    """Extract clean arXiv ID from various input formats."""
    arxiv_id = arxiv_id.strip()
    # Handle full URLs
    for prefix in ["https://arxiv.org/abs/", "https://arxiv.org/pdf/",
                    "http://arxiv.org/abs/", "http://arxiv.org/pdf/",
                    "https://ar5iv.labs.arxiv.org/html/"]:
        if arxiv_id.startswith(prefix):
            arxiv_id = arxiv_id[len(prefix):]
    # Remove version suffix
    arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
    # Remove trailing slashes or .pdf
    arxiv_id = arxiv_id.rstrip('/').replace('.pdf', '')
    return arxiv_id


def _fetch_html(arxiv_id: str) -> str:
    """Fetch HTML version of paper from ar5iv."""
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    try:
        resp = httpx.get(url, timeout=20, follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # Some papers don't have ar5iv versions yet
            return ""
        raise


def _extract_sections(html: str) -> dict:
    """Extract paper sections from ar5iv HTML."""
    soup = BeautifulSoup(html, "html.parser")
    sections = {}

    # Title
    title_tag = soup.find("h1", class_="ltx_title")
    if title_tag:
        sections["title"] = title_tag.get_text(strip=True)

    # Abstract
    abstract_tag = soup.find("div", class_="ltx_abstract")
    if abstract_tag:
        sections["abstract"] = abstract_tag.get_text(strip=True)
        # Remove "Abstract" prefix
        if sections["abstract"].lower().startswith("abstract"):
            sections["abstract"] = sections["abstract"][8:].strip()

    # Find all section headers and their content
    all_sections = soup.find_all(["h2", "h3"])
    for i, header in enumerate(all_sections):
        header_text = header.get_text(strip=True).lower()
        # Remove numbering prefix (e.g., "2.1" "3.")
        clean_header = re.sub(r'^[\d.]+\s*', '', header_text)

        # Collect content until next header
        content_parts = []
        sibling = header.find_next_sibling()
        while sibling and sibling.name not in ["h2", "h3"]:
            text = sibling.get_text(strip=True)
            if text and len(text) > 20:  # Skip tiny fragments
                content_parts.append(text)
            sibling = sibling.find_next_sibling()

        content = "\n".join(content_parts)
        if not content:
            continue

        # Categorize section
        if any(kw in clean_header for kw in ["method", "approach", "model", "architecture", "framework"]):
            sections.setdefault("methodology", "")
            sections["methodology"] += f"\n### {header.get_text(strip=True)}\n{content}\n"
        elif any(kw in clean_header for kw in ["experiment", "setup", "training", "implementation"]):
            sections.setdefault("experiments", "")
            sections["experiments"] += f"\n### {header.get_text(strip=True)}\n{content}\n"
        elif any(kw in clean_header for kw in ["result", "discussion", "analysis", "evaluation", "performance"]):
            sections.setdefault("results", "")
            sections["results"] += f"\n### {header.get_text(strip=True)}\n{content}\n"
        elif any(kw in clean_header for kw in ["conclusion", "summary", "future"]):
            sections.setdefault("conclusion", "")
            sections["conclusion"] += f"\n### {header.get_text(strip=True)}\n{content}\n"
        elif any(kw in clean_header for kw in ["related", "background", "prior", "literature"]):
            sections.setdefault("related_work", "")
            sections["related_work"] += f"\n### {header.get_text(strip=True)}\n{content}\n"
        elif any(kw in clean_header for kw in ["intro"]):
            sections.setdefault("introduction", "")
            sections["introduction"] += f"\n### {header.get_text(strip=True)}\n{content}\n"
        else:
            sections.setdefault("other", "")
            sections["other"] += f"\n### {header.get_text(strip=True)}\n{content}\n"

    return sections


def _truncate_section(text: str, max_chars: int = 3000) -> str:
    """Truncate section text while preserving sentence boundaries."""
    if len(text) <= max_chars:
        return text
    # Find last sentence boundary before limit
    truncated = text[:max_chars]
    last_period = truncated.rfind('.')
    if last_period > max_chars * 0.7:
        truncated = truncated[:last_period + 1]
    return truncated + f"\n[... truncated, {len(text)} total chars]"


def read_paper(arxiv_id: str, sections: str = "all") -> str:
    """Read and extract structured content from an arXiv paper."""
    clean_id = _normalize_arxiv_id(arxiv_id)

    html = _fetch_html(clean_id)
    if not html:
        return f"Paper {clean_id} not available on ar5iv. Try fetching the abstract URL directly: https://arxiv.org/abs/{clean_id}"

    extracted = _extract_sections(html)
    if not extracted:
        return f"Could not extract sections from paper {clean_id}. The HTML may have an unusual structure."

    # Filter requested sections
    requested = [s.strip().lower() for s in sections.split(",")]
    if "all" in requested:
        requested = ["title", "abstract", "introduction", "methodology",
                      "experiments", "results", "conclusion"]

    output_parts = [f"# Paper: {clean_id}"]
    if "title" in extracted:
        output_parts.append(f"**{extracted['title']}**\n")

    for section_name in requested:
        if section_name in extracted:
            content = _truncate_section(extracted[section_name])
            output_parts.append(f"## {section_name.title()}\n{content}")

    result = "\n\n".join(output_parts)

    # Add available sections info
    available = [k for k in extracted.keys() if k != "title"]
    output_parts.append(f"\n---\nAvailable sections: {', '.join(available)}")

    return "\n\n".join(output_parts)


def extract_paper_details(arxiv_id: str, focus: str = "setup") -> str:
    """Extract specific structured details from a paper."""
    clean_id = _normalize_arxiv_id(arxiv_id)

    html = _fetch_html(clean_id)
    if not html:
        return f"Paper {clean_id} not available on ar5iv."

    extracted = _extract_sections(html)

    if focus == "setup":
        # Extract experimental setup: hyperparameters, datasets, hardware
        relevant = ""
        for key in ["experiments", "methodology", "other"]:
            if key in extracted:
                relevant += extracted[key]
        return _extract_setup_info(relevant, extracted.get("title", ""))

    elif focus == "baselines":
        # Extract what methods/baselines were compared
        relevant = ""
        for key in ["experiments", "results", "introduction"]:
            if key in extracted:
                relevant += extracted[key]
        return _extract_baselines(relevant, extracted.get("title", ""))

    elif focus == "findings":
        # Extract key numerical results
        relevant = ""
        for key in ["results", "conclusion", "abstract"]:
            if key in extracted:
                relevant += extracted[key]
        return _extract_findings(relevant, extracted.get("title", ""))

    elif focus == "limitations":
        relevant = ""
        for key in ["conclusion", "results", "other"]:
            if key in extracted:
                relevant += extracted[key]
        return _extract_limitations(relevant, extracted.get("title", ""))

    return f"Unknown focus: {focus}. Use: setup, baselines, findings, limitations"


def _extract_setup_info(text: str, title: str) -> str:
    """Extract experimental setup details."""
    lines = [f"# Experimental Setup — {title}"]

    # Find dataset mentions
    datasets = set()
    for pattern in [r'(?:CIFAR|ImageNet|MNIST|SST|GLUE|SQuAD|WikiText|Penn Treebank|PTB|SVHN|STL)[\w-]*',
                    r'(?:dataset|benchmark|corpus)\s*:?\s*(\w[\w\s-]{2,30})']:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            datasets.add(match.group().strip())
    if datasets:
        lines.append(f"\n**Datasets**: {', '.join(datasets)}")

    # Find hyperparameters
    hyperparams = []
    for pattern in [r'(?:learning rate|lr)\s*[=:]\s*[\d.e-]+',
                    r'(?:batch size|batch_size)\s*[=:]\s*\d+',
                    r'(?:epochs?|iterations?)\s*[=:]\s*\d+',
                    r'(?:dropout|weight.?decay|momentum)\s*[=:]\s*[\d.]+',
                    r'(?:hidden.?size|embed.?dim)\s*[=:]\s*\d+']:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            hyperparams.append(match.group())
    if hyperparams:
        lines.append(f"\n**Hyperparameters**:")
        for hp in hyperparams[:15]:
            lines.append(f"  - {hp}")

    # Extract relevant text chunks (max 2000 chars)
    lines.append(f"\n**Raw setup text** (first 2000 chars):\n{text[:2000]}")

    return "\n".join(lines)


def _extract_baselines(text: str, title: str) -> str:
    """Extract baseline/comparison methods."""
    lines = [f"# Baselines & Comparisons — {title}"]

    # Look for method names in comparison context
    methods = set()
    for pattern in [r'(?:compared?\s+(?:to|with|against))\s+([\w\s,+-]+?)(?:\.|,|\()',
                    r'(?:baseline|benchmark)s?\s*:?\s*([\w\s,+-]+?)(?:\.|,|\()',
                    r'(?:outperform|surpass|exceed)s?\s+([\w\s,+-]+?)(?:\.|,)']:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            methods.add(match.group(1).strip()[:50])
    if methods:
        lines.append(f"\n**Methods compared**: {'; '.join(methods)}")

    lines.append(f"\n**Comparison context** (first 2000 chars):\n{text[:2000]}")
    return "\n".join(lines)


def _extract_findings(text: str, title: str) -> str:
    """Extract key numerical results."""
    lines = [f"# Key Findings — {title}"]

    # Extract metric-value pairs
    metrics = []
    for match in re.finditer(r'(\w[\w\s-]{1,30}?)\s*(?:of|=|:|\bis\b)\s*([\d.]+)\s*(%|\\%)?', text):
        name = match.group(1).strip()
        value = match.group(2)
        pct = match.group(3) or ""
        if name.lower() not in {"of", "is", "a", "the", "in", "to", "and", "with"}:
            metrics.append(f"  - {name}: {value}{pct}")
    if metrics:
        lines.append(f"\n**Reported metrics**:")
        for m in metrics[:20]:
            lines.append(m)

    lines.append(f"\n**Results text** (first 2000 chars):\n{text[:2000]}")
    return "\n".join(lines)


def _extract_limitations(text: str, title: str) -> str:
    """Extract limitations and future work."""
    lines = [f"# Limitations — {title}"]
    lines.append(f"\n**Conclusion/limitations text** (first 2000 chars):\n{text[:2000]}")
    return "\n".join(lines)


# ── Registration ───────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "read_paper": read_paper,
    "extract_paper_details": extract_paper_details,
}
