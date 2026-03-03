"""
資料集 / 題目 MCP Server (模擬)
================================
提供以下 tools 給 Agent 呼叫：
- search_hf_datasets: 搜尋 Hugging Face 資料集
- fetch_leetcode_problem: 取得 LeetCode 題目
- fetch_humaneval: 取得 HumanEval benchmark 題目
"""

import json
from urllib.request import urlopen, Request
from urllib.parse import quote


def search_hf_datasets(query: str, limit: int = 5) -> dict:
    """搜尋 Hugging Face 資料集"""
    q = quote(query)
    url = f"https://huggingface.co/api/datasets?search={q}&sort=downloads&direction=-1&limit={limit}"
    req = Request(url, headers={"User-Agent": "ResearchAgent/1.0"})
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    datasets = []
    for d in data:
        datasets.append({
            "id": d.get("id", ""),
            "downloads": d.get("downloads", 0),
            "tags": d.get("tags", [])[:5],
            "description": d.get("description", "")[:200],
        })
    return {"source": "Hugging Face", "count": len(datasets), "datasets": datasets}


def fetch_leetcode_problem(slug: str = "two-sum") -> dict:
    """取得 LeetCode 題目 (透過公開 GraphQL API)"""
    url = "https://leetcode.com/graphql"
    query_body = json.dumps({
        "query": """query questionData($titleSlug: String!) {
            question(titleSlug: $titleSlug) {
                questionId title difficulty content
                topicTags { name }
                codeSnippets { lang code }
            }
        }""",
        "variables": {"titleSlug": slug},
    }).encode()
    req = Request(url, data=query_body, headers={
        "Content-Type": "application/json",
        "User-Agent": "ResearchAgent/1.0",
    })
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())

    q = data.get("data", {}).get("question", {})
    if not q:
        return {"error": f"Problem '{slug}' not found"}

    # 取得 Python3 code snippet
    python_snippet = ""
    for s in q.get("codeSnippets", []):
        if s.get("lang") == "Python3":
            python_snippet = s.get("code", "")
            break

    return {
        "source": "LeetCode",
        "id": q.get("questionId"),
        "title": q.get("title"),
        "difficulty": q.get("difficulty"),
        "tags": [t["name"] for t in q.get("topicTags", [])],
        "python_template": python_snippet,
        "description_html": (q.get("content") or "")[:500],
    }


def fetch_humaneval_sample() -> dict:
    """取得 HumanEval benchmark 範例題目 (OpenAI 的 code generation benchmark)"""
    # HumanEval 在 GitHub 上公開，這裡提供靜態範例
    return {
        "source": "HumanEval",
        "description": "OpenAI HumanEval benchmark - 164 Python function completion tasks",
        "download_url": "https://huggingface.co/datasets/openai/openai_humaneval",
        "sample_problem": {
            "task_id": "HumanEval/0",
            "prompt": 'from typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',
            "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n",
            "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n",
        },
    }


# ── Tool 定義 ─────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_hf_datasets",
            "description": "Search Hugging Face for ML datasets. Returns dataset IDs, download counts, and tags.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g., 'question answering')"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_leetcode_problem",
            "description": "Fetch a LeetCode problem by slug. Returns problem description, difficulty, tags, and Python template.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slug": {"type": "string", "description": "Problem slug (e.g., 'two-sum', 'reverse-linked-list')"},
                },
                "required": ["slug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_humaneval_sample",
            "description": "Get a sample from OpenAI's HumanEval benchmark (164 Python coding tasks for evaluating code generation).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_FUNCTIONS = {
    "search_hf_datasets": search_hf_datasets,
    "fetch_leetcode_problem": fetch_leetcode_problem,
    "fetch_humaneval_sample": fetch_humaneval_sample,
}
