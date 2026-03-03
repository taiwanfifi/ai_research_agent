"""
Meta Skill — Self-Evolution Engine
=====================================
Generates new MCP servers and evolves existing skills based on performance.
This is the self-modifying core of the system.
"""

import importlib.util
import json
import os
import re
import time

from core.llm import MiniMaxClient
from skills.base_skill import Skill


# Template: existing paper_search.py as few-shot example for generating new MCP servers
MCP_TEMPLATE = '''"""
{description}
===========================
Auto-generated MCP server.
"""

import json
from urllib.request import urlopen, Request
from urllib.parse import quote


{functions}


# ── Tool Definitions ─────────────────────────────────────────────────
TOOLS = {tools_json}

TOOL_FUNCTIONS = {tool_functions}
'''

# Banned patterns in generated code
BANNED_PATTERNS = [
    r'import\s+subprocess', r'import\s+shutil', r'os\.system',
    r'os\.remove', r'os\.unlink', r'os\.rmdir', r'eval\(',
    r'exec\(', r'__import__', r'open\(.*(w|a)\)',
    r'import\s+socket', r'import\s+ctypes',
]


class MetaSkill:
    """Generates MCP servers and evolves skills."""

    def __init__(self, llm: MiniMaxClient, generated_dir: str, tool_registry=None):
        self.llm = llm
        self.generated_dir = generated_dir
        self.tool_registry = tool_registry
        os.makedirs(generated_dir, exist_ok=True)

    def generate_mcp_server(self, description: str, tool_specs: list[dict] = None) -> dict:
        """
        Generate a new MCP server module using LLM.

        Args:
            description: What the MCP server should do
            tool_specs: Optional list of tool specs [{"name": ..., "description": ..., "params": ...}]

        Returns:
            {"success": bool, "path": str, "tools": list[str], "error": str}
        """
        prompt = f"""Generate a Python MCP server module for the following purpose:

{description}

{f"Tool specifications: {json.dumps(tool_specs, indent=2)}" if tool_specs else "Design appropriate tool functions based on the description."}

Requirements:
1. Follow this exact structure with TOOLS list and TOOL_FUNCTIONS dict
2. Only use Python standard library (urllib, json, xml.etree, etc.)
3. All API calls should use urllib.request with proper error handling
4. Functions should return dicts (not JSON strings)
5. Include proper docstrings
6. Handle timeouts (15s) and errors gracefully

Example pattern (from paper_search.py):

```python
def search_something(query: str, limit: int = 5) -> dict:
    \"\"\"Search for something.\"\"\"
    q = quote(query)
    url = f"https://api.example.com/search?q={{q}}&limit={{limit}}"
    req = Request(url, headers={{"User-Agent": "ResearchAgent/1.0"}})
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return {{"source": "Example", "results": data}}

TOOLS = [
    {{
        "type": "function",
        "function": {{
            "name": "search_something",
            "description": "Search for something.",
            "parameters": {{
                "type": "object",
                "properties": {{
                    "query": {{"type": "string", "description": "Search query"}},
                    "limit": {{"type": "integer", "description": "Max results", "default": 5}},
                }},
                "required": ["query"],
            }},
        }},
    }},
]

TOOL_FUNCTIONS = {{"search_something": search_something}}
```

Generate ONLY the Python code, no markdown fences or explanations."""

        response = self.llm.chat([
            {"role": "system", "content": "You are a Python code generator. Output only valid Python code."},
            {"role": "user", "content": prompt},
        ])

        code = response["choices"][0]["message"]["content"]

        # Strip markdown code fences if present
        code = re.sub(r'^```python\s*', '', code)
        code = re.sub(r'\s*```$', '', code)

        # Security validation
        errors = self._validate_code(code)
        if errors:
            return {"success": False, "path": "", "tools": [], "error": f"Validation failed: {'; '.join(errors)}"}

        # Write to file
        module_name = re.sub(r'[^a-z0-9_]', '_', description.lower().split('.')[0][:40])
        path = os.path.join(self.generated_dir, f"{module_name}.py")
        with open(path, "w") as f:
            f.write(code)

        # Try to import and register
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            tools = getattr(module, "TOOLS", [])
            tool_functions = getattr(module, "TOOL_FUNCTIONS", {})

            if not tools or not tool_functions:
                os.unlink(path)
                return {"success": False, "path": "", "tools": [],
                        "error": "Module missing TOOLS or TOOL_FUNCTIONS"}

            # Register with tool registry if available
            if self.tool_registry:
                self.tool_registry.register_module(module, source=path)

            tool_names = [t["function"]["name"] for t in tools]
            return {"success": True, "path": path, "tools": tool_names}

        except Exception as e:
            if os.path.exists(path):
                os.unlink(path)
            return {"success": False, "path": "", "tools": [], "error": str(e)}

    def _validate_code(self, code: str) -> list[str]:
        """Validate generated code for safety."""
        errors = []

        # Check for banned patterns
        for pattern in BANNED_PATTERNS:
            if re.search(pattern, code):
                errors.append(f"Banned pattern: {pattern}")

        # Check syntax
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")

        # Check for required structure
        if "TOOLS" not in code:
            errors.append("Missing TOOLS list")
        if "TOOL_FUNCTIONS" not in code:
            errors.append("Missing TOOL_FUNCTIONS dict")

        return errors

    def evolve_skill(self, skill: Skill, performance_log: str = "") -> Skill:
        """
        Evolve a skill based on its performance history.

        Uses LLM to analyze failure patterns and improve the skill's
        prompt, workflow steps, or success criteria.
        """
        prompt = f"""A research skill needs improvement. Analyze its performance and suggest improvements.

Skill: {skill.name} (v{skill.version})
Description: {skill.description}
Current prompt: {skill.prompt[:500]}
Workflow steps: {json.dumps(skill.workflow_steps)}
Success criteria: {skill.success_criteria}

Performance:
- Runs: {skill.runs}, Successes: {skill.successes}, Failures: {skill.failures}
- Success rate: {skill.success_rate():.0%}
- Average time: {skill.avg_elapsed_s:.1f}s
- Recent failures: {json.dumps(skill.failure_log[-5:])}

{f"Additional context: {performance_log}" if performance_log else ""}

Respond with a JSON object containing the improved skill:
{{
  "prompt": "improved system prompt",
  "workflow_steps": ["step 1", "step 2", ...],
  "success_criteria": "updated criteria",
  "changes_made": "brief description of what was changed and why"
}}"""

        response = self.llm.chat([
            {"role": "system", "content": "You improve AI agent skills. Respond with JSON only."},
            {"role": "user", "content": prompt},
        ])

        content = response["choices"][0]["message"]["content"]
        json_match = re.search(r'\{[\s\S]*\}', content)
        if not json_match:
            return skill

        try:
            improvements = json.loads(json_match.group())
        except json.JSONDecodeError:
            return skill

        # Create evolved version
        evolved = Skill(
            name=skill.name,
            version=skill.version + 1,
            description=skill.description,
            prompt=improvements.get("prompt", skill.prompt),
            tools=skill.tools,
            workflow_steps=improvements.get("workflow_steps", skill.workflow_steps),
            success_criteria=improvements.get("success_criteria", skill.success_criteria),
            worker_type=skill.worker_type,
        )

        print(f"  [MetaSkill] Evolved {skill.name} v{skill.version} → v{evolved.version}")
        if "changes_made" in improvements:
            print(f"  [MetaSkill] Changes: {improvements['changes_made']}")

        return evolved
