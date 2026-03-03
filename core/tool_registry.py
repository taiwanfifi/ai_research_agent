"""
Dynamic Tool Registry
======================
Manages tool definitions and implementations with hot-reload support.
Dynamic tool management with hot-reload support.
"""

import importlib
import importlib.util
import json
import os
from typing import Callable


class ToolRegistry:
    """Dynamic tool registry with registration, lookup, and hot-reload."""

    def __init__(self):
        self._tools: list[dict] = []          # OpenAI-format tool definitions
        self._functions: dict[str, Callable] = {}  # name -> callable
        self._sources: dict[str, str] = {}    # name -> source module/path

    @property
    def tools(self) -> list[dict]:
        """All registered tool definitions."""
        return list(self._tools)

    @property
    def functions(self) -> dict[str, Callable]:
        """All registered tool functions."""
        return dict(self._functions)

    def register(self, tool_def: dict, func: Callable, source: str = "manual"):
        """Register a single tool definition + implementation."""
        name = tool_def["function"]["name"]
        # Update if already registered
        self._tools = [t for t in self._tools if t["function"]["name"] != name]
        self._tools.append(tool_def)
        self._functions[name] = func
        self._sources[name] = source

    def register_module(self, module, source: str = None):
        """Register all tools from a module with TOOLS and TOOL_FUNCTIONS attributes."""
        tools = getattr(module, "TOOLS", [])
        functions = getattr(module, "TOOL_FUNCTIONS", {})
        src = source or getattr(module, "__name__", "unknown")
        for tool_def in tools:
            name = tool_def["function"]["name"]
            if name in functions:
                self.register(tool_def, functions[name], source=src)

    def register_from_path(self, path: str):
        """Hot-load a Python module from file path and register its tools."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Module not found: {path}")

        module_name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.register_module(module, source=path)

    def unregister(self, name: str):
        """Remove a tool by name."""
        self._tools = [t for t in self._tools if t["function"]["name"] != name]
        self._functions.pop(name, None)
        self._sources.pop(name, None)

    def execute(self, name: str, arguments: dict) -> str:
        """Execute a tool by name and return JSON result string."""
        func = self._functions.get(name)
        if not func:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            result = func(**arguments)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def list_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._functions.keys())

    def get_info(self) -> list[dict]:
        """Get info about all registered tools."""
        return [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "source": self._sources.get(t["function"]["name"], "unknown"),
            }
            for t in self._tools
        ]

    def load_builtin_servers(self, mcp_dir: str):
        """Load all MCP server modules from a directory."""
        if not os.path.isdir(mcp_dir):
            return
        for fname in sorted(os.listdir(mcp_dir)):
            if fname.endswith(".py") and not fname.startswith("_"):
                self.register_from_path(os.path.join(mcp_dir, fname))

    def __len__(self):
        return len(self._functions)

    def __contains__(self, name: str):
        return name in self._functions
