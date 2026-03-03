"""
Tests for workspace scoping — mission-isolated file I/O
=========================================================
Tests that code_runner tools are properly scoped to mission
workspace, and that CodeVersionStore tracks the same files.

Run: python3 -m pytest tests/test_workspace_scoping.py -v
  or: python3 tests/test_workspace_scoping.py
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_servers.code_runner import create_workspace_tools, _DEFAULT_WORKSPACE
from core.code_store import CodeVersionStore
from core.tool_registry import ToolRegistry


class TestCreateWorkspaceTools(unittest.TestCase):
    """Factory function produces correctly scoped closures."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ws = os.path.join(self.tmpdir, "mission_ws")
        self.tools = create_workspace_tools(self.ws)

    def test_returns_three_tools(self):
        self.assertIn("write_file", self.tools)
        self.assertIn("read_file", self.tools)
        self.assertIn("run_python_code", self.tools)

    def test_write_file_to_scoped_workspace(self):
        result = self.tools["write_file"]("test.py", "x = 1")
        self.assertTrue(result["success"])
        self.assertTrue(result["path"].startswith(self.ws))
        self.assertTrue(os.path.exists(result["path"]))

    def test_read_file_from_scoped_workspace(self):
        self.tools["write_file"]("data.txt", "hello world")
        result = self.tools["read_file"]("data.txt")
        self.assertTrue(result["success"])
        self.assertEqual(result["content"], "hello world")

    def test_read_nonexistent_file(self):
        result = self.tools["read_file"]("nope.txt")
        self.assertFalse(result["success"])

    def test_run_python_code_cwd_is_workspace(self):
        result = self.tools["run_python_code"]("import os; print(os.getcwd())")
        self.assertTrue(result["success"])
        self.assertIn(self.ws, result["stdout"])

    def test_run_python_code_can_import_written_file(self):
        self.tools["write_file"]("mymod.py", "VALUE = 42")
        result = self.tools["run_python_code"]("from mymod import VALUE; print(VALUE)")
        self.assertTrue(result["success"])
        self.assertIn("42", result["stdout"])

    def test_run_python_code_timeout(self):
        result = self.tools["run_python_code"]("import time; time.sleep(10)", timeout=1)
        self.assertFalse(result["success"])
        self.assertIn("Timeout", result["stderr"])

    def test_path_traversal_blocked(self):
        """write_file should use basename only, blocking path traversal."""
        result = self.tools["write_file"]("../../etc/passwd", "hacked")
        self.assertTrue(result["success"])
        # Should write to workspace/passwd, not ../../etc/passwd
        self.assertTrue(result["path"].startswith(self.ws))
        self.assertTrue(result["path"].endswith("passwd"))

    def test_two_workspaces_isolated(self):
        """Two different mission workspaces don't interfere."""
        ws2 = os.path.join(self.tmpdir, "mission_ws_2")
        tools2 = create_workspace_tools(ws2)

        self.tools["write_file"]("shared_name.py", "content_A")
        tools2["write_file"]("shared_name.py", "content_B")

        result_a = self.tools["read_file"]("shared_name.py")
        result_b = tools2["read_file"]("shared_name.py")

        self.assertEqual(result_a["content"], "content_A")
        self.assertEqual(result_b["content"], "content_B")


class TestRegistryScoping(unittest.TestCase):
    """Tool functions are properly replaced in ToolRegistry."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ws = os.path.join(self.tmpdir, "workspace")

    def test_scoped_tools_override_defaults(self):
        from mcp_servers import code_runner

        registry = ToolRegistry()
        registry.register_module(code_runner)

        # Before scoping: source is the module name
        info = {i["name"]: i for i in registry.get_info()}
        self.assertNotIn("@mission", info["write_file"]["source"])

        # Scope to mission workspace
        scoped = code_runner.create_workspace_tools(self.ws)
        for tool_def in code_runner.TOOLS:
            name = tool_def["function"]["name"]
            if name in scoped:
                registry.register(tool_def, scoped[name],
                                  source=f"code_runner@mission_test")

        # After scoping: source updated
        info = {i["name"]: i for i in registry.get_info()}
        self.assertIn("@mission", info["write_file"]["source"])

        # Execution goes to scoped workspace
        result = json.loads(
            registry.execute("write_file", {"filename": "t.py", "content": "pass"})
        )
        self.assertTrue(result["success"])
        self.assertIn(self.ws, result["path"])

    def test_non_code_tools_preserved(self):
        from mcp_servers import code_runner

        registry = ToolRegistry()
        registry.register_module(code_runner)

        scoped = code_runner.create_workspace_tools(self.ws)
        for tool_def in code_runner.TOOLS:
            name = tool_def["function"]["name"]
            if name in scoped:
                registry.register(tool_def, scoped[name], source="scoped")

        # pip_install and detect_hardware should still be registered
        self.assertIn("pip_install", registry)
        self.assertIn("detect_hardware", registry)


class TestCodeStoreAlignment(unittest.TestCase):
    """CodeVersionStore and scoped write_file operate on the same directory."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ws = os.path.join(self.tmpdir, "workspace")
        self.tools = create_workspace_tools(self.ws)
        self.code_store = CodeVersionStore(self.ws)

    def test_write_and_track_same_directory(self):
        # write_file writes to workspace
        result = self.tools["write_file"]("algo.py", "def sort(arr): return sorted(arr)")
        actual_path = result["path"]

        # code_store tracks in same workspace
        self.code_store.track_write("algo.py", "def sort(arr): return sorted(arr)")

        # Both should reference the same workspace
        self.assertTrue(actual_path.startswith(self.ws))
        store_dir = os.path.join(self.ws, ".code_store", "algo")
        self.assertTrue(os.path.isdir(store_dir))

    def test_full_coder_simulation(self):
        """Simulate the exact tracked_executor flow from coder.py."""
        from mcp_servers import code_runner

        registry = ToolRegistry()
        registry.register_module(code_runner)

        # Scope
        scoped = code_runner.create_workspace_tools(self.ws)
        for tool_def in code_runner.TOOLS:
            name = tool_def["function"]["name"]
            if name in scoped:
                registry.register(tool_def, scoped[name], source="scoped")

        code_store = CodeVersionStore(self.ws)
        code_store.set_current_cycle(5)
        base_executor = registry.execute

        # Build tracked executor (same as coder.py does)
        def tracked_executor(func_name, func_args):
            result = base_executor(func_name, func_args)
            if func_name == "write_file":
                try:
                    filename = func_args.get("filename", "")
                    content = func_args.get("content", "")
                    if filename and content:
                        code_store.track_write(filename, content, reason="coder")
                except Exception:
                    pass
            return result

        # Execute
        result_json = tracked_executor("write_file", {
            "filename": "model.py",
            "content": "class Net: pass",
        })
        result = json.loads(result_json)

        # Verify alignment
        self.assertTrue(result["success"])
        self.assertIn(self.ws, result["path"])

        history = code_store.get_history("model.py")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["cycle"], 5)

        # Code can be read back
        read_result = json.loads(
            registry.execute("read_file", {"filename": "model.py"})
        )
        self.assertIn("Net", read_result["content"])

        # Code can be executed
        run_result = json.loads(
            registry.execute("run_python_code", {
                "code": "from model import Net; print(Net)"
            })
        )
        self.assertTrue(run_result["success"])


if __name__ == "__main__":
    unittest.main()
