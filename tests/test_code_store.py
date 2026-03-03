"""
Tests for core.code_store — CodeVersionStore
=============================================
Run: python3 -m pytest tests/test_code_store.py -v
  or: python3 tests/test_code_store.py
"""

import json
import os
import tempfile
import unittest

# Ensure imports work from project root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.code_store import CodeVersionStore, ModuleInfo


class TestTrackWrite(unittest.TestCase):
    """Version snapshot, diff, and manifest creation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ws = os.path.join(self.tmpdir, "workspace")
        self.store = CodeVersionStore(self.ws)

    def test_first_write_creates_v001(self):
        self.store.track_write("model.py", "x = 1", reason="init")
        h = self.store.get_history("model.py")
        self.assertEqual(len(h), 1)
        self.assertEqual(h[0]["version"], "v001")
        self.assertEqual(h[0]["reason"], "init")
        self.assertFalse(h[0]["has_diff"])

    def test_second_write_creates_diff(self):
        self.store.track_write("model.py", "x = 1")
        self.store.track_write("model.py", "x = 2")
        h = self.store.get_history("model.py")
        self.assertEqual(len(h), 2)
        self.assertEqual(h[1]["version"], "v002")
        self.assertTrue(h[1]["has_diff"])

        # Diff file exists
        diff_path = os.path.join(
            self.ws, ".code_store", "model", "v001_v002.diff"
        )
        self.assertTrue(os.path.exists(diff_path))

    def test_snapshot_files_exist(self):
        self.store.track_write("solver.py", "def solve(): pass")
        self.store.track_write("solver.py", "def solve(x): return x")
        d = os.path.join(self.ws, ".code_store", "solver")
        self.assertTrue(os.path.exists(os.path.join(d, "v001.py")))
        self.assertTrue(os.path.exists(os.path.join(d, "v002.py")))

    def test_best_effort_never_raises(self):
        """track_write should never raise, even with weird input."""
        # Empty content
        self.store.track_write("", "")
        # None-like
        self.store.track_write("x.py", "valid")  # just to ensure no crash


class TestModuleMap(unittest.TestCase):
    """AST parsing into module maps."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CodeVersionStore(os.path.join(self.tmpdir, "ws"))

    def test_parses_functions_and_classes(self):
        code = '''
import os

def hello(name: str) -> str:
    """Greet someone."""
    return f"Hello {name}"

class Model:
    """A model."""
    def forward(self, x):
        return x
'''
        self.store.track_write("example.py", code)
        map_path = os.path.join(
            self.store.store_dir, "example", "module_map.json"
        )
        with open(map_path) as f:
            modules = json.load(f)

        names = [m["name"] for m in modules]
        self.assertIn("hello", names)
        self.assertIn("Model", names)
        self.assertIn("top_level", names)

        # Check function signature
        hello = next(m for m in modules if m["name"] == "hello")
        self.assertIn("name: str", hello["signature"])
        self.assertEqual(hello["kind"], "function")

        # Check class has methods
        model = next(m for m in modules if m["name"] == "Model")
        self.assertIn("forward", model["calls"])
        self.assertEqual(model["kind"], "class")

    def test_syntax_error_graceful_degradation(self):
        """SyntaxError should produce a single top_level block, not crash."""
        self.store.track_write("bad.py", "def foo(:\n  pass")
        h = self.store.get_history("bad.py")
        self.assertEqual(len(h), 1)  # Still tracked

    def test_changed_modules_detected(self):
        self.store.track_write("m.py", "def a(): pass\ndef b(): pass")
        self.store.track_write("m.py", "def a(): pass\ndef b(x): return x\ndef c(): pass")
        h = self.store.get_history("m.py")
        changed = h[1].get("modules_changed", [])
        # b was modified, c was added
        self.assertTrue(any("b" in c for c in changed))
        self.assertTrue(any("c" in c for c in changed))


class TestWorkspaceSummary(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CodeVersionStore(os.path.join(self.tmpdir, "ws"))

    def test_empty_workspace(self):
        self.assertEqual(self.store.get_workspace_summary(), "")

    def test_summary_includes_tracked_files(self):
        self.store.track_write("a.py", "x = 1", reason="test")
        self.store.track_write("b.py", "y = 2", reason="test2")
        summary = self.store.get_workspace_summary()
        self.assertIn("a.py", summary)
        self.assertIn("b.py", summary)
        self.assertIn("v001", summary)


class TestFixContext(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CodeVersionStore(os.path.join(self.tmpdir, "ws"))

    def test_fix_context_targets_error_line(self):
        code = '''def setup():
    pass

def process(data):
    result = data / 0  # line 5
    return result

def cleanup():
    pass
'''
        self.store.track_write("worker.py", code)
        ctx = self.store.get_fix_context("worker.py", "ZeroDivisionError on line 5")
        self.assertIn("process", ctx)

    def test_fix_context_includes_diff_on_v2(self):
        self.store.track_write("f.py", "def a(): pass")
        self.store.track_write("f.py", "def a(): return 1")
        ctx = self.store.get_fix_context("f.py", "line 1")
        self.assertIn("Recent changes", ctx)

    def test_nonexistent_file_returns_empty(self):
        self.assertEqual(self.store.get_fix_context("nope.py", "err"), "")


class TestGetModuleCode(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CodeVersionStore(os.path.join(self.tmpdir, "ws"))

    def test_extract_single_module(self):
        code = "def foo():\n    return 42\n\ndef bar():\n    return 99\n"
        self.store.track_write("m.py", code)
        extracted = self.store.get_module_code("m.py", "foo")
        self.assertIn("42", extracted)
        self.assertNotIn("99", extracted)

    def test_nonexistent_module_returns_empty(self):
        self.store.track_write("m.py", "x = 1")
        self.assertEqual(self.store.get_module_code("m.py", "nope"), "")


class TestCycleTracking(unittest.TestCase):
    """Cycle-based write tracking for InsightDAG linking."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CodeVersionStore(os.path.join(self.tmpdir, "ws"))

    def test_set_and_query_cycle(self):
        self.store.set_current_cycle(3)
        self.store.track_write("a.py", "x = 1")
        self.store.track_write("b.py", "y = 2")

        self.store.set_current_cycle(5)
        self.store.track_write("a.py", "x = 2")

        writes_3 = self.store.get_cycle_writes(3)
        writes_5 = self.store.get_cycle_writes(5)
        writes_99 = self.store.get_cycle_writes(99)

        self.assertEqual(len(writes_3), 2)
        self.assertEqual(len(writes_5), 1)
        self.assertEqual(writes_5[0]["version"], "v002")
        self.assertEqual(len(writes_99), 0)

    def test_explicit_cycle_overrides_current(self):
        self.store.set_current_cycle(10)
        self.store.track_write("f.py", "pass", cycle=7)
        writes = self.store.get_cycle_writes(7)
        self.assertEqual(len(writes), 1)
        self.assertEqual(self.store.get_cycle_writes(10), [])


class TestInsightLinking(unittest.TestCase):
    """Reverse link: code version → insight ID."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = CodeVersionStore(os.path.join(self.tmpdir, "ws"))

    def test_link_insight_updates_manifest(self):
        self.store.track_write("m.py", "x = 1")
        h = self.store.get_history("m.py")
        self.assertEqual(h[0]["insight_id"], "")  # unlinked initially

        self.store.link_insight("m.py", "v001", "i0042")
        h = self.store.get_history("m.py")
        self.assertEqual(h[0]["insight_id"], "i0042")

    def test_link_nonexistent_version_no_crash(self):
        self.store.track_write("m.py", "x = 1")
        self.store.link_insight("m.py", "v999", "i0001")  # should not crash

    def test_link_nonexistent_file_no_crash(self):
        self.store.link_insight("nope.py", "v001", "i0001")  # should not crash

    def test_multiple_versions_different_insights(self):
        self.store.track_write("m.py", "v1")
        self.store.track_write("m.py", "v2")
        self.store.link_insight("m.py", "v001", "i0001")
        self.store.link_insight("m.py", "v002", "i0002")
        h = self.store.get_history("m.py")
        self.assertEqual(h[0]["insight_id"], "i0001")
        self.assertEqual(h[1]["insight_id"], "i0002")


if __name__ == "__main__":
    unittest.main()
