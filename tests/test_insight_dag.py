"""
Tests for core.insight_dag — InsightDAG
========================================
Run: python3 -m pytest tests/test_insight_dag.py -v
  or: python3 tests/test_insight_dag.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.insight_dag import InsightDAG, InsightNode


class TestAddAndRetrieve(unittest.TestCase):

    def test_add_returns_sequential_ids(self):
        dag = InsightDAG()
        id1 = dag.add(cycle=1, worker="explorer", task="search", success=True, content="found papers")
        id2 = dag.add(cycle=2, worker="coder", task="implement", success=True, content="wrote code")
        self.assertEqual(id1, "i0001")
        self.assertEqual(id2, "i0002")

    def test_node_fields_populated(self):
        dag = InsightDAG()
        dag.add(cycle=3, worker="coder", task="write model", success=False, content="syntax error")
        node = dag.nodes["i0001"]
        self.assertEqual(node.cycle, 3)
        self.assertEqual(node.worker, "coder")
        self.assertFalse(node.success)
        self.assertEqual(node.relevance, 0.3)  # failures start at 0.3
        self.assertIn("failure", node.tags)
        self.assertIn("coder", node.tags)

    def test_success_relevance_default(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="explorer", task="t", success=True, content="ok")
        self.assertEqual(dag.nodes["i0001"].relevance, 0.5)

    def test_references_validated(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="a", task="t", success=True, content="first")
        dag.add(cycle=2, worker="b", task="t", success=True, content="second",
                references=["i0001", "i9999"])  # i9999 doesn't exist
        node = dag.nodes["i0002"]
        self.assertEqual(node.references, ["i0001"])

    def test_task_truncated(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="a", task="x" * 200, success=True, content="ok")
        self.assertEqual(len(dag.nodes["i0001"].task), 100)


class TestCodeRefs(unittest.TestCase):
    """Forward link: insight → code versions."""

    def test_add_with_code_refs(self):
        dag = InsightDAG()
        refs = [
            {"filename": "model.py", "version": "v001", "modules_changed": []},
            {"filename": "utils.py", "version": "v002", "modules_changed": ["~load"]},
        ]
        dag.add(cycle=1, worker="coder", task="write", success=True,
                content="wrote code", code_refs=refs)
        node = dag.nodes["i0001"]
        self.assertEqual(len(node.code_refs), 2)
        self.assertEqual(node.code_refs[0]["filename"], "model.py")

    def test_add_without_code_refs(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="explorer", task="search", success=True, content="ok")
        self.assertEqual(dag.nodes["i0001"].code_refs, [])

    def test_panoramic_view_shows_code_refs(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="coder", task="write", success=True,
                content="ok", code_refs=[{"filename": "m.py", "version": "v003", "modules_changed": []}])
        view = dag.get_panoramic_view()
        self.assertIn("m.py@v003", view)

    def test_panoramic_view_no_code_for_explorer(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="explorer", task="search", success=True, content="ok")
        view = dag.get_panoramic_view()
        self.assertNotIn("[code:", view)


class TestPanoramicView(unittest.TestCase):

    def test_empty_dag(self):
        dag = InsightDAG()
        self.assertEqual(dag.get_panoramic_view(), "(no insights yet)")

    def test_sorted_by_relevance(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="a", task="t", success=True, content="low")
        dag.add(cycle=2, worker="b", task="t", success=True, content="high")
        dag.nodes["i0001"].relevance = 0.2
        dag.nodes["i0002"].relevance = 0.9

        view = dag.get_panoramic_view()
        pos_high = view.index("high")
        pos_low = view.index("low")
        self.assertLess(pos_high, pos_low)

    def test_archived_excluded(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="a", task="t", success=True, content="alive")
        dag.add(cycle=2, worker="b", task="t", success=True, content="dead")
        dag.nodes["i0002"].archived = True
        view = dag.get_panoramic_view()
        self.assertIn("alive", view)
        self.assertNotIn("dead", view)

    def test_max_items_respected(self):
        dag = InsightDAG()
        for i in range(10):
            dag.add(cycle=i, worker="a", task="t", success=True, content=f"item{i}")
        view = dag.get_panoramic_view(max_items=3)
        # Should have at most 3 insight blocks
        self.assertEqual(view.count("[i0"), 3)


class TestDistillation(unittest.TestCase):

    def test_boost_selected(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="a", task="t", success=True, content="important")
        dag.add(cycle=2, worker="b", task="t", success=True, content="meh")
        dag.update_from_distillation(top_ids=["i0001"])
        self.assertAlmostEqual(dag.nodes["i0001"].relevance, 0.7)  # 0.5 + 0.2
        self.assertAlmostEqual(dag.nodes["i0002"].relevance, 0.4)  # 0.5 * 0.8

    def test_boost_capped_at_1(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="a", task="t", success=True, content="x")
        dag.nodes["i0001"].relevance = 0.95
        dag.update_from_distillation(top_ids=["i0001"])
        self.assertEqual(dag.nodes["i0001"].relevance, 1.0)

    def test_decay_archives_low_relevance(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="a", task="t", success=True, content="will die")
        dag.nodes["i0001"].relevance = 0.12
        dag.update_from_distillation(top_ids=[])  # not selected
        # 0.12 * 0.8 = 0.096 < 0.1 → archived
        self.assertTrue(dag.nodes["i0001"].archived)

    def test_connections_added(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="a", task="t", success=True, content="A")
        dag.add(cycle=2, worker="b", task="t", success=True, content="B")
        dag.update_from_distillation(
            top_ids=["i0001"],
            connections=[{"from": "i0002", "to": "i0001"}],
        )
        self.assertIn("i0001", dag.nodes["i0002"].references)

    def test_invalid_top_ids_ignored(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="a", task="t", success=True, content="x")
        dag.update_from_distillation(top_ids=["i9999"])  # doesn't exist
        # Should not crash, i0001 should decay
        self.assertAlmostEqual(dag.nodes["i0001"].relevance, 0.4)

    def test_archived_nodes_untouched(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="a", task="t", success=True, content="x")
        dag.nodes["i0001"].archived = True
        old_rel = dag.nodes["i0001"].relevance
        dag.update_from_distillation(top_ids=["i0001"])
        self.assertEqual(dag.nodes["i0001"].relevance, old_rel)


class TestFilters(unittest.TestCase):

    def test_get_by_worker(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="coder", task="t", success=True, content="a")
        dag.add(cycle=2, worker="explorer", task="t", success=True, content="b")
        dag.add(cycle=3, worker="coder", task="t", success=True, content="c")
        coders = dag.get_by_worker("coder")
        self.assertEqual(len(coders), 2)
        self.assertTrue(all(n.worker == "coder" for n in coders))

    def test_get_by_worker_excludes_archived(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="coder", task="t", success=True, content="a")
        dag.nodes["i0001"].archived = True
        self.assertEqual(len(dag.get_by_worker("coder")), 0)

    def test_get_failures(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="a", task="t", success=True, content="ok")
        dag.add(cycle=2, worker="b", task="t", success=False, content="fail1")
        dag.add(cycle=3, worker="c", task="t", success=False, content="fail2")
        failures = dag.get_failures(limit=5)
        self.assertEqual(len(failures), 2)
        self.assertFalse(any(f.success for f in failures))
        # Most recent first
        self.assertEqual(failures[0].cycle, 3)

    def test_get_failures_limit(self):
        dag = InsightDAG()
        for i in range(5):
            dag.add(cycle=i, worker="a", task="t", success=False, content=f"f{i}")
        self.assertEqual(len(dag.get_failures(limit=2)), 2)


class TestCounts(unittest.TestCase):

    def test_active_and_total(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="a", task="t", success=True, content="x")
        dag.add(cycle=2, worker="b", task="t", success=True, content="y")
        dag.nodes["i0002"].archived = True
        self.assertEqual(dag.active_count(), 1)
        self.assertEqual(dag.total_count(), 2)


class TestSerialization(unittest.TestCase):

    def test_roundtrip(self):
        dag = InsightDAG()
        dag.add(cycle=1, worker="coder", task="write", success=True,
                content="ok", code_refs=[{"filename": "m.py", "version": "v001", "modules_changed": []}])
        dag.add(cycle=2, worker="explorer", task="search", success=False,
                content="fail", references=["i0001"])
        dag.nodes["i0001"].relevance = 0.8

        data = dag.to_dict()
        dag2 = InsightDAG.from_dict(data)

        self.assertEqual(dag2._next_id, dag._next_id)
        self.assertEqual(len(dag2.nodes), 2)
        self.assertEqual(dag2.nodes["i0001"].relevance, 0.8)
        self.assertEqual(dag2.nodes["i0001"].code_refs[0]["filename"], "m.py")
        self.assertEqual(dag2.nodes["i0002"].references, ["i0001"])

    def test_legacy_migration(self):
        old = [
            {"cycle": 1, "worker": "explorer", "task": "search papers",
             "success": True, "insight": "found 3 papers on attention"},
            {"cycle": 2, "worker": "coder", "task": "implement",
             "success": False, "insight": "code crashed"},
        ]
        dag = InsightDAG.from_legacy_list(old)
        self.assertEqual(dag.total_count(), 2)
        self.assertEqual(dag.nodes["i0001"].worker, "explorer")
        self.assertEqual(dag.nodes["i0002"].success, False)
        self.assertIn("found 3 papers", dag.nodes["i0001"].content)

    def test_empty_dag_serialization(self):
        dag = InsightDAG()
        data = dag.to_dict()
        dag2 = InsightDAG.from_dict(data)
        self.assertEqual(dag2.total_count(), 0)
        self.assertEqual(dag2._next_id, 1)


if __name__ == "__main__":
    unittest.main()
