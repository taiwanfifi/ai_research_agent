"""
Insight DAG — Directed Acyclic Graph for Research Insights
============================================================
Replaces the flat insights list with a structured knowledge graph.
Each insight has relevance scoring, cross-references, tags,
code version links, and archival state — enabling smarter
distillation and retrieval.

Bidirectional link with CodeVersionStore:
  InsightNode.code_refs → [{filename, version, modules_changed}]
  CodeVersionStore manifest → {insight_id} per version
"""

import time
import uuid
from dataclasses import dataclass, field, asdict


@dataclass
class InsightNode:
    """A single insight in the knowledge DAG."""
    id: str
    cycle: int
    timestamp: str
    worker: str
    task: str
    success: bool
    content: str
    references: list[str] = field(default_factory=list)  # IDs of referenced insights
    relevance: float = 0.5  # 0.0 - 1.0
    tags: list[str] = field(default_factory=list)
    archived: bool = False
    code_refs: list[dict] = field(default_factory=list)  # [{filename, version, modules_changed}]


class InsightDAG:
    """
    DAG-structured knowledge graph for research insights.

    Each insight can reference others, has a relevance score that
    gets updated during distillation, and can be archived (but
    never deleted) when relevance drops below threshold.
    """

    def __init__(self):
        self.nodes: dict[str, InsightNode] = {}
        self._next_id: int = 1

    def add(self, cycle: int, worker: str, task: str, success: bool,
            content: str, tags: list[str] = None,
            references: list[str] = None,
            code_refs: list[dict] = None) -> str:
        """Add a new insight node. Returns the node ID."""
        node_id = f"i{self._next_id:04d}"
        self._next_id += 1

        # Auto-generate tags from worker and success status
        auto_tags = [worker]
        if not success:
            auto_tags.append("failure")
        if tags:
            auto_tags.extend(tags)
        auto_tags = list(dict.fromkeys(auto_tags))  # dedupe

        # Validate references exist
        valid_refs = [r for r in (references or []) if r in self.nodes]

        node = InsightNode(
            id=node_id,
            cycle=cycle,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            worker=worker,
            task=task[:100],
            success=success,
            content=content,
            references=valid_refs,
            relevance=0.5 if success else 0.3,
            tags=auto_tags,
            archived=False,
            code_refs=code_refs or [],
        )
        self.nodes[node_id] = node
        return node_id

    def get_panoramic_view(self, max_items: int = 20) -> str:
        """
        Format active insights sorted by relevance for LLM consumption.
        Returns a human-readable text block.
        """
        active = [n for n in self.nodes.values() if not n.archived]
        if not active:
            return "(no insights yet)"

        # Sort by relevance (desc), then by cycle (desc)
        active.sort(key=lambda n: (n.relevance, n.cycle), reverse=True)
        active = active[:max_items]

        lines = []
        for node in active:
            marker = "\u2713" if node.success else "\u2717"
            refs = ""
            if node.references:
                refs = f" [refs: {', '.join(node.references)}]"
            code = ""
            if node.code_refs:
                code_parts = [f"{c['filename']}@{c['version']}" for c in node.code_refs]
                code = f" [code: {', '.join(code_parts)}]"
            rel = f"rel={node.relevance:.2f}"
            tags = f" #{' #'.join(node.tags)}" if node.tags else ""
            lines.append(
                f"[{node.id} | Cycle {node.cycle} | {node.worker} | "
                f"{marker} | {rel}{tags}{refs}{code}]\n{node.content}"
            )

        return "\n\n".join(lines)

    def update_from_distillation(
        self,
        top_ids: list[str],
        connections: list[dict] = None,
        decay_factor: float = 0.8,
    ):
        """
        Update relevance scores after a distillation round.

        Args:
            top_ids: IDs of insights the LLM marked as important
            connections: List of {"from": id, "to": id} new references
            decay_factor: Multiply non-selected insights' relevance by this
        """
        valid_top = set(i for i in top_ids if i in self.nodes)

        for node_id, node in self.nodes.items():
            if node.archived:
                continue

            if node_id in valid_top:
                # Boost selected insights
                node.relevance = min(1.0, node.relevance + 0.2)
            else:
                # Decay unselected
                node.relevance *= decay_factor

            # Archive if relevance drops too low
            if node.relevance < 0.1:
                node.archived = True

        # Add new reference connections
        if connections:
            for conn in connections:
                from_id = conn.get("from", "")
                to_id = conn.get("to", "")
                if from_id in self.nodes and to_id in self.nodes:
                    node = self.nodes[from_id]
                    if to_id not in node.references:
                        node.references.append(to_id)

    def get_by_worker(self, worker: str) -> list[InsightNode]:
        """Get active insights by worker type."""
        return [
            n for n in self.nodes.values()
            if n.worker == worker and not n.archived
        ]

    def get_failures(self, limit: int = 5) -> list[InsightNode]:
        """Get recent failure insights."""
        failures = [
            n for n in self.nodes.values()
            if not n.success and not n.archived
        ]
        failures.sort(key=lambda n: n.cycle, reverse=True)
        return failures[:limit]

    def active_count(self) -> int:
        """Count of non-archived insights."""
        return sum(1 for n in self.nodes.values() if not n.archived)

    def total_count(self) -> int:
        """Total insight count including archived."""
        return len(self.nodes)

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize for checkpoint."""
        return {
            "next_id": self._next_id,
            "nodes": {
                node_id: asdict(node)
                for node_id, node in self.nodes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "InsightDAG":
        """Restore from checkpoint."""
        dag = cls()
        dag._next_id = data.get("next_id", 1)
        for node_id, node_data in data.get("nodes", {}).items():
            dag.nodes[node_id] = InsightNode(**node_data)
        return dag

    @classmethod
    def from_legacy_list(cls, insights: list[dict]) -> "InsightDAG":
        """
        Migrate from the old flat insights list format.
        Preserves all data with auto-generated IDs and default relevance.
        """
        dag = cls()
        for ins in insights:
            dag.add(
                cycle=ins.get("cycle", 0),
                worker=ins.get("worker", "unknown"),
                task=ins.get("task", ""),
                success=ins.get("success", True),
                content=ins.get("insight", str(ins)),
            )
        return dag
