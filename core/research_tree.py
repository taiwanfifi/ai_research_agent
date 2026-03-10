"""
Research Tree — Progressive Agentic Tree Search
=================================================
Replaces linear hypothesis exploration with branching + backtracking.
Each node is a research direction (hypothesis + approach).
UCB1-like selection balances exploitation vs exploration.

Integrates with existing supervisor loop — tree GUIDES decisions,
doesn't replace the cycle-based execution.

Based on: AI Scientist v2 (progressive tree search, 4x medals),
AIDE (solution tree with LLM patches), FunSearch (island populations).
"""

import math
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ResearchNode:
    """A single node in the research tree."""
    id: str
    hypothesis: str          # What this branch tests
    approach: str = ""       # How to test it (method/setup)
    parent_id: str = ""      # "" for root
    children: list = field(default_factory=list)  # child node IDs
    score: float = 0.0       # Quality score (0-1), updated from results
    visits: int = 0          # How many cycles spent on this branch
    status: str = "unexplored"  # unexplored, exploring, completed, pruned
    results: dict = field(default_factory=dict)  # Key metrics
    cycle_started: int = 0
    cycle_ended: int = 0
    depth: int = 0
    debug_depth: int = 0     # Consecutive debug/fix attempts (cap at 3)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "approach": self.approach,
            "parent_id": self.parent_id,
            "children": self.children,
            "score": self.score,
            "visits": self.visits,
            "status": self.status,
            "results": self.results,
            "cycle_started": self.cycle_started,
            "cycle_ended": self.cycle_ended,
            "depth": self.depth,
            "debug_depth": self.debug_depth,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchNode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ResearchTree:
    """Progressive tree search for research direction exploration.

    Usage in supervisor loop:
        tree = ResearchTree(goal)
        while cycles_remain:
            branch = tree.select_next()      # UCB1 selection
            # ... dispatch worker on branch.hypothesis ...
            tree.update_score(branch.id, score, results)
            if score < parent_score * 0.7:
                tree.prune(branch.id)         # Backtrack
            else:
                tree.expand(branch.id, children)  # Deepen
    """

    # UCB1 exploration constant — higher = more exploration
    C = 1.4

    # Max tree depth (root=0, max children depth)
    MAX_DEPTH = 3

    # Max children per node
    MAX_CHILDREN = 4

    # Prune threshold: if child score < parent * this, prune
    PRUNE_RATIO = 0.6

    # Max consecutive debug/fix attempts before auto-abandon (from AIDE)
    DEBUG_DEPTH_CAP = 3

    def __init__(self, goal: str):
        self.goal = goal
        self.nodes: dict[str, ResearchNode] = {}
        self.root_id = ""
        self.total_visits = 0
        self._create_root(goal)

    def _create_root(self, goal: str):
        root = ResearchNode(
            id=f"N_{uuid.uuid4().hex[:6]}",
            hypothesis=goal,
            status="exploring",
            depth=0,
        )
        self.root_id = root.id
        self.nodes[root.id] = root

    @property
    def root(self) -> ResearchNode:
        return self.nodes[self.root_id]

    def select_next(self) -> ResearchNode | None:
        """UCB1-based selection of next branch to explore.

        Returns the most promising unexplored/exploring node,
        balancing exploitation (score) and exploration (unvisited).
        """
        candidates = []
        for node in self.nodes.values():
            if node.status in ("pruned", "completed"):
                continue
            # Unexplored nodes get priority
            if node.status == "unexplored":
                candidates.append((float('inf'), node))
                continue
            # UCB1 for exploring nodes with visits
            if node.visits > 0 and self.total_visits > 0:
                exploit = node.score
                explore = self.C * math.sqrt(
                    math.log(self.total_visits) / node.visits
                )
                ucb = exploit + explore
            else:
                ucb = float('inf')
            candidates.append((ucb, node))

        if not candidates:
            return None

        # Sort by UCB (descending), break ties by fewer visits
        candidates.sort(key=lambda x: (-x[0], x[1].visits))
        return candidates[0][1]

    def expand(self, parent_id: str, hypotheses: list[dict]) -> list[str]:
        """Create child nodes from hypotheses.

        Args:
            parent_id: Node to expand
            hypotheses: List of {"claim": str, "approach": str}

        Returns:
            List of new node IDs
        """
        parent = self.nodes.get(parent_id)
        if not parent:
            return []

        if parent.depth >= self.MAX_DEPTH:
            return []

        # Limit children
        remaining_slots = self.MAX_CHILDREN - len(parent.children)
        hypotheses = hypotheses[:remaining_slots]

        new_ids = []
        for h in hypotheses:
            node = ResearchNode(
                id=f"N_{uuid.uuid4().hex[:6]}",
                hypothesis=h.get("claim", h.get("hypothesis", "")),
                approach=h.get("approach", h.get("experiment", "")),
                parent_id=parent_id,
                depth=parent.depth + 1,
                status="unexplored",
            )
            self.nodes[node.id] = node
            parent.children.append(node.id)
            new_ids.append(node.id)

        return new_ids

    def update_score(self, node_id: str, score: float,
                     results: dict | None = None, cycle: int = 0):
        """Update node score after execution.

        Also propagates score changes upward to parents.
        """
        node = self.nodes.get(node_id)
        if not node:
            return

        node.visits += 1
        self.total_visits += 1

        # Running average score
        if node.visits == 1:
            node.score = score
            node.cycle_started = cycle
        else:
            # Exponential moving average (recent results weighted more)
            alpha = 0.6
            node.score = alpha * score + (1 - alpha) * node.score

        if results:
            node.results.update(results)

        node.cycle_ended = cycle

        # Propagate to parent
        self._propagate_score(node_id)

    def _propagate_score(self, node_id: str):
        """Update parent scores based on best child performance."""
        node = self.nodes.get(node_id)
        if not node or not node.parent_id:
            return

        parent = self.nodes.get(node.parent_id)
        if not parent:
            return

        # Parent score = max of children scores (optimistic)
        child_scores = []
        for cid in parent.children:
            child = self.nodes.get(cid)
            if child and child.visits > 0 and child.status != "pruned":
                child_scores.append(child.score)

        if child_scores:
            best_child = max(child_scores)
            # Parent score is blend of own + best child
            if parent.visits > 0:
                parent.score = max(parent.score, best_child * 0.9)

    def prune(self, node_id: str):
        """Mark a branch as pruned (dead end)."""
        node = self.nodes.get(node_id)
        if node:
            node.status = "pruned"
            # Also prune all descendants
            for cid in node.children:
                self.prune(cid)

    def complete(self, node_id: str):
        """Mark a branch as completed (results obtained)."""
        node = self.nodes.get(node_id)
        if node:
            node.status = "completed"

    def should_backtrack(self, node_id: str) -> bool:
        """Check if current branch should be abandoned.

        Returns True if:
        - Score dropped below parent * PRUNE_RATIO
        - Too many visits without improvement
        """
        node = self.nodes.get(node_id)
        if not node or not node.parent_id:
            return False

        parent = self.nodes.get(node.parent_id)
        if not parent or parent.visits == 0:
            return False

        # Score regression check
        if node.visits >= 2 and node.score < parent.score * self.PRUNE_RATIO:
            return True

        # Stagnation check: 3+ visits, no improvement over parent
        if node.visits >= 3 and node.score <= parent.score:
            return True

        return False

    def get_active_branch(self) -> ResearchNode | None:
        """Get the currently active (exploring) deepest node."""
        exploring = [n for n in self.nodes.values()
                     if n.status == "exploring"]
        if not exploring:
            return None
        # Deepest first, then most recent
        exploring.sort(key=lambda n: (-n.depth, -n.cycle_ended))
        return exploring[0]

    def get_branch_context(self, node_id: str) -> str:
        """Get formatted context for a branch (ancestry + siblings)."""
        node = self.nodes.get(node_id)
        if not node:
            return ""

        parts = []

        # Ancestry (root → current)
        ancestry = self._get_ancestry(node_id)
        if len(ancestry) > 1:
            parts.append("## Research Tree — Current Branch")
            for i, anc in enumerate(ancestry):
                indent = "  " * i
                status_icon = {"exploring": "→", "completed": "✓",
                               "pruned": "✗", "unexplored": "○"}.get(anc.status, "?")
                score_str = f" (score: {anc.score:.2f})" if anc.visits > 0 else ""
                parts.append(f"{indent}{status_icon} {anc.hypothesis[:80]}{score_str}")

        # Siblings (alternative branches at same level)
        if node.parent_id:
            parent = self.nodes.get(node.parent_id)
            if parent:
                siblings = [self.nodes[cid] for cid in parent.children
                            if cid != node_id and cid in self.nodes]
                if siblings:
                    parts.append("\n**Alternative branches (siblings):**")
                    for sib in siblings:
                        icon = {"completed": "✓", "pruned": "✗",
                                "unexplored": "○", "exploring": "→"}.get(sib.status, "?")
                        score_str = f" score={sib.score:.2f}" if sib.visits > 0 else ""
                        parts.append(f"  {icon} {sib.hypothesis[:60]}{score_str}")

        return "\n".join(parts) if parts else ""

    def _get_ancestry(self, node_id: str) -> list[ResearchNode]:
        """Get path from root to node."""
        path = []
        current = self.nodes.get(node_id)
        while current:
            path.append(current)
            if current.parent_id:
                current = self.nodes.get(current.parent_id)
            else:
                break
        path.reverse()
        return path

    def get_tree_summary(self) -> str:
        """Get human-readable tree summary for prompts."""
        if len(self.nodes) <= 1:
            return ""

        parts = ["## Research Tree"]
        self._print_subtree(self.root_id, parts, indent=0)

        # Stats
        active = sum(1 for n in self.nodes.values() if n.status == "exploring")
        pruned = sum(1 for n in self.nodes.values() if n.status == "pruned")
        completed = sum(1 for n in self.nodes.values() if n.status == "completed")
        parts.append(f"\nTree: {len(self.nodes)} nodes "
                     f"({active} active, {completed} done, {pruned} pruned)")

        return "\n".join(parts)

    def _print_subtree(self, node_id: str, parts: list, indent: int):
        node = self.nodes.get(node_id)
        if not node:
            return

        prefix = "  " * indent
        icon = {"exploring": "→", "completed": "✓", "pruned": "✗",
                "unexplored": "○"}.get(node.status, "?")
        score_str = f" [{node.score:.2f}]" if node.visits > 0 else ""
        parts.append(f"{prefix}{icon} {node.hypothesis[:70]}{score_str}")

        for cid in node.children:
            self._print_subtree(cid, parts, indent + 1)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "root_id": self.root_id,
            "total_visits": self.total_visits,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchTree":
        tree = cls.__new__(cls)
        tree.goal = d["goal"]
        tree.root_id = d["root_id"]
        tree.total_visits = d.get("total_visits", 0)
        tree.nodes = {
            nid: ResearchNode.from_dict(nd)
            for nid, nd in d.get("nodes", {}).items()
        }
        return tree

    def get_best_branch(self) -> ResearchNode | None:
        """Get the highest-scoring completed branch."""
        completed = [n for n in self.nodes.values()
                     if n.status == "completed" and n.visits > 0]
        if not completed:
            return None
        return max(completed, key=lambda n: n.score)

    def has_unexplored(self) -> bool:
        """Check if there are unexplored branches."""
        return any(n.status == "unexplored" for n in self.nodes.values())

    def increment_debug_depth(self, node_id: str) -> bool:
        """Increment debug counter for a node after a fix attempt.

        Returns True if under cap, False if cap reached (should abandon).
        From AIDE: "5 drafts + greedy + 3-depth debug cap"
        """
        node = self.nodes.get(node_id)
        if not node:
            return False
        node.debug_depth += 1
        if node.debug_depth >= self.DEBUG_DEPTH_CAP:
            print(f"  [ResearchTree] Debug depth cap ({self.DEBUG_DEPTH_CAP}) "
                  f"reached for {node.hypothesis[:50]} — auto-abandoning")
            self.prune(node_id)
            return False
        return True

    def reset_debug_depth(self, node_id: str):
        """Reset debug counter (called on successful execution)."""
        node = self.nodes.get(node_id)
        if node:
            node.debug_depth = 0
