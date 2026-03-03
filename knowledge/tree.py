"""
Self-Organizing Knowledge Tree
================================
B-tree-like auto-organizing knowledge store.
When a directory exceeds the threshold, it asks the LLM to cluster items
into subcategories and reorganizes automatically.
"""

import json
import os
import shutil
import time

from knowledge.index import KnowledgeIndex
from knowledge.categories import DEFAULT_CATEGORIES, REORG_THRESHOLD, SPLIT_TARGET


class KnowledgeTree:
    """Auto-organizing knowledge tree with LLM-based reorganization."""

    def __init__(self, root_dir: str, llm_client=None):
        self.root_dir = root_dir
        self.llm = llm_client
        os.makedirs(root_dir, exist_ok=True)

        # Initialize default category directories
        for cat in DEFAULT_CATEGORIES:
            cat_dir = os.path.join(root_dir, cat)
            os.makedirs(cat_dir, exist_ok=True)

        # Root index
        self.root_index = KnowledgeIndex(root_dir)

    def add(self, category: str, item_id: str, content: str, metadata: dict = None):
        """
        Add a knowledge item.

        Args:
            category: Top-level category (papers, experiments, methods, code, reports)
            item_id: Unique identifier (e.g., "attention_is_all_you_need")
            content: Full content to store
            metadata: Optional metadata (title, summary, keywords, etc.)
        """
        meta = metadata or {}
        cat_dir = os.path.join(self.root_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        # Write content file
        ext = meta.get("extension", ".md")
        filepath = os.path.join(cat_dir, f"{item_id}{ext}")
        with open(filepath, "w") as f:
            f.write(content)

        # Update index
        idx = KnowledgeIndex(cat_dir)
        idx.add_item(item_id, {
            "title": meta.get("title", item_id),
            "summary": meta.get("summary", content[:200]),
            "keywords": meta.get("keywords", []),
            "file": f"{item_id}{ext}",
            "size": len(content),
        })

        # Check if reorganization needed
        if idx.count() > REORG_THRESHOLD and self.llm:
            self._auto_reorganize(cat_dir, idx)

        return filepath

    def get(self, category: str, item_id: str) -> str | None:
        """Read a knowledge item's content."""
        cat_dir = os.path.join(self.root_dir, category)
        idx = KnowledgeIndex(cat_dir)
        item = idx.get_item(item_id)
        if not item:
            return None
        filepath = os.path.join(cat_dir, item["file"])
        if os.path.exists(filepath):
            with open(filepath) as f:
                return f.read()
        return None

    def search(self, query: str, categories: list[str] = None) -> list[dict]:
        """Search across knowledge tree."""
        results = []
        cats = categories or list(DEFAULT_CATEGORIES.keys())

        for cat in cats:
            cat_dir = os.path.join(self.root_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            results.extend(self._search_recursive(cat_dir, query, category=cat))

        return results

    def _search_recursive(self, directory: str, query: str, category: str = "", depth: int = 0) -> list[dict]:
        """Recursively search indexes."""
        results = []
        idx = KnowledgeIndex(directory)

        for item_id, meta in idx.search(query):
            results.append({
                "category": category,
                "item_id": item_id,
                "path": os.path.join(directory, meta.get("file", "")),
                "depth": depth,
                **meta,
            })

        # Search subcategories
        for subcat in idx.get_subcategories():
            subdir = os.path.join(directory, subcat)
            if os.path.isdir(subdir):
                results.extend(self._search_recursive(
                    subdir, query,
                    category=f"{category}/{subcat}",
                    depth=depth + 1,
                ))

        return results

    def get_summary(self, depth: int = 2) -> dict:
        """Get a tree overview for the supervisor."""
        return self._summarize_dir(self.root_dir, depth)

    def _summarize_dir(self, directory: str, max_depth: int, current_depth: int = 0) -> dict:
        """Recursively summarize directory tree."""
        idx = KnowledgeIndex(directory)
        items = idx.list_items()

        summary = {
            "path": os.path.relpath(directory, self.root_dir),
            "item_count": len(items),
            "items": [
                {"id": k, "title": v.get("title", k)}
                for k, v in list(items.items())[:10]  # Show first 10
            ],
        }

        if current_depth < max_depth:
            subcats = idx.get_subcategories()
            # Also check actual subdirectories
            if os.path.isdir(directory):
                for d in sorted(os.listdir(directory)):
                    full = os.path.join(directory, d)
                    if os.path.isdir(full) and not d.startswith("_") and d not in subcats:
                        subcats.append(d)

            if subcats:
                summary["subcategories"] = {}
                for sc in subcats:
                    subdir = os.path.join(directory, sc)
                    if os.path.isdir(subdir):
                        summary["subcategories"][sc] = self._summarize_dir(
                            subdir, max_depth, current_depth + 1
                        )

        return summary

    def _auto_reorganize(self, directory: str, idx: KnowledgeIndex):
        """Ask LLM to cluster items into subcategories."""
        if not self.llm:
            return

        items = idx.list_items()
        item_list = "\n".join([
            f"- {item_id}: {meta.get('title', item_id)} — {meta.get('summary', '')[:100]}"
            for item_id, meta in items.items()
        ])

        prompt = f"""You have {len(items)} knowledge items that need to be organized into {SPLIT_TARGET} subcategories.

Items:
{item_list}

Respond with ONLY a JSON object mapping each subcategory name to a list of item_ids:
{{
  "subcategory_name": ["item_id_1", "item_id_2"],
  ...
}}

Rules:
- Use short, descriptive subcategory names (lowercase, underscores)
- Create exactly {SPLIT_TARGET} subcategories
- Every item must be assigned to exactly one subcategory
"""

        try:
            response = self.llm.chat([
                {"role": "system", "content": "You organize knowledge items into categories. Respond with JSON only."},
                {"role": "user", "content": prompt},
            ])

            content = response["choices"][0]["message"]["content"]
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                return
            clusters = json.loads(json_match.group())

            # Move items to subcategories
            for subcat, item_ids in clusters.items():
                subdir = os.path.join(directory, subcat)
                os.makedirs(subdir, exist_ok=True)
                sub_idx = KnowledgeIndex(subdir)

                for item_id in item_ids:
                    item_meta = idx.get_item(item_id)
                    if not item_meta:
                        continue

                    # Move file
                    old_path = os.path.join(directory, item_meta.get("file", ""))
                    new_path = os.path.join(subdir, item_meta.get("file", ""))
                    if os.path.exists(old_path):
                        shutil.move(old_path, new_path)

                    # Update indexes
                    sub_idx.add_item(item_id, item_meta)
                    idx.remove_item(item_id)

            # Update subcategory list
            idx.set_subcategories(list(clusters.keys()))
            print(f"  [Knowledge] Reorganized {directory} into {list(clusters.keys())}")

        except Exception as e:
            print(f"  [Knowledge] Reorg failed: {e}")

    def search_cross(self, query: str, other_dirs: list[dict],
                      categories: list[str] = None) -> list[dict]:
        """
        Search across other missions' knowledge trees.

        Args:
            query: Search query
            other_dirs: List of {"mission_id": str, "knowledge_dir": str, "goal": str}
            categories: Optional category filter

        Returns:
            List of results, each tagged with source_mission
        """
        results = []
        for info in other_dirs:
            kdir = info["knowledge_dir"]
            if not os.path.isdir(kdir):
                continue
            other_tree = KnowledgeTree(kdir, llm_client=None)
            hits = other_tree.search(query, categories=categories)
            for hit in hits:
                hit["source_mission"] = info["mission_id"]
                hit["source_goal"] = info.get("goal", "")
            results.extend(hits)
        return results

    def list_categories(self) -> list[str]:
        """List top-level categories."""
        return [
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and not d.startswith("_")
        ]

    def stats(self) -> dict:
        """Get overall knowledge tree statistics."""
        total = 0
        by_category = {}
        for cat in self.list_categories():
            cat_dir = os.path.join(self.root_dir, cat)
            idx = KnowledgeIndex(cat_dir)
            count = idx.count()
            by_category[cat] = count
            total += count
        return {"total_items": total, "by_category": by_category}
