"""
Knowledge Index Management
============================
Manages _index.json files for lazy loading and search.
"""

import json
import os
import time


class KnowledgeIndex:
    """Manages a directory's _index.json for knowledge items."""

    def __init__(self, directory: str):
        self.directory = directory
        self.index_path = os.path.join(directory, "_index.json")
        os.makedirs(directory, exist_ok=True)
        self._cache = None

    def _load(self) -> dict:
        if self._cache is not None:
            return self._cache
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                self._cache = json.load(f)
        else:
            self._cache = {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "item_count": 0,
                "items": {},
                "subcategories": [],
            }
        return self._cache

    def _save(self):
        data = self._load()
        data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        data["item_count"] = len(data["items"])
        with open(self.index_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_item(self, item_id: str, metadata: dict):
        """Add or update an item in the index."""
        data = self._load()
        data["items"][item_id] = {
            **metadata,
            "added_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save()

    def remove_item(self, item_id: str):
        """Remove an item from the index."""
        data = self._load()
        data["items"].pop(item_id, None)
        self._save()

    def get_item(self, item_id: str) -> dict | None:
        """Get item metadata."""
        return self._load()["items"].get(item_id)

    def list_items(self) -> dict:
        """List all items."""
        return dict(self._load()["items"])

    def count(self) -> int:
        """Number of items in this index."""
        return len(self._load()["items"])

    def search(self, query: str) -> list[tuple[str, dict]]:
        """Simple keyword search across item titles and keywords."""
        query_lower = query.lower()
        results = []
        for item_id, meta in self._load()["items"].items():
            searchable = " ".join([
                item_id,
                meta.get("title", ""),
                meta.get("summary", ""),
                " ".join(meta.get("keywords", [])),
            ]).lower()
            if query_lower in searchable:
                results.append((item_id, meta))
        return results

    def set_subcategories(self, subcats: list[str]):
        """Update the list of subcategories."""
        data = self._load()
        data["subcategories"] = subcats
        self._save()

    def get_subcategories(self) -> list[str]:
        """Get subcategory names."""
        return self._load().get("subcategories", [])

    def invalidate_cache(self):
        """Force reload from disk on next access."""
        self._cache = None
