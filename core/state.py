"""
JSON-File State Store
======================
Persistent state management using JSON files.
Supports namespaces, checkpointing, and crash recovery.
"""

import json
import os
import time
from typing import Any


class StateStore:
    """JSON-file state store with namespaced keys."""

    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)

    def _path(self, namespace: str, key: str) -> str:
        ns_dir = os.path.join(self.state_dir, namespace)
        os.makedirs(ns_dir, exist_ok=True)
        return os.path.join(ns_dir, f"{key}.json")

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        """Read a value from state."""
        path = self._path(namespace, key)
        if not os.path.exists(path):
            return default
        with open(path) as f:
            data = json.load(f)
        return data.get("value", default)

    def set(self, namespace: str, key: str, value: Any):
        """Write a value to state."""
        path = self._path(namespace, key)
        data = {
            "value": value,
            "updated_at": time.time(),
            "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def delete(self, namespace: str, key: str):
        """Delete a key from state."""
        path = self._path(namespace, key)
        if os.path.exists(path):
            os.unlink(path)

    def list_keys(self, namespace: str) -> list[str]:
        """List all keys in a namespace."""
        ns_dir = os.path.join(self.state_dir, namespace)
        if not os.path.isdir(ns_dir):
            return []
        return [f[:-5] for f in os.listdir(ns_dir) if f.endswith(".json")]

    def list_namespaces(self) -> list[str]:
        """List all namespaces."""
        return [
            d for d in os.listdir(self.state_dir)
            if os.path.isdir(os.path.join(self.state_dir, d))
        ]

    def checkpoint(self, namespace: str, data: dict):
        """Save a checkpoint with timestamp."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.set(namespace, f"checkpoint_{ts}", data)
        self.set(namespace, "latest_checkpoint", data)

    def load_checkpoint(self, namespace: str) -> dict | None:
        """Load the latest checkpoint."""
        return self.get(namespace, "latest_checkpoint")

    def clear_namespace(self, namespace: str):
        """Remove all keys in a namespace."""
        ns_dir = os.path.join(self.state_dir, namespace)
        if os.path.isdir(ns_dir):
            for f in os.listdir(ns_dir):
                os.unlink(os.path.join(ns_dir, f))
