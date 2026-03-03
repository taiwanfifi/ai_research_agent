"""
Code Version Store — Git-like Version Tracking + AST Module Map
================================================================
Tracks every file write with versioned snapshots, unified diffs,
and AST-parsed module maps so the coder worker gets precise,
modular context instead of full-file rewrites.

Bidirectional link with InsightDAG:
    Each version records the cycle it was written in and (optionally)
    the insight_id generated from that cycle's work. InsightDAG nodes
    store code_refs pointing back here. Together they form a unified
    research memory graph.

Storage layout per tracked file:
    workspace/.code_store/{filename_stem}/
        v001.py, v002.py, ...        <- version snapshots
        v001_v002.diff, ...          <- unified diffs
        manifest.json                <- version history + reasons + links
        module_map.json              <- AST-parsed module map
"""

import ast
import difflib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict


@dataclass
class ModuleInfo:
    """A single code module (function, class, or top-level block)."""
    name: str
    kind: str  # "function", "class", "top_level"
    signature: str = ""
    docstring: str = ""
    start_line: int = 0
    end_line: int = 0
    calls: list[str] = field(default_factory=list)


class CodeVersionStore:
    """
    Git-like version tracking for workspace files.

    All operations are best-effort: tracking failures never block
    the original write_file operation.
    """

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.store_dir = os.path.join(workspace_dir, ".code_store")
        os.makedirs(self.store_dir, exist_ok=True)
        self._current_cycle: int = 0

    def set_current_cycle(self, cycle: int):
        """Set the current supervisor cycle (called before each dispatch)."""
        self._current_cycle = cycle

    # ── Public API ────────────────────────────────────────────────

    def track_write(self, filename: str, content: str, reason: str = "",
                    cycle: int = None):
        """
        Called after every write_file: save snapshot, compute diff,
        parse AST, update manifest.  Best-effort — never raises.

        Args:
            cycle: Supervisor cycle number. Uses _current_cycle if not specified.
        """
        try:
            stem = self._stem(filename)
            file_dir = os.path.join(self.store_dir, stem)
            os.makedirs(file_dir, exist_ok=True)

            manifest = self._load_manifest(file_dir)
            version = len(manifest.get("versions", [])) + 1
            ver_str = f"v{version:03d}"

            # Save snapshot
            ext = os.path.splitext(filename)[1] or ".py"
            snapshot_path = os.path.join(file_dir, f"{ver_str}{ext}")
            with open(snapshot_path, "w") as f:
                f.write(content)

            # Compute diff against previous version
            diff_text = ""
            if version > 1:
                prev_str = f"v{version - 1:03d}"
                prev_path = os.path.join(file_dir, f"{prev_str}{ext}")
                if os.path.exists(prev_path):
                    with open(prev_path) as f:
                        old_content = f.read()
                    diff_text = self._compute_diff(old_content, content, filename)
                    diff_path = os.path.join(file_dir, f"{prev_str}_{ver_str}.diff")
                    with open(diff_path, "w") as f:
                        f.write(diff_text)

            # Parse module map
            module_map = self._parse_module_map(content)
            map_path = os.path.join(file_dir, "module_map.json")
            with open(map_path, "w") as f:
                json.dump([asdict(m) for m in module_map], f, indent=2)

            # Identify changed modules
            changed_modules = []
            if version > 1:
                old_map_path = os.path.join(file_dir, "_prev_module_map.json")
                if os.path.exists(old_map_path):
                    with open(old_map_path) as f:
                        old_map_data = json.load(f)
                    old_map = [ModuleInfo(**d) for d in old_map_data]
                    changed_modules = self._identify_changed_modules(old_map, module_map)

            # Save current map as _prev for next comparison
            with open(os.path.join(file_dir, "_prev_module_map.json"), "w") as f:
                json.dump([asdict(m) for m in module_map], f, indent=2)

            # Update manifest
            actual_cycle = cycle if cycle is not None else self._current_cycle
            versions = manifest.get("versions", [])
            versions.append({
                "version": ver_str,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "reason": reason,
                "has_diff": bool(diff_text),
                "modules_changed": changed_modules,
                "module_count": len(module_map),
                "cycle": actual_cycle,
                "insight_id": "",  # filled by link_insight() after extraction
            })
            manifest["versions"] = versions
            manifest["filename"] = filename
            manifest["latest"] = ver_str
            self._save_manifest(file_dir, manifest)

        except Exception:
            pass  # Best-effort — never crash the write

    def get_workspace_summary(self) -> str:
        """One-line-per-file overview for supervisor context."""
        try:
            lines = []
            if not os.path.exists(self.store_dir):
                return ""
            for stem in sorted(os.listdir(self.store_dir)):
                file_dir = os.path.join(self.store_dir, stem)
                if not os.path.isdir(file_dir) or stem.startswith("."):
                    continue
                manifest = self._load_manifest(file_dir)
                if not manifest.get("versions"):
                    continue
                filename = manifest.get("filename", stem)
                latest = manifest["versions"][-1]
                ver = latest["version"]
                n_modules = latest.get("module_count", "?")
                reason = latest.get("reason", "")[:60]
                lines.append(f"  {filename} ({ver}, {n_modules} modules) — {reason}")
            if not lines:
                return ""
            return "## Workspace Files (version-tracked)\n" + "\n".join(lines)
        except Exception:
            return ""

    def get_fix_context(self, filename: str, error_msg: str) -> str:
        """
        Smart context for bug fixes: parse traceback, find the
        failing module, return only that module's code + diff + I/O contract.
        """
        try:
            stem = self._stem(filename)
            file_dir = os.path.join(self.store_dir, stem)
            if not os.path.isdir(file_dir):
                return ""

            # Parse traceback for line numbers
            error_lines = self._extract_error_lines(error_msg)

            # Load module map
            map_path = os.path.join(file_dir, "module_map.json")
            if not os.path.exists(map_path):
                return ""
            with open(map_path) as f:
                modules = [ModuleInfo(**d) for d in json.load(f)]

            # Find modules containing error lines
            target_modules = []
            if error_lines:
                for mod in modules:
                    for line_no in error_lines:
                        if mod.start_line <= line_no <= mod.end_line:
                            if mod not in target_modules:
                                target_modules.append(mod)
            if not target_modules:
                target_modules = modules  # fallback: show all

            # Load latest source
            manifest = self._load_manifest(file_dir)
            if not manifest.get("versions"):
                return ""
            latest = manifest["versions"][-1]
            ext = os.path.splitext(filename)[1] or ".py"
            source_path = os.path.join(file_dir, f"{latest['version']}{ext}")
            if not os.path.exists(source_path):
                return ""
            with open(source_path) as f:
                source_lines = f.readlines()

            parts = [f"## Fix Context for {filename}"]
            parts.append(f"Error: {error_msg[:300]}")

            # Show targeted modules
            for mod in target_modules:
                code = "".join(source_lines[mod.start_line - 1:mod.end_line])
                parts.append(f"\n### Module: {mod.name} ({mod.kind}, lines {mod.start_line}-{mod.end_line})")
                if mod.signature:
                    parts.append(f"Signature: {mod.signature}")
                if mod.docstring:
                    parts.append(f"Docstring: {mod.docstring[:200]}")
                parts.append(f"```python\n{code}```")

            # Show latest diff if available
            if len(manifest["versions"]) >= 2:
                prev = manifest["versions"][-2]["version"]
                curr = latest["version"]
                diff_path = os.path.join(file_dir, f"{prev}_{curr}.diff")
                if os.path.exists(diff_path):
                    with open(diff_path) as f:
                        diff_text = f.read()
                    if diff_text:
                        parts.append(f"\n### Recent changes ({prev} → {curr})")
                        parts.append(f"```diff\n{diff_text[:1500]}```")

            return "\n".join(parts)
        except Exception:
            return ""

    def get_module_code(self, filename: str, module_name: str) -> str:
        """Extract a single module's source code."""
        try:
            stem = self._stem(filename)
            file_dir = os.path.join(self.store_dir, stem)
            manifest = self._load_manifest(file_dir)
            if not manifest.get("versions"):
                return ""

            latest = manifest["versions"][-1]
            ext = os.path.splitext(filename)[1] or ".py"
            source_path = os.path.join(file_dir, f"{latest['version']}{ext}")
            if not os.path.exists(source_path):
                return ""

            with open(source_path) as f:
                source_lines = f.readlines()

            map_path = os.path.join(file_dir, "module_map.json")
            if not os.path.exists(map_path):
                return ""
            with open(map_path) as f:
                modules = [ModuleInfo(**d) for d in json.load(f)]

            for mod in modules:
                if mod.name == module_name:
                    return "".join(source_lines[mod.start_line - 1:mod.end_line])
            return ""
        except Exception:
            return ""

    def get_history(self, filename: str) -> list[dict]:
        """Version history list for a file."""
        try:
            stem = self._stem(filename)
            file_dir = os.path.join(self.store_dir, stem)
            manifest = self._load_manifest(file_dir)
            return manifest.get("versions", [])
        except Exception:
            return []

    def get_cycle_writes(self, cycle: int) -> list[dict]:
        """
        Get all version writes that happened in a specific cycle.
        Returns [{filename, version, modules_changed, reason}].
        Used by supervisor to link code versions to insights.
        """
        try:
            results = []
            if not os.path.exists(self.store_dir):
                return results
            for stem in os.listdir(self.store_dir):
                file_dir = os.path.join(self.store_dir, stem)
                if not os.path.isdir(file_dir) or stem.startswith("."):
                    continue
                manifest = self._load_manifest(file_dir)
                for v in manifest.get("versions", []):
                    if v.get("cycle") == cycle:
                        results.append({
                            "filename": manifest.get("filename", stem),
                            "version": v["version"],
                            "modules_changed": v.get("modules_changed", []),
                            "reason": v.get("reason", ""),
                        })
            return results
        except Exception:
            return []

    def link_insight(self, filename: str, version: str, insight_id: str):
        """
        Retroactively link a code version to an insight ID.
        Creates the reverse direction of the InsightNode→code_refs link.
        """
        try:
            stem = self._stem(filename)
            file_dir = os.path.join(self.store_dir, stem)
            manifest = self._load_manifest(file_dir)
            for v in manifest.get("versions", []):
                if v["version"] == version:
                    v["insight_id"] = insight_id
            self._save_manifest(file_dir, manifest)
        except Exception:
            pass  # Best-effort

    # ── Internal helpers ──────────────────────────────────────────

    def _parse_module_map(self, source: str) -> list[ModuleInfo]:
        """
        Parse Python source with ast to extract functions, classes,
        and top-level blocks.  Falls back to a single top_level
        block on SyntaxError.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Graceful degradation: treat whole file as one block
            line_count = source.count("\n") + 1
            return [ModuleInfo(
                name="top_level", kind="top_level",
                start_line=1, end_line=line_count,
            )]

        modules = []
        source_lines = source.splitlines()

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                sig = self._get_function_signature(node)
                doc = ast.get_docstring(node) or ""
                calls = self._extract_calls(node)
                modules.append(ModuleInfo(
                    name=node.name,
                    kind="function",
                    signature=sig,
                    docstring=doc[:200],
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    calls=calls,
                ))
            elif isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node) or ""
                methods = []
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                sig = f"class {node.name}({', '.join(self._get_base_names(node))})" if node.bases else f"class {node.name}"
                modules.append(ModuleInfo(
                    name=node.name,
                    kind="class",
                    signature=sig,
                    docstring=doc[:200],
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    calls=methods,  # for classes, store method names
                ))

        # Capture top-level code blocks (imports, assignments, etc.)
        top_level_lines = set(range(1, len(source_lines) + 1))
        for mod in modules:
            for ln in range(mod.start_line, mod.end_line + 1):
                top_level_lines.discard(ln)

        if top_level_lines:
            sorted_lines = sorted(top_level_lines)
            # Group contiguous top-level lines
            if sorted_lines:
                modules.insert(0, ModuleInfo(
                    name="top_level",
                    kind="top_level",
                    start_line=sorted_lines[0],
                    end_line=sorted_lines[-1],
                ))

        return modules

    def _compute_diff(self, old: str, new: str, filename: str = "") -> str:
        """Compute unified diff between two versions."""
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{filename}", tofile=f"b/{filename}",
            lineterm="",
        )
        return "".join(diff)

    def _identify_changed_modules(
        self, old_map: list[ModuleInfo], new_map: list[ModuleInfo]
    ) -> list[str]:
        """Compare two module maps and return names of changed modules."""
        old_by_name = {m.name: m for m in old_map}
        new_by_name = {m.name: m for m in new_map}

        changed = []
        # New or modified
        for name, new_mod in new_by_name.items():
            old_mod = old_by_name.get(name)
            if not old_mod:
                changed.append(f"+{name}")  # added
            elif (old_mod.signature != new_mod.signature or
                  old_mod.start_line != new_mod.start_line or
                  old_mod.end_line != new_mod.end_line):
                changed.append(f"~{name}")  # modified

        # Deleted
        for name in old_by_name:
            if name not in new_by_name:
                changed.append(f"-{name}")  # removed

        return changed

    def _stem(self, filename: str) -> str:
        """Normalize filename to a safe directory name."""
        return os.path.splitext(os.path.basename(filename))[0]

    def _load_manifest(self, file_dir: str) -> dict:
        path = os.path.join(file_dir, "manifest.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self, file_dir: str, data: dict):
        path = os.path.join(file_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Build a human-readable function signature."""
        args = []
        for arg in node.args.args:
            ann = ""
            if arg.annotation:
                ann = f": {ast.unparse(arg.annotation)}"
            args.append(f"{arg.arg}{ann}")
        ret = ""
        if node.returns:
            ret = f" -> {ast.unparse(node.returns)}"
        return f"def {node.name}({', '.join(args)}){ret}"

    def _get_base_names(self, node: ast.ClassDef) -> list[str]:
        """Extract base class names."""
        names = []
        for base in node.bases:
            try:
                names.append(ast.unparse(base))
            except Exception:
                names.append("?")
        return names

    def _extract_calls(self, node: ast.AST) -> list[str]:
        """Extract function call names from an AST node."""
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        return list(dict.fromkeys(calls))  # dedupe, preserve order

    def _extract_error_lines(self, error_msg: str) -> list[int]:
        """Parse traceback to find line numbers."""
        lines = []
        for match in re.finditer(r'line (\d+)', error_msg):
            lines.append(int(match.group(1)))
        return lines
