"""
Execution Log — Single Source of Truth for Tool Execution Results
=================================================================
Records every tool execution result into workspace/execution_log.json.
The LLM summary can ONLY reference numbers from this log, preventing
fabrication at the architectural level (not just enforcement).

Reuses ResultVerifier._extract_labeled_numbers() for metric parsing.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict


@dataclass
class ExecutionEntry:
    """One tool execution result."""
    cycle: int
    worker: str
    tool_name: str           # "run_python_code", "write_file"
    timestamp: str
    success: bool
    stdout: str              # raw (up to 5000 chars)
    stderr: str
    parsed_metrics: dict     # {"accuracy": 85.3, "loss": 0.42}
    files_written: list      # [str]
    returncode: int


class ExecutionLog:
    """
    Append-only log of tool executions within a mission.
    Persisted to workspace/execution_log.json.
    """

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self._entries: list[ExecutionEntry] = []
        self._log_path = os.path.join(workspace_dir, "execution_log.json")
        # Load existing if resuming
        self._load()

    def record(self, cycle: int, worker: str, tool_name: str,
               result_dict: dict) -> ExecutionEntry:
        """
        Record a tool execution result.

        Args:
            cycle: Current supervisor cycle
            worker: Worker name (coder, reviewer, etc.)
            tool_name: Tool that was called
            result_dict: Raw tool result (parsed JSON or string)

        Returns:
            The created ExecutionEntry
        """
        # Parse result_dict based on tool type
        if isinstance(result_dict, str):
            try:
                result_dict = json.loads(result_dict)
            except (json.JSONDecodeError, TypeError):
                result_dict = {"stdout": result_dict, "success": True}

        stdout = ""
        stderr = ""
        returncode = 0
        success = True
        files_written = []

        if tool_name == "run_python_code":
            stdout = (result_dict.get("stdout", "") or "")[:5000]
            stderr = (result_dict.get("stderr", "") or "")[:2000]
            returncode = result_dict.get("returncode", 0)
            success = returncode == 0
        elif tool_name == "write_file":
            success = result_dict.get("success", False)
            path = result_dict.get("path", "")
            if path:
                files_written.append(path)
            stdout = f"wrote {path}" if success else f"failed to write {path}"
        else:
            stdout = str(result_dict)[:2000]
            success = True

        # Parse metrics from stdout
        parsed_metrics = {}
        if stdout:
            parsed_metrics = self._parse_metrics(stdout)

        entry = ExecutionEntry(
            cycle=cycle,
            worker=worker,
            tool_name=tool_name,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            success=success,
            stdout=stdout,
            stderr=stderr,
            parsed_metrics=parsed_metrics,
            files_written=files_written,
            returncode=returncode,
        )
        self._entries.append(entry)
        self._save()
        return entry

    def get_summary_for_prompt(self, max_chars: int = 3000) -> str:
        """
        THE KEY METHOD: Format execution data for injection into LLM summary.

        Returns a structured text block showing:
        - All parsed metrics across runs
        - Files written
        - Errors encountered
        """
        if not self._entries:
            return "(no tool executions recorded)"

        parts = []

        # Collect all metrics across runs
        all_metrics: dict[str, list] = {}
        all_files: list[str] = []
        errors: list[str] = []
        run_count = 0

        for e in self._entries:
            if e.tool_name == "run_python_code":
                run_count += 1
                if not e.success and e.stderr:
                    errors.append(f"[cycle {e.cycle}] {e.stderr[:200]}")
                for k, v in e.parsed_metrics.items():
                    all_metrics.setdefault(k, []).append(
                        {"value": v, "cycle": e.cycle, "worker": e.worker}
                    )
            if e.files_written:
                all_files.extend(e.files_written)

        # Format metrics section
        if all_metrics:
            parts.append("METRICS (from code execution stdout):")
            for metric_name, values in all_metrics.items():
                if len(values) == 1:
                    v = values[0]
                    parts.append(f"  {metric_name}: {v['value']}")
                else:
                    # Show progression
                    val_strs = [f"{v['value']} (cycle {v['cycle']})" for v in values]
                    parts.append(f"  {metric_name}: {' → '.join(val_strs)}")

        # Format files section
        unique_files = list(dict.fromkeys(all_files))  # dedupe preserving order
        if unique_files:
            parts.append(f"\nFILES WRITTEN ({len(unique_files)}):")
            for f in unique_files:
                parts.append(f"  - {f}")

        # Format execution summary
        success_count = sum(1 for e in self._entries
                           if e.tool_name == "run_python_code" and e.success)
        parts.append(f"\nEXECUTION SUMMARY: {success_count}/{run_count} code runs succeeded")

        # Format recent errors
        if errors:
            parts.append(f"\nERRORS ({len(errors)}):")
            for err in errors[-3:]:
                parts.append(f"  - {err}")

        result = "\n".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n...(truncated)"
        return result

    def get_latest_metrics(self, n: int = 5) -> list[dict]:
        """Get the most recent parsed metrics from the last n code executions."""
        code_entries = [e for e in self._entries
                        if e.tool_name == "run_python_code" and e.parsed_metrics]
        result = []
        for e in code_entries[-n:]:
            result.append({
                "cycle": e.cycle,
                "worker": e.worker,
                "metrics": e.parsed_metrics,
                "success": e.success,
            })
        return result

    def to_dict(self) -> dict:
        """Serialize for checkpoint."""
        return {
            "entries": [asdict(e) for e in self._entries],
        }

    @classmethod
    def from_dict(cls, d: dict, workspace_dir: str) -> "ExecutionLog":
        """Restore from checkpoint data."""
        log = cls.__new__(cls)
        log.workspace_dir = workspace_dir
        log._log_path = os.path.join(workspace_dir, "execution_log.json")
        log._entries = []
        for ed in d.get("entries", []):
            log._entries.append(ExecutionEntry(**ed))
        return log

    # ── Metric parsing (reuses ResultVerifier pattern) ────────────

    @staticmethod
    def _parse_metrics(stdout: str) -> dict:
        """
        Extract labeled numbers from stdout.
        Reuses the same pattern as ResultVerifier._extract_labeled_numbers().
        """
        import re

        results = {}
        _SKIP_LABELS = {
            "epoch", "step", "iteration", "batch", "iter", "len", "size",
            "count", "length", "total", "trainable", "param", "params",
            "rank", "alpha", "dropout", "seed", "num", "dim", "hidden",
            "layer", "layers", "index", "idx", "id", "version", "max",
            "min", "lr", "wd", "warmup", "vocab", "embedding", "head",
            "heads", "width", "height", "channels",
        }

        # Pattern: "label: value" or "label = value"
        for match in re.finditer(r'([\w\-]+)\s*[:=]\s*(\d+\.?\d*)\s*%?', stdout):
            label = match.group(1).strip()
            if label.lower() in _SKIP_LABELS:
                continue
            try:
                value = float(match.group(2))
                # Use the last occurrence of each label (most recent result)
                results[label.lower()] = value
            except ValueError:
                continue

        # Pattern: table rows "| label | value |"
        for match in re.finditer(r'\|\s*([\w\s\-]+?)\s*\|\s*(\d+\.?\d*)\s*%?\s*\|', stdout):
            label = match.group(1).strip()
            if not label or label.replace(" ", "").replace("-", "").isdigit():
                continue
            try:
                value = float(match.group(2))
                results[label.lower()] = value
            except ValueError:
                continue

        return results

    # ── Persistence ───────────────────────────────────────────────

    def _save(self):
        """Persist to disk."""
        try:
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            with open(self._log_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load(self):
        """Load existing entries from disk."""
        if os.path.exists(self._log_path):
            try:
                with open(self._log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for ed in data.get("entries", []):
                    self._entries.append(ExecutionEntry(**ed))
            except Exception:
                pass
