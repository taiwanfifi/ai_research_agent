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
import re
import time
from dataclasses import dataclass, field, asdict


# Pre-compiled regex patterns (avoid re-compilation per call)
_METRIC_PATTERN = re.compile(r'([\w\-]+)\s*[:=]\s*(\d+\.?\d*)\s*%?')
_TABLE_PATTERN = re.compile(r'\|\s*([\w\s\-]+?)\s*\|\s*(\d+\.?\d*)\s*%?\s*\|')

_SKIP_LABELS = {
    "epoch", "step", "iteration", "batch", "iter", "len", "size",
    "count", "length", "total", "trainable", "param", "params",
    "rank", "alpha", "dropout", "seed", "num", "dim", "hidden",
    "layer", "layers", "index", "idx", "id", "version", "max",
    "min", "lr", "wd", "warmup", "vocab", "embedding", "head",
    "heads", "width", "height", "channels",
    # Hyperparameters (Round 10.1 fix)
    "r", "n", "k", "d", "b", "t", "m", "x", "e",
    "learning_rate", "batch_size", "seq_len", "max_len", "max_length",
    "lora_alpha", "lora_dropout", "lora_r", "lora_rank",
    "num_epochs", "epochs", "steps", "n_epochs", "gradient",
    "weight_decay", "momentum", "beta", "beta1", "beta2", "gamma",
    "temperature", "top_k", "top_p", "beam", "beams",
    "samples", "batches", "train", "validation", "val", "test",
    # File-related labels (prevent matching "py: 6499" from file listings)
    "py", "json", "png", "jpg", "csv", "txt", "md", "yaml", "yml",
    "bytes", "kb", "mb", "gb", "lines", "files", "dirs", "code_store",
    "__pycache__", "pycache",
}


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

    Optimized for minimal overhead:
    - Deferred disk writes (batch at flush/checkpoint, not per-record)
    - Cached summary generation
    - Compact JSON serialization
    - Pre-compiled regex patterns
    """

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self._entries: list[ExecutionEntry] = []
        self._log_path = os.path.join(workspace_dir, "execution_log.json")
        self._dirty = False          # Track if entries changed since last save
        self._summary_cache = None   # Cached summary string
        self._summary_count = -1     # Entry count when cache was built
        # Load existing if resuming
        self._load()

    def record(self, cycle: int, worker: str, tool_name: str,
               result_dict: dict) -> ExecutionEntry:
        """Record a tool execution result (deferred disk write)."""
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

        # Lazy metric parsing — only for code runs with stdout
        parsed_metrics = {}
        if stdout and tool_name == "run_python_code":
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
        self._dirty = True
        self._summary_cache = None  # Invalidate cache
        return entry

    def flush(self):
        """Write to disk if dirty. Call at end of worker run or checkpoint."""
        if self._dirty:
            self._save()
            self._dirty = False

    def get_summary_for_prompt(self, max_chars: int = 3000) -> str:
        """
        THE KEY METHOD: Format execution data for injection into LLM summary.
        Cached — only regenerated when entries change.
        """
        if not self._entries:
            return "(no tool executions recorded)"

        # Return cached if unchanged
        if self._summary_cache is not None and self._summary_count == len(self._entries):
            return self._summary_cache

        parts = []
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

        if all_metrics:
            parts.append("METRICS (from code execution stdout — showing FINAL value per metric):")
            for metric_name, values in all_metrics.items():
                latest = values[-1]  # Always use latest value
                parts.append(f"  {metric_name}: {latest['value']} (cycle {latest['cycle']})")

        # Include JSON result files from workspace for cross-verification
        import glob as glob_mod
        result_files = sorted(glob_mod.glob(os.path.join(self.workspace_dir, "results*.json"))
                              + glob_mod.glob(os.path.join(self.workspace_dir, "analysis*.json")))
        if result_files:
            parts.append("\nRESULT FILES (ground truth for verification):")
            for rf in result_files[:8]:  # Max 8 files
                try:
                    import json as json_mod
                    with open(rf) as fh:
                        data = json_mod.load(fh)
                    fname = os.path.basename(rf)
                    # Compact representation
                    compact = json_mod.dumps(data, indent=None, ensure_ascii=False)
                    parts.append(f"  {fname}: {compact[:300]}")
                except Exception:
                    pass

        unique_files = list(dict.fromkeys(all_files))
        if unique_files:
            parts.append(f"\nFILES WRITTEN ({len(unique_files)}):")
            for f in unique_files:
                parts.append(f"  - {f}")

        success_count = sum(1 for e in self._entries
                           if e.tool_name == "run_python_code" and e.success)
        parts.append(f"\nEXECUTION SUMMARY: {success_count}/{run_count} code runs succeeded")

        if errors:
            parts.append(f"\nERRORS ({len(errors)}):")
            for err in errors[-3:]:
                parts.append(f"  - {err}")

        result = "\n".join(parts)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n...(truncated)"

        # Cache
        self._summary_cache = result
        self._summary_count = len(self._entries)
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
        log._dirty = False
        log._summary_cache = None
        log._summary_count = -1
        for ed in d.get("entries", []):
            log._entries.append(ExecutionEntry(**ed))
        return log

    # ── Metric parsing (pre-compiled regex) ────────────────────

    @staticmethod
    def _parse_metrics(stdout: str) -> dict:
        """Extract labeled numbers from stdout."""
        results = {}

        for match in _METRIC_PATTERN.finditer(stdout):
            label = match.group(1).strip()
            if label.lower() in _SKIP_LABELS or len(label) <= 1:
                continue
            try:
                value = float(match.group(2))
                results[label.lower()] = value
            except ValueError:
                continue

        for match in _TABLE_PATTERN.finditer(stdout):
            label = match.group(1).strip()
            if not label or label.replace(" ", "").replace("-", "").isdigit():
                continue
            try:
                value = float(match.group(2))
                results[label.lower()] = value
            except ValueError:
                continue

        return results

    # ── Persistence ───────────────────────────────────────────

    def _save(self):
        """Persist to disk (compact JSON, no indent)."""
        try:
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            with open(self._log_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, separators=(",", ":"))
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
