#!/usr/bin/env python3
"""Dashboard API server — read-only access to missions/ data.

Usage:
    python3 visual/server.py [--port PORT] [--missions PATH]
"""

import argparse
import json
import os
import re
import glob
import urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

VISUAL_DIR = Path(__file__).resolve().parent
DEFAULT_MISSIONS = VISUAL_DIR.parent / "missions"
STATIC_DIR = VISUAL_DIR / "static"


def _read_json(path):
    """Read a JSON file, return None on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_text(path):
    """Read a text file, return None on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(mission_dir):
    """Load latest checkpoint, unwrapping the StateStore wrapper."""
    cp = _read_json(mission_dir / "state" / "mission" / "latest_checkpoint.json")
    if cp is None:
        return None
    # StateStore wraps in {"value": {...}, "updated_at": ..., "updated_iso": ...}
    if isinstance(cp, dict) and "value" in cp:
        return cp["value"]
    return cp


def _load_historical_checkpoints(mission_dir):
    """Load all checkpoint_*.json files sorted by filename (chronological)."""
    pattern = str(mission_dir / "state" / "mission" / "checkpoint_*.json")
    files = sorted(glob.glob(pattern))
    checkpoints = []
    for f in files:
        cp = _read_json(f)
        if cp is None:
            continue
        val = cp.get("value", cp) if isinstance(cp, dict) else cp
        val["_filename"] = os.path.basename(f)
        checkpoints.append(val)
    return checkpoints


def _synthesize_dag(completed_tasks):
    """Create pseudo-DAG from completed_tasks when insight_dag is missing."""
    nodes = {}
    for i, task in enumerate(completed_tasks or []):
        nid = f"i{i:04d}"
        nodes[nid] = {
            "id": nid,
            "cycle": i + 1,
            "timestamp": "",
            "worker": task.get("worker", "explorer"),
            "task": (task.get("task", "") or "")[:100],
            "success": task.get("success", True),
            "content": task.get("output", "") or "",
            "references": [f"i{i-1:04d}"] if i > 0 else [],
            "relevance": 0.5,
            "tags": [task.get("worker", "explorer")] + (["failure"] if not task.get("success", True) else []),
            "archived": False,
            "code_refs": [],
            "synthetic": True,
        }
    return {"next_id": len(completed_tasks or []), "nodes": nodes}


def _get_missions(missions_dir):
    """List all missions with basic info."""
    results = []
    if not missions_dir.is_dir():
        return results
    for entry in sorted(missions_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("mission_"):
            continue
        manifest = _read_json(entry / "mission.json")
        if manifest is None:
            manifest = {"mission_id": entry.name}
        cp = _load_checkpoint(entry)
        # Check for mission score
        score_data = _read_json(entry / "workspace" / "mission_score.json")
        info = {
            "id": entry.name,
            "goal": manifest.get("goal", ""),
            "slug": manifest.get("slug", ""),
            "status": manifest.get("status", "unknown"),
            "language": manifest.get("language", "en"),
            "created_at": manifest.get("created_at", ""),
            "cycle": cp.get("cycle", 0) if cp else 0,
            "max_cycles": cp.get("max_cycles", 0) if cp else 0,
            "state": cp.get("state", "") if cp else "",
            "task_count": len(cp.get("completed_tasks", [])) if cp else 0,
            "error_count": len(cp.get("errors", [])) if cp else 0,
            "score": score_data.get("overall") if score_data else None,
            "grade": score_data.get("grade") if score_data else None,
        }
        results.append(info)
    return results


def _get_mission_detail(missions_dir, mission_id):
    """Full mission state: manifest + latest checkpoint."""
    mdir = missions_dir / mission_id
    if not mdir.is_dir():
        return None
    manifest = _read_json(mdir / "mission.json") or {}
    cp = _load_checkpoint(mdir) or {}
    return {"manifest": manifest, "checkpoint": cp}


def _get_insights(missions_dir, mission_id):
    """InsightDAG nodes + edges, or synthesized from completed_tasks."""
    mdir = missions_dir / mission_id
    cp = _load_checkpoint(mdir)
    if cp is None:
        return {"next_id": 0, "nodes": {}}

    dag = cp.get("insight_dag")
    if dag and dag.get("nodes"):
        return dag

    return _synthesize_dag(cp.get("completed_tasks", []))


def _get_code(missions_dir, mission_id):
    """All tracked files with version history + module maps."""
    mdir = missions_dir / mission_id
    code_store = mdir / "workspace" / ".code_store"
    if not code_store.is_dir():
        return []
    results = []
    for stem_dir in sorted(code_store.iterdir()):
        if not stem_dir.is_dir():
            continue
        manifest = _read_json(stem_dir / "manifest.json")
        module_map = _read_json(stem_dir / "module_map.json")
        results.append({
            "stem": stem_dir.name,
            "manifest": manifest,
            "module_map": module_map,
        })
    return results


def _get_diff(missions_dir, mission_id, file_stem, v1, v2):
    """Specific diff content between two versions."""
    mdir = missions_dir / mission_id
    diff_path = mdir / "workspace" / ".code_store" / file_stem / f"{v1}_{v2}.diff"
    return _read_text(diff_path)


def _get_knowledge(missions_dir, mission_id):
    """Knowledge tree with category indices."""
    mdir = missions_dir / mission_id
    knowledge_dir = mdir / "knowledge"
    if not knowledge_dir.is_dir():
        return {}
    result = {}
    for cat_dir in sorted(knowledge_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        index = _read_json(cat_dir / "_index.json")
        result[cat_dir.name] = index or {"item_count": 0, "items": {}}
    return result


def _get_reports(missions_dir, mission_id):
    """Report list with metadata."""
    mdir = missions_dir / mission_id
    reports_dir = mdir / "reports"
    if not reports_dir.is_dir():
        return []
    results = []
    for f in sorted(reports_dir.iterdir()):
        if f.suffix == ".md":
            stat = f.stat()
            results.append({
                "filename": f.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            })
    return results


def _get_report_content(missions_dir, mission_id, filename):
    """Report markdown content."""
    mdir = missions_dir / mission_id
    # Sanitize filename to prevent path traversal
    safe_name = Path(filename).name
    return _read_text(mdir / "reports" / safe_name)


def _get_mission_score(missions_dir, mission_id):
    """Mission quality score from mission_score.json."""
    mdir = missions_dir / mission_id
    return _read_json(mdir / "workspace" / "mission_score.json")


def _get_comparisons(missions_dir):
    """List all A/B comparison results."""
    comp_dir = missions_dir / "_comparisons"
    if not comp_dir.is_dir():
        return []
    results = []
    for f in sorted(comp_dir.iterdir()):
        if f.suffix == ".json":
            data = _read_json(f)
            if data:
                data["_filename"] = f.name
                results.append(data)
    return results


def _get_workspace_files(missions_dir, mission_id):
    """List all files in workspace (excluding .code_store internals)."""
    mdir = missions_dir / mission_id
    workspace = mdir / "workspace"
    if not workspace.is_dir():
        return []
    results = []
    for root, dirs, files in os.walk(workspace):
        # Skip .code_store version files but include manifests
        rel_root = os.path.relpath(root, workspace)
        for f in sorted(files):
            filepath = Path(root) / f
            rel_path = os.path.relpath(filepath, workspace)
            # Skip __pycache__ and temp files
            if "__pycache__" in rel_path or f.startswith("tmp"):
                continue
            stat = filepath.stat()
            ext = filepath.suffix.lower()
            file_type = "code" if ext == ".py" else "image" if ext in (".png", ".jpg", ".svg") else "data" if ext in (".csv", ".json", ".txt") else "other"
            results.append({
                "path": rel_path,
                "name": f,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "type": file_type,
                "ext": ext,
            })
    return results


def _get_workspace_file_content(missions_dir, mission_id, filepath):
    """Read a workspace file's content (text files only, images served as binary)."""
    mdir = missions_dir / mission_id
    workspace = mdir / "workspace"
    # Sanitize to prevent path traversal
    safe_path = os.path.normpath(filepath)
    if safe_path.startswith(".."):
        return None
    full_path = workspace / safe_path
    if not full_path.is_file():
        return None
    return full_path


def _get_timeline(missions_dir, mission_id):
    """Cycle-by-cycle history from historical checkpoints."""
    mdir = missions_dir / mission_id
    checkpoints = _load_historical_checkpoints(mdir)
    timeline = []
    for cp in checkpoints:
        timeline.append({
            "filename": cp.get("_filename", ""),
            "cycle": cp.get("cycle", 0),
            "max_cycles": cp.get("max_cycles", 0),
            "state": cp.get("state", ""),
            "completed_tasks": cp.get("completed_tasks", []),
            "task_queue": cp.get("task_queue", []),
            "errors": cp.get("errors", []),
            "last_action": cp.get("last_action", ""),
            "saved_at": cp.get("saved_at", ""),
            "reports_generated": cp.get("reports_generated", 0),
        })
    return timeline


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

def _load_api_config():
    """Load MiniMax API config for Q&A feature."""
    key = os.environ.get("MINIMAX_API_KEY", "")
    if not key:
        keyfile = VISUAL_DIR.parent.parent / "apikey.txt"
        if keyfile.exists():
            with open(keyfile) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("sk-"):
                        key = line
                        break
    return {
        "api_key": key,
        "base_url": os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
        "model": os.environ.get("MINIMAX_MODEL", "MiniMax-M2.5-highspeed"),
    }

API_CONFIG = _load_api_config()


def _llm_qa(question: str, context: str) -> str:
    """Ask MiniMax LLM a question with context."""
    if not API_CONFIG["api_key"]:
        return "Error: No API key configured. Set MINIMAX_API_KEY environment variable."

    payload = json.dumps({
        "model": API_CONFIG["model"],
        "messages": [
            {"role": "system", "content": (
                "You are a research assistant helping a project manager understand "
                "AI research results. Explain clearly and concisely. "
                "Use the provided context to answer questions. "
                "If the context contains code, explain what it does. "
                "If it contains results, interpret them. "
                "Respond in the same language as the question."
            )},
            {"role": "user", "content": f"## Context\n{context[:4000]}\n\n## Question\n{question}"},
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{API_CONFIG['base_url']}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_CONFIG['api_key']}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"


class DashboardHandler(SimpleHTTPRequestHandler):
    """Regex-based API router + static file server."""

    missions_dir = DEFAULT_MISSIONS

    # Route table: (pattern, handler_method_name)
    ROUTES = [
        (r"^/api/missions$", "_api_missions"),
        (r"^/api/comparisons$", "_api_comparisons"),
        (r"^/api/mission/([^/]+)$", "_api_mission_detail"),
        (r"^/api/mission/([^/]+)/score$", "_api_mission_score"),
        (r"^/api/mission/([^/]+)/insights$", "_api_insights"),
        (r"^/api/mission/([^/]+)/code$", "_api_code"),
        (r"^/api/mission/([^/]+)/code/([^/]+)/diff/([^/]+)/([^/]+)$", "_api_diff"),
        (r"^/api/mission/([^/]+)/knowledge$", "_api_knowledge"),
        (r"^/api/mission/([^/]+)/reports$", "_api_reports"),
        (r"^/api/mission/([^/]+)/reports/(.+)$", "_api_report_content"),
        (r"^/api/mission/([^/]+)/timeline$", "_api_timeline"),
        (r"^/api/mission/([^/]+)/workspace$", "_api_workspace_files"),
        (r"^/api/mission/([^/]+)/workspace/(.+)$", "_api_workspace_file"),
    ]

    def do_POST(self):
        path = unquote(self.path.split("?")[0])
        if path == "/api/qa":
            self._api_qa()
        else:
            self._json_response({"error": "Not found"}, 404)

    def do_GET(self):
        path = unquote(self.path.split("?")[0])

        # Try API routes
        for pattern, method_name in self.ROUTES:
            m = re.match(pattern, path)
            if m:
                handler = getattr(self, method_name)
                handler(*m.groups())
                return

        # Serve static files from visual/static/
        if path == "/" or path == "":
            path = "/index.html"
        file_path = STATIC_DIR / path.lstrip("/")
        if file_path.is_file():
            self._serve_file(file_path)
        else:
            self._json_response({"error": "Not found"}, 404)

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, file_path):
        ext = file_path.suffix.lower()
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
        }
        ct = content_types.get(ext, "application/octet-stream")
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        except Exception:
            self._json_response({"error": "File read error"}, 500)

    # --- API handlers ---

    def _api_missions(self):
        self._json_response(_get_missions(self.missions_dir))

    def _api_comparisons(self):
        self._json_response(_get_comparisons(self.missions_dir))

    def _api_mission_detail(self, mission_id):
        data = _get_mission_detail(self.missions_dir, mission_id)
        if data is None:
            self._json_response({"error": "Mission not found"}, 404)
        else:
            self._json_response(data)

    def _api_mission_score(self, mission_id):
        data = _get_mission_score(self.missions_dir, mission_id)
        if data is None:
            self._json_response({"error": "No score available"}, 404)
        else:
            self._json_response(data)

    def _api_insights(self, mission_id):
        self._json_response(_get_insights(self.missions_dir, mission_id))

    def _api_code(self, mission_id):
        self._json_response(_get_code(self.missions_dir, mission_id))

    def _api_diff(self, mission_id, file_stem, v1, v2):
        content = _get_diff(self.missions_dir, mission_id, file_stem, v1, v2)
        if content is None:
            self._json_response({"error": "Diff not found"}, 404)
        else:
            self._json_response({"diff": content})

    def _api_knowledge(self, mission_id):
        self._json_response(_get_knowledge(self.missions_dir, mission_id))

    def _api_reports(self, mission_id):
        self._json_response(_get_reports(self.missions_dir, mission_id))

    def _api_report_content(self, mission_id, filename):
        content = _get_report_content(self.missions_dir, mission_id, filename)
        if content is None:
            self._json_response({"error": "Report not found"}, 404)
        else:
            self._json_response({"filename": filename, "content": content})

    def _api_timeline(self, mission_id):
        self._json_response(_get_timeline(self.missions_dir, mission_id))

    def _api_workspace_files(self, mission_id):
        self._json_response(_get_workspace_files(self.missions_dir, mission_id))

    def _api_workspace_file(self, mission_id, filepath):
        full_path = _get_workspace_file_content(self.missions_dir, mission_id, filepath)
        if full_path is None:
            self._json_response({"error": "File not found"}, 404)
            return
        ext = full_path.suffix.lower()
        if ext in (".png", ".jpg", ".jpeg", ".gif", ".svg"):
            self._serve_file(full_path)
        else:
            content = _read_text(full_path)
            if content is None:
                self._json_response({"error": "Cannot read file"}, 500)
            else:
                self._json_response({"path": filepath, "content": content})

    def _api_qa(self):
        """Handle Q&A questions via LLM."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            question = data.get("question", "")
            context = data.get("context", "")
            if not question:
                self._json_response({"error": "No question provided"}, 400)
                return
            answer = _llm_qa(question, context)
            self._json_response({"answer": answer})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def log_message(self, format, *args):
        """Quieter logging — only show API requests."""
        msg = format % args
        if "/api/" in msg:
            super().log_message(format, *args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Research AI Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--missions", type=str, default=None, help="Path to missions/ directory")
    args = parser.parse_args()

    missions_dir = Path(args.missions) if args.missions else DEFAULT_MISSIONS
    DashboardHandler.missions_dir = missions_dir

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"Dashboard server starting on http://localhost:{args.port}")
    print(f"Reading missions from: {missions_dir}")
    print(f"Serving static files from: {STATIC_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
