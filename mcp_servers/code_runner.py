"""
程式執行 MCP Server
===========================
提供以下 tools 給 Agent 呼叫：
- run_python_code: 在 subprocess 中執行 Python 程式碼（支援外部套件）
- write_file: 寫入檔案到 workspace
- read_file: 讀取 workspace 中的檔案
- pip_install: 安裝 Python 套件
- detect_hardware: 偵測 GPU / 硬體環境

Workspace scoping:
    By default, tools operate in a global workspace (ai_research_agent/workspace/).
    Use create_workspace_tools(path) to get mission-scoped tool functions
    bound to a specific workspace directory.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile

# Default workspace (backward-compat fallback for standalone use)
_DEFAULT_WORKSPACE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workspace")
os.makedirs(_DEFAULT_WORKSPACE, exist_ok=True)

# Default timeout: 300s (5 minutes) for GPU workloads
DEFAULT_TIMEOUT = int(os.environ.get("CODE_TIMEOUT", "300"))

# Memory limit per subprocess (in bytes). Default 6GB — prevents Mac OOM reboot.
_MEM_LIMIT_BYTES = int(os.environ.get("CODE_MEM_LIMIT", str(6 * 1024**3)))

# Track spawned process PIDs for mission-end cleanup
_active_pids: set = set()


# ── Workspace-scoped tool factory ────────────────────────────────────

def create_workspace_tools(workspace_dir: str) -> dict:
    """
    Create tool functions scoped to a specific workspace directory.

    Returns a dict of {name: function} that can replace the default
    module-level functions in the ToolRegistry after registration.

    This is the key to mission-isolated file I/O: each mission gets
    its own workspace, and all code execution happens there.
    """
    os.makedirs(workspace_dir, exist_ok=True)

    def _run_python_code(code: str, timeout: int = None) -> dict:
        """Execute Python code in a subprocess within the scoped workspace.
        Uses process groups to ensure all child processes are cleaned up."""
        if timeout is None:
            timeout = DEFAULT_TIMEOUT
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=workspace_dir, delete=False
        ) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        # Set memory limit via resource module (soft limit only, hard stays unchanged)
        def _set_mem_limit():
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_AS,
                                   (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))
            except (ImportError, ValueError, OSError):
                pass  # Non-Unix or limit not supported

        proc = subprocess.Popen(
            [sys.executable, tmp_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=workspace_dir,
            start_new_session=True,  # New process group for cleanup
            preexec_fn=_set_mem_limit,
        )
        _active_pids.add(proc.pid)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return {
                "success": proc.returncode == 0,
                "stdout": stdout[:5000],
                "stderr": stderr[:3000],
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            # Kill entire process group on timeout
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
            proc.wait(timeout=5)
            return {"success": False, "stdout": "", "stderr": f"Timeout after {timeout}s", "returncode": -1}
        finally:
            _active_pids.discard(proc.pid)
            # Kill any lingering children in the process group
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _write_file(filename: str, content: str) -> dict:
        """Write a file to the scoped workspace."""
        safe_name = os.path.basename(filename)
        path = os.path.join(workspace_dir, safe_name)
        with open(path, "w") as f:
            f.write(content)
        return {"success": True, "path": path, "size": len(content)}

    def _read_file(filename: str) -> dict:
        """Read a file from the scoped workspace."""
        safe_name = os.path.basename(filename)
        path = os.path.join(workspace_dir, safe_name)
        if not os.path.exists(path):
            return {"success": False, "error": f"File not found: {safe_name}"}
        with open(path) as f:
            content = f.read()
        return {"success": True, "filename": safe_name, "content": content[:5000], "size": len(content)}

    return {
        "run_python_code": _run_python_code,
        "write_file": _write_file,
        "read_file": _read_file,
    }


# ── Default (unscoped) functions — used when registered as a module ──

def run_python_code(code: str, timeout: int = None) -> dict:
    """在 subprocess 中執行 Python 程式碼（可用 torch, numpy 等已安裝套件）
    Uses process groups to ensure all child processes are cleaned up."""
    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=_DEFAULT_WORKSPACE, delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    def _set_mem_limit():
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS,
                               (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))
        except (ImportError, ValueError, OSError):
            pass

    proc = subprocess.Popen(
        [sys.executable, tmp_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=_DEFAULT_WORKSPACE,
        start_new_session=True,
        preexec_fn=_set_mem_limit,
    )
    _active_pids.add(proc.pid)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return {
            "success": proc.returncode == 0,
            "stdout": stdout[:5000],
            "stderr": stderr[:3000],
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        proc.wait(timeout=5)
        return {"success": False, "stdout": "", "stderr": f"Timeout after {timeout}s", "returncode": -1}
    finally:
        _active_pids.discard(proc.pid)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def write_file(filename: str, content: str) -> dict:
    """寫入檔案到 workspace"""
    safe_name = os.path.basename(filename)
    path = os.path.join(_DEFAULT_WORKSPACE, safe_name)
    with open(path, "w") as f:
        f.write(content)
    return {"success": True, "path": path, "size": len(content)}


def read_file(filename: str) -> dict:
    """讀取 workspace 中的檔案"""
    safe_name = os.path.basename(filename)
    path = os.path.join(_DEFAULT_WORKSPACE, safe_name)
    if not os.path.exists(path):
        return {"success": False, "error": f"File not found: {safe_name}"}
    with open(path) as f:
        content = f.read()
    return {"success": True, "filename": safe_name, "content": content[:5000], "size": len(content)}


def pip_install(packages: str) -> dict:
    """安裝 Python 套件（例如 'torch numpy transformers'）"""
    pkg_list = packages.strip().split()
    if not pkg_list:
        return {"success": False, "error": "No packages specified"}

    # Safety: block obviously dangerous packages
    blocked = {"os", "sys", "shutil", "subprocess", "ctypes", "socket"}
    for p in pkg_list:
        base = p.split("==")[0].split(">=")[0].split("<=")[0].lower()
        if base in blocked:
            return {"success": False, "error": f"Package '{base}' is blocked for safety"}

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + pkg_list,
            capture_output=True, text=True, timeout=120,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:2000],
            "installed": pkg_list,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "pip install timed out after 120s"}


def detect_hardware() -> dict:
    """偵測當前硬體環境（GPU、記憶體、已安裝套件）"""
    from config import HW_ENV
    return HW_ENV


# ── Tool 定義 ─────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Execute Python code in a subprocess. Can use any installed package (torch, numpy, etc). Default timeout 300s.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 300, max 600)", "default": 300},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename (no path, written to workspace/)"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename to read"},
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pip_install",
            "description": "Install Python packages using pip. Example: 'torch numpy' or 'transformers>=4.30'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "packages": {"type": "string", "description": "Space-separated package names (e.g. 'torch numpy matplotlib')"},
                },
                "required": ["packages"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_hardware",
            "description": "Detect available GPU, memory, and installed ML packages on this machine.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_FUNCTIONS = {
    "run_python_code": run_python_code,
    "write_file": write_file,
    "read_file": read_file,
    "pip_install": pip_install,
    "detect_hardware": detect_hardware,
}
