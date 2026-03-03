"""
程式執行 MCP Server (模擬)
===========================
提供以下 tools 給 Agent 呼叫：
- run_python_code: 在 subprocess 中執行 Python 程式碼
- write_file: 寫入檔案到 workspace
- read_file: 讀取 workspace 中的檔案
"""

import os
import subprocess
import tempfile

WORKSPACE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "workspace")
os.makedirs(WORKSPACE, exist_ok=True)


def run_python_code(code: str, timeout: int = 30) -> dict:
    """在 subprocess 中安全執行 Python 程式碼"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir=WORKSPACE, delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=WORKSPACE,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:2000],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"Timeout after {timeout}s", "returncode": -1}
    finally:
        os.unlink(tmp_path)


def write_file(filename: str, content: str) -> dict:
    """寫入檔案到 workspace"""
    # 安全檢查：不允許路徑穿越
    safe_name = os.path.basename(filename)
    path = os.path.join(WORKSPACE, safe_name)
    with open(path, "w") as f:
        f.write(content)
    return {"success": True, "path": path, "size": len(content)}


def read_file(filename: str) -> dict:
    """讀取 workspace 中的檔案"""
    safe_name = os.path.basename(filename)
    path = os.path.join(WORKSPACE, safe_name)
    if not os.path.exists(path):
        return {"success": False, "error": f"File not found: {safe_name}"}
    with open(path) as f:
        content = f.read()
    return {"success": True, "filename": safe_name, "content": content[:5000], "size": len(content)}


# ── Tool 定義 ─────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python_code",
            "description": "Execute Python code in a sandboxed subprocess. Returns stdout, stderr, and exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
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
]

TOOL_FUNCTIONS = {
    "run_python_code": run_python_code,
    "write_file": write_file,
    "read_file": read_file,
}
