"""
Configuration management for AI Research Agent
================================================
Centralizes API keys, paths, runtime limits, and hardware detection.
"""

import os
import platform
import subprocess
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── API Configuration ───────────────────────────────────────────────
def _load_api_key() -> str:
    """Load MiniMax API key from apikey.txt or environment."""
    if key := os.environ.get("MINIMAX_API_KEY"):
        return key
    keyfile = os.path.join(os.path.dirname(BASE_DIR), "apikey.txt")
    if os.path.exists(keyfile):
        with open(keyfile) as f:
            for line in f:
                line = line.strip()
                if line.startswith("sk-"):
                    return line
    raise RuntimeError("No MiniMax API key found. Set MINIMAX_API_KEY or add to apikey.txt")

API_KEY = _load_api_key()
BASE_URL = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.5-highspeed")

# ── Runtime Limits ──────────────────────────────────────────────────
MAX_TURNS = int(os.environ.get("MAX_TURNS", "10"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4096"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))
API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "120"))
MAX_RETRIES = 3

# ── Paths ───────────────────────────────────────────────────────────
MISSIONS_DIR = os.path.join(BASE_DIR, "missions")
MCP_SERVERS_DIR = os.path.join(BASE_DIR, "mcp_servers")
GENERATED_MCP_DIR = os.path.join(MCP_SERVERS_DIR, "generated")
SKILLS_DIR = os.path.join(BASE_DIR, "skills")

# Ensure shared directories exist
for d in [MISSIONS_DIR, GENERATED_MCP_DIR, SKILLS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Hardware Detection ─────────────────────────────────────────────
def _detect_hardware() -> dict:
    """Detect GPU and compute environment at startup."""
    hw = {
        "platform": platform.system(),        # Darwin / Linux / Windows
        "arch": platform.machine(),            # arm64 / x86_64
        "python": platform.python_version(),
        "gpu": "none",
        "gpu_name": "",
        "gpu_memory": "",
        "device": "cpu",                       # torch device string
        "packages": [],                        # available ML packages
    }

    # Check installed packages (use importlib to avoid side-effect crashes)
    import importlib.util
    for pkg in ["torch", "numpy", "transformers", "jax", "tensorflow"]:
        if importlib.util.find_spec(pkg) is not None:
            hw["packages"].append(pkg)

    # ── CUDA (NVIDIA) ─────────────────────────────────────────────
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                parts = out.stdout.strip().split("\n")[0].split(", ")
                hw["gpu"] = "cuda"
                hw["gpu_name"] = parts[0].strip()
                hw["gpu_memory"] = f"{parts[1].strip()} MB" if len(parts) > 1 else ""
                hw["device"] = "cuda"
        except Exception:
            pass

    # ── MPS (Apple Silicon) ───────────────────────────────────────
    if hw["gpu"] == "none" and hw["platform"] == "Darwin" and hw["arch"] == "arm64":
        # Apple Silicon always has MPS, check if torch supports it
        if "torch" in hw["packages"]:
            try:
                import torch
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    hw["gpu"] = "mps"
                    hw["device"] = "mps"
            except Exception:
                hw["gpu"] = "mps"
                hw["device"] = "mps"
        else:
            # torch not installed but hardware is capable
            hw["gpu"] = "mps (torch not installed)"
            hw["device"] = "cpu"

        # Get chip name from sysctl
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode == 0:
                hw["gpu_name"] = out.stdout.strip()
        except Exception:
            pass

        # Get memory
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode == 0:
                mem_gb = int(out.stdout.strip()) / (1024**3)
                hw["gpu_memory"] = f"{mem_gb:.0f} GB (unified)"
        except Exception:
            pass

    return hw


HW_ENV = _detect_hardware()

# Build a concise summary string that gets injected into worker prompts
_lines = [f"Platform: {HW_ENV['platform']} {HW_ENV['arch']}"]
if HW_ENV["gpu"] != "none":
    _lines.append(f"GPU: {HW_ENV['gpu'].upper()} — {HW_ENV['gpu_name']}")
    if HW_ENV["gpu_memory"]:
        _lines.append(f"GPU Memory: {HW_ENV['gpu_memory']}")
    _lines.append(f"Torch device: {HW_ENV['device']}")
else:
    _lines.append("GPU: none (CPU only)")
if HW_ENV["packages"]:
    _lines.append(f"Installed ML packages: {', '.join(HW_ENV['packages'])}")
else:
    _lines.append("Installed ML packages: none (only Python standard library available)")
HW_ENV_SUMMARY = "\n".join(_lines)
