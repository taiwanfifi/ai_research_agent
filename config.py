"""
Configuration management for AI Research Agent
================================================
Centralizes API keys, paths, and runtime limits.
"""

import os

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
MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.5")

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

# Legacy paths — kept for agent.py backward compatibility
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge_base")
STATE_DIR = os.path.join(BASE_DIR, ".state")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

# Ensure shared directories exist
for d in [MISSIONS_DIR, GENERATED_MCP_DIR, SKILLS_DIR]:
    os.makedirs(d, exist_ok=True)
