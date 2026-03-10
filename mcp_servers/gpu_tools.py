"""
GPU Tools — Remote GPU execution for heavy compute tasks.
==========================================================
Provides tools for workers to run code on remote GPUs via vast.ai.
Handles instance lifecycle, file transfer, and cost tracking.
"""

import json
import os

from core.gpu_manager import (
    search_gpus, create_instance, show_instances, get_ssh_command,
    execute_remote, stop_instance, destroy_instance, transfer_files,
    get_usage_summary, GPUSession, log_session, wait_for_ssh,
)


# ── Tool definitions (OpenAI format) ──────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "gpu_search",
            "description": "Search for available GPU instances on vast.ai. Returns cheapest offers matching requirements. Use before gpu_create to find the right GPU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_ram_gb": {
                        "type": "integer",
                        "description": "Minimum GPU VRAM in GB (default 24). Use 24 for LoRA/7B models, 40 for full 13B, 80 for 70B.",
                        "default": 24,
                    },
                    "max_cost_per_hour": {
                        "type": "number",
                        "description": "Maximum cost in $/hr (default 0.35). RTX 4090 ≈ $0.25/hr, A100 ≈ $0.50/hr.",
                        "default": 0.35,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gpu_run",
            "description": "Run Python code on a remote GPU. Automatically finds cheapest GPU, creates instance, runs code, retrieves output, and destroys instance. Use for: training models, fine-tuning, heavy inference. Cost is tracked automatically. ALWAYS prefer this over local CPU for tasks >5 minutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute on GPU. Must be self-contained (include all imports, data download, training, and result saving). Print results to stdout — they will be captured.",
                    },
                    "min_ram_gb": {
                        "type": "integer",
                        "description": "Minimum GPU VRAM in GB (default 24)",
                        "default": 24,
                    },
                    "max_cost_per_hour": {
                        "type": "number",
                        "description": "Maximum cost in $/hr (default 0.35)",
                        "default": 0.35,
                    },
                    "timeout_minutes": {
                        "type": "integer",
                        "description": "Maximum runtime in minutes before auto-destroy (default 30)",
                        "default": 30,
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gpu_status",
            "description": "Show current GPU instances and cumulative usage/cost summary. Use to check if instances are running, and to track total spend.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


# ── Tool implementations ──────────────────────────────────

def _gpu_search(min_ram_gb: int = 24, max_cost_per_hour: float = 0.35,
                **kwargs) -> str:
    """Search for GPU offers."""
    try:
        offers = search_gpus(
            min_ram_gb=min_ram_gb,
            max_cost_per_hour=max_cost_per_hour,
        )
        if not offers:
            return json.dumps({
                "error": f"No GPUs found with ≥{min_ram_gb}GB RAM and ≤${max_cost_per_hour}/hr. "
                         "Try increasing max_cost_per_hour or decreasing min_ram_gb.",
            })
        return json.dumps({"offers": offers[:5]}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _gpu_run(code: str, min_ram_gb: int = 24, max_cost_per_hour: float = 0.35,
             timeout_minutes: int = 30, **kwargs) -> str:
    """Run code on a remote GPU."""
    from core.gpu_manager import quick_gpu_run

    # Get mission_id from workspace context if available
    mission_id = kwargs.get("_mission_id", "unknown")

    try:
        result = quick_gpu_run(
            code=code,
            mission_id=mission_id,
            min_ram_gb=min_ram_gb,
            max_cost=max_cost_per_hour,
            timeout_minutes=timeout_minutes,
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "cost": 0})


def _gpu_status(**kwargs) -> str:
    """Show instances and usage summary."""
    try:
        instances = show_instances()
        usage = get_usage_summary()

        instance_list = []
        for inst in instances:
            instance_list.append({
                "id": inst.get("id"),
                "gpu": inst.get("gpu_name", "?"),
                "status": inst.get("actual_status", inst.get("status_msg", "?")),
                "cost_per_hour": inst.get("dph_total", 0),
            })

        return json.dumps({
            "active_instances": instance_list,
            "usage_summary": usage,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


TOOL_FUNCTIONS = {
    "gpu_search": _gpu_search,
    "gpu_run": _gpu_run,
    "gpu_status": _gpu_status,
}
