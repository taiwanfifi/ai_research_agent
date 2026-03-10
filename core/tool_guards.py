"""
Tool Guards — Deterministic Preflight Checks
=============================================
Enforce hard constraints mechanically in the tool layer.
Rules that can be checked without LLM judgment go here,
not in prompts where the LLM may ignore them.

Returns structured preflight errors with suggested fixes.
"""

import json
import os
import re


def guard_run_python_code(code: str, workspace_dir: str = "",
                          **kwargs) -> dict | None:
    """Preflight checks before executing Python code.

    Returns None if OK, or a structured error dict if blocked.
    """
    # 1. Block exec(open(...)) — bypasses timeout, untrackable
    if re.search(r'exec\s*\(\s*open\s*\(', code):
        return {
            "blocked": True,
            "error_type": "banned_pattern",
            "reason": "exec(open(...)) bypasses timeout and tracking",
            "suggested_actions": [
                "Paste the code directly into run_python_code"
            ],
        }

    # 2. Block plt.show() — hangs in headless subprocess
    if "plt.show()" in code and "plt.savefig" not in code:
        return {
            "blocked": True,
            "error_type": "banned_pattern",
            "reason": "plt.show() blocks execution in subprocess",
            "suggested_actions": [
                "Replace plt.show() with plt.savefig('figure.png', dpi=150, bbox_inches='tight')",
                "Add plt.close() after savefig",
            ],
        }

    # 3. Block bitsandbytes optimizer on Mac
    if ("adamw_bnb" in code or "bitsandbytes" in code.lower()) and \
       "optim" in code:
        # Check if it's actually setting an optimizer, not just importing
        if re.search(r'optim\s*=\s*["\']adamw_bnb', code):
            return {
                "blocked": True,
                "error_type": "env_incompatible",
                "reason": "bitsandbytes optimizers crash on Mac (NoneType error)",
                "suggested_actions": [
                    "Use optim='adamw_torch' instead"
                ],
            }

    # 4. Warn if training without dataset subset (likely timeout)
    if _looks_like_training(code) and not _has_dataset_limit(code):
        # Advisory, not blocking — return as warning
        return {
            "blocked": False,
            "error_type": "timeout_risk",
            "reason": "Training on full dataset may exceed 600s timeout",
            "suggested_actions": [
                "Add dataset.select(range(2000)) for training subset",
                "Or add explicit timing to monitor progress",
            ],
        }

    # 5. Block evaluation_strategy (removed in transformers 4.50+)
    if "evaluation_strategy" in code and "eval_strategy" not in code:
        return {
            "blocked": True,
            "error_type": "deprecated_api",
            "reason": "evaluation_strategy removed in transformers 4.50+",
            "suggested_actions": [
                "Replace evaluation_strategy with eval_strategy"
            ],
        }

    # 6. Block HuggingFace load_dataset for standard vision datasets (causes 'list'.lower() error)
    if _uses_hf_for_vision(code):
        return {
            "blocked": True,
            "error_type": "known_bug",
            "reason": "HuggingFace load_dataset for CIFAR-10/MNIST causes 'list'.lower() error. Use torchvision instead.",
            "suggested_actions": [
                "Replace `from datasets import load_dataset` with `import torchvision`",
                "Use `torchvision.datasets.CIFAR10(root, train=True, download=True, transform=transform)`",
                "Use `torch.utils.data.Subset(dataset, range(N))` for subsetting",
            ],
        }

    # 7. Check for PEFT/LoRA training without CPU forcing (regular fine-tuning can use MPS)
    if _is_peft_training(code) and not _forces_cpu(code):
        return {
            "blocked": False,
            "error_type": "env_incompatible",
            "reason": "PEFT/LoRA training crashes on MPS (NoneType errors). Regular fine-tuning without PEFT can use MPS.",
            "suggested_actions": [
                "Add os.environ['CUDA_VISIBLE_DEVICES'] = '' at top of script",
                "Or remove PEFT/LoRA and use regular fine-tuning with MPS",
            ],
        }

    return None


def guard_pip_install(packages: str, **kwargs) -> dict | None:
    """Preflight checks before installing packages."""
    pkg_list = packages.strip().split()

    # Block bitsandbytes on Mac
    for p in pkg_list:
        base = p.split("==")[0].split(">=")[0].split("<=")[0].lower()
        if base == "bitsandbytes":
            return {
                "blocked": False,
                "error_type": "env_incompatible",
                "reason": "bitsandbytes does not work on macOS",
                "suggested_actions": [
                    "Use standard PyTorch optimizers instead",
                    "Skip bitsandbytes quantization on this platform",
                ],
            }

    return None


def guard_write_file(filename: str = "", content: str = "",
                     workspace_dir: str = "", **kwargs) -> dict | None:
    """Preflight checks before writing a file."""
    # Missing required args — let the tool itself fail with a clear message
    if not filename or not content:
        return None

    # Block HuggingFace load_dataset for vision datasets in Python files
    if filename.endswith(".py") and _uses_hf_for_vision(content):
        return {
            "blocked": True,
            "error_type": "known_bug",
            "reason": "HuggingFace load_dataset for CIFAR-10/MNIST causes 'list'.lower() error. Use torchvision instead.",
            "suggested_actions": [
                "Use torchvision.datasets.CIFAR10/MNIST instead of load_dataset",
                "Use torch.utils.data.Subset for subsetting",
            ],
        }

    # Block writing outside workspace (path traversal)
    if ".." in filename or "/" in filename:
        safe = os.path.basename(filename)
        if safe != filename:
            return {
                "blocked": True,
                "error_type": "path_traversal",
                "reason": f"Cannot write outside workspace. Use basename only.",
                "suggested_actions": [
                    f"Use filename='{safe}' instead of '{filename}'"
                ],
            }

    return None


# ── Guard registry ──────────────────────────────────────────────

GUARDS = {
    "run_python_code": guard_run_python_code,
    "pip_install": guard_pip_install,
    "write_file": guard_write_file,
}


def run_guard(tool_name: str, arguments: dict,
              workspace_dir: str = "") -> dict | None:
    """Run preflight guard for a tool. Returns None if OK, error dict if not.

    The caller should:
    - If result is None: proceed with tool execution
    - If result["blocked"] is True: return error to LLM, do NOT execute
    - If result["blocked"] is False: inject warning, still execute
    """
    guard_fn = GUARDS.get(tool_name)
    if not guard_fn:
        return None
    return guard_fn(workspace_dir=workspace_dir, **arguments)


# ── Helpers ─────────────────────────────────────────────────────

def _looks_like_training(code: str) -> bool:
    """Heuristic: does this code do model training?"""
    indicators = ["Trainer(", ".train()", "num_train_epochs", "training_args",
                   "TrainingArguments", "for epoch in"]
    return sum(1 for i in indicators if i in code) >= 2


def _has_dataset_limit(code: str) -> bool:
    """Check if code limits dataset size."""
    patterns = [".select(range(", "[:2000]", "[:1000]", "[:500]",
                "max_samples", "subset", ".head("]
    return any(p in code for p in patterns)


def _is_peft_training(code: str) -> bool:
    """Check if code does PEFT/LoRA training."""
    has_peft = any(k in code for k in ["LoraConfig", "get_peft_model",
                                        "from peft", "TaskType"])
    has_train = any(k in code for k in [".train()", "Trainer("])
    return has_peft and has_train


def _uses_hf_for_vision(code: str) -> bool:
    """Check if code uses HuggingFace load_dataset for standard vision datasets."""
    has_load_dataset = "load_dataset" in code
    vision_datasets = ["cifar", "mnist", "svhn", "fashion_mnist", "imagenet"]
    has_vision = any(d in code.lower() for d in vision_datasets)
    return has_load_dataset and has_vision


def _forces_cpu(code: str) -> bool:
    """Check if code explicitly forces CPU."""
    patterns = ["CUDA_VISIBLE_DEVICES", "device('cpu')",
                'device("cpu")', "no_cuda=True", "use_cpu=True"]
    return any(p in code for p in patterns)
