"""
Error Pattern Matcher — Deterministic Error Classification
===========================================================
Instant regex-based error classification with fix hints.
Zero LLM cost. Eliminates guesswork in failure analysis.

From ICSE 2025: only 17% of LLM code errors are auto-fixable.
This module classifies errors to decide: fix, amnesia, alternatives, or backtrack.
"""

import re
from dataclasses import dataclass


@dataclass
class ErrorPattern:
    name: str
    category: str          # architecture, device, training, import, data, syntax
    patterns: list[str]    # regex patterns to match
    fix_hint: str          # specific fix guidance
    auto_fixable: bool     # can be fixed with a simple edit?
    severity: str = "high" # high, medium, low


ERROR_PATTERNS = [
    # ── Shape/Dimension errors (most common, NOT auto-fixable) ──
    ErrorPattern(
        name="shape_mismatch",
        category="architecture",
        patterns=[
            r"RuntimeError.*size mismatch",
            r"RuntimeError.*mat1 and mat2",
            r"RuntimeError.*Expected.*channels.*got",
            r"RuntimeError.*shape.*is invalid for input of size",
        ],
        fix_hint=(
            "Layer dimension mismatch. For CIFAR-10 after 5 MaxPool: spatial=1x1, "
            "FC input=512. For MNIST after 2 MaxPool(2): spatial=7x7. "
            "Print x.shape after each layer to debug. "
            "Use nn.AdaptiveAvgPool2d((1,1)) before Flatten to handle any spatial size."
        ),
        auto_fixable=False,
    ),
    ErrorPattern(
        name="linear_input_mismatch",
        category="architecture",
        patterns=[
            r"RuntimeError.*in_features.*expected.*but got",
        ],
        fix_hint=(
            "Linear layer input size wrong. Common fix: add a dummy forward pass "
            "with x=torch.randn(1,3,32,32) and print x.shape before the Linear layer "
            "to find the correct input size."
        ),
        auto_fixable=False,
    ),

    # ── Device errors (auto-fixable) ──
    ErrorPattern(
        name="device_mismatch",
        category="device",
        patterns=[
            r"RuntimeError.*expected.*device.*got",
            r"RuntimeError.*Input type.*and weight type.*should be the same",
            r"Expected.*tensor.*cuda.*cpu",
        ],
        fix_hint="Move all tensors and model to same device. Add .to(device) for model, inputs, and targets.",
        auto_fixable=True,
    ),

    # ── Training errors ──
    ErrorPattern(
        name="nan_loss",
        category="training",
        patterns=[
            r"loss.*nan",
            r"NaN.*loss",
            r"RuntimeError.*element.*nan",
        ],
        fix_hint="Learning rate too high or numerical instability. Try lr *= 0.1 or add gradient clipping.",
        auto_fixable=False,
    ),
    ErrorPattern(
        name="no_grad",
        category="training",
        patterns=[
            r"RuntimeError.*element.*does not require grad",
            r"RuntimeError.*grad can be implicitly created",
        ],
        fix_hint="Model parameters don't have requires_grad=True, or using no_grad context during training.",
        auto_fixable=True,
    ),
    ErrorPattern(
        name="adamw_import",
        category="training",
        patterns=[
            r"FutureWarning.*AdamW",
            r"ImportError.*optim\.AdamW",
        ],
        fix_hint="Use torch.optim.AdamW (not transformers.AdamW). Import: from torch.optim import AdamW",
        auto_fixable=True,
    ),

    # ── Data loading errors ──
    ErrorPattern(
        name="data_type_error",
        category="data",
        patterns=[
            r"expected string or bytes-like object.*got.*list",
            r"'list' object has no attribute 'lower'",
            r"TypeError.*expected str.*got list",
        ],
        fix_hint=(
            "Data loading: passing a list where a string is expected. "
            "FIX: Use torchvision.datasets (CIFAR10, MNIST) instead of HuggingFace datasets. "
            "If using HF: ensure tokenizer receives individual strings, not lists — "
            "use `dataset.map(lambda x: tokenizer(x['text']), batched=False)` "
            "or access `example['text']` not `batch['text']`. "
            "Also check: split='train' not ['train']."
        ),
        auto_fixable=True,
    ),
    ErrorPattern(
        name="dataset_not_found",
        category="data",
        patterns=[
            r"FileNotFoundError.*data",
            r"RuntimeError.*Dataset not found",
            r"OSError.*Can't load dataset",
        ],
        fix_hint="Dataset download failed. Add download=True to dataset constructor. Check internet connectivity.",
        auto_fixable=True,
    ),
    ErrorPattern(
        name="tokenizer_error",
        category="data",
        patterns=[
            r"Can't load tokenizer",
            r"OSError.*is not a local folder",
            r"HTTPError.*huggingface",
        ],
        fix_hint="HuggingFace model/tokenizer name wrong or download failed. Check model name spelling.",
        auto_fixable=True,
    ),

    # ── Import errors (auto-fixable) ──
    ErrorPattern(
        name="import_error",
        category="import",
        patterns=[
            r"ModuleNotFoundError.*No module named",
            r"ImportError.*cannot import name",
        ],
        fix_hint="Missing import. Check that all required packages are installed and import paths are correct.",
        auto_fixable=True,
    ),

    # ── Memory errors ──
    ErrorPattern(
        name="oom",
        category="memory",
        patterns=[
            r"CUDA out of memory",
            r"RuntimeError.*out of memory",
            r"MemoryError",
        ],
        fix_hint=(
            "Out of memory. Reduce batch_size by 50% or use fewer samples. "
            "For transformers: use gradient_checkpointing or fp16. "
            "For PEFT/LoRA: always use CPU, never MPS."
        ),
        auto_fixable=False,
    ),

    # ── MPS/Apple Silicon errors ──
    ErrorPattern(
        name="mps_error",
        category="device",
        patterns=[
            r"NotImplementedError.*MPS",
            r"RuntimeError.*MPS backend",
            r"mps.*not.*support",
        ],
        fix_hint="MPS (Apple GPU) doesn't support this operation. Use device='cpu' instead.",
        auto_fixable=True,
    ),

    # ── Timeout/process errors ──
    ErrorPattern(
        name="timeout",
        category="execution",
        patterns=[
            r"TimeoutError",
            r"timed out",
            r"exceeded.*timeout",
        ],
        fix_hint=(
            "Execution timed out. Reduce: num_epochs, num_samples, or model complexity. "
            "Use 1-2 epochs and 1000 samples for fast iteration."
        ),
        auto_fixable=False,
    ),

    # ── Syntax errors (auto-fixable) ──
    ErrorPattern(
        name="syntax_error",
        category="syntax",
        patterns=[
            r"SyntaxError",
            r"IndentationError",
        ],
        fix_hint="Syntax or indentation error in generated code. Fix the specific line.",
        auto_fixable=True,
    ),
]


def classify_error(stderr: str) -> ErrorPattern | None:
    """Classify an error by matching against known patterns.

    Returns the first matching ErrorPattern, or None if no match.
    Zero LLM cost — pure regex.
    """
    if not stderr:
        return None

    for pattern in ERROR_PATTERNS:
        for regex in pattern.patterns:
            if re.search(regex, stderr, re.IGNORECASE):
                return pattern

    return None


def get_escalation_level(consecutive_failures: int,
                         error_pattern: ErrorPattern | None) -> str:
    """Determine escalation level based on failure count + error type.

    Returns:
        "RETRY_WITH_FIX" — simple fix, try again (level 1)
        "RETRY_WITH_AMNESIA" — compress failure context, try fresh (level 2)
        "GENERATE_ALTERNATIVES" — try 3 fundamentally different approaches (level 3)
        "BACKTRACK_TO_PARENT" — abandon this branch entirely (level 4)
    """
    # Auto-fixable errors get more retries
    if error_pattern and error_pattern.auto_fixable:
        if consecutive_failures <= 2:
            return "RETRY_WITH_FIX"
        elif consecutive_failures == 3:
            return "RETRY_WITH_AMNESIA"
        else:
            return "GENERATE_ALTERNATIVES"

    # Non-auto-fixable errors escalate faster
    if consecutive_failures <= 1:
        return "RETRY_WITH_FIX"
    elif consecutive_failures == 2:
        return "RETRY_WITH_AMNESIA"
    elif consecutive_failures == 3:
        return "GENERATE_ALTERNATIVES"
    else:
        return "BACKTRACK_TO_PARENT"


def compress_failure_history(failures: list[str]) -> str:
    """Amnesia: compress old failure details, keep only recent 2.

    Prevents LLM from getting stuck in failure context loops.
    """
    if len(failures) <= 2:
        return "\n".join(failures)

    summary = f"[{len(failures) - 2} earlier attempts failed with similar errors]"
    return f"{summary}\n" + "\n".join(failures[-2:])


def should_abandon_direction(error_history: list[str], max_similar: int = 3) -> bool:
    """Check if last N errors are same category — method is fundamentally broken.

    From AutoKaggle: if 3 errors are same category, stop trying.
    """
    if len(error_history) < max_similar:
        return False

    recent = error_history[-max_similar:]
    categories = []
    for err in recent:
        pattern = classify_error(err)
        cat = pattern.category if pattern else "unknown"
        categories.append(cat)

    # Same category N times → abandon
    return len(set(categories)) == 1 and categories[0] != "unknown"
