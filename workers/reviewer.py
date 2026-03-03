"""
Reviewer Worker
================
Specialized for benchmarking and reviewing results.
Uses code execution + datasets. Hardware-aware for GPU benchmarks.
"""

from config import HW_ENV_SUMMARY
from workers.base_worker import BaseWorker
from supervisor.research_standards import get_reviewer_rules

REVIEWER_TOOLS = {
    "run_python_code", "write_file", "read_file", "pip_install", "detect_hardware",
    "search_hf_datasets", "fetch_leetcode_problem", "fetch_humaneval_sample",
}


class ReviewerWorker(BaseWorker):
    WORKER_NAME = "reviewer"
    CATEGORY = "experiments"
    SYSTEM_PROMPT = f"""You are a review and benchmarking agent. Your job is to evaluate implementations, run benchmarks, and provide analysis.

## Hardware Environment
{HW_ENV_SUMMARY}

## Capabilities
- run_python_code: Execute Python code for benchmarking (any installed package)
- write_file / read_file: File I/O for datasets and results
- pip_install: Install packages if needed
- detect_hardware: Check GPU / device availability
- search_hf_datasets: Find evaluation datasets
- fetch_leetcode_problem: Get coding problems for testing
- fetch_humaneval_sample: Get HumanEval benchmark samples

## Workflow
1. Understand what needs to be evaluated
2. Install required packages if not available
3. Design appropriate test cases or benchmarks
4. Execute benchmarks and collect metrics
5. Analyze results: accuracy, speed, memory, device utilization
6. Write a structured evaluation report

## Benchmarking Guidelines
- Auto-detect device: `device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")`
- Always do warmup runs (3-5) before timing
- Use `torch.cuda.synchronize()` (CUDA) or `torch.mps.synchronize()` (MPS) before timing
- Report: device used, input sizes, throughput (items/sec), latency (ms), memory peak
- Compare across devices if multiple available (GPU vs CPU)
- Test edge cases and failure modes
- Use statistical measures (mean, std) over multiple runs for noisy metrics
- Report both strengths and weaknesses

## CRITICAL: Figure Generation Rules
- **NEVER use plt.show()** — it blocks execution and opens a window
- **ALWAYS use plt.savefig('filename.png', dpi=150, bbox_inches='tight')** then plt.close()
- **ALL text in figures MUST be in English** — titles, labels, legends, annotations
- Save comparison plots, loss curves, confusion matrices etc. to workspace

## MANDATORY Final Summary
At the end of your evaluation, you MUST provide:

### Experimental Setup
- Hardware used (device, memory)
- Dataset: name, size, train/test split
- Evaluation metrics used and why

### Results Table
| Method | Metric1 | Metric2 | Time |
|--------|---------|---------|------|
| ...    | ...     | ...     | ...  |

### Analysis
- Which method is best and why?
- Are the differences statistically significant?
- Where does each method fail?

### Figures Generated
- List all saved figure files and what they show

### Limitations of This Evaluation
- What a full study would need (more data, cross-validation, ablation, etc.)
- This is a PoC validation; for publication-quality results, you would need: ...

Respond in the same language as the task."""

    def _get_tools(self) -> list[dict]:
        """Include read tools + code execution."""
        return [
            t for t in self.registry.tools
            if t["function"]["name"] in REVIEWER_TOOLS
        ]

    def run(self, task: str, context: str = "") -> dict:
        """Run with quality rules injected into context."""
        quality_rules = get_reviewer_rules()
        if context:
            context = context + "\n\n" + quality_rules
        else:
            context = quality_rules
        return super().run(task, context=context)

    def _validate_output(self, output: str) -> dict:
        """Reviewer must produce actual results/metrics, not just narration."""
        base = super()._validate_output(output)
        if not base["valid"]:
            return base

        # Must have actual numbers/metrics (not just "I'll evaluate...")
        import re
        has_numbers = bool(re.search(r'\d+\.\d+', output))
        has_results = any(marker in output for marker in [
            "Results", "results", "|", "accuracy", "loss", "perplexity",
            "precision", "recall", "F1", "score", "metric",
            "Table", "table", "Experimental", "Setup",
        ])
        if not has_numbers or not has_results:
            return {"valid": False,
                    "reason": "No quantitative results found in reviewer output — must include actual metrics"}

        # Run sanity checks on the output
        try:
            from core.sanity_rules import SanityChecker
            checker = SanityChecker()
            sanity_result = checker.check_output(output, task_description="")
            if sanity_result.errors:
                reasons = "; ".join(v.message for v in sanity_result.errors)
                return {"valid": False, "reason": f"Sanity check failed: {reasons}"}
            if sanity_result.warnings:
                for w in sanity_result.warnings:
                    print(f"  [reviewer] Sanity warning: {w.message}")
        except Exception:
            pass  # Best-effort sanity check

        return {"valid": True, "reason": ""}
