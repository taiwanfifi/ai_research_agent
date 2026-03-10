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
    "gpu_search", "gpu_run", "gpu_status",
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
1. FIRST: read_file to check for existing results JSON files (results_*.json, *_results.json)
2. If results exist: load them, compute statistics, generate charts — do NOT retrain
3. If results are missing or incomplete: run minimal evaluation code to fill gaps
4. Analyze: statistical tests (t-test, Cohen's d), effect sizes, confidence intervals
5. Generate comparison charts with plt.savefig
6. Save analysis_summary.json with ALL statistics (this is the single source of truth for scoring)
7. Write a structured evaluation report
8. IMPORTANT: analysis_summary.json must include every numerical claim in your report

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
- **Figure annotations**: Add significance markers (*, **, ***) on comparison charts. Include 95% CI error bars.

## CRITICAL: analysis_summary.json Structure
Your analysis_summary.json MUST include: methods (with mean, std, per-seed values), statistics (paired_t_test with t_stat and p_value, cohens_d, significant bool), best_method name, and conclusion sentence.
This file is the SINGLE SOURCE OF TRUTH — the final report reads from it.

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

### Discussion & Interpretation (CRITICAL)
Go beyond statistics — explain the WHY:
- **Why** did the winning method outperform? What mechanism explains it?
- **Surprising results**: Anything unexpected? Why might it have happened?
- **Practical implications**: When would a practitioner choose method A over B?
- **Connection to theory**: Do results align with or contradict existing literature?

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

    def _validate_with_llm_judge(self, task, full_output, stdout_capture,
                                  tool_calls_log, messages, elapsed):
        """Reviewer override: must have run code even in LLM judge mode."""
        if tool_calls_log:
            ran_code = any(tc.get("name") == "run_python_code" for tc in tool_calls_log)
            if not ran_code:
                return {
                    "success": False,
                    "output": full_output,
                    "messages": messages,
                    "worker": self.WORKER_NAME,
                    "elapsed_s": round(elapsed, 1),
                    "tool_calls": tool_calls_log,
                    "error": "Reviewer must run code to verify results, not just quote numbers from prior output",
                }
        return super()._validate_with_llm_judge(
            task, full_output, stdout_capture, tool_calls_log, messages, elapsed,
        )

    def _validate_output(self, output: str) -> dict:
        """Reviewer must produce actual results/metrics and must have run code."""
        base = super()._validate_output(output)
        if not base["valid"]:
            return base

        # Must have actual numbers/metrics (not just "I'll evaluate...")
        import re
        has_numbers = bool(re.search(r'\d+\.\d+', output))
        has_results = any(marker in output for marker in [
            "Results", "results", "|", "accuracy", "loss", "perplexity",
            "precision", "recall", "F1", "score", "metric",
        ])
        if not has_numbers or not has_results:
            return {"valid": False,
                    "reason": "No quantitative results found in reviewer output — must include actual metrics"}

        # Reviewer must have actually executed code, not just quoted numbers
        if hasattr(self, '_last_tool_calls'):
            ran_code = any(tc.get("name") == "run_python_code" for tc in self._last_tool_calls)
            if not ran_code:
                return {"valid": False,
                        "reason": "Reviewer must run code to verify results, not just quote numbers from prior output"}

        # Run sanity checks on the output — errors block
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
