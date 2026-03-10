"""
Coder Worker
==============
Specialized for code writing and experiments.
Uses run_python_code, write_file, read_file, pip_install, detect_hardware.

Hardware-aware: reads HW_ENV_SUMMARY from config so the LLM knows
what GPU is available and writes appropriate device code.

Version-tracked: when a CodeVersionStore is provided, every write_file
call is automatically tracked with snapshots, diffs, and AST module maps.
"""

import json
from config import HW_ENV_SUMMARY
from workers.base_worker import BaseWorker
from supervisor.research_standards import get_coder_rules
from core.code_recipes import format_recipes_for_prompt

CODER_TOOLS = {"run_python_code", "write_file", "read_file", "pip_install", "detect_hardware",
               "list_modules", "edit_function",
               "gpu_search", "gpu_run", "gpu_status"}


class CoderWorker(BaseWorker):
    WORKER_NAME = "coder"
    CATEGORY = "code"
    SYSTEM_PROMPT = f"""You are a coding agent specialized in implementing algorithms and running experiments.

## Hardware Environment
{HW_ENV_SUMMARY}

## Tools
- run_python_code: Execute Python code (any installed package)
- write_file / read_file: File I/O in workspace directory
- list_modules: See all functions/classes in a file with line ranges
- edit_function: Replace a SINGLE function/class without touching the rest
- pip_install: Install packages (e.g. "torch numpy matplotlib")
- detect_hardware: Check GPU/device availability

## Workflow
1. CHECK CONTEXT for Code Recipes — if provided, USE THEM. Do NOT rewrite from scratch.
   - If a simple_cnn recipe is available, use it instead of ResNet/VGG (unless task explicitly names a specific architecture).
2. SPEC COMPLIANCE: Use EXACT hyperparameters from the task (lr, momentum, weight_decay, epochs, seeds, samples). NEVER substitute your own values.
3. pip_install if needed → write code with write_file
4. SMOKE TEST FIRST: run a quick 1-batch forward pass to verify imports, data loading, and tensor shapes BEFORE full training. Fix any errors here — it's much cheaper than debugging mid-training.
5. Execute full training with run_python_code
6. If task says "run" or "train": you MUST execute and show results, not just write a file
7. If code crashes, fix AND re-run — don't just fix and declare done
8. Use edit_function (not write_file) to modify existing files. list_modules first.
9. ALWAYS save final code with write_file
9. For CIFAR-10/MNIST: use torchvision.datasets (NOT HuggingFace datasets)

## Figure Rules
- NEVER plt.show() — use plt.savefig('name.png', dpi=150, bbox_inches='tight') then plt.close()
- ALL text in figures in English

## Results Capture (CRITICAL)
- ALWAYS save results to a JSON file FIRST, then print summary. Training stdout gets truncated.
- Pattern: `json.dump(results, open('results_xxx.json', 'w'))` BEFORE printing.
- Print key metrics as `metric_name: value` (e.g. `accuracy: 85.3`) for auto-capture.
- If training has multiple seeds, save ALL seed results in one JSON.

## GPU Available (vast.ai)
- Use gpu_run for: fine-tuning, full dataset training, anything needing GPU or >10 min compute
- gpu_run handles full lifecycle: find GPU → create → run → get results → destroy
- Code must be SELF-CONTAINED: all imports, pip installs, data downloads, training, result printing
- Costs tracked automatically. RTX 4090 ≈ $0.25/hr, A100 ≈ $0.50/hr

## Time Budget
- 600s timeout per run_python_code call (LOCAL CPU). Training on 2000 samples ≈ 60-100s per seed on CPU.
- For heavy tasks: use gpu_run instead (no timeout, remote execution).
- Train ALL 5 seeds in ONE run_python_code call (loop over seeds), save to ONE JSON file.
- Always add timing: `import time; t0=time.time()` ... `print(f"training_time: {{time.time()-t0:.1f}}s")`
- Budget: 5 seeds × ~100s = ~500s. Keep each seed under 100s. If tight, reduce epochs or samples.
- Save JSON after EACH seed (not all at end) so partial results survive timeout.

## MANDATORY Final Summary
1. **Files Created/Modified**: List all files saved with write_file
2. **Architecture**: Model/algorithm with key hyperparameters
3. **Results Table**: All numerical results
4. **Key Findings**: 2-3 sentences
5. **Limitations**: What this doesn't cover
6. **Reproducibility**: Seeds, epochs, commands to re-run

Respond in the same language as the task."""

    def __init__(self, llm, registry, event_bus=None, knowledge_tree=None,
                 code_store=None):
        super().__init__(llm, registry, event_bus, knowledge_tree)
        self.code_store = code_store

    def _get_tool_executor(self):
        """Wrap write_file calls to auto-track with code store.
        Chains through super() to get ToolGuard preflight checks."""
        base_executor = super()._get_tool_executor()
        if not self.code_store:
            return base_executor

        code_store = self.code_store

        def tracked_executor(func_name: str, func_args: dict) -> str:
            result = base_executor(func_name, func_args)
            if func_name == "write_file":
                try:
                    filename = func_args.get("filename", "")
                    content = func_args.get("content", "")
                    if filename and content:
                        code_store.track_write(
                            filename, content,
                            reason=f"coder write (task context)",
                        )
                except Exception:
                    pass
            elif func_name == "edit_function":
                try:
                    parsed = json.loads(result) if isinstance(result, str) else result
                    if parsed.get("success"):
                        filename = func_args.get("filename", "")
                        fn_name = func_args.get("function_name", "")
                        # Re-read the full file to track the new version
                        import os
                        ws = getattr(self, '_workspace_dir', None)
                        if ws:
                            fpath = os.path.join(ws, os.path.basename(filename))
                        else:
                            fpath = parsed.get("path", "")
                        if fpath and os.path.exists(fpath):
                            with open(fpath) as f:
                                code_store.track_write(
                                    filename, f.read(),
                                    reason=f"edit_function: {fn_name}",
                                )
                except Exception:
                    pass
            return result

        return tracked_executor

    def run(self, task: str, context: str = "") -> dict:
        """Run with code store context, quality rules, and code recipes injected."""
        # Inject pre-vetted code recipes FIRST (highest priority — before quality rules)
        recipes = format_recipes_for_prompt(task)
        if recipes:
            recipe_block = (
                "# ⚠️ MANDATORY: USE THESE PRE-TESTED CODE SNIPPETS\n"
                "# Do NOT write data loading, training loops, or multi-seed eval from scratch.\n"
                "# Copy-paste these recipes and modify only what's needed for the task.\n\n"
                + recipes
            )
            if context:
                context = recipe_block + "\n\n" + context
            else:
                context = recipe_block

        # Inject quality rules
        quality_rules = get_coder_rules()
        if context:
            context = context + "\n\n" + quality_rules
        else:
            context = quality_rules

        # Inject workspace summary and fix context
        if self.code_store:
            extra_context_parts = []

            summary = self.code_store.get_workspace_summary()
            if summary:
                extra_context_parts.append(summary)

            # If this looks like a bug fix, provide targeted context
            if any(kw in task.lower() for kw in ("fix", "error", "bug", "fail", "debug")):
                # Try to find the relevant file from the task description
                import re
                file_match = re.search(r'(\w+\.py)', task)
                if file_match:
                    fix_ctx = self.code_store.get_fix_context(
                        file_match.group(1), task
                    )
                    if fix_ctx:
                        extra_context_parts.append(fix_ctx)

            if extra_context_parts:
                context = context + "\n\n" + "\n\n".join(extra_context_parts) if context else "\n\n".join(extra_context_parts)

        return super().run(task, context=context)

    def _get_tools(self) -> list[dict]:
        """Only include code execution tools."""
        return [
            t for t in self.registry.tools
            if t["function"]["name"] in CODER_TOOLS
        ]

    def _validate_with_llm_judge(self, task, full_output, stdout_capture,
                                  tool_calls_log, messages, elapsed):
        """Coder override: check workspace files BEFORE running LLM judge."""
        # Filesystem check — must have real files regardless of validation mode
        has_real_files = False
        if hasattr(self, '_workspace_dir') and self._workspace_dir:
            import os, glob
            ws_files = glob.glob(os.path.join(self._workspace_dir, '*'))
            real_files = [f for f in ws_files
                          if not os.path.basename(f).startswith(('tmp_', '.'))
                          and '__pycache__' not in f
                          and '.code_store' not in f]
            has_real_files = len(real_files) > 0

        if not has_real_files and tool_calls_log:
            has_real_files = any(
                tc.get("name") == "write_file" and tc.get("file_written")
                for tc in tool_calls_log
            )

        if not has_real_files:
            return {
                "success": False,
                "output": full_output,
                "messages": messages,
                "worker": self.WORKER_NAME,
                "elapsed_s": round(elapsed, 1),
                "tool_calls": tool_calls_log,
                "error": "No files found in workspace — code was not saved via write_file",
            }

        # Spec compliance check (discrepancy monitor)
        # stdout_capture is a list — join to string for spec compliance check
        spec_output = "\n".join(stdout_capture) if isinstance(stdout_capture, list) else (stdout_capture or full_output)
        spec_warnings = self._check_spec_compliance(task, spec_output)
        if spec_warnings:
            for w in spec_warnings:
                print(f"  [coder] Spec: {w}")

        # Filesystem OK → run LLM judge
        result = super()._validate_with_llm_judge(
            task, full_output, stdout_capture, tool_calls_log, messages, elapsed,
        )

        # Attach spec warnings to result for judge visibility
        if spec_warnings and isinstance(result, dict):
            result["spec_warnings"] = spec_warnings
        return result

    @staticmethod
    def _check_spec_compliance(task: str, output) -> list[str]:
        """Deterministic spec compliance: extract numbers from task, verify in output.

        Returns list of warning strings. Empty if all specs found.
        """
        import re
        # Defensive: handle list input (stdout_capture is a list)
        if isinstance(output, list):
            output = "\n".join(str(s) for s in output)
        output = str(output or "")
        warnings = []

        # Extract spec numbers from task description
        # Patterns: "3 epochs", "2000 samples", "seed 42", "lr 2e-4", etc.
        spec_patterns = [
            (r'(\d+)\s*epoch', 'epochs'),
            (r'(\d+)\s*sample', 'samples'),
            (r'(\d+)\s*seed', 'seeds'),
            (r'seed[s]?\s*[\(:]?\s*([\d,\s]+)', 'seed_values'),
            (r'learning.?rate\s*[=:]?\s*([\d.e\-]+)', 'learning_rate'),
            (r'batch.?size\s*[=:]?\s*(\d+)', 'batch_size'),
        ]

        specs = {}
        for pattern, name in spec_patterns:
            match = re.search(pattern, task.lower())
            if match:
                specs[name] = match.group(1)

        if not specs:
            return []  # No specs to check

        # Check each spec against output
        for name, value in specs.items():
            if name == 'seed_values':
                # Check each seed appears
                seed_nums = re.findall(r'\d+', value)
                for seed in seed_nums:
                    if seed not in output:
                        warnings.append(f"Seed {seed} specified in task but not found in output")
            elif name == 'epochs':
                # Check epoch count in training output
                epoch_matches = re.findall(r'epoch[s]?\s*[=:]?\s*(\d+)', output.lower())
                num_epochs = re.findall(r'num_train_epochs\s*[=:]?\s*(\d+)', output.lower())
                if num_epochs and int(num_epochs[-1]) != int(value):
                    warnings.append(f"Task says {value} epochs but code uses {num_epochs[-1]}")
            elif name == 'samples':
                # Check sample count
                sample_matches = re.findall(r'select\(range\((\d+)\)', output)
                sample_matches += re.findall(r'(\d+)\s*(?:train|training)\s*sample', output.lower())
                if sample_matches:
                    actual = sample_matches[-1]
                    if int(actual) != int(value):
                        warnings.append(f"Task says {value} samples but output shows {actual}")

        return warnings

    def _validate_output(self, output: str) -> dict:
        """Coder must have real files in workspace, not just text patterns.

        Note: In LLM judge modes (llm_full, llm_critical, hybrid), this method
        is NOT called — _validate_with_llm_judge() handles everything.
        """
        base = super()._validate_output(output)
        if not base["valid"]:
            return base

        # Check actual workspace for files (not text patterns)
        has_real_files = False
        if hasattr(self, '_workspace_dir') and self._workspace_dir:
            import os, glob
            ws_files = glob.glob(os.path.join(self._workspace_dir, '*'))
            # Filter out __pycache__, .code_store, tmp_ files
            real_files = [f for f in ws_files
                          if not os.path.basename(f).startswith(('tmp_', '.'))
                          and '__pycache__' not in f
                          and '.code_store' not in f]
            has_real_files = len(real_files) > 0

        # Fallback: check tool_calls_log if available (set during run())
        if not has_real_files and hasattr(self, '_last_tool_calls'):
            has_real_files = any(
                tc.get("name") == "write_file" and tc.get("file_written")
                for tc in self._last_tool_calls
            )

        if not has_real_files:
            return {"valid": False, "reason": "No files found in workspace — code was not saved via write_file"}

        # Run sanity checks on the output — errors now block (keyword mode only)
        if self.validation_mode == "keyword":
            try:
                from core.sanity_rules import SanityChecker
                checker = SanityChecker()
                sanity_result = checker.check_output(output, task_description="")
                if sanity_result.errors:
                    reasons = "; ".join(v.message for v in sanity_result.errors)
                    return {"valid": False, "reason": f"Sanity check failed: {reasons}"}
                for v in sanity_result.violations:
                    if v.severity == "warning":
                        print(f"  [coder] Sanity warning: {v.message}")
            except Exception:
                pass  # Best-effort sanity check

        return {"valid": True, "reason": ""}
