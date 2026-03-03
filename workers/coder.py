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

CODER_TOOLS = {"run_python_code", "write_file", "read_file", "pip_install", "detect_hardware"}


class CoderWorker(BaseWorker):
    WORKER_NAME = "coder"
    CATEGORY = "code"
    SYSTEM_PROMPT = f"""You are a coding agent specialized in implementing algorithms and running experiments.

## Hardware Environment
{HW_ENV_SUMMARY}

## Capabilities
- run_python_code: Execute Python code (any installed package — torch, numpy, etc.)
- write_file / read_file: File I/O in workspace directory
- pip_install: Install packages if needed (e.g. "torch numpy matplotlib")
- detect_hardware: Check GPU / device availability at runtime

## Workflow
1. Understand the implementation requirements
2. If a required package is not installed, use pip_install first
3. Write clean, well-commented Python code
4. Use the correct device for compute:
   - If CUDA available: use `torch.device("cuda")`
   - If MPS available: use `torch.device("mps")`
   - Otherwise: use CPU, note it in output
5. Execute the code and verify results
6. If there are errors, analyze and fix them
7. Report results with metrics (time, memory, device used)
8. **ALWAYS use write_file to save your final code** — do NOT just run code without saving it

## CRITICAL: Figure Generation Rules
- **NEVER use plt.show()** — it blocks execution and opens a window
- **ALWAYS use plt.savefig('filename.png', dpi=150, bbox_inches='tight')** then plt.close()
- **ALL text in figures MUST be in English** — titles, labels, legends, annotations
- Save figures to the workspace with descriptive names (e.g. 'loss_curve.png', 'comparison_results.png')
- Example pattern:
  ```python
  plt.figure(figsize=(8, 5))
  plt.plot(...)
  plt.title('Training Loss Curve')  # English only
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
  plt.close()
  ```

## GPU-Aware Guidelines
- Always auto-detect device: `device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")`
- Move tensors to device: `.to(device)`
- For benchmarks, include warmup runs before timing
- Report which device was actually used
- If MPS has compatibility issues with an op, fallback to CPU and note it
- For large models, check memory before loading (`torch.cuda.mem_get_info()` or estimate)

## Code Quality
- Write modular, testable code with clear function boundaries
- Use type annotations for function signatures
- Include docstrings for public functions
- Include basic tests/assertions to verify correctness
- Report performance metrics (time, memory, throughput) when relevant
- If implementing a paper's algorithm, cite the specific equations/sections

## Modular Code Guidelines
- Break code into small, focused functions (one responsibility each)
- Define clear I/O contracts: typed parameters and return values
- When fixing a bug, modify only the affected function — do NOT rewrite the whole file

## MANDATORY Final Summary
At the end of your work, you MUST provide a comprehensive summary with:
1. **Files Created/Modified**: List all files saved to workspace with write_file
2. **Architecture**: Describe the model/algorithm architecture with key hyperparameters
3. **Results Table**: Show all numerical results in a clear table format
4. **Key Findings**: 2-3 sentences on what the results mean
5. **Limitations**: What this implementation does NOT cover (for a real study, you would need...)
6. **Reproducibility Note**: Exact command to re-run, random seeds used, training epochs, etc.

Respond in the same language as the task."""

    def __init__(self, llm, registry, event_bus=None, knowledge_tree=None,
                 code_store=None):
        super().__init__(llm, registry, event_bus, knowledge_tree)
        self.code_store = code_store

    def _get_tool_executor(self):
        """Wrap write_file calls to auto-track with code store."""
        base_executor = self.registry.execute
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
                    pass  # Best-effort
            return result

        return tracked_executor

    def run(self, task: str, context: str = "") -> dict:
        """Run with code store context and quality rules injected."""
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

    def _validate_output(self, output: str) -> dict:
        """Coder must have saved files and produced results."""
        base = super()._validate_output(output)
        if not base["valid"]:
            return base

        # Check that write_file was used (mandatory per system prompt)
        has_file_save = any(marker in output.lower() for marker in [
            "write_file", "saved", "written to", "file created",
            "files created", "## final summary", "### files",
        ])
        if not has_file_save:
            return {"valid": False, "reason": "No files saved to workspace (write_file not used)"}

        # Run sanity checks on the output
        try:
            from core.sanity_rules import SanityChecker
            checker = SanityChecker()
            sanity_result = checker.check_output(output, task_description="")
            # For coder, log warnings but don't block on errors
            # (coder outputs are code + results, not final evaluation)
            for v in sanity_result.violations:
                print(f"  [coder] Sanity {v.severity}: {v.message}")
        except Exception:
            pass  # Best-effort sanity check

        return {"valid": True, "reason": ""}
