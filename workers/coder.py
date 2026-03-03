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

## Output Format
- Show the code you wrote
- Show execution results (stdout/stderr)
- Summarize what was implemented, which device was used, and key metrics
- Note any limitations or next steps

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
        """Run with code store context injected."""
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
