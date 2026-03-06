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

CODER_TOOLS = {"run_python_code", "write_file", "read_file", "pip_install", "detect_hardware",
               "list_modules", "edit_function"}


class CoderWorker(BaseWorker):
    WORKER_NAME = "coder"
    CATEGORY = "code"
    SYSTEM_PROMPT = f"""You are a coding agent specialized in implementing algorithms and running experiments.

## Hardware Environment
{HW_ENV_SUMMARY}

## Capabilities
- run_python_code: Execute Python code (any installed package — torch, numpy, etc.)
- write_file / read_file: File I/O in workspace directory
- list_modules: See all functions/classes in a file with line ranges (use before editing!)
- edit_function: Replace a SINGLE function/class in a file without touching the rest
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
9. **When task says "run" or "train"**: Execute the training code DIRECTLY in run_python_code.
   Do NOT just write a .py file — you must also EXECUTE it and show the results.
   The correct pattern: write the script with write_file, then run it with run_python_code.
   Do NOT use exec(open(...)) or subprocess — paste the actual training code into run_python_code.
10. **If training crashes, fix AND re-run**: Don't just fix the code and declare done.
    After fixing, always re-execute to verify the fix produces real metrics.

## CRITICAL: Editing Existing Files
- **NEVER rewrite a whole file to fix a bug or change one function**
- Instead: list_modules(filename) → read the function → edit_function(filename, func_name, new_code)
- write_file is for NEW files only. edit_function is for modifying EXISTING files.
- If you rewrite a 200-line file to change 5 lines, you will likely break something.

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

## CRITICAL: Match Model Class to Task Type
- **Classification** (sentiment, NLI, topic): Use `AutoModelForSequenceClassification`, NOT `AutoModelForCausalLM`
  - GPT-2/DistilGPT-2 classification: `AutoModelForSequenceClassification.from_pretrained("distilgpt2", num_labels=2)`
  - Set `model.config.pad_token_id = tokenizer.pad_token_id`
  - Loss and labels are handled automatically by the model
- **Text generation** (summarization, translation, chat): Use `AutoModelForCausalLM`
- **NEVER use causal LM loss (next-token prediction) to train a classifier** — it will produce 0.0 accuracy
- When using PEFT/LoRA with classification: `task_type=TaskType.SEQ_CLS` in LoraConfig
- **SST-2 column names**: HuggingFace uses 'sentence', but saved JSON files may use 'text'. Always check actual column names with `list(dataset.column_names)` or `data[0].keys()` before processing.

## CRITICAL: HuggingFace TrainingArguments Compatibility
- Use `eval_strategy` NOT `evaluation_strategy` (deprecated in transformers >= 4.46, removed in 4.50+)
- Same for `save_strategy`, `logging_strategy` — use the short names
- Example:
  ```python
  TrainingArguments(
      output_dir="./output",
      eval_strategy="epoch",      # NOT evaluation_strategy
      save_strategy="epoch",      # NOT save_strategy="epoch" (this one is fine)
      num_train_epochs=2,
      per_device_train_batch_size=16,
  )
  ```

## GPU/Device Guidelines
- **PEFT/LoRA fine-tuning**: ALWAYS use CPU. MPS DOES NOT WORK (crashes with NoneType errors). Add this at the TOP of every training script:
  ```python
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  device = torch.device("cpu")
  ```
  Do NOT auto-detect device for fine-tuning. Do NOT try MPS. It WILL fail.
- **Inference/non-training**: Auto-detect: `device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")`
- Move tensors to device: `.to(device)`
- For benchmarks, include warmup runs before timing
- Report which device was actually used
- For large models, check memory before loading (`torch.cuda.mem_get_info()` or estimate)

## Structured Output
When printing results, use the format: `metric_name: value`
Examples: `accuracy: 85.3`, `loss: 0.42`, `training_time: 12.5s`, `f1_score: 0.91`
This ensures metrics are automatically captured by the execution log.

## Time Budget
- Code execution has a 600s (10min) timeout. Plan accordingly:
  - Training ONE model on 2000 samples for 2 epochs ≈ 200-400s on CPU
  - **ALWAYS use dataset subsets** — `dataset.select(range(2000))` for training. Full datasets WILL timeout.
  - Do NOT try to train multiple configurations in a single run_python_code call
  - If training 3 seeds, do them ONE AT A TIME in separate run_python_code calls
  - Always add timing: `import time; t0=time.time()` ... `print(f"training_time: {{time.time()-t0:.1f}}s")`
  - **NEVER use subprocess.run/Popen/call** — it is BLOCKED at the system level and will return an error. Paste training code directly into run_python_code.
  - **NEVER use exec(open(...).read())** — paste the code directly.

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

        # Filesystem OK → run LLM judge
        return super()._validate_with_llm_judge(
            task, full_output, stdout_capture, tool_calls_log, messages, elapsed,
        )

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
