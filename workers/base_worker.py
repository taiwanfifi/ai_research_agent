"""
Base Worker
============
Abstract worker that wraps MiniMaxClient.agent_loop() with
worker-specific system prompts, auto-saves results to knowledge tree,
and publishes completion events.
"""

import json
import time
from core.llm import MiniMaxClient, strip_think
from core.tool_registry import ToolRegistry
from core.event_bus import EventBus, EventType


class BaseWorker:
    """Base class for specialized workers."""

    # Override in subclasses
    WORKER_NAME = "base"
    SYSTEM_PROMPT = "You are a helpful assistant."
    CATEGORY = "reports"  # Knowledge tree category for results

    def __init__(self, llm: MiniMaxClient, registry: ToolRegistry,
                 event_bus: EventBus = None, knowledge_tree=None):
        self.llm = llm
        self.registry = registry
        self.event_bus = event_bus
        self.knowledge = knowledge_tree
        self.max_turns = 10
        self.mission_id: str = ""
        self.result_verifier = None  # Set by supervisor
        self.execution_log = None    # Set by supervisor (structured pipeline)
        self.llm_judge = None        # Set by supervisor (Round 11)
        self.validation_mode = "keyword"  # Set by supervisor (Round 11)
        self.enable_monologue = False     # Set by supervisor (inner monologue)
        self._current_cycle: int = 0  # Set by supervisor before dispatch
        self._workspace_dir: str = ""    # Set by supervisor before dispatch

    def _inner_monologue(self, task: str, context: str = "") -> dict:
        """
        Inner monologue: reflect on a task before executing it.
        Returns {"action": "proceed"|"modify"|"pushback", "reasoning": str, "modified_task": str}
        """
        if not self.enable_monologue:
            return {"action": "proceed", "reasoning": "", "modified_task": task}

        # Gather workspace state for context
        workspace_info = ""
        if self._workspace_dir:
            import os, glob
            try:
                files = [os.path.basename(f) for f in glob.glob(os.path.join(self._workspace_dir, '*'))
                         if '__pycache__' not in f and not os.path.basename(f).startswith('.')]
                workspace_info = f"\nWorkspace files: {', '.join(files[:15]) if files else '(empty)'}"
            except Exception:
                pass

        # Gather platform info so worker knows hardware constraints
        import platform, os as _os
        platform_info = f"Platform: {platform.system()} {platform.machine()}"
        if platform.system() == "Darwin" and "arm" in platform.machine().lower():
            platform_info += " (Apple Silicon — MPS backend, no CUDA, bitsandbytes incompatible)"
        mem_gb = _os.sysconf('SC_PAGE_SIZE') * _os.sysconf('SC_PHYS_PAGES') / (1024**3)
        platform_info += f", RAM: {mem_gb:.0f}GB"

        # Gather recent failure history (if any)
        failure_context = ""
        if hasattr(self, '_recent_failures') and self._recent_failures:
            failure_context = "\n## Recent failures from similar tasks\n"
            for f in self._recent_failures[-3:]:
                failure_context += f"- {f}\n"

        # Gather available tools for this worker
        try:
            tool_names = [t["function"]["name"] for t in self._get_tools()]
            tools_str = ", ".join(tool_names)
        except Exception:
            tools_str = "(unknown)"

        prompt = f"""You are the inner voice of a {self.WORKER_NAME} agent. Before executing a task, you must reflect.

## Task assigned to you
{task}

## Context from supervisor
{context if context else '(none)'}

## Your tools (you HAVE these capabilities)
{tools_str}

## Environment
{platform_info}
{workspace_info}
{failure_context}
## Your reflection process
Think about:
1. Is this task clearly defined? What information am I missing?
2. Is the approach correct? (e.g., right model class, right device, right data split)
3. Are there technical risks given the environment? (e.g., MPS incompatibility, timeout limits)
4. Do NOT pushback just because a task seems hard — you have tools. Only pushback if the task is fundamentally misguided.

## Your response (MUST be valid JSON)
Return ONLY one of:

If task is fine:
{{"action": "proceed", "reasoning": "Brief note on why this is sound"}}

If task needs adjustment:
{{"action": "modify", "reasoning": "What's wrong and why", "modified_task": "The improved task description"}}

If task should not be done:
{{"action": "pushback", "reasoning": "Why this task is misguided and what should be done instead"}}"""

        try:
            response = self.llm.chat([
                {"role": "system", "content": "You reflect honestly. Output ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ])

            # Extract text content from API response
            raw = response["choices"][0]["message"]["content"]
            text = strip_think(raw).strip()
            # Handle markdown code blocks
            if "```" in text:
                import re
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
                if match:
                    text = match.group(1)

            result = json.loads(text)
            action = result.get("action", "proceed")
            if action not in ("proceed", "modify", "pushback"):
                action = "proceed"

            print(f"  [{self.WORKER_NAME}] Inner monologue: {action}")
            if action != "proceed":
                print(f"  [{self.WORKER_NAME}] Reasoning: {result.get('reasoning', '')[:150]}")

            return {
                "action": action,
                "reasoning": result.get("reasoning", ""),
                "modified_task": result.get("modified_task", task),
            }
        except Exception as e:
            print(f"  [{self.WORKER_NAME}] Inner monologue failed ({e}), proceeding")
            return {"action": "proceed", "reasoning": "", "modified_task": task}

    def _get_tool_executor(self):
        """Return the tool executor callable with ToolGuard preflight checks.
        Override in subclasses to add additional wrapping (e.g. code_store tracking)."""
        base_executor = self.registry.execute
        workspace_dir = getattr(self, '_workspace_dir', '') or ''

        def guarded_executor(func_name: str, func_args: dict) -> str:
            from core.tool_guards import run_guard
            guard_result = run_guard(func_name, func_args, workspace_dir=workspace_dir)
            if guard_result:
                if guard_result.get("blocked"):
                    print(f"  [{self.WORKER_NAME}] GUARD BLOCKED {func_name}: {guard_result['reason']}")
                    return json.dumps({
                        "success": False,
                        "error": f"Preflight check failed: {guard_result['reason']}",
                        "suggested_actions": guard_result.get("suggested_actions", []),
                    })
                else:
                    # Warning only — log and continue
                    print(f"  [{self.WORKER_NAME}] GUARD WARNING {func_name}: {guard_result['reason']}")
            return base_executor(func_name, func_args)

        return guarded_executor

    def run(self, task: str, context: str = "") -> dict:
        """
        Execute a task and return results.

        Args:
            task: The task description
            context: Optional context from supervisor (prior knowledge, constraints)

        Returns:
            dict with keys: success, output, messages, worker, elapsed_s
        """
        if self.event_bus:
            self.event_bus.emit(EventType.WORKER_STARTED, {
                "worker": self.WORKER_NAME, "task": task,
            }, source=self.WORKER_NAME)

        # ── Inner Monologue: reflect before acting ──────────────
        monologue = self._inner_monologue(task, context)
        if monologue["action"] == "pushback":
            return {
                "success": False,
                "output": "",
                "messages": [],
                "worker": self.WORKER_NAME,
                "elapsed_s": 0,
                "error": f"PUSHBACK: {monologue['reasoning']}",
                "pushback": True,
                "pushback_reasoning": monologue["reasoning"],
            }
        if monologue["action"] == "modify":
            task = monologue["modified_task"]
            print(f"  [{self.WORKER_NAME}] Task modified by inner monologue")

        full_prompt = self.SYSTEM_PROMPT

        # Inject Decision Protocol for workers with risky tools
        from core.decision_envelope import DECISION_PROMPT, RISKY_TOOLS
        worker_tools = {t["function"]["name"] for t in self._get_tools()}
        if worker_tools & RISKY_TOOLS:
            full_prompt += f"\n\n{DECISION_PROMPT}"

        if context:
            full_prompt += f"\n\n## Context from supervisor:\n{context}"

        # Filter tools to only those this worker needs
        tools_defs = self._get_tools()

        output_parts = []
        tool_results_parts = []  # Capture tool results separately
        tool_calls_log = []  # Track actual tool calls made
        decision_envelopes = []  # Track structured decisions

        def on_response(turn, content, latency):
            output_parts.append(content)
            # Parse decision envelope from assistant text
            from core.decision_envelope import parse_envelope
            envelope = parse_envelope(content)
            if envelope:
                decision_envelopes.append(envelope)
            print(f"  [{self.WORKER_NAME}] Turn {turn} ({latency:.0f}ms): {content[:100]}...")

        def on_tool_call(name, args):
            tool_calls_log.append({"name": name, "args": args})
            print(f"  [{self.WORKER_NAME}] Tool: {name}")

        stdout_capture = []  # Capture stdout for LLM judge

        def on_tool_result(name, result):
            # Record to execution log (structured pipeline)
            if self.execution_log and name in ("run_python_code", "write_file"):
                try:
                    if isinstance(result, dict):
                        parsed = result
                    elif isinstance(result, str):
                        try:
                            parsed = json.loads(result)
                        except (json.JSONDecodeError, ValueError):
                            parsed = {"stdout": result[:5000]}
                    else:
                        parsed = {"stdout": str(result)[:5000]}
                    self.execution_log.record(
                        cycle=self._current_cycle,
                        worker=self.WORKER_NAME,
                        tool_name=name,
                        result_dict=parsed if isinstance(parsed, dict) else {"stdout": str(parsed)},
                    )
                except Exception as e:
                    print(f"  [{self.WORKER_NAME}] WARNING: execution_log.record failed: {e}")

            # Capture important tool results (code execution output, search results)
            if result and len(result.strip()) > 20:
                # Only keep first 1000 chars of each tool result to avoid bloat
                tool_results_parts.append(f"[{name}] {result[:1000]}")

            # Capture write_file results for verification
            if name == "write_file" and result and tool_calls_log:
                try:
                    parsed = json.loads(result) if isinstance(result, str) else result
                    if isinstance(parsed, dict) and parsed.get("success"):
                        tool_calls_log[-1]["file_written"] = parsed.get("path", "")
                except (json.JSONDecodeError, AttributeError):
                    pass

            # Capture stdout for result verification (both keyword and LLM judge modes)
            if name == "run_python_code" and result:
                try:
                    parsed = json.loads(result) if isinstance(result, str) else result
                    stdout = parsed.get("stdout", "") if isinstance(parsed, dict) else ""
                    if stdout:
                        stdout_capture.append(stdout)
                        # Feed to keyword-based verifier if active
                        if self.result_verifier:
                            self.result_verifier.capture(
                                cycle=self._current_cycle,
                                worker=self.WORKER_NAME,
                                stdout=stdout,
                            )
                except (json.JSONDecodeError, AttributeError):
                    pass

            # Run sanity checks on raw stdout (keyword mode only)
            if name == "run_python_code" and result and self.validation_mode == "keyword":
                try:
                    parsed = json.loads(result) if isinstance(result, str) else result
                    stdout = parsed.get("stdout", "") if isinstance(parsed, dict) else ""
                    if stdout:
                        from core.sanity_rules import SanityChecker
                        sc = SanityChecker()
                        check = sc.check_output(stdout)
                        if check.violations:
                            for v in check.violations:
                                print(f"  [{self.WORKER_NAME}] Stdout sanity {v.severity}: {v.message}")
                except (json.JSONDecodeError, AttributeError, Exception):
                    pass

        # Prepare execution log summary for structured summary forcing
        exec_log_summary = None
        if self.execution_log:
            exec_log_summary = self.execution_log.get_summary_for_prompt()

        t0 = time.perf_counter()
        try:
            messages = self.llm.agent_loop(
                task=task,
                system_prompt=full_prompt,
                tools_defs=tools_defs,
                tool_executor=self._get_tool_executor(),
                max_turns=self.max_turns,
                on_response=on_response,
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
                execution_log_summary=exec_log_summary,
            )
            elapsed = time.perf_counter() - t0

            # Combine LLM text responses with key tool results
            full_output = "\n\n".join(output_parts)
            # If LLM output is thin but tool results are rich, append them
            if tool_results_parts and len(full_output) < 500:
                full_output += "\n\n## Tool Execution Results\n" + "\n\n".join(tool_results_parts[-5:])

            # Store tool calls log and decision envelopes for downstream use
            self._last_tool_calls = tool_calls_log
            self._last_envelopes = decision_envelopes

            # ── Validation: LLM Judge or Keyword-based ──────────────
            if self.llm_judge and self.validation_mode in ("llm_full", "llm_critical", "hybrid"):
                result = self._validate_with_llm_judge(
                    task, full_output, stdout_capture, tool_calls_log,
                    messages, elapsed,
                )
            else:
                result = self._validate_with_keywords(
                    full_output, tool_calls_log, messages, elapsed,
                )

            # Auto-save to knowledge tree (only if valid)
            if result["success"]:
                judge_summary = ""
                if result.get("judge_result"):
                    judge_summary = result["judge_result"].get("summary", "")
                self._save_to_knowledge(task, full_output, judge_summary=judge_summary)

            if self.event_bus:
                self.event_bus.emit(EventType.WORKER_FINISHED, {
                    "worker": self.WORKER_NAME, "task": task, "success": result["success"],
                }, source=self.WORKER_NAME)

        except Exception as e:
            elapsed = time.perf_counter() - t0
            result = {
                "success": False,
                "output": str(e),
                "messages": [],
                "worker": self.WORKER_NAME,
                "elapsed_s": round(elapsed, 1),
                "error": str(e),
            }

            if self.event_bus:
                self.event_bus.emit(EventType.TASK_FAILED, {
                    "worker": self.WORKER_NAME, "task": task, "error": str(e),
                }, source=self.WORKER_NAME)

        # Flush execution log to disk (deferred writes)
        if self.execution_log:
            self.execution_log.flush()

        return result

    def _get_tools(self) -> list[dict]:
        """Get tool definitions for this worker. Override to filter."""
        return self.registry.tools

    def _save_to_knowledge(self, task: str, output: str,
                           judge_summary: str = ""):
        """Save worker output to knowledge tree.

        Uses LLM judge summary (Round 11) when available, otherwise
        the LAST substantial response for the knowledge summary.
        """
        if not self.knowledge:
            return

        # Prefer LLM judge summary if available
        summary = judge_summary if judge_summary else self._extract_summary(output)

        # Skip knowledge entry if output is essentially empty/procedural
        if len(summary) < 50:
            print(f"  [{self.WORKER_NAME}] Skipping knowledge save — output too short/procedural")
            return

        item_id = f"{self.WORKER_NAME}_{int(time.time())}"
        self.knowledge.add(
            category=self.CATEGORY,
            item_id=item_id,
            content=f"# Task\n{task}\n\n# Result\n{output}",
            metadata={
                "title": task[:100],
                "summary": summary[:500],
                "keywords": [self.WORKER_NAME],
                "worker": self.WORKER_NAME,
                "mission_id": self.mission_id,
            },
        )

        if self.event_bus:
            self.event_bus.emit(EventType.KNOWLEDGE_ADDED, {
                "category": self.CATEGORY, "item_id": item_id,
            }, source=self.WORKER_NAME)

    def _validate_with_llm_judge(self, task: str, full_output: str,
                                stdout_capture: list, tool_calls_log: list,
                                messages: list, elapsed: float) -> dict:
        """Validate using LLM Judge (Round 11). Falls back to keywords on failure."""
        print(f"  [{self.WORKER_NAME}] Running LLM Judge evaluation...")
        judge_result = self.llm_judge.evaluate_worker_output(
            task=task,
            output=full_output,
            stdout_parts=stdout_capture,
            tool_calls=tool_calls_log,
            worker_name=self.WORKER_NAME,
        )

        # If LLM call failed, fall back to keyword validation
        if not judge_result.get("_parse_ok", True):
            print(f"  [{self.WORKER_NAME}] LLM Judge failed, falling back to keyword validation")
            return self._validate_with_keywords(
                full_output, tool_calls_log, messages, elapsed,
            )

        # Determine success: task_completed is primary, is_substantive is secondary
        task_completed = judge_result.get("task_completed", True)
        is_substantive = judge_result.get("is_substantive", True)
        has_files = bool(judge_result.get("has_code_output", False))

        # Hard evidence: if tool_calls show files were written, that's substantive work
        files_written = [tc.get("file_written", "") for tc in tool_calls_log
                        if tc.get("file_written")]
        if files_written:
            has_files = True

        error_msg = ""

        # A task is valid if it was completed OR produced files/code output
        # "not substantive" alone doesn't fail — it just means procedural narration
        is_valid = task_completed or has_files
        if not is_valid and not is_substantive:
            error_msg = "Output validation failed: LLM Judge determined task was not completed and output is not substantive"
        elif not is_valid:
            error_msg = "Output validation failed: LLM Judge determined task was not completed"

        # Check for explicit placeholder/fabrication signals in stdout
        stdout_text = " ".join(stdout_capture).lower()
        placeholder_signals = ["placeholder data", "representative data", "using placeholder",
                               "using representative", "fabricated", "dummy data", "fake data",
                               "simulated results", "not real results"]
        for signal in placeholder_signals:
            if signal in stdout_text and is_valid:
                is_valid = False
                error_msg = f"Output uses {signal} — must produce real experimental results"
                print(f"  [{self.WORKER_NAME}] BLOCKED: {error_msg}")
                break

        # Check for contradicted claims (fabrication detection)
        # Require 2+ contradictions to block — single contradictions are often false positives
        # from the judge misinterpreting descriptive statements as numerical claims
        contradicted = [c for c in judge_result.get("claims_vs_stdout", [])
                       if c.get("status") == "contradicted"]
        if len(contradicted) >= 2 and is_valid:
            contradiction_msgs = [c.get("claim", "") for c in contradicted]
            is_valid = False
            error_msg = f"Fabrication detected — claims contradict stdout: {'; '.join(contradiction_msgs)}"
            for c in contradicted:
                print(f"  [{self.WORKER_NAME}] FABRICATION BLOCKED: {c.get('claim', '')}")
        elif contradicted:
            # Single contradiction — warn but don't block
            for c in contradicted:
                print(f"  [{self.WORKER_NAME}] WARNING: Possible fabrication (not blocking): {c.get('claim', '')}")

        # Calculate verification score from claims
        claims = judge_result.get("claims_vs_stdout", [])
        verification_score = 1.0
        if claims:
            verified = sum(1 for c in claims if c.get("status") == "verified")
            verification_score = verified / len(claims)

        result = {
            "success": is_valid,
            "output": full_output,
            "messages": messages,
            "worker": self.WORKER_NAME,
            "elapsed_s": round(elapsed, 1),
            "tool_calls": tool_calls_log,
            "verification_score": verification_score,
            "judge_result": judge_result,
        }
        if error_msg:
            result["error"] = error_msg
            print(f"  [{self.WORKER_NAME}] VALIDATION FAILED: {error_msg}")

        # Log quality concerns
        for concern in judge_result.get("quality_concerns", []):
            print(f"  [{self.WORKER_NAME}] Quality concern: {concern}")

        return result

    def _validate_with_keywords(self, full_output: str, tool_calls_log: list,
                                 messages: list, elapsed: float) -> dict:
        """Original keyword-based validation + ResultVerifier (pre-Round 11)."""
        validation = self._validate_output(full_output)

        result = {
            "success": validation["valid"],
            "output": full_output,
            "messages": messages,
            "worker": self.WORKER_NAME,
            "elapsed_s": round(elapsed, 1),
            "tool_calls": tool_calls_log,
        }
        if not validation["valid"]:
            result["error"] = f"Output validation failed: {validation['reason']}"
            print(f"  [{self.WORKER_NAME}] VALIDATION FAILED: {validation['reason']}")

        # Run result verification (check claims vs stdout) — ENFORCING
        if validation["valid"] and self.result_verifier:
            try:
                verification = self.result_verifier.verify_output(full_output)
                result["verification_score"] = verification.score
                if verification.contradicted:
                    contradiction_msgs = [c.raw_text for c in verification.contradicted]
                    result["success"] = False
                    result["error"] = f"Fabrication detected — claims contradict stdout: {'; '.join(contradiction_msgs)}"
                    for c in verification.contradicted:
                        print(f"  [{self.WORKER_NAME}] FABRICATION BLOCKED: {c.raw_text}")
                elif verification.score == 0.0 and len(verification.claims) > 2:
                    result["verification_warning"] = "No claims could be verified against stdout"
                    for w in verification.warnings:
                        print(f"  [{self.WORKER_NAME}] Verify warning: {w}")
                else:
                    if verification.warnings:
                        for w in verification.warnings:
                            print(f"  [{self.WORKER_NAME}] Verify warning: {w}")
            except Exception:
                pass

        return result

    def _validate_output(self, output: str) -> dict:
        """Validate that worker output is substantive, not just procedural narration.

        Returns {"valid": bool, "reason": str}.
        Subclasses can override for worker-specific validation.
        """
        if not output or len(output.strip()) < 100:
            return {"valid": False, "reason": "Output is too short (<100 chars)"}

        # Check for procedural-only output: ALL text is just "Let me..." / "I'll..."
        # with no substantive content anywhere in the output
        paragraphs = [p.strip() for p in output.split("\n\n") if p.strip()]
        procedural_markers = [
            "let me ", "i'll ", "i will ", "let's ", "now i ", "now let",
            "讓我", "我來", "我將", "接下來",
        ]

        substantive_paragraphs = []
        for p in paragraphs:
            first_line = p.split("\n")[0].lower().strip()
            is_procedural = any(first_line.startswith(m) for m in procedural_markers)
            if not is_procedural and len(p) > 80:
                substantive_paragraphs.append(p)

        if len(substantive_paragraphs) < 1:
            return {"valid": False, "reason": "Output is entirely procedural narration with no substantive content"}

        return {"valid": True, "reason": ""}

    def _extract_summary(self, output: str) -> str:
        """Extract the most informative part of the output for knowledge summary.

        Strategy: look for the mandatory final summary section (added in Round 1),
        or fall back to the LAST substantial paragraph instead of first 200 chars.
        """
        if not output:
            return ""

        # Look for known summary section headers
        summary_markers = [
            "### Key Findings", "### Results Table", "### Analysis",
            "## Final Summary", "### Top Papers", "### Files Created",
            "### Experimental Setup", "## Summary", "### Architecture",
        ]
        for marker in summary_markers:
            idx = output.find(marker)
            if idx >= 0:
                # Return from this marker onwards, capped
                return output[idx:idx + 500].strip()

        # Fall back: use the last substantial paragraph (>100 chars)
        paragraphs = [p.strip() for p in output.split("\n\n") if len(p.strip()) > 100]
        if paragraphs:
            return paragraphs[-1][:500]

        # Last resort: last 300 chars (more likely to have results than first 200)
        return output[-300:].strip() if len(output) > 300 else output.strip()
