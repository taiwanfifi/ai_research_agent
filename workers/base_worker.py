"""
Base Worker
============
Abstract worker that wraps MiniMaxClient.agent_loop() with
worker-specific system prompts, auto-saves results to knowledge tree,
and publishes completion events.
"""

import json
import time
from core.llm import MiniMaxClient
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

    def _get_tool_executor(self):
        """Return the tool executor callable. Override in subclasses to wrap."""
        return self.registry.execute

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

        full_prompt = self.SYSTEM_PROMPT
        if context:
            full_prompt += f"\n\n## Context from supervisor:\n{context}"

        # Filter tools to only those this worker needs
        tools_defs = self._get_tools()

        output_parts = []
        tool_results_parts = []  # Capture tool results separately

        def on_response(turn, content, latency):
            output_parts.append(content)
            print(f"  [{self.WORKER_NAME}] Turn {turn} ({latency:.0f}ms): {content[:100]}...")

        def on_tool_call(name, args):
            print(f"  [{self.WORKER_NAME}] Tool: {name}")

        def on_tool_result(name, result):
            # Capture important tool results (code execution output, search results)
            if result and len(result.strip()) > 20:
                # Only keep first 1000 chars of each tool result to avoid bloat
                tool_results_parts.append(f"[{name}] {result[:1000]}")

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
            )
            elapsed = time.perf_counter() - t0

            # Combine LLM text responses with key tool results
            full_output = "\n\n".join(output_parts)
            # If LLM output is thin but tool results are rich, append them
            if tool_results_parts and len(full_output) < 500:
                full_output += "\n\n## Tool Execution Results\n" + "\n\n".join(tool_results_parts[-5:])

            # Validate output quality before declaring success
            validation = self._validate_output(full_output)

            result = {
                "success": validation["valid"],
                "output": full_output,
                "messages": messages,
                "worker": self.WORKER_NAME,
                "elapsed_s": round(elapsed, 1),
            }
            if not validation["valid"]:
                result["error"] = f"Output validation failed: {validation['reason']}"
                print(f"  [{self.WORKER_NAME}] VALIDATION FAILED: {validation['reason']}")

            # Auto-save to knowledge tree (only if valid)
            if validation["valid"]:
                self._save_to_knowledge(task, full_output)

            if self.event_bus:
                self.event_bus.emit(EventType.WORKER_FINISHED, {
                    "worker": self.WORKER_NAME, "task": task, "success": True,
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

        return result

    def _get_tools(self) -> list[dict]:
        """Get tool definitions for this worker. Override to filter."""
        return self.registry.tools

    def _save_to_knowledge(self, task: str, output: str):
        """Save worker output to knowledge tree.

        Uses the LAST substantial response (which should contain the
        mandatory final summary) for the knowledge summary, rather than
        the first 200 chars which is usually procedural narration.
        """
        if not self.knowledge:
            return

        # Extract a meaningful summary from the output
        summary = self._extract_summary(output)

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
