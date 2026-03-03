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

        def on_response(turn, content, latency):
            output_parts.append(content)
            print(f"  [{self.WORKER_NAME}] Turn {turn} ({latency:.0f}ms): {content[:100]}...")

        def on_tool_call(name, args):
            print(f"  [{self.WORKER_NAME}] Tool: {name}")

        t0 = time.perf_counter()
        try:
            messages = self.llm.agent_loop(
                task=task,
                system_prompt=full_prompt,
                tools_defs=tools_defs,
                tool_executor=self.registry.execute,
                max_turns=self.max_turns,
                on_response=on_response,
                on_tool_call=on_tool_call,
            )
            elapsed = time.perf_counter() - t0
            full_output = "\n\n".join(output_parts)

            result = {
                "success": True,
                "output": full_output,
                "messages": messages,
                "worker": self.WORKER_NAME,
                "elapsed_s": round(elapsed, 1),
            }

            # Auto-save to knowledge tree
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
        """Save worker output to knowledge tree."""
        if not self.knowledge:
            return
        item_id = f"{self.WORKER_NAME}_{int(time.time())}"
        self.knowledge.add(
            category=self.CATEGORY,
            item_id=item_id,
            content=f"# Task\n{task}\n\n# Result\n{output}",
            metadata={
                "title": task[:100],
                "summary": output[:200],
                "keywords": [self.WORKER_NAME],
                "worker": self.WORKER_NAME,
                "mission_id": self.mission_id,
            },
        )

        if self.event_bus:
            self.event_bus.emit(EventType.KNOWLEDGE_ADDED, {
                "category": self.CATEGORY, "item_id": item_id,
            }, source=self.WORKER_NAME)
