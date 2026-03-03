"""
MiniMax LLM Client
===================
Extracted from agent.py — provides the core LLM interface for all components.
"""

import json
import re
import time
import httpx


def strip_think(text: str) -> str:
    """Remove MiniMax's <think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class MiniMaxClient:
    """Stateless MiniMax API client."""

    def __init__(self, api_key: str, base_url: str = "https://api.minimax.io/v1",
                 model: str = "MiniMax-M2.5", max_tokens: int = 4096,
                 temperature: float = 0.3, timeout: int = 120, max_retries: int = 3):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def chat(self, messages: list, tools: list = None) -> dict:
        """Call MiniMax chat completions API with retry."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools

        for attempt in range(self.max_retries):
            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.ReadTimeout:
                if attempt < self.max_retries - 1:
                    print(f"  [Retry] Timeout, retrying ({attempt+2}/{self.max_retries})...")
                    time.sleep(2)
                else:
                    raise

    def agent_loop(self, task: str, system_prompt: str, tools_defs: list,
                   tool_executor, max_turns: int = 10,
                   on_response=None, on_tool_call=None, on_tool_result=None):
        """
        Run a complete agent loop.

        Args:
            task: The user task/query
            system_prompt: System prompt for the agent
            tools_defs: List of tool definitions (OpenAI format)
            tool_executor: Callable(name, args) -> str that executes tools
            max_turns: Maximum iterations
            on_response: Optional callback(turn, content, latency_ms)
            on_tool_call: Optional callback(name, args)
            on_tool_result: Optional callback(name, result)

        Returns:
            List of all messages (conversation history)
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        for turn in range(1, max_turns + 1):
            t0 = time.perf_counter()
            response = self.chat(messages, tools=tools_defs)
            latency = (time.perf_counter() - t0) * 1000

            choice = response["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason", "")

            messages.append(message)

            # Handle text response
            if message.get("content"):
                clean_content = strip_think(message["content"])
                if clean_content and on_response:
                    on_response(turn, clean_content, latency)

            # Handle tool calls
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    try:
                        func_args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        func_args = {}

                    if on_tool_call:
                        on_tool_call(func_name, func_args)

                    result = tool_executor(func_name, func_args)

                    if on_tool_result:
                        on_tool_result(func_name, result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
                continue

            # No tool calls + stop → done
            if finish_reason == "stop":
                break

        return messages
