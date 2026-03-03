"""
MiniMax LLM Client
===================
Provides the core LLM interface for all components.
Handles retries, message truncation, and 400 recovery.
"""

import json
import re
import time
import httpx


def strip_think(text: str) -> str:
    """Remove MiniMax's <think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Max chars per tool result to avoid blowing up context
_MAX_TOOL_RESULT = 3000
# Max total chars across all messages before we trim old ones
_MAX_CONTEXT_CHARS = 60000


def _estimate_chars(messages: list) -> int:
    """Rough estimate of total chars in message list."""
    total = 0
    for m in messages:
        total += len(m.get("content") or "")
        for tc in m.get("tool_calls", []):
            total += len(tc.get("function", {}).get("arguments", ""))
    return total


def _trim_messages(messages: list, max_chars: int = _MAX_CONTEXT_CHARS) -> list:
    """
    Trim middle messages if context is too large.
    Keep: system prompt (first) + last N messages.
    """
    if _estimate_chars(messages) <= max_chars:
        return messages

    # Always keep system prompt
    system = [messages[0]] if messages and messages[0].get("role") == "system" else []
    rest = messages[len(system):]

    # Keep removing oldest non-system messages until under limit
    while rest and _estimate_chars(system + rest) > max_chars:
        rest.pop(0)

    trimmed = system + rest
    if len(trimmed) < len(messages):
        # Insert a note so LLM knows context was trimmed
        if system:
            trimmed.insert(1, {
                "role": "user",
                "content": "[Note: earlier conversation was trimmed to fit context limit. Continue from here.]",
            })
    return trimmed


class MiniMaxClient:
    """Stateless MiniMax API client with auto-retry and context management."""

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
        """Call MiniMax chat completions API with retry and 400 recovery."""
        # Trim if too long
        messages = _trim_messages(messages)

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

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 400 and attempt < self.max_retries - 1:
                    # 400 usually means context too long — aggressively trim
                    print(f"  [LLM] 400 error, trimming context and retrying "
                          f"({attempt+2}/{self.max_retries})...")
                    messages = _trim_messages(messages,
                                              max_chars=_MAX_CONTEXT_CHARS // (attempt + 2))
                    payload["messages"] = messages
                    # Also try without tools on last retry
                    if attempt == self.max_retries - 2 and tools:
                        print(f"  [LLM] Dropping tools for retry...")
                        payload.pop("tools", None)
                    time.sleep(1)
                    continue
                raise

            except httpx.ReadTimeout:
                if attempt < self.max_retries - 1:
                    print(f"  [LLM] Timeout, retrying ({attempt+2}/{self.max_retries})...")
                    time.sleep(2)
                else:
                    raise

    def agent_loop(self, task: str, system_prompt: str, tools_defs: list,
                   tool_executor, max_turns: int = 10,
                   on_response=None, on_tool_call=None, on_tool_result=None):
        """
        Run a complete agent loop with auto-truncation of tool results.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        for turn in range(1, max_turns + 1):
            t0 = time.perf_counter()
            try:
                response = self.chat(messages, tools=tools_defs)
            except Exception as e:
                print(f"  [LLM] API error on turn {turn}: {e}")
                break

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

                    # Truncate long tool results to prevent context explosion
                    if len(result) > _MAX_TOOL_RESULT:
                        result = result[:_MAX_TOOL_RESULT] + "\n...(truncated)"

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
