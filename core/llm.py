"""
MiniMax LLM Client
===================
Provides the core LLM interface for all components.
Handles retries, context compaction (summarization), and 400 recovery.

Context management strategy (like Claude Code's compact mode):
- Proactive: when context exceeds threshold, summarize old messages
- Reactive: on 400 error, compact → trim → drop tools (escalating)
- Tool results are always truncated to prevent individual explosions
- All trimming/compaction preserves tool_call_id consistency
"""

import json
import re
import time
import httpx


def strip_think(text: str) -> str:
    """Remove MiniMax's <think> reasoning blocks."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── Constants ────────────────────────────────────────────────────────

# Max chars per tool result to avoid blowing up context
_MAX_TOOL_RESULT = 4000

# Proactive compaction threshold — compact BEFORE hitting API limits
# MiniMax-M2.5-highspeed: 204,800 tokens (~600K chars)
# We compact early to leave room for response + tools
_COMPACT_THRESHOLD = 100_000

# Hard limit for emergency trimming (if compaction fails)
_MAX_CONTEXT_CHARS = 150_000


# ── Helper functions ────────────────────────────────────────────────

def _estimate_chars(messages: list) -> int:
    """Rough estimate of total chars in message list."""
    total = 0
    for m in messages:
        total += len(m.get("content") or "")
        for tc in m.get("tool_calls", []):
            total += len(tc.get("function", {}).get("arguments", ""))
    return total


def _fix_tool_call_args(tc: dict) -> dict | None:
    """
    Validate and fix a tool_call's function arguments.
    MiniMax API returns 400 if arguments is not valid JSON.
    Returns fixed tool_call, or None if unfixable.
    """
    func = tc.get("function", {})
    args_str = func.get("arguments", "")

    # Already valid JSON
    try:
        json.loads(args_str)
        return tc
    except (json.JSONDecodeError, TypeError):
        pass

    # Try common fixes
    # 1. Truncated JSON — try to close it
    if isinstance(args_str, str) and args_str.strip():
        # Try wrapping in proper JSON
        for fix in [
            args_str + '"}',           # unclosed string
            args_str + '"}]}',         # unclosed array
            args_str + '"}}}',         # deeply nested
            '{"code": ""}',            # give up, use empty
        ]:
            try:
                json.loads(fix)
                fixed_tc = json.loads(json.dumps(tc))  # deep copy
                fixed_tc["function"]["arguments"] = fix
                return fixed_tc
            except (json.JSONDecodeError, TypeError):
                continue

    # Unfixable — convert to a minimal valid JSON
    fixed_tc = json.loads(json.dumps(tc))
    fixed_tc["function"]["arguments"] = "{}"
    return fixed_tc


def _sanitize_messages(messages: list) -> list:
    """
    Fix all known causes of MiniMax 400 errors in message history:

    1. Orphaned tool_call_ids: tool messages without matching assistant,
       or assistant tool_calls without matching tool results.
    2. Invalid JSON in tool_call arguments: LLM sometimes generates
       truncated/broken JSON, which the API rejects on replay.

    Both issues are common after context trimming or when the LLM
    generates malformed output.
    """
    # Pass 0: fix broken JSON in all tool_call arguments
    fixed_messages = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            fixed_tcs = []
            for tc in m["tool_calls"]:
                fixed = _fix_tool_call_args(tc)
                if fixed:
                    fixed_tcs.append(fixed)
            if fixed_tcs:
                cleaned = dict(m)
                cleaned["tool_calls"] = fixed_tcs
                fixed_messages.append(cleaned)
            elif m.get("content"):
                cleaned = dict(m)
                cleaned.pop("tool_calls", None)
                fixed_messages.append(cleaned)
        else:
            fixed_messages.append(m)

    messages = fixed_messages

    # Pass 1: collect all tool_call_ids from both sides
    assistant_tc_ids = set()
    tool_result_ids = set()

    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls", []):
                tc_id = tc.get("id")
                if tc_id:
                    assistant_tc_ids.add(tc_id)
        elif m.get("role") == "tool":
            tc_id = m.get("tool_call_id")
            if tc_id:
                tool_result_ids.add(tc_id)

    # Valid pairs: tool_call_id exists on BOTH sides
    valid_tc_ids = assistant_tc_ids & tool_result_ids

    # Pass 2: filter messages, keeping only complete pairs
    result = []
    for m in messages:
        if m.get("role") == "tool":
            if m.get("tool_call_id") in valid_tc_ids:
                result.append(m)
            # else: orphaned tool message → drop
        elif m.get("role") == "assistant" and m.get("tool_calls"):
            # Keep only tool_calls that have matching tool results
            valid_tcs = [tc for tc in m["tool_calls"]
                         if tc.get("id") in valid_tc_ids]
            if valid_tcs:
                cleaned = dict(m)
                cleaned["tool_calls"] = valid_tcs
                result.append(cleaned)
            elif m.get("content"):
                # No valid tool_calls left — keep as text-only
                cleaned = dict(m)
                cleaned.pop("tool_calls", None)
                result.append(cleaned)
            # else: empty message → drop
        else:
            result.append(m)

    return result


def _trim_messages(messages: list, max_chars: int = _MAX_CONTEXT_CHARS) -> list:
    """
    Emergency trim: drop old messages if context is too large.
    This is the FALLBACK when compaction fails.
    Keep: system prompt (first) + last N messages.
    Always sanitizes tool_call_id pairs after trimming.
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

    # CRITICAL: fix any broken tool_call_id pairs
    trimmed = _sanitize_messages(trimmed)

    if len(trimmed) < len(messages):
        if system:
            trimmed.insert(1, {
                "role": "user",
                "content": "[Earlier conversation was trimmed to fit context limit. Continue from here.]",
            })
    return trimmed


def _find_clean_split(rest: list, keep_recent: int = 8) -> int:
    """
    Find index in 'rest' (messages after system prompt) to split [old | recent].
    Tries to split at a user or assistant boundary so tool_call groups stay intact.
    """
    if len(rest) <= keep_recent:
        return 0  # nothing to split

    target = len(rest) - keep_recent

    # Scan backward from target to find a valid split point
    for i in range(target, max(0, target - 10), -1):
        role = rest[i].get("role")
        if role in ("user", "assistant"):
            return i

    # Scan forward from target
    for i in range(target, len(rest)):
        role = rest[i].get("role")
        if role in ("user", "assistant"):
            return i

    return max(0, target)


# ── Main Client ─────────────────────────────────────────────────────

class MiniMaxClient:
    """MiniMax API client with auto-retry, context compaction, and recovery."""

    def __init__(self, api_key: str, base_url: str = "https://api.minimax.io/v1",
                 model: str = "MiniMax-M2.5", max_tokens: int = 4096,
                 temperature: float = 0.3, timeout: int = 120, max_retries: int = 4):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    # ── Context Compaction (like Claude Code's /compact) ──────────────

    def compact_messages(self, messages: list, keep_recent: int = 8) -> list:
        """
        Summarize old messages while keeping recent context intact.
        Analogous to Claude Code's compact mode.

        Strategy:
        1. Keep system prompt (first message)
        2. Summarize old exchanges into a compact summary via LLM
        3. Keep recent N messages verbatim (with tool_call_ids intact)
        4. Sanitize to ensure no orphaned tool_call_ids

        Returns new message list with fewer messages but same information.
        """
        total_chars = _estimate_chars(messages)
        if total_chars <= _COMPACT_THRESHOLD and len(messages) <= keep_recent + 5:
            return messages  # no need to compact

        # Split system from rest
        system = [messages[0]] if messages and messages[0].get("role") == "system" else []
        rest = messages[len(system):]

        if len(rest) <= keep_recent:
            return messages

        # Find clean split point
        split_idx = _find_clean_split(rest, keep_recent=keep_recent)
        old_messages = rest[:split_idx]
        recent_messages = rest[split_idx:]

        if not old_messages:
            return messages

        # Summarize old messages via LLM
        print(f"  [LLM] Compacting: {len(old_messages)} old messages → summary "
              f"(keeping {len(recent_messages)} recent)...")
        summary = self._summarize_for_compact(old_messages)

        # Build compacted message list
        compacted = system + [
            {
                "role": "user",
                "content": (
                    "[Conversation Summary — earlier exchanges were compacted]\n"
                    f"{summary}\n"
                    "[End Summary — continue from recent messages below]"
                ),
            },
        ] + recent_messages

        # CRITICAL: sanitize tool_call_id pairs
        compacted = _sanitize_messages(compacted)

        old_chars = total_chars
        new_chars = _estimate_chars(compacted)
        reduction = ((old_chars - new_chars) / old_chars * 100) if old_chars else 0
        print(f"  [LLM] Compacted: {old_chars:,} → {new_chars:,} chars "
              f"(-{reduction:.0f}%, {len(old_messages)} msgs summarized)")

        return compacted

    def _summarize_for_compact(self, messages: list) -> str:
        """
        Use the LLM to summarize a batch of old messages into compact text.
        Keeps the summary request itself small to avoid nested 400 errors.
        """
        # Build a condensed text representation of messages
        parts = []
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content", "") or ""
            tool_calls = m.get("tool_calls", [])

            if role == "tool":
                # Tool results are often huge — aggressively shorten for summary
                parts.append(f"[Tool result]: {content[:300]}")
            elif role == "assistant":
                if tool_calls:
                    for tc in tool_calls:
                        fname = tc.get("function", {}).get("name", "")
                        fargs = tc.get("function", {}).get("arguments", "")[:150]
                        parts.append(f"[Called {fname}]: {fargs}")
                if content:
                    clean = strip_think(content)
                    if clean:
                        parts.append(f"[Assistant]: {clean[:400]}")
            elif role == "user":
                parts.append(f"[User]: {content[:400]}")

        conversation_text = "\n".join(parts)
        # Hard cap so the summary request itself stays well under API limits
        if len(conversation_text) > 12000:
            conversation_text = conversation_text[:12000] + "\n...(older content omitted)"

        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": (
                            "Summarize this agent conversation history concisely. "
                            "You MUST preserve:\n"
                            "- Key research findings, paper titles, and results\n"
                            "- Code that was written (key function names, logic)\n"
                            "- Execution outputs and benchmark numbers\n"
                            "- Errors encountered and how they were resolved\n"
                            "- Decisions made and their rationale\n"
                            "- Current state of the task\n\n"
                            "Use bullet points. Be factual and specific. "
                            "This summary will replace the original messages — "
                            "the agent needs to continue working from this context."
                        )},
                        {"role": "user", "content": conversation_text},
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.1,
                },
                timeout=60,
            )
            resp.raise_for_status()
            result = resp.json()
            summary = strip_think(result["choices"][0]["message"]["content"])
            print(f"  [LLM] Summary generated ({len(summary)} chars)")
            return summary

        except Exception as e:
            print(f"  [LLM] Summarization API failed ({e}), using mechanical fallback")
            # Fallback: extract key lines without LLM
            fallback = []
            for m in messages:
                content = m.get("content", "") or ""
                role = m.get("role", "?")
                if content and role != "tool":
                    clean = strip_think(content) if role == "assistant" else content
                    if clean:
                        fallback.append(f"- [{role}]: {clean[:120]}")
                elif role == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        fname = tc.get("function", {}).get("name", "")
                        fallback.append(f"- [called {fname}]")
            return "Previous conversation (auto-extracted):\n" + "\n".join(fallback[-20:])

    # ── Chat API ──────────────────────────────────────────────────────

    def chat(self, messages: list, tools: list = None) -> dict:
        """
        Call MiniMax chat completions with retry, compaction, and recovery.

        Recovery strategy on 400 (escalating):
        1. Sanitize tool_call_ids (often the real cause!) → retry
        2. Compact (summarize old messages) → retry
        3. Aggressive trim + sanitize → retry
        """
        # Proactive: trim if way over hard limit, then sanitize
        messages = _trim_messages(messages)
        messages = _sanitize_messages(messages)

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
                error_body = ""
                try:
                    error_body = e.response.text[:500]
                except Exception:
                    pass

                if status == 400 and attempt < self.max_retries - 1:
                    total_chars = _estimate_chars(messages)
                    print(f"  [LLM] 400 error (context ~{total_chars:,} chars), "
                          f"recovery {attempt+1}/{self.max_retries-1}...")
                    if error_body:
                        print(f"  [LLM] Error detail: {error_body[:200]}")

                    if attempt == 0:
                        # Strategy 1: Sanitize tool_call_ids (most common cause!)
                        print(f"  [LLM] → Strategy: sanitize tool_call_ids + compact")
                        messages = _sanitize_messages(messages)
                        if total_chars > _COMPACT_THRESHOLD:
                            messages = self.compact_messages(messages, keep_recent=8)
                    elif attempt == 1:
                        # Strategy 2: Aggressive trim + sanitize
                        print(f"  [LLM] → Strategy: aggressive trim + sanitize")
                        messages = _trim_messages(messages,
                                                  max_chars=_MAX_CONTEXT_CHARS // 3)
                        messages = _sanitize_messages(messages)
                    else:
                        # Strategy 3: Nuclear — drop tools + extreme trim
                        print(f"  [LLM] → Strategy: drop tools + extreme trim")
                        messages = _trim_messages(messages,
                                                  max_chars=_MAX_CONTEXT_CHARS // 5)
                        messages = _sanitize_messages(messages)
                        payload.pop("tools", None)

                    payload["messages"] = messages
                    time.sleep(1)
                    continue
                raise

            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.WriteTimeout,
                    httpx.PoolTimeout, httpx.TimeoutException):
                if attempt < self.max_retries - 1:
                    print(f"  [LLM] Timeout, retrying ({attempt+2}/{self.max_retries})...")
                    time.sleep(2)
                else:
                    raise

    # ── Agent Loop ────────────────────────────────────────────────────

    def agent_loop(self, task: str, system_prompt: str, tools_defs: list,
                   tool_executor, max_turns: int = 10,
                   on_response=None, on_tool_call=None, on_tool_result=None,
                   execution_log_summary: str = None):
        """
        Run a complete agent loop with proactive context compaction.

        Before each LLM call, checks if context is getting large and
        compacts automatically — like Claude Code's auto-compact feature.
        This prevents 400 errors instead of just reacting to them.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task},
        ]

        for turn in range(1, max_turns + 1):
            # ── Proactive compaction ───────────────────────────────
            total_chars = _estimate_chars(messages)
            if total_chars > _COMPACT_THRESHOLD:
                print(f"  [LLM] Context large ({total_chars:,} chars > "
                      f"{_COMPACT_THRESHOLD:,} threshold), compacting...")
                messages = self.compact_messages(messages, keep_recent=10)

            # ── Reserve last 2 turns for summary (no tools) ───────
            # This prevents workers from exhausting all turns on tool
            # calls and never producing a text summary.
            is_summary_turn = (turn >= max_turns - 1)
            current_tools = None if is_summary_turn else tools_defs

            if is_summary_turn and turn == max_turns - 1:
                # Build summary prompt — with or without execution log data
                summary_content = (
                    "[System] You are running low on turns. "
                    "Stop using tools and write your FINAL SUMMARY now.\n\n"
                )

                # If execution log data is available, inject ground truth
                if execution_log_summary and execution_log_summary.strip() != "(no tool executions recorded)":
                    summary_content += (
                        "ACTUAL EXECUTION RESULTS (ground truth from execution_log.json):\n"
                        f"{execution_log_summary}\n\n"
                        "CRITICAL: Use ONLY the numbers shown above. "
                        "If a metric is not listed, say 'not measured'.\n"
                    )
                else:
                    summary_content += (
                        "CRITICAL RULES:\n"
                        "1. ONLY report numbers that appeared in tool execution results above\n"
                        "2. If you did not run code or find papers, say so explicitly — do NOT invent numbers\n"
                        "3. For each claim, reference which tool call produced it\n"
                        "4. If experiments failed or were not completed, report the failure honestly\n"
                    )

                summary_content += (
                    "5. Structure with ### headers. Include file names of files you actually created.\n\n"
                    "It is MUCH better to report 'experiment not completed' than to fabricate results."
                )

                messages.append({
                    "role": "user",
                    "content": summary_content,
                })
                print(f"  [LLM] Turn {turn}/{max_turns}: forcing summary (no tools)")

            t0 = time.perf_counter()
            try:
                response = self.chat(messages, tools=current_tools)
            except Exception as e:
                print(f"  [LLM] API error on turn {turn}: {e}")
                # Emergency: compact aggressively and try one more time
                try:
                    print(f"  [LLM] Emergency compaction + retry...")
                    messages = self.compact_messages(messages, keep_recent=4)
                    messages = _trim_messages(messages,
                                              max_chars=_MAX_CONTEXT_CHARS // 3)
                    messages = _sanitize_messages(messages)
                    response = self.chat(messages, tools=current_tools)
                except Exception as e2:
                    print(f"  [LLM] Recovery failed: {e2}")
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
