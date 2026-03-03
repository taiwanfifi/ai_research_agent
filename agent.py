#!/usr/bin/env python3
"""
AI Research Agent — MiniMax 驅動
==================================
一個能搜論文、寫 code、跑實驗的研究 Agent。

用法:
    python3 agent.py "搜尋最新的 LLM agent 論文，然後寫一個簡單的 transformer attention 實作並測試"
    python3 agent.py "找 HumanEval benchmark，解一道題目並驗證"
    python3 agent.py  (互動模式)
"""

import json
import sys
import time

# ── 載入核心模組 ────────────────────────────────────────────────────
from config import API_KEY, BASE_URL, MODEL, MAX_TURNS, MAX_TOKENS, TEMPERATURE
from core.llm import MiniMaxClient, strip_think
from core.tool_registry import ToolRegistry
from mcp_servers import paper_search, dataset_fetch, code_runner

# ── 初始化 LLM client ──────────────────────────────────────────────
client = MiniMaxClient(
    api_key=API_KEY, base_url=BASE_URL, model=MODEL,
    max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
)

# ── 初始化 Tool Registry ───────────────────────────────────────────
registry = ToolRegistry()
registry.register_module(paper_search)
registry.register_module(dataset_fetch)
registry.register_module(code_runner)

# ── 保留向後相容的全局變數 ──────────────────────────────────────────
ALL_TOOLS = registry.tools
ALL_FUNCTIONS = registry.functions

SYSTEM_PROMPT = """你是一個 AI 研究助手。你可以：

1. **搜尋論文**: 用 search_arxiv、search_semantic_scholar、search_openalex 搜尋學術論文
2. **取得資料集/題目**: 用 search_hf_datasets 搜尋資料集、fetch_leetcode_problem 取得程式題、fetch_humaneval_sample 取得 benchmark
3. **寫程式/跑實驗**: 用 run_python_code 執行 Python 程式碼、write_file 寫檔案、read_file 讀檔案

工作流程：
- 先理解使用者的研究需求
- 搜尋相關論文了解現有方法
- 寫程式碼實作或實驗
- 執行並驗證結果
- 如果有錯誤，自動修正並重試

注意：
- run_python_code 中不要 import 非標準庫（只有 Python 內建庫可用）
- 回答用繁體中文
- 每一步都說明你在做什麼和為什麼
"""


def run_agent(task: str):
    """執行 Agent Loop"""
    print(f"\n{'='*60}")
    print(f"  AI Research Agent (MiniMax-M2.5)")
    print(f"{'='*60}")
    print(f"  任務: {task}")
    print(f"  可用工具: {len(registry)} 個")
    print(f"  最大輪數: {MAX_TURNS}")

    def on_response(turn, content, latency):
        print(f"\n  Agent ({latency:.0f}ms):")
        for line in content.split("\n"):
            print(f"    {line}")

    def on_tool_call(name, args):
        print(f"\n  [Tool] {name}({json.dumps(args, ensure_ascii=False)[:100]})")

    def on_tool_result(name, result):
        print(f"  [Result] {result[:200]}...")

    messages = client.agent_loop(
        task=task,
        system_prompt=SYSTEM_PROMPT,
        tools_defs=registry.tools,
        tool_executor=registry.execute,
        max_turns=MAX_TURNS,
        on_response=on_response,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
    )

    print(f"\n{'='*60}")
    print(f"  Agent 完成")
    print(f"{'='*60}")
    return messages


def interactive_mode():
    """互動模式"""
    print(f"\n{'='*60}")
    print(f"  AI Research Agent — 互動模式")
    print(f"  LLM: MiniMax-M2.5")
    print(f"  工具: {len(registry)} 個 ({', '.join(registry.list_names())})")
    print(f"  輸入 'quit' 退出")
    print(f"{'='*60}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("\n  You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        messages.append({"role": "user", "content": user_input})

        for turn in range(MAX_TURNS):
            response = client.chat(messages, tools=registry.tools)
            choice = response["choices"][0]
            message = choice["message"]
            messages.append(message)

            if message.get("content"):
                clean = strip_think(message["content"])
                if clean:
                    print(f"\n  Agent: {clean}")

            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    func_args = json.loads(tc["function"]["arguments"]) if tc["function"].get("arguments") else {}
                    print(f"  [Tool] {func_name}")
                    result = registry.execute(func_name, func_args)
                    print(f"  [Result] {result[:150]}...")
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                continue

            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        run_agent(task)
    else:
        interactive_mode()
