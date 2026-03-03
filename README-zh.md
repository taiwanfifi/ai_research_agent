# AI 自動研究 Agent（繁體中文說明）

## 這個系統到底在幹嘛？

簡單說：**你丟一句話給它，它會自己查論文、寫程式、跑實驗、寫報告**。

整個流程全自動，你不需要動手。它背後有一個「主管」（Supervisor）會把你的目標拆成一堆小任務，然後分配給三個「工人」去做：

| 工人 | 他會做什麼 | 實際用的工具 |
|------|-----------|-------------|
| **Explorer（探索者）** | 搜論文、搜 dataset、搜 GitHub repo | arXiv API、Semantic Scholar API、OpenAlex API、HuggingFace API、GitHub API |
| **Coder（寫程式的）** | 寫 Python 程式碼、執行、debug | `run_python_code`（真的會跑 subprocess）、`write_file`、`read_file` |
| **Reviewer（審查者）** | 跑 benchmark、評估結果、分析好壞 | 跟 Coder 一樣的工具 + HuggingFace datasets + LeetCode/HumanEval 題目 |

### 舉個例子

你輸入：
```
python3 main.py "研究 Flash Attention 優化方法，然後實作一個簡化版並跑 benchmark"
```

系統會自動做這些事：

1. **Supervisor 拆解任務**（用 LLM 決定）：
   - 任務 1 → Explorer：去 arXiv 和 Semantic Scholar 搜 Flash Attention 相關論文
   - 任務 2 → Explorer：搜 GitHub 上的 Flash Attention 實作
   - 任務 3 → Coder：根據論文寫一個簡化版 Flash Attention（Python）
   - 任務 4 → Coder：寫一個 naive attention 做對照
   - 任務 5 → Reviewer：跑 benchmark 比較兩個版本的速度差異

2. **每個工人會自己呼叫工具**：
   - Explorer 真的會打 arXiv API 拿到論文標題、作者、摘要
   - Coder 真的會寫 Python code 然後 `run_python_code` 跑起來，有 bug 會自己修
   - Reviewer 真的會寫測試程式碼跑 benchmark 然後分析結果

3. **所有結果自動存到知識庫**（Knowledge Tree）

4. **最後產出一份 Markdown 報告**

---

## 怎麼裝、怎麼跑

### 第一步：Clone

```bash
git clone git@github.com:taiwanfifi/ai_research_agent.git
cd ai_research_agent
```

### 第二步：設定 API Key

這個系統用 [MiniMax](https://www.minimax.io/) 的 LLM（模型 MiniMax-M2.5）。你需要一組 API Key：

```bash
# 方法 A：環境變數
export MINIMAX_API_KEY="sk-你的key"

# 方法 B：寫到檔案（放在上一層目錄）
echo "sk-你的key" > ../apikey.txt
```

### 第三步：直接跑

不用裝任何套件！全部用 Python 標準庫。只要你有 Python 3.11+：

```bash
python3 main.py "你的研究目標"
```

---

## 所有用法

### 1. 開一個新的研究任務

```bash
# 基本用法（英文報告）
python3 main.py "research Flash Attention optimization methods"

# 中文報告
python3 main.py --zh "研究 Flash Attention 優化方法"

# 開啟跨任務知識（會參考之前做過的任務的成果）
python3 main.py --cross "基於之前的研究繼續深入"

# 全部組合
python3 main.py --zh --cross "深入研究 KV Cache 壓縮技術"
```

### 2. 恢復之前的任務（Resume）

每次跑的任務都會存下來。你可以用模糊匹配找回它：

```bash
# 用關鍵字（匹配 slug）
python3 main.py --resume flash_attention

# 用時間（匹配 timestamp）
python3 main.py --resume 20260303

# 不帶參數 = 恢復最近一次
python3 main.py --resume

# 恢復 + 改方向（本來在研究 Flash Attention，現在只想看 v2）
python3 main.py --resume flash --direction "只研究 Flash Attention v2 的 IO-aware 實作"
```

如果匹配到多個任務，會列出來讓你選：

```
  Multiple missions match 'attention':
    [1] mission_20260303_185200_flash_attention_search
        Goal: 研究 Flash Attention 優化方法
    [2] mission_20260303_190000_attention_implement
        Goal: 實作 efficient attention

  Select [1-2]: _
```

### 3. 查看任務列表

```bash
python3 main.py --list-missions
```

輸出像這樣：

```
  Missions (2)
  ============================================================
  mission_20260303_185200_flash_attention_search
    Goal: 研究 Flash Attention 優化方法
    Status: finished  Language: zh

  mission_20260303_190000_attention_implement
    Goal: 實作 efficient attention
    Direction: 只研究 Flash Attention v2
    Status: running  Language: en [cross]
```

### 4. 查看狀態

```bash
python3 main.py --status
```

### 5. 互動模式

```bash
python3 main.py --interactive
# 或者什麼都不加
python3 main.py
```

進入互動模式後可以用的指令：

```
  > 研究 KV Cache 壓縮技術              ← 直接打字 = 開新任務
  > /resume flash_attention              ← 恢復之前的任務
  > /resume flash "改成只看 v2"          ← 恢復 + 改方向
  > /missions                            ← 列出所有任務
  > /cross                               ← 開/關 跨任務知識
  > /zh                                  ← 切換 英文 ↔ 繁體中文 報告
  > /status                              ← 當前任務狀態
  > /report                              ← 立刻產生一份報告
  > quit                                 ← 離開
```

---

## 跑完東西在哪裡？

這是最重要的部分。每個任務的所有檔案都在 `missions/` 資料夾裡，**完全隔離**：

```
missions/
└── mission_20260303_185200_flash_attention_search/
    │
    ├── mission.json          ← 任務的基本資訊
    │
    ├── knowledge/            ← ★ 知識庫（最重要的產出）
    │   ├── papers/           ← 搜到的論文摘要
    │   ├── code/             ← 寫的程式碼 + 執行結果
    │   ├── experiments/      ← benchmark 數據 + 評估結果
    │   ├── methods/          ← 發現的演算法、技術
    │   └── reports/          ← 任務過程中的知識報告
    │
    ├── reports/              ← ★ 最終報告
    │   ├── progress_en_20260303_192000.md   ← 英文報告
    │   └── progress_zh_20260303_192000.md   ← 中文報告
    │
    ├── workspace/            ← 工作區（暫存檔案）
    │
    └── state/                ← checkpoint（用來 resume 的）
```

### 具體去看什麼

#### 看報告

```bash
# 看最新的中文報告
cat missions/mission_20260303_*/reports/progress_zh_*.md
```

報告長這樣：

```markdown
# 研究任務報告

產生時間：2026-03-03 19:20:00

## 任務目標

研究 Flash Attention 優化方法，然後實作一個簡化版並跑 benchmark

## 進度（5/5 項任務）

### 已完成
- [explorer] 搜尋 Flash Attention 相關論文（23.5秒）
- [explorer] 搜尋 GitHub 上的參考實作（15.2秒）
- [coder] 實作簡化版 Flash Attention（45.8秒）
- [coder] 實作 naive attention 對照版（12.3秒）
- [reviewer] 跑 benchmark 比較效能（38.1秒）

## 知識庫統計

- 總計：8 項
- papers：3 項
- code：3 項
- experiments：2 項

## 後續步驟

（無待執行任務）
```

#### 看搜到的論文

```bash
ls missions/mission_20260303_*/knowledge/papers/
# explorer_1709649120.md
# explorer_1709649135.md

cat missions/mission_20260303_*/knowledge/papers/explorer_*.md
```

每個檔案裡面會有搜到的論文標題、作者、摘要、引用數等等。

#### 看寫的程式碼

```bash
ls missions/mission_20260303_*/knowledge/code/
# coder_1709649200.md
# coder_1709649250.md

cat missions/mission_20260303_*/knowledge/code/coder_*.md
```

裡面會有完整的 Python 程式碼 + 執行結果。

#### 看實驗 / benchmark 結果

```bash
cat missions/mission_20260303_*/knowledge/experiments/reviewer_*.md
```

裡面會有 benchmark 方法、測試結果、數據分析。

#### 看任務 manifest

```bash
cat missions/mission_20260303_*/mission.json
```

```json
{
  "mission_id": "mission_20260303_185200_flash_attention_search",
  "goal": "研究 Flash Attention 優化方法",
  "direction": "研究 Flash Attention 優化方法",
  "slug": "flash_attention_search",
  "created_at": "2026-03-03T18:52:00",
  "language": "zh",
  "cross_knowledge": false,
  "status": "finished"
}
```

---

## 典型工作流程

### 流程一：從零開始的完整研究

```bash
# Step 1: 先做文獻調研
python3 main.py --zh "調研 2024-2026 年 efficient attention 的最新進展"

# Step 2: 看報告，決定下一步
cat missions/mission_*_efficient_attention*/reports/progress_zh_*.md

# Step 3: 開一個新任務，開啟 cross-knowledge 讓它參考第一次的成果
python3 main.py --zh --cross "實作 Flash Attention v2 並跟 naive attention 做 benchmark"

# Step 4: 看程式碼和實驗結果
cat missions/mission_*_flash_attention*/knowledge/code/*.md
cat missions/mission_*_flash_attention*/knowledge/experiments/*.md
```

### 流程二：中途改方向

```bash
# 原本在研究 Flash Attention
python3 main.py "research Flash Attention optimization"

# 跑到一半想改成只看 IO-aware 的部分
# Ctrl+C 中斷

# 用 resume + 改方向繼續
python3 main.py --resume flash --direction "focus only on IO-aware memory optimization in Flash Attention v2"
```

### 流程三：互動式研究（推薦新手用）

```bash
python3 main.py

# 進入互動模式：
  > 搜尋最近的 speculative decoding 論文
  # （系統跑完，你看到報告）

  > /zh
  # Report language: zh（切換成中文）

  > 實作一個簡單的 speculative decoding
  # （系統開一個新任務，寫程式跑實驗）

  > /missions
  # 看到兩個任務

  > /resume speculative "改成用 Medusa 方法"
  # 恢復第一個任務，方向改成 Medusa

  > quit
```

### 流程四：快速單次查論文

如果你只是想快速查個東西，不需要完整 mission，用 legacy agent：

```bash
python3 agent.py "搜尋 2025 年最新的 LLM 量化方法論文"
```

這個不會建立 mission 目錄，結果直接印在 terminal。

---

## 實際能力與限制

### 它真的能做到的事

- 搜尋 arXiv / Semantic Scholar / OpenAlex 論文（**真的打 API，不是假的**）
- 搜尋 HuggingFace 上的 dataset
- 搜尋 GitHub repo 和 code（有速率限制：每分鐘 10 次）
- 用 `subprocess` 跑 Python 程式碼（**真的會執行，不是模擬**）
- 寫檔案到 workspace
- 讀取 workspace 的檔案
- 抓 LeetCode 題目
- 抓 HumanEval benchmark 範例

### 限制

- **只能用 Python 標準庫**：`run_python_code` 不能 import numpy、torch 等外部套件
- **沒有網頁瀏覽**：不能打開任意 URL，只能用上面列的那些 API
- **沒有 PDF 下載**：能找到論文的 arXiv 連結，但不會去下載 PDF 全文
- **30 秒 timeout**：程式碼執行最長 30 秒，跑不完就會被 kill
- **LLM 品質取決於 MiniMax**：如果 LLM 回答品質不好，寫出來的程式碼也會不好
- **每個 worker 最多 10 輪對話**：一個任務裡如果 bug 太多修不完，就會放棄

---

## 環境變數

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `MINIMAX_API_KEY` | 無（必填） | MiniMax API Key |
| `MINIMAX_BASE_URL` | `https://api.minimax.io/v1` | API 網址 |
| `MINIMAX_MODEL` | `MiniMax-M2.5` | 用哪個模型 |
| `MAX_TURNS` | `10` | 每個 worker 最多幾輪對話 |
| `MAX_TOKENS` | `4096` | 每次回覆最多幾個 token |
| `TEMPERATURE` | `0.3` | 溫度（越低越確定性） |
| `API_TIMEOUT` | `120` | API 超時（秒） |

---

## 專案結構

```
ai_research_agent/
├── main.py                  ← 主入口（你跑的就是這個）
├── agent.py                 ← Legacy 單一 agent（快速測試用）
├── config.py                ← API key、路徑、參數設定
│
├── core/
│   ├── mission.py           ← Mission 管理（建立/列表/搜尋/恢復）
│   ├── llm.py               ← MiniMax LLM 客戶端
│   ├── tool_registry.py     ← 工具註冊中心
│   ├── event_bus.py         ← 事件系統
│   └── state.py             ← JSON 狀態持久化
│
├── supervisor/
│   ├── supervisor.py        ← 主管（拆任務、分配工人、管進度）
│   ├── planner.py           ← 任務拆解（goal → 小任務清單）
│   └── reporter.py          ← 報告生成（英文/中文）
│
├── workers/
│   ├── base_worker.py       ← Worker 基底類別
│   ├── explorer.py          ← 探索者（搜論文/搜 dataset）
│   ├── coder.py             ← 寫程式的（寫 code/跑 code）
│   └── reviewer.py          ← 審查者（跑 benchmark/分析結果）
│
├── knowledge/
│   ├── tree.py              ← 知識樹（自動分類、跨任務搜尋）
│   ├── index.py             ← 索引管理
│   └── categories.py        ← 分類定義
│
├── mcp_servers/              ← 實際的工具實作
│   ├── paper_search.py      ← arXiv + Semantic Scholar + OpenAlex
│   ├── code_runner.py       ← Python 程式碼執行 + 檔案讀寫
│   ├── dataset_fetch.py     ← HuggingFace + LeetCode + HumanEval
│   └── github_search.py     ← GitHub 搜尋
│
├── skills/                   ← 技能系統（可自我進化）
│   ├── registry.py
│   ├── meta_skill.py
│   └── builtin/
│
└── missions/                 ← ★ 所有任務的資料都在這裡
    ├── mission_20260303_185200_flash_attention_search/
    │   ├── mission.json
    │   ├── knowledge/       ← 論文、程式碼、實驗結果
    │   ├── reports/         ← 最終報告
    │   ├── workspace/       ← 暫存
    │   └── state/           ← checkpoint
    └── mission_20260303_190000_.../
```

---

## 常見問題

### Q: 跑一次要花多少錢 / 多久？

每個任務大概跑 3-7 個小任務，每個小任務最多 10 輪 LLM 對話。以 MiniMax-M2.5 的定價來說成本很低。時間取決於任務複雜度，通常 2-10 分鐘。

### Q: 可以換成 GPT-4 / Claude 嗎？

理論上可以，但需要改 `core/llm.py` 裡的 `MiniMaxClient` 換成 OpenAI 或 Anthropic 的 API。介面是相容的（都是 OpenAI function calling 格式）。

### Q: 程式碼跑不了外部套件怎麼辦？

目前只支援 Python 標準庫。如果你需要跑 numpy/torch，可以改 `mcp_servers/code_runner.py` 裡面的 `run_python_code`，讓它用你本機的 Python 環境。

### Q: 知識庫滿了會怎樣？

知識樹有自動重組機制：當一個分類超過 20 筆時，會用 LLM 自動拆成子分類。你不用管它。
