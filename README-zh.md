# AI 自動研究 Agent（繁體中文說明）

## 這個系統到底在幹嘛？

簡單說：**你丟一句話給它，它會自己查論文、寫程式、跑實驗、寫報告**。

整個流程全自動，你不需要動手。它背後有一個「主管」（Supervisor），像真正的研究人員一樣 **自主思考、靈活應變**：做得不好會重來，搜得不夠會再搜，code 跑失敗會自己修，有階段成果就寫報告。

### 三個工人

| 工人 | 他會做什麼 | 實際用的工具 |
|------|-----------|-------------|
| **Explorer（探索者）** | 搜論文、搜 dataset、搜 GitHub repo | arXiv API、Semantic Scholar API、OpenAlex API、HuggingFace API、GitHub API |
| **Coder（寫程式的）** | 寫 Python 程式碼、執行、debug、用 GPU 加速 | `run_python_code`、`pip_install`、`write_file`、`read_file`（全部鎖定在 mission workspace） |
| **Reviewer（審查者）** | 跑 benchmark、評估結果、分析好壞 | 跟 Coder 一樣的工具 + HuggingFace datasets + LeetCode/HumanEval 題目 |

### 硬體感知

系統啟動時會自動偵測你的硬體，告訴 Coder/Reviewer：

- **Apple Silicon Mac** → 自動用 `torch.device("mps")` 加速
- **NVIDIA GPU** → 自動用 `torch.device("cuda")`
- **沒有 GPU** → 用 CPU，不會傻傻寫 GPU code 然後報錯

### 記憶蒸餾系統（InsightDAG）

**不是滑動視窗！** Supervisor 有一套 DAG 結構的研究記憶：

```
每個 worker 完成任務後 → LLM 提取一條 insight（心得）
                        → 加入 InsightDAG（有引用關係、relevance 分數）

每輪決策前 → LLM 看所有 active insights（按 relevance 排序）
           → 挑出最重要的 → 提升 relevance
           → 不重要的 → 衰減 → 太低就歸檔
           → 產生 working memory（當前研究理解）

效果：
  - 重要的舊 insight 會一直活著（像研究日誌裡的關鍵發現）
  - 失敗的嘗試會自然衰減消失（但不刪除，可追溯）
  - 新的 insight 如果呼應了舊的，會建立引用關係
  - 就像真正研究者的記憶：重要的越記越牢，瑣碎的自然忘掉
```

### 程式碼版本追蹤（CodeVersionStore）

Coder 寫的每一個檔案都自動做版本追蹤：

```
workspace/.code_store/model/
├── v001.py              ← 第一版完整原始碼
├── v002.py              ← 第二版
├── v001_v002.diff       ← 兩版之間的 diff
├── manifest.json        ← 版本歷史 + 原因
└── module_map.json      ← AST 解析的模組地圖（函數、類別、signature）
```

**Debug 時的威力**：不是把整個檔案丟給 LLM，而是：
1. 解析 traceback → 找出是哪個函數出錯
2. 只給那個函數的原始碼 + 最近的 diff + I/O contract
3. LLM 拿到精準的 context，修 bug 更快更準

### 任務隔離 Workspace

每個 mission 的檔案 I/O 完全隔離：

```
missions/mission_20260303_185200_flash_attention/
├── workspace/            ← write_file / read_file / run_python_code 都在這裡
│   ├── model.py          ← Coder 寫的程式碼
│   └── .code_store/      ← 版本追蹤資料
└── ...
```

`write_file`、`read_file`、`run_python_code` 在系統啟動時會被 **重新綁定到 mission workspace**。不同任務的檔案絕對不會互相干擾。

### Supervisor 怎麼運作

**不是固定流水線！** 每一輪 Supervisor 會看 working memory（蒸餾過的研究理解），然後自己決定下一步：

```
每一輪（cycle）：
  1. 蒸餾所有 insights → 更新 working memory
  2. 看 working memory → 決定下一步

  "論文搜得不夠"        → 派 Explorer 再搜
  "有想法了，來寫 code"  → 派 Coder 實作（自動版本追蹤）
  "code 跑失敗了"       → 派 Coder 帶著精準 fix context 去修
  "跑出來了，測一下"     → 派 Reviewer 跑 benchmark
  "結果不好"            → 派 Coder 改進，或 Explorer 找更好的方法
  "方向不對"            → 重新規劃（replan）
  "有階段成果了"         → 寫一份中間報告
  "目標達成"            → 寫最終報告 → 結束
```

**每一輪都會存 checkpoint**（包含完整的 InsightDAG + working memory），所以：
- `Ctrl+C` 隨時中斷都安全
- `--resume` 從斷點精確繼續（恢復所有 insight、relevance 分數、版本追蹤）
- 關機、斷電、網路斷都不怕

### 舉個例子

你輸入：
```
python3 main.py --zh "研究 Flash Attention 優化方法，然後實作一個簡化版並跑 benchmark"
```

系統可能會這樣跑（**每一步都是 LLM 自己決定的**，不是寫死的）：

```
Cycle 1: search_more → Explorer 去 arXiv 搜 Flash Attention 論文
         [Memory] Insight i0001: 找到 3 篇核心論文...（relevance: 0.5）

Cycle 2: search_more → Explorer 去 GitHub 搜參考實作
         [Memory] Insight i0002: GitHub 上有 2 個高星實作...（relevance: 0.5）
         [Memory] 蒸餾 → i0001, i0002 被選為重要 → relevance 提升

Cycle 3: implement → Coder 根據論文寫 Flash Attention
         [CodeStore] model.py v001 tracked (3 modules)

Cycle 4: fix_code → Coder 拿到 fix context（只有出錯的函數 + diff）
         [CodeStore] model.py v002 tracked, diff: ~attention_forward

Cycle 5: implement → Coder 寫 naive attention 做對照
         [CodeStore] baseline.py v001 tracked

Cycle 6: benchmark → Reviewer 跑 benchmark 比較兩個版本
         [Memory] Insight i0005: Flash Attention 快 2.3x...

Cycle 7: done → 最終報告
```

如果中途 Ctrl+C 斷在 Cycle 4：
```bash
python3 main.py --resume flash
# → 從 Cycle 5 繼續
# → InsightDAG 完整恢復（4 個 insights, relevance 分數都在）
# → CodeVersionStore 知道 model.py 已經是 v002
```

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

Python 3.11+ 即可。如果你要跑 ML 相關的實驗，建議先裝 torch：

```bash
# 基本（不跑 GPU code 也行）
python3 main.py "你的研究目標"

# 建議：裝好 ML 套件讓 Coder 能寫 torch code
pip install torch numpy matplotlib
```

系統會自動偵測哪些套件已裝、有什麼 GPU，然後告訴 LLM。

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

# 恢復 + 改方向
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

每個任務的所有檔案都在 `missions/` 資料夾裡，**完全隔離**：

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
    │   ├── progress_en_20260303_192000.md
    │   └── progress_zh_20260303_192000.md
    │
    ├── workspace/            ← ★ 工作區（Coder 寫的程式碼都在這）
    │   ├── model.py          ← 實際程式碼
    │   ├── baseline.py
    │   └── .code_store/      ← 自動版本追蹤
    │       ├── model/
    │       │   ├── v001.py, v002.py     ← 每版快照
    │       │   ├── v001_v002.diff       ← 版本差異
    │       │   ├── manifest.json        ← 版本歷史
    │       │   └── module_map.json      ← AST 模組地圖
    │       └── baseline/
    │           └── ...
    │
    └── state/                ← checkpoint（InsightDAG + working memory）
```

### 看報告

```bash
cat missions/mission_20260303_*/reports/progress_zh_*.md
```

### 看搜到的論文

```bash
cat missions/mission_20260303_*/knowledge/papers/explorer_*.md
```

### 看寫的程式碼

```bash
# 看知識庫裡的程式碼記錄
cat missions/mission_20260303_*/knowledge/code/coder_*.md

# 看實際的 Python 檔案（workspace 裡）
cat missions/mission_20260303_*/workspace/*.py

# 看版本歷史
cat missions/mission_20260303_*/workspace/.code_store/*/manifest.json

# 看 AST 模組地圖（函數、類別、signature）
cat missions/mission_20260303_*/workspace/.code_store/*/module_map.json
```

### 看實驗 / benchmark 結果

```bash
cat missions/mission_20260303_*/knowledge/experiments/reviewer_*.md
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

# Step 4: 看程式碼和版本追蹤
cat missions/mission_*_flash_attention*/workspace/*.py
cat missions/mission_*_flash_attention*/workspace/.code_store/*/manifest.json
```

### 流程二：中途改方向

```bash
# 原本在研究 Flash Attention
python3 main.py "research Flash Attention optimization"

# 跑到一半想改成只看 IO-aware 的部分
# Ctrl+C 中斷

# 用 resume + 改方向繼續（InsightDAG 和版本追蹤全部保留）
python3 main.py --resume flash --direction "focus only on IO-aware memory optimization in Flash Attention v2"
```

### 流程三：互動式研究（推薦新手用）

```bash
python3 main.py

  > 搜尋最近的 speculative decoding 論文
  # （系統跑完，InsightDAG 累積了 insights）

  > /zh
  # Report language: zh

  > 實作一個簡單的 speculative decoding
  # （新任務，Coder 寫的 code 自動版本追蹤）

  > /resume speculative "改成用 Medusa 方法"
  # 恢復第一個任務，方向改成 Medusa

  > quit
```

---

## 實際能力與限制

### 它真的能做到的事

- 搜尋 arXiv / Semantic Scholar / OpenAlex 論文（**真的打 API，不是假的**）
- 搜尋 HuggingFace 上的 dataset
- 搜尋 GitHub repo 和 code（有速率限制：每分鐘 10 次）
- 用 `subprocess` 跑 Python 程式碼（**真的會執行，不是模擬**）
- **可以用 torch、numpy 等外部套件**（系統會偵測已安裝的套件）
- **可以用 `pip_install` 工具裝套件**（Coder 發現缺套件會自己裝）
- **自動偵測 GPU**：MPS（Mac）或 CUDA（NVIDIA）
- **每個 mission 的檔案 I/O 完全隔離**
- **程式碼自動版本追蹤**：每次 write_file 都存快照 + diff + AST 模組地圖
- **智慧 debug context**：bug fix 時只提供出錯函數的 code + diff，不是整個檔案
- **研究記憶 DAG**：重要 insight 越記越牢，瑣碎的自然衰減歸檔
- **code 跑失敗會自己修**：Supervisor 看到 error 會派 Coder 帶著精準 context 重寫
- **結果不好會迭代改進**：不會跑一次就結束

### 限制

- **沒有網頁瀏覽**：不能打開任意 URL，只能用上面列的那些 API
- **沒有 PDF 下載**：能找到論文的 arXiv 連結，但不會去下載 PDF 全文
- **5 分鐘 timeout**：程式碼執行最長 300 秒，超大模型訓練可能不夠
- **LLM 品質取決於 MiniMax**：如果 LLM 回答品質不好，寫出來的程式碼也會不好
- **每個 worker 單次任務最多 10 輪對話**：但 Supervisor 可以多次派同一個 worker
- **最多 30 個 cycle**：Supervisor 最多跑 30 輪（可用 `--max-cycles` 調整）

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
| `CODE_TIMEOUT` | `300` | 程式碼執行超時（秒） |

---

## 專案結構

```
ai_research_agent/
├── main.py                  ← 主入口（mission workspace scoping 在這裡接線）
├── config.py                ← API key、路徑、參數設定、硬體偵測
│
├── core/
│   ├── mission.py           ← Mission 管理（建立/列表/搜尋/恢復）
│   ├── llm.py               ← MiniMax LLM 客戶端
│   ├── tool_registry.py     ← 工具註冊中心
│   ├── event_bus.py         ← 事件系統
│   ├── state.py             ← JSON 狀態持久化
│   ├── code_store.py        ← ★ Git-like 版本追蹤 + AST 模組地圖
│   └── insight_dag.py       ← ★ DAG 知識圖 + relevance 評分
│
├── supervisor/
│   ├── supervisor.py        ← 主管（InsightDAG 記憶蒸餾 + 自適應決策）
│   ├── planner.py           ← 任務拆解（goal → 小任務清單）
│   └── reporter.py          ← 報告生成（英文/中文 + working memory）
│
├── workers/
│   ├── base_worker.py       ← Worker 基底類別（tool executor hook）
│   ├── explorer.py          ← 探索者（搜論文/搜 dataset）
│   ├── coder.py             ← 寫程式的（自動版本追蹤 + 智慧 fix context）
│   └── reviewer.py          ← 審查者（跑 benchmark/分析結果）
│
├── knowledge/
│   ├── tree.py              ← 知識樹（自動分類、跨任務搜尋）
│   ├── index.py             ← 索引管理
│   └── categories.py        ← 分類定義
│
├── mcp_servers/              ← 實際的工具實作
│   ├── paper_search.py      ← arXiv + Semantic Scholar + OpenAlex
│   ├── code_runner.py       ← ★ Python 執行 + workspace scoping factory
│   ├── dataset_fetch.py     ← HuggingFace + LeetCode + HumanEval
│   └── github_search.py     ← GitHub 搜尋
│
├── skills/                   ← 技能系統（可自我進化）
│   ├── registry.py
│   ├── meta_skill.py
│   └── builtin/
│
└── missions/                 ← ★ 所有任務的資料都在這裡
    └── mission_<timestamp>_<slug>/
        ├── mission.json
        ├── state/           ← checkpoint（InsightDAG + working memory）
        ├── knowledge/       ← 論文、程式碼、實驗結果
        ├── workspace/       ← Coder 的隔離工作區 + .code_store/
        └── reports/         ← 最終報告
```

---

## 常見問題

### Q: 跑一次要花多少錢 / 多久？

每個任務大概跑 3-10 個 cycle，每個 cycle 有 1-2 次 LLM 呼叫（蒸餾 + 決策 + worker）。以 MiniMax-M2.5 的定價來說成本很低。時間取決於任務複雜度，通常 2-10 分鐘。

### Q: 可以換成 GPT-4 / Claude 嗎？

理論上可以，但需要改 `core/llm.py` 裡的 `MiniMaxClient` 換成 OpenAI 或 Anthropic 的 API。介面是相容的（都是 OpenAI function calling 格式）。

### Q: 知識庫滿了會怎樣？

知識樹有自動重組機制：當一個分類超過 20 筆時，會用 LLM 自動拆成子分類。你不用管它。

### Q: InsightDAG 會不會無限增長？

不會。InsightDAG 有自動衰減機制：每次蒸餾時，不重要的 insight 的 relevance 會乘以 0.8 衰減，低於 0.1 就歸檔。歸檔的 insight 不參與決策但保留可追溯。通常 active insights 維持在 10-20 個。

### Q: 版本追蹤會不會佔很多空間？

每個版本就是一個 Python 檔案的副本（通常幾 KB），外加一個 diff 檔和 JSON manifest。一個典型的 10-cycle 任務可能產生 5-10 個版本，大概 50-100 KB。完全可忽略。

### Q: 舊的 checkpoint 可以用嗎？

可以。系統有向後相容：舊的 checkpoint 用 flat list 存 insights，系統會自動用 `InsightDAG.from_legacy_list()` 遷移成 DAG 格式。不需要手動轉換。
