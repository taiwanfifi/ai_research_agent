# AI 自動研究 Agent 架構設計

> LLM: MiniMax-M2.5 (OpenAI 相容 API)
> 目標: 一個能自動搜論文、寫 code、跑實驗、優化方法的研究 Agent

---

## 一、先釐清概念：Aider vs OpenHands vs MCP vs Skills

### Aider 是什麼？
- **AI 結對程式設計工具**（terminal 裡的 AI coding partner）
- 你給它任務 → 它用 LLM 生成/修改程式碼 → 自動 git commit
- 有 lint 自動修復 + test 自動跑的迴圈
- **本質上是一個 coding agent**，類似 Claude Code 的角色
- 支援 100+ LLM（透過 litellm），**可以接 MiniMax**
- 可作為 Python library 嵌入：`from aider.coders import Coder`
- **沒有 MCP 支援**，沒有 plugin 系統

### OpenHands 是什麼？
- **開源 AI 軟體工程師**（比 aider 更完整的 agent 平台）
- SWE-bench 77.6% 通過率
- 有 Docker 沙箱、瀏覽器自動化、Jupyter、檔案編輯
- **已內建 MCP 客戶端**，可以連接外部 MCP server
- 有 REST API + WebSocket + Web UI + CLI
- 支援 100+ LLM（透過 litellm），**可以接 MiniMax**
- 有 plugin 系統（Jupyter, VSCode, Agent Skills）
- 有 microagents（領域知識注入）

### MCP 是什麼？(Model Context Protocol)
- **工具擴充協議** — 讓 Agent 能呼叫外部工具
- 像 USB 插頭：任何 Agent 只要支援 MCP 客戶端，就能用任何 MCP Server
- 每個 MCP Server 提供特定功能（搜論文、查資料庫、操作 GitHub 等）
- **MCP = 讓 Agent 獲得新能力的標準介面**

### Skills 是什麼？
- **內建於 Agent 的專業知識/流程**
- 通常是一段 prompt + 工具組合，告訴 Agent「遇到這種任務要怎麼做」
- 例如：「做文獻回顧」的 skill = 搜 arXiv → 讀摘要 → 整理引用 → 寫 related work
- **Skills 用 MCP tools，但 Skills ≠ MCP**

### 三者的關係
```
┌─────────────────────────────────────────────────┐
│              Agent (大腦/指揮官)                  │
│         OpenHands 或 Aider 或自建                │
│                                                  │
│   Skills (技能/流程)                              │
│   ├─ "搜論文" skill                              │
│   ├─ "寫 code" skill                             │
│   ├─ "跑實驗" skill                              │
│   └─ "優化方法" skill                            │
│                                                  │
│   MCP Tools (工具箱)                             │
│   ├─ arXiv MCP Server      → 搜預印本           │
│   ├─ Semantic Scholar MCP   → 引用分析           │
│   ├─ Dataset MCP Server     → 下載題目/資料集     │
│   ├─ Code Sandbox MCP       → 執行程式碼         │
│   └─ Benchmark MCP Server   → 跑評測            │
└─────────────────────────────────────────────────┘
```

---

## 二、Aider vs OpenHands 比較

| 面向 | Aider | OpenHands |
|------|-------|-----------|
| **定位** | 結對程式設計工具 | 完整 AI 軟體工程師 |
| **本質** | 「幫你寫 code」 | 「替你完成整個任務」 |
| **MCP 支援** | 無 | **有（內建 MCP 客戶端）** |
| **沙箱** | 無（直接在你的機器跑） | **Docker / K8s 隔離沙箱** |
| **瀏覽器** | 無 | **有（Playwright）** |
| **Web UI** | 簡易 Streamlit | **完整 React UI** |
| **Python API** | `Coder.run("task")` | SDK + REST API |
| **Git 整合** | 自動 commit | 有 |
| **測試迴圈** | lint → fix → test → fix | action → observe → repeat |
| **Benchmark** | 自有 leaderboard | **SWE-bench 77.6%** |
| **擴充性** | 低（無 plugin） | **高（MCP + plugin + microagent）** |
| **適合當「大腦」** | 適合純 coding | **更適合（已有完整 agent loop）** |
| **MiniMax 支援** | 透過 litellm | 透過 litellm |

**結論：OpenHands 更適合當「大腦」Agent，因為它已經有 MCP 客戶端、沙箱、瀏覽器。**
**Aider 適合當「coding 專用工具」，被 OpenHands 或其他 Agent 呼叫。**

---

## 三、你缺的東西：Datasets / 題目來源

你說得對，還少了一個「下載 datasets 或題目」的來源。以下是可用的免費 API：

### 程式題目類
| 來源 | API | 說明 |
|------|-----|------|
| **LeetCode** | 有 GraphQL API | 題目、解答、難度分類 |
| **Codeforces** | `codeforces.com/api/` | 競程題目 + 測試案例 |
| **HumanEval** | GitHub dataset | OpenAI 的 code generation benchmark (164 題) |
| **MBPP** | GitHub dataset | Google 的 Python 程式測試集 (974 題) |
| **SWE-bench** | GitHub dataset | 真實 GitHub issue 修復任務 |
| **APPS** | GitHub dataset | 10,000 程式題 (多難度) |

### 研究資料集類
| 來源 | API | 說明 |
|------|-----|------|
| **Hugging Face Datasets** | `huggingface.co/api/datasets` | 數萬個 ML 資料集 |
| **Kaggle** | `kaggle.com/api/v1/` | 競賽資料集 (需免費 API Key) |
| **UCI ML Repository** | 有 API | 經典 ML 資料集 |
| **Papers with Code** | GitHub 開放資料 | 論文對應的 benchmark + dataset |

### QA / NLP 類
| 來源 | API | 說明 |
|------|-----|------|
| **SQuAD** | GitHub dataset | 閱讀理解 QA |
| **MMLU** | Hugging Face | 多領域知識測試 |
| **GSM8K** | Hugging Face | 數學推理 |
| **ARC** | Hugging Face | 科學推理 |

---

## 四、建議架構：MiniMax 驅動的 AI 研究 Agent

### 方案：輕量自建 Agent + MCP Servers

不用完整部署 OpenHands（太重），而是：
1. **自建一個輕量 Agent Loop**（Python，用 MiniMax API）
2. **把各功能做成 MCP Server**（標準化、可替換）
3. **定義 Skills**（研究流程的 prompt template）

```
┌────────────────────────────────────────────────────────────┐
│                 Research Agent (Python)                     │
│                 LLM: MiniMax-M2.5                          │
│                                                            │
│  Agent Loop:                                               │
│  1. 接收研究任務                                            │
│  2. 判斷需要什麼工具 (function calling / prompt)            │
│  3. 呼叫 MCP tools                                         │
│  4. 觀察結果                                               │
│  5. 決定下一步 → 回到 2                                     │
│                                                            │
│  Skills (研究流程模板):                                     │
│  ├─ literature_review  → 搜論文 + 分析引用 + 寫綜述        │
│  ├─ code_experiment    → 寫 code + 跑測試 + 記錄結果       │
│  ├─ method_discovery   → 搜最新方法 + 比較 SOTA + 找 gap   │
│  └─ benchmark_eval     → 下載題目 + 跑 code + 計算分數     │
└──────────┬──────────┬──────────┬──────────┬────────────────┘
           │          │          │          │
    ┌──────▼───┐ ┌────▼────┐ ┌──▼──────┐ ┌▼──────────┐
    │ 論文搜尋  │ │ Code    │ │ Dataset │ │ Benchmark │
    │ MCP      │ │ MCP     │ │ MCP     │ │ MCP       │
    │          │ │         │ │         │ │           │
    │ • arXiv  │ │ • 執行  │ │ • HF    │ │ • 評測    │
    │ • S2     │ │ • 測試  │ │ • LCode │ │ • 比較    │
    │ • OAlex  │ │ • Git   │ │ • Kaggle│ │ • 報告    │
    │ • CORE   │ │ • Lint  │ │ • SWE-b │ │           │
    └──────────┘ └─────────┘ └─────────┘ └───────────┘
```

### 為什麼不直接用 OpenHands？

| 考量 | 自建輕量 Agent | 直接用 OpenHands |
|------|---------------|-----------------|
| 複雜度 | 低（幾百行 Python） | 高（需部署 Docker + 多服務） |
| MiniMax 整合 | 直接呼叫 API | 需配置 litellm provider |
| 可控性 | 完全自己控制 | 框架限制 |
| 學習成本 | 低 | 中高 |
| 功能完整度 | 需自建 | 開箱即用 |
| 適合階段 | **先驗證概念** | 後期正式部署 |

**建議：先用自建輕量 Agent 驗證概念，確認 MiniMax 能跑通整個流程後，再考慮遷移到 OpenHands。**

---

## 五、實作計畫

### Phase 1: 基礎設施（本資料夾）
```
ai_research_agent/
├── ARCHITECTURE.md          ← 本文件
├── main.py                  ← 主入口 (CLI + mission 管理)
├── mcp_servers/
│   ├── paper_search.py      ← 論文搜尋 MCP (arXiv + S2 + OpenAlex)
│   ├── code_runner.py       ← 程式執行 MCP (subprocess sandbox)
│   ├── dataset_fetch.py     ← 資料集下載 MCP (HF + LeetCode)
│   └── benchmark.py         ← 評測 MCP
├── skills/
│   ├── literature_review.md ← 文獻回顧流程
│   ├── code_experiment.md   ← 程式實驗流程
│   └── method_discovery.md  ← 方法探索流程
├── workspace/               ← Agent 的工作目錄
└── results/                 ← 實驗結果
```

### Phase 2: 驗證
- 給 Agent 一個簡單研究任務
- 看它能不能：搜論文 → 寫 code → 跑實驗 → 優化

### Phase 3: 擴展
- 接入 OpenHands 作為更強的 coding backend
- 加入更多 dataset 來源
- 建立自動化 benchmark pipeline
