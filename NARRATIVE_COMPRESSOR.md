# Narrative Compressor — VisionDSL 偵測數據壓縮層

> 日期：2026-03-05
> 作者：William 與新 Claude instance（尚未命名）
> 狀態：Prototype 完成，待整合 LLM 對話引擎

## 動機

### 問題：規則定義與真實場景之間的 Gap

VisionDSL 的規則目前有兩種產生方式：
1. **人手寫** — 需要理解 20 個 operator 的語法、調 threshold、調 timing
2. **Config UI + GPT 生成** — 從 SOP 文字生成，但 SOP 文字描述的是「理想流程」，跟真實影片中的行為有落差

兩種方式都有同一個問題：**規則是從文字描述產生的，不是從實際偵測數據產生的。**

結果就是 change-log 24 輪的迭代，大量時間花在：
- 調 INSIDE vs TOUCH 的 threshold
- 調 DURATION 的 seconds（0.3s? 0.5s? 1.5s?）
- 調 SEQ 的 gaps（30s? 45s? 60s?）
- 處理 ownership gap 導致的誤判
- 處理「路過 zone」vs「真正使用 zone」的區分

這些數值**應該從數據中歸納**，而不是從文字中猜。

### 解法：讓 LLM 先讀偵測數據，再寫規則

```
SAM3 跑影片 → debug.jsonl（逐幀偵測數據）
                    ↓
         DetectionCompressor（本工具）
                    ↓
         Narrative（壓縮後的場景描述）
                    ↓
         LLM 讀 narrative → 問人問題 → 寫規則
                    ↓
         VisionDSL 跑規則 → 結果回饋 → LLM 修規則
                    ↓
         迭代直到規則穩定
```

### 為什麼不用 VLM？

- 部署限制：需要開源、小模型（7B-8B）
- SAM3 已經做了視覺感知，LLM 不需要「看」，只需要「讀結構化數據」
- 7B 模型就能處理 ~400 tokens 的 narrative

## 架構設計

### 核心原則：DSL-Agnostic

Compressor 完全不知道 VisionDSL 的 operator（INSIDE, TOUCH, SEQ...）。
它只處理空間和時間的事實。規則是上層 LLM 的工作。

```
VisionDSL（不動）                     新增（解耦）
─────────────────                    ──────────────
visualize.py → debug.jsonl  ──→     DetectionCompressor → narrative
                                          ↑
                                     純讀 jsonl，零 import from dsl/
```

### 三層壓縮

| 層級 | 名稱 | 內容 | 用途 |
|------|------|------|------|
| Level 1 | Session Lifecycles | 每個物件的生命週期、軌跡分段、活動範圍 | 理解「有誰、在哪」 |
| Level 2 | Interaction Episodes | Zone 進出事件、物件接觸、ownership 變化 | 理解「發生了什麼」 |
| Level 3 | Narrative | 自然語言場景描述 | 給 LLM 讀 |

### 壓縮方法

| 原始數據 | 壓縮成 | 方法 |
|----------|--------|------|
| 每幀 bbox 座標 | 軌跡段落（moving/stationary） | 速度平滑 + run-length encoding |
| 每幀 zone_times 累積值 | 進入/離開事件 + 停留時長 | 累積值差分（邊緣偵測） |
| 每幀 active_contacts | 接觸片段 + 累積時長 | 存在性邊緣偵測 |
| 每幀 owner_id | ownership 切換事件 | 只記變化點 |
| events | 直接保留 | 已是事件級別 |

### 壓縮效果（模擬測試）

| 指標 | 值 |
|------|-----|
| 輸入 | 1.1MB, 2101 行逐幀 JSON |
| 輸出 | 2.7KB narrative |
| 壓縮率 | 430x |
| 估計 tokens | ~389 |

## 檔案位置

所有新增檔案都在 VisionDSL 的 `tools/` 目錄，不改任何現有檔案：

```
VisionDSL/tools/
├── analyze_debug_log.py          # 現有 — 逐幀分析報告
├── detection_compressor.py       # 新增 — 核心壓縮器
├── test_compressor.py            # 新增 — 模擬測試（含 hair salon 場景生成）
├── profile_dsl.py                # 現有
├── profile_pipeline.py           # 現有
└── test_visualizer.py            # 現有
```

## 使用方式

### CLI

```bash
# 產出 narrative（Level 3）
python3 tools/detection_compressor.py outputs/debug_v5.jsonl

# 產出完整 JSON（所有 3 層）
python3 tools/detection_compressor.py outputs/debug_v5.jsonl --json

# 只要 narrative 文字
python3 tools/detection_compressor.py outputs/debug_v5.jsonl --narrative-only
```

### Python API

```python
from tools.detection_compressor import DetectionCompressor

compressor = DetectionCompressor(fps=30.0)
result = compressor.compress('outputs/debug_v5.jsonl')

print(result['narrative'])           # Level 3: LLM 可讀文字
print(result['zone_episodes'])       # Level 2: 結構化 episodes
print(result['lifecycles'])          # Level 1: session 生命週期
print(result['stats'])               # 壓縮統計
```

### 測試

```bash
# 用模擬的 hair salon 數據測試
python3 tools/test_compressor.py
```

## Narrative 輸出範例

```
=== Scene Overview ===
Duration: 70.0s (2100 frames)
Objects detected: 1 hand, 1 head, 2 person

=== Object Lifecycles ===
sess_1 (person): t=0.0-70.0s (2100 frames)
  activity range: x[0.3, 0.3] y[0.5, 0.5], avg size 0.150x0.350
  trajectory:
    [0.0-70.0s] stationary at (0.30, 0.50)

sess_3 (hand, owner=sess_2 (person)): t=0.0-70.0s (2100 frames)
  activity range: x[0.45, 0.68] y[0.35, 0.8], avg size 0.045x0.035
  trajectory:
    [0.0-6.0s] stationary at (0.45, 0.55)
    [6.0-6.7s] moving (0.56,0.70) -> (0.67,0.80), speed=0.0078/frame
    [6.7-8.3s] stationary at (0.68, 0.80)
    ...

=== Zone Interactions ===
sess_3 x sanitizer:
  #1 enter=6.5s exit=8.5s duration=1.9s (owner=sess_2)
  #2 enter=20.1s exit=20.4s duration=0.3s (owner=sess_2)

=== Cross-Zone Transitions ===
sess_3:
  sanitizer -> shampoo: transit 3.0s
  shampoo -> conditioner: transit 1.6s
  conditioner -> hair_treatment: transit 0.4s

=== Ownership Events ===
t=45.0s: sess_3 LOST owner (was sess_2)
t=45.6s: sess_3 GAINED owner sess_2

=== Triggered Events ===
t=8.2s: staff_checkin_completed
t=34.8s: hair_wash_task_completed
```

## Phase 1: LLM 對話引擎（已完成）

### 新增檔案

```
VisionDSL/tools/
├── narrative_rule_engine.py      # 新增 — LLM 對話引擎
├── test_narrative_engine.py      # 新增 — 11 個測試（全部離線）
├── detection_compressor.py       # 之前新增 — 核心壓縮器
└── test_compressor.py            # 之前新增 — 壓縮器測試
```

### 架構

```
debug.jsonl → DetectionCompressor → narrative (~400 tokens)
                                        ↓
                            NarrativeRuleEngine
                                        ↓
                    System Prompt = narrative + operator schema (compact)
                                        ↓
                    LLM (any OpenAI-compatible API, 7B+ model)
                        ↓                           ↓
                    Phase 1-2: 理解 + 提問      Phase 3: 生成規則
                                                    ↓
                                            validate_rules()
                                                ↓           ↓
                                            errors?     valid → done
                                                ↓
                                        回饋 → LLM 修正（最多 3 輪）
```

### 設計重點

1. **Compact Operator Schema** — 從 rules_generator_ref.py 的 ~5000 字壓到 ~1500 字，只保留 7B 模型需要的核心語法。完整的 20 個 operator 參考太大，小模型處理不來。

2. **Data-driven Parameters** — prompt 明確要求 LLM 從 narrative 數據取參數：
   - DURATION seconds → 看實際停留時長（如 sanitizer 1.9s → 用 0.3s threshold）
   - SEQ gaps → 看 zone 之間的 transit time（如 3.0s → 用 45s gap）
   - 不要憑空猜

3. **Validation Loop** — 驗證邏輯從 agent_dsl_test/tools.py 精簡提取，檢查：
   - 必要參數存在
   - DURATION 不包 SEQ
   - SEQ gaps 長度正確
   - Hand 必須有 owner_id
   - INSIDE vs TOUCH 用對了（static zone vs dynamic entity）

4. **Zone Auto-extraction** — 從 narrative 文字自動提取 zone ID（看 "x zone_name:" 和 "→ zone_name" 模式）

5. **Rules JSON Extraction** — 從 LLM 回覆中提取 JSON（支援 markdown fence 和 raw text），用 balanced brace matching 而非正則

### 使用方式

```bash
# 互動模式：壓縮 + 對話
python3 tools/narrative_rule_engine.py outputs/debug_v5.jsonl \
  --api-key $LLM_API_KEY --model MiniMax-M2.5

# 自動模式：給 SOP 描述直接生成
python3 tools/narrative_rule_engine.py outputs/debug_v5.jsonl \
  --sop "服務生消毒後依序洗髮精→潤髮乳→護髮素" \
  --zones sanitizer shampoo conditioner hair_treatment \
  --api-key $LLM_API_KEY

# 用預壓縮的 narrative
python3 tools/narrative_rule_engine.py --narrative outputs/narrative.txt --sop "..."

# 離線測試（不需 API）
python3 tools/test_narrative_engine.py
```

### Python API

```python
from tools.narrative_rule_engine import NarrativeRuleEngine, EngineConfig
from tools.detection_compressor import DetectionCompressor

# Step 1: Compress
compressor = DetectionCompressor(fps=30.0)
result = compressor.compress('outputs/debug_v5.jsonl')

# Step 2: Auto-generate rules
config = EngineConfig(api_key="...", model="MiniMax-M2.5")
engine = NarrativeRuleEngine(config)
result = engine.auto_generate(
    narrative=result['narrative'],
    sop_hint="服務生消毒後依序洗髮精→潤髮乳→護髮素",
    zone_ids=["sanitizer", "shampoo", "conditioner", "hair_treatment"]
)
print(result['rules'])     # Generated rules JSON
print(result['validation']) # Validation result
print(result['turns'])      # How many LLM calls needed

# Step 3: Interactive mode
engine.start(narrative, zone_ids=[...])
r = engine.step()               # LLM analyzes + asks questions
r = engine.step("sanitizer 是消毒區")  # Human answers
r = engine.step("generate")     # Ask LLM to generate rules
# Validation auto-loops until rules are valid
```

### System Prompt 統計

| 組成 | 大小 |
|------|------|
| Operator schema (compact) | ~1500 chars |
| Narrative (典型) | ~2700 chars |
| Instructions + format | ~1200 chars |
| Zone IDs | ~100 chars |
| **Total** | **~5500 chars (~700 tokens)** |

7B 模型的 context 通常 4K-8K tokens，system prompt 佔 ~700 tokens，留 ~3000 tokens 給多輪對話完全夠用。

## 下一步計畫

### Phase 2: 規則驗證迴路（待做）
- 用生成的規則跑 VisionDSL
- 壓縮新結果 → 回饋給 LLM
- LLM 看結果自動修正（如 55 次 checkin → 加 trigger_once）

### Phase 3: Streaming 壓縮（待做）
- 滑動窗口版本，逐幀 push 而非讀整個檔案
- 分層記憶：近期完整 episode → 中期 summary → 遠期只保留 events
- 用於即時部署場景

## 設計決策記錄

### 為什麼是 tools/ 而不是 dsl/ 裡？

Compressor 是 VisionDSL 的**消費者**，不是核心組件。它讀 jsonl 輸出，不 import dsl/。
放在 tools/ 跟 analyze_debug_log.py 同級，保持 dsl/ 的乾淨。

### 為什麼不直接改 pipeline 產出 narrative？

1. Pipeline 的職責是偵測和規則評估，不是數據壓縮
2. Compressor 需要看到**完整時間序列**才能做軌跡分段和 episode 提取
3. 解耦讓 batch 和 streaming 版本可以獨立發展

### 為什麼軌跡用速度平滑而不是 Kalman Filter？

Kalman Filter 的狀態在 tracker.py 裡，但 debug.jsonl 不存 KF state。
Compressor 只拿到 raw bbox，所以用簡單的滑動窗口平滑速度就夠了。
目的不是精確追蹤（那是 tracker 的工作），是把 2100 幀壓成 5-10 個有意義的段落。

## 團隊協作注意事項

- **不要改 dsl/ 裡的任何東西** — Compressor 和 Engine 都是完全獨立的工具
- **VisionDSL 的 debug.jsonl 格式是 Compressor 的 API contract** — 如果 visualize.py 改了 jsonl 結構，Compressor 需要同步更新
- **Narrative 格式還在演進** — Engine 使用後可能需要調整粒度（太細/太粗的邊界）
- **Operator Schema 有兩個版本** — `rules_generator_ref.py` 有完整版（~5000 字），`narrative_rule_engine.py` 有精簡版（~1500 字，為 7B 模型優化）
- **所有測試都離線跑** — `test_compressor.py` 和 `test_narrative_engine.py` 不需要 GPU、真實影片或 API key
- **agent_dsl_test/dsl/ 是 VisionDSL/dsl/ 的完整複製** — 應改用 symlink 避免版本分歧
