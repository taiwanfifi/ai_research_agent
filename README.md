# Self-Evolving AI Research Agent

A multi-layer autonomous research system that can search papers, write code, run experiments, and self-evolve its own skills — all orchestrated by an LLM supervisor.

## Architecture

```
┌─────────────────────────────────────────────┐
│              main.py (CLI)                  │
│  --resume / --zh / --cross / --interactive  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           MissionManager                     │
│  create / list / find (fuzzy) / load / save │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│            Supervisor                        │
│  plan → execute cycles → report             │
│  (cross-knowledge · bilingual reports)      │
├─────────┬────────────┬──────────────────────┤
│ Explorer│   Coder    │     Reviewer         │
│ (papers)│   (code)   │   (benchmarks)       │
└─────────┴────────────┴──────────────────────┘
        │          │            │
┌───────▼──────────▼────────────▼─────────────┐
│          MCP Tool Servers                    │
│  arXiv · Semantic Scholar · OpenAlex         │
│  HuggingFace · GitHub · Code Runner         │
└─────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────┐
│       Knowledge Tree (auto-organizing)       │
│  papers/ · methods/ · code/ · experiments/  │
└─────────────────────────────────────────────┘
```

Each mission gets its own isolated directory:

```
missions/
└── mission_20260303_185200_flash_attention_search/
    ├── mission.json      # manifest (goal, direction, language, status)
    ├── state/            # checkpoints
    ├── knowledge/        # papers, code, experiments, methods
    ├── workspace/        # temporary files
    └── reports/          # progress_en_*.md / progress_zh_*.md
```

## Setup

### 1. Clone

```bash
git clone git@github.com:taiwanfifi/ai_research_agent.git
cd ai_research_agent
```

### 2. API Key

The system uses [MiniMax](https://www.minimax.io/) as the LLM backend. Set your API key:

```bash
# Option A: environment variable
export MINIMAX_API_KEY="sk-your-key-here"

# Option B: file (in parent directory)
echo "sk-your-key-here" > ../apikey.txt
```

### 3. Dependencies

Python 3.11+ required. No third-party packages needed — the system uses only the standard library + HTTP calls to the MiniMax API.

## Usage

### Start a New Research Mission

```bash
# Basic — English report (default)
python3 main.py "research Flash Attention optimization methods"

# Chinese report (繁體中文)
python3 main.py --zh "研究 Flash Attention 優化方法"

# With cross-mission knowledge (reference other missions' findings)
python3 main.py --cross "compare attention mechanisms for local inference"

# Combine flags
python3 main.py --zh --cross "深入研究 KV Cache 壓縮技術"
```

What happens:
1. A new `missions/mission_<timestamp>_<slug>/` directory is created
2. The supervisor decomposes your goal into tasks
3. Workers execute tasks (search papers, write code, run benchmarks)
4. A progress report is generated

### Resume a Mission

Missions can be resumed by **fuzzy matching** on timestamp, slug, or goal text:

```bash
# Resume by slug keyword
python3 main.py --resume flash_attention

# Resume by timestamp prefix
python3 main.py --resume 20260303_19

# Resume the most recent mission (no argument)
python3 main.py --resume

# Resume and change direction
python3 main.py --resume attention --direction "focus only on Flash Attention v2"

# Resume with new direction (positional syntax)
python3 main.py --resume flash "改成只研究 IO-aware 實作"

# Resume + switch to Chinese reports
python3 main.py --resume flash --zh

# Resume + enable cross-knowledge
python3 main.py --resume flash --cross
```

If multiple missions match, you'll be prompted to choose:

```
  Multiple missions match 'attention':
    [1] mission_20260303_185200_flash_attention_search
        Goal: research Flash Attention optimization methods
    [2] mission_20260303_190000_attention_implement
        Goal: implement efficient attention mechanism

  Select [1-2]: _
```

### List & Inspect Missions

```bash
# List all missions
python3 main.py --list-missions

# Show status of the most recent mission
python3 main.py --status

# Generate a report for a specific mission
python3 main.py --resume flash --report
```

### Interactive Mode

```bash
python3 main.py --interactive
# or simply
python3 main.py
```

Inside interactive mode:

```
  > research efficient KV cache compression          # start new mission
  > /resume flash_attention                           # resume by keyword
  > /resume flash "focus on v2 implementation"        # resume + new direction
  > /missions                                         # list all missions
  > /cross                                            # toggle cross-knowledge on/off
  > /zh                                               # toggle English ↔ 繁體中文
  > /status                                           # current mission status
  > /report                                           # generate progress report
  > quit                                              # exit
```

### Legacy Agent (Quick Test)

The original single-file agent still works for quick experiments:

```bash
python3 agent.py "搜尋最新的 LLM agent 論文"
python3 agent.py   # interactive mode
```

## Examples

### Example 1: Literature Survey

```bash
python3 main.py "survey recent advances in efficient attention mechanisms (2024-2026)"
```

The supervisor will:
- Dispatch **explorer** to search arXiv, Semantic Scholar, and OpenAlex
- Organize found papers into `knowledge/papers/`
- Dispatch **reviewer** to compare and summarize findings
- Generate `reports/progress_en_<timestamp>.md`

### Example 2: Code Implementation

```bash
python3 main.py "implement Flash Attention v2 from scratch in Python and benchmark against naive attention"
```

The supervisor will:
- **Explorer** searches for the Flash Attention paper and reference implementations
- **Coder** implements the algorithm
- **Reviewer** runs benchmarks comparing performance
- All artifacts saved in the mission's `knowledge/` and `workspace/`

### Example 3: Multi-Mission Workflow with Cross-Knowledge

```bash
# Mission 1: Survey the field
python3 main.py "survey KV cache compression techniques"

# Mission 2: Build on Mission 1's findings
python3 main.py --cross "implement the most promising KV cache compression method"

# Mission 3: Benchmark everything (Chinese report)
python3 main.py --cross --zh "對所有 KV cache 方法做完整 benchmark"
```

### Example 4: Resume and Pivot

```bash
# Start a broad mission
python3 main.py "research efficient inference techniques"

# Later, narrow the focus
python3 main.py --resume inference --direction "focus only on speculative decoding methods"
```

### Example 5: Interactive Research Session

```bash
python3 main.py -i
```

```
  > survey quantization methods for LLMs
  ...
  (mission completes)

  > /zh
  Report language: zh

  > /cross
  Cross-knowledge: on

  > 基於前面的研究，實作 GPTQ 量化演算法
  ...
  (new mission with cross-knowledge, Chinese reports)

  > /resume quantization "改成專注 AWQ 而非 GPTQ"
  ...
  (resumes first mission with new direction)

  > /missions
  Missions (2)
    mission_20260303_185200_quantization_survey
    mission_20260303_190500_gptq_implementation

  > /report
  (generates report for active mission)

  > quit
```

## Configuration

Environment variables for tuning:

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIMAX_API_KEY` | — | API key (required) |
| `MINIMAX_BASE_URL` | `https://api.minimax.io/v1` | API endpoint |
| `MINIMAX_MODEL` | `MiniMax-M2.5` | Model name |
| `MAX_TURNS` | `10` | Max LLM turns per worker |
| `MAX_TOKENS` | `4096` | Max tokens per response |
| `TEMPERATURE` | `0.3` | Sampling temperature |
| `API_TIMEOUT` | `120` | API timeout (seconds) |

## Project Structure

```
ai_research_agent/
├── main.py                      # CLI entry point (mission-aware)
├── agent.py                     # Legacy single-agent mode
├── config.py                    # API keys, paths, runtime limits
│
├── core/
│   ├── mission.py               # MissionManager + MissionContext
│   ├── llm.py                   # MiniMax LLM client
│   ├── tool_registry.py         # Dynamic tool management
│   ├── event_bus.py             # Pub/sub event system
│   └── state.py                 # JSON-file state persistence
│
├── supervisor/
│   ├── supervisor.py            # LLM-driven orchestrator
│   ├── planner.py               # Goal → task decomposition
│   └── reporter.py              # Bilingual report generation
│
├── workers/
│   ├── base_worker.py           # Abstract worker base class
│   ├── explorer.py              # Paper/dataset search
│   ├── coder.py                 # Code implementation
│   └── reviewer.py              # Benchmarking & evaluation
│
├── knowledge/
│   ├── tree.py                  # Auto-organizing knowledge tree
│   ├── index.py                 # Knowledge index management
│   └── categories.py            # Default categories & thresholds
│
├── skills/
│   ├── registry.py              # Skill lifecycle management
│   ├── meta_skill.py            # Self-evolution engine
│   └── builtin/                 # Built-in skill definitions
│
├── mcp_servers/
│   ├── paper_search.py          # arXiv, Semantic Scholar, OpenAlex
│   ├── code_runner.py           # Python sandbox execution
│   ├── dataset_fetch.py         # HuggingFace datasets
│   ├── github_search.py         # GitHub repos & code search
│   └── generated/               # Auto-generated tool servers
│
└── missions/                    # Mission data (git-ignored runtime)
    └── mission_<timestamp>_<slug>/
        ├── mission.json
        ├── state/
        ├── knowledge/
        ├── workspace/
        └── reports/
```

## License

MIT
