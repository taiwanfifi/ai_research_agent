# Self-Evolving AI Research Agent

A multi-layer autonomous research system that can search papers, write code, run experiments, and self-evolve its own skills — all orchestrated by an LLM supervisor with structured memory distillation.

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
│  Each mission → isolated directory           │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│            Supervisor                        │
│  Memory distillation (InsightDAG)           │
│  → reflect → decide → dispatch → extract    │
│  (cross-knowledge · bilingual reports)      │
├─────────┬────────────┬──────────────────────┤
│ Explorer│   Coder    │     Reviewer         │
│ (papers)│  (code +   │   (benchmarks)       │
│         │ versioning)│                      │
└─────────┴────────────┴──────────────────────┘
        │          │            │
┌───────▼──────────▼────────────▼─────────────┐
│   MCP Tool Servers (mission-scoped)         │
│  arXiv · Semantic Scholar · OpenAlex        │
│  HuggingFace · GitHub · Code Runner         │
│  write_file / read_file / run_python_code   │
│  → scoped to mission workspace at runtime   │
└─────────────────────────────────────────────┘
        │                    │
┌───────▼────────────┐ ┌────▼────────────────┐
│  Knowledge Tree    │ │  CodeVersionStore   │
│  (auto-organizing) │ │  (AST module maps,  │
│  papers/ methods/  │ │   diffs, snapshots)  │
│  code/ experiments/│ │  .code_store/        │
└────────────────────┘ └─────────────────────┘
```

### Memory Distillation (InsightDAG)

Instead of a sliding window over recent outputs, the supervisor maintains a **DAG of research insights** with relevance scoring:

1. After each worker completes, LLM extracts a structured insight (what was learned, key numbers, next steps)
2. Insights are added to the DAG with references to related prior insights
3. Before each decision, LLM distills ALL active insights into working memory — promoting important ones, decaying irrelevant ones, archiving dead ends
4. Old but important insights persist; recent but useless ones decay — like a real researcher's memory

### Code Version Tracking (CodeVersionStore)

Every file the coder writes is automatically version-tracked:

- **Snapshots**: `v001.py`, `v002.py`, ... — full source at each version
- **Diffs**: `v001_v002.diff` — what changed between versions
- **AST Module Maps**: parsed function/class boundaries, signatures, docstrings, call graphs
- **Fix Context**: when debugging, provides only the failing module's code + recent diff instead of the entire file

### Mission-Scoped Workspace

Each mission gets fully isolated file I/O:

```
missions/
└── mission_20260303_185200_flash_attention_search/
    ├── mission.json          # manifest (goal, direction, language, status)
    ├── state/                # checkpoints (full supervisor state)
    ├── knowledge/            # papers, code, experiments, methods
    ├── workspace/            # code files written by coder (scoped)
    │   ├── model.py          # actual code
    │   └── .code_store/      # version tracking data
    │       └── model/
    │           ├── v001.py, v002.py
    │           ├── v001_v002.diff
    │           ├── manifest.json
    │           └── module_map.json
    └── reports/              # progress_en_*.md / progress_zh_*.md
```

`write_file`, `read_file`, and `run_python_code` are all scoped to the mission workspace at runtime via closure-based tool function replacement. This means:
- Code written by the coder lands in the mission directory
- `run_python_code` executes with `cwd` set to the mission workspace (so `import model` works)
- CodeVersionStore tracks the same files that the coder actually writes
- Different missions never interfere with each other's files

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

For ML experiments, optionally install:

```bash
pip install torch numpy matplotlib
```

The system auto-detects available packages and GPU (CUDA / MPS / CPU).

## Usage

### Start a New Research Mission

```bash
# Basic — English report (default)
python3 main.py "research Flash Attention optimization methods"

# Chinese report
python3 main.py --zh "研究 Flash Attention 優化方法"

# With cross-mission knowledge (reference other missions' findings)
python3 main.py --cross "compare attention mechanisms for local inference"

# Combine flags
python3 main.py --zh --cross "深入研究 KV Cache 壓縮技術"
```

What happens:
1. A new `missions/mission_<timestamp>_<slug>/` directory is created
2. Code tools (write_file, read_file, run_python_code) are scoped to the mission workspace
3. The supervisor decomposes your goal into tasks
4. Workers execute tasks (search papers, write code, run benchmarks)
5. Every cycle: extract insight → distill memory → decide next action → checkpoint
6. A progress report is generated

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

# Resume + switch to Chinese reports
python3 main.py --resume flash --zh
```

If multiple missions match, you'll be prompted to choose.

### List & Inspect Missions

```bash
python3 main.py --list-missions
python3 main.py --status
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

## Examples

### Example 1: Literature Survey

```bash
python3 main.py "survey recent advances in efficient attention mechanisms (2024-2026)"
```

### Example 2: Code Implementation + Benchmark

```bash
python3 main.py "implement Flash Attention v2 from scratch and benchmark against naive attention"
```

The supervisor will search papers → write code → run benchmarks → iterate on results → generate report. All code is version-tracked with AST module maps for intelligent debugging.

### Example 3: Multi-Mission with Cross-Knowledge

```bash
python3 main.py "survey KV cache compression techniques"
python3 main.py --cross "implement the most promising KV cache compression method"
python3 main.py --cross --zh "對所有 KV cache 方法做完整 benchmark"
```

### Example 4: Resume and Pivot

```bash
python3 main.py "research efficient inference techniques"
# Later, narrow the focus:
python3 main.py --resume inference --direction "focus only on speculative decoding methods"
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIMAX_API_KEY` | — | API key (required) |
| `MINIMAX_BASE_URL` | `https://api.minimax.io/v1` | API endpoint |
| `MINIMAX_MODEL` | `MiniMax-M2.5` | Model name |
| `MAX_TURNS` | `10` | Max LLM turns per worker |
| `MAX_TOKENS` | `4096` | Max tokens per response |
| `TEMPERATURE` | `0.3` | Sampling temperature |
| `API_TIMEOUT` | `120` | API timeout (seconds) |
| `CODE_TIMEOUT` | `300` | Code execution timeout (seconds) |

## Project Structure

```
ai_research_agent/
├── main.py                      # CLI entry point (mission-aware)
├── config.py                    # API keys, paths, runtime limits, hardware detection
│
├── core/
│   ├── mission.py               # MissionManager + MissionContext
│   ├── llm.py                   # MiniMax LLM client
│   ├── tool_registry.py         # Dynamic tool management
│   ├── event_bus.py             # Pub/sub event system
│   ├── state.py                 # JSON-file state persistence
│   ├── code_store.py            # Git-like version tracking + AST module maps
│   └── insight_dag.py           # DAG knowledge graph + relevance scoring
│
├── supervisor/
│   ├── supervisor.py            # LLM-driven orchestrator (InsightDAG memory)
│   ├── planner.py               # Goal → task decomposition
│   └── reporter.py              # Bilingual report generation
│
├── workers/
│   ├── base_worker.py           # Abstract worker (tool executor hook)
│   ├── explorer.py              # Paper/dataset search
│   ├── coder.py                 # Code implementation (version-tracked)
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
│   ├── code_runner.py           # Python sandbox + workspace scoping
│   ├── dataset_fetch.py         # HuggingFace datasets
│   ├── github_search.py         # GitHub repos & code search
│   └── generated/               # Auto-generated tool servers
│
└── missions/                    # Mission data (git-ignored runtime)
    └── mission_<timestamp>_<slug>/
        ├── mission.json
        ├── state/               # full checkpoint (InsightDAG + working memory)
        ├── knowledge/
        ├── workspace/           # scoped code I/O + .code_store/
        └── reports/
```

## License

MIT
