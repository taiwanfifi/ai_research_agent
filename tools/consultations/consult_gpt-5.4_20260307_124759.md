# Peer Consultation: gpt-5.4
Date: 2026-03-07 12:47
Total tokens: 18375 (~$0.55)

---
## Turn 1

### Question
I'm building an autonomous AI research agent called "Opus". It decomposes research goals into tasks, dispatches them to specialized workers (explorer, coder, reviewer), and produces research reports with real experiments.

## Current Architecture
- **Supervisor**: LLM-driven adaptive loop. Decomposes goal → task queue → dispatches workers → checkpoints. Max 10-12 cycles.
- **Workers**: Each is an LLM agent with tools. Explorer searches papers. Coder writes+runs Python. Reviewer benchmarks+visualizes.
- **LLM**: MiniMax-M2.5-highspeed (204K context). Each worker gets ~10 tool-calling turns.
- **Validation**: LLM Judge evaluates worker output quality. ResultVerifier cross-checks claims vs stdout.
- **Learning**: EvolutionStore persists learnings across missions (JSON, word-overlap dedup).
- **Scoring**: Rule-based MissionScorer (6 dimensions: literature/code/results/verification/artifacts/report).

## Concrete Problems (with data)

### 1. Prompt Bloat — Rules accumulate, LLM ignores them
The coder prompt is 160+ lines. Every environment bug (eval_strategy deprecated, bitsandbytes crashes on Mac, subprocess bypasses timeout) becomes a new rule. The LLM increasingly ignores rules deep in the prompt. We need a meta-prompt or layered architecture.

### 2. No Error Analysis — Supervisor just retries or skips
When a task fails, supervisor either retries the same task or skips it. It doesn't analyze stderr, diagnose root cause, or adapt strategy. From 36 missions (263 tasks, 69 failures):
- 67% of failures had diagnostic info in stderr that an LLM could have used
- 33% were environment bugs, 33% task-too-big, 22% wrong approach, 11% dependency failures

### 3. No Pre-Execution Judgment
Workers execute tasks blindly. They don't check if dependencies are met, if referenced files exist, or if the task is feasible within timeout. A dry-run A/B test showed:
- Strategy B (dependency+file check): catches 4.3% of failures
- Strategy C (B + resource estimation + friction buffer): catches 53.6% but 31.6% precision (too many false positives)

### 4. No Post-Observation Checkpoint
After each tool result, the worker just continues to the next step. It doesn't pause to ask "does this match my expectation?" — unlike how a human developer would check output before proceeding.

### 5. Evolution Store is Surface-Level
Stores string patterns like "use eval_strategy not evaluation_strategy". No causal understanding. Dedup is word-overlap based. The LLM may or may not read these learnings.

## What I've Studied
I analyzed a related system (kael_daemon) that has:
- **Two-layer cognition**: THINK → QUESTION → ACT (structured question before execution)
- **Friction buffer**: Accumulates "[FRICTION: what failed | what would be better]" — injected into next cycle
- **Anti-coasting detection**: Monitors if agent is just doing easy work, injects provocative questions
- **Capability decay**: Unused skills decay, pushing exploration of neglected areas
- **Goal verification with tool evidence**: Can't declare done without proof

## My Questions
I want to make Opus genuinely better, not just add more patches. The core issue: it executes blindly without judgment.

Given this architecture, what are the 3 highest-leverage changes I should make? Not incremental patches — structural changes that would address the root causes. Be specific about mechanisms.

### Answer
Here are the 3 highest-leverage changes. They all attack the same root flaw: **Opus has action, but not control theory**. It needs explicit prediction, diagnosis, and evidence gates.

---

## 1) Replace giant prompts with a **policy engine + stateful checklists**
Your prompt bloat is a design failure, not a prompting problem.

### Change
Move operational rules out of the worker prompt into a **retrieval-backed policy layer** with typed triggers.

### Mechanism
Before every worker action, run:

- **Context classifier** → labels current step:
  - task type: search / code / benchmark / report
  - environment: mac/linux, transformers, subprocess, plotting, etc.
  - risk flags: timeout risk, dependency risk, file risk, API risk

- **Policy retrieval** → fetch only relevant rules from a structured store:
  ```json
  {
    "trigger": {"tool":"python", "lib":"transformers", "symptom":"deprecated arg"},
    "checklist": [
      "Inspect installed package version",
      "Check current API docs or introspect signature",
      "Prefer eval_strategy over evaluation_strategy"
    ],
    "hard_constraints": [
      "Do not retry same command unchanged"
    ]
  }
  ```

- Inject into worker as a **short step-local checklist**, not a monolithic prompt.

### Why this works
LLMs ignore long static prompts because most tokens are irrelevant. You need **situational control**, not more instructions.

### Non-negotiable
Stop appending “lessons learned” into prompts. If a rule can’t be triggered by context, it shouldn’t be loaded.

---

## 2) Add a mandatory **Plan → Predict → Execute → Observe → Diagnose** loop for every tool call
Right now workers are blind. Fix that at the action granularity.

### Change
Every tool invocation must pass through a structured cognition loop. No direct tool calls.

### Mechanism
For each step, require the worker to emit:

1. **Plan**
   - what action it will take
2. **Prediction**
   - expected output / files / runtime / success criteria
3. **Preflight**
   - file existence
   - dependency/import check
   - estimated runtime class: <30s / 30–300s / >300s
   - feasibility under current budget
4. **Execute**
5. **Observe**
   - compare actual vs predicted
6. **Diagnose**
   - classify mismatch:
     - env bug
     - missing dependency
     - wrong approach
     - task too large
     - bad assumption
7. **Adapt**
   - retry same / retry modified / decompose / escalate / abort

### Enforcement
- Tool wrapper rejects execution if Plan/Predict/Preflight fields are missing.
- Supervisor receives the diagnosis object, not just “failed”.

### Why this works
This directly fixes:
- no pre-execution judgment
- no post-observation checkpoint
- no error analysis

And your own data says stderr already contains useful diagnosis in 67% of failures.

---

## 3) Replace “retry/skip” supervision with an **explicit failure-oriented controller**
Your supervisor is currently a task dispatcher. It needs to become a **recovery manager**.

### Change
Introduce a separate **Critic/Recovery module** that owns failure handling and task resizing.

### Mechanism
On any failed or low-confidence step, the controller must produce:

- **Failure class**
- **Root-cause hypothesis**
- **Recovery action**
  - patch command
  - install dependency
  - rewrite approach
  - split task into smaller subtasks
  - switch worker
  - terminate branch

Use a simple transition table:
- env bug → patch environment or use workaround
- task-too-big → decompose automatically
- wrong approach → generate alternative method
- dependency failure → verify/install/mock/replace

Also add **anti-coasting**:
- if 2 cycles produce no new artifact, no new evidence, or no reduced uncertainty, force a recovery review instead of continuing.

### Why this works
Your current loop rewards motion, not progress. This change makes failure informative and prevents useless retries.

---

## Hard truth
Do **not** invest next in a smarter EvolutionStore. Your memory is downstream of bad cognition. If the agent doesn’t explicitly predict, diagnose, and adapt, better memory just stores more shallow junk.

First fix the control loop. Then memory becomes useful.

---
## Turn 2

### Question
You mentioned (from your previous response). Let me zoom in on the prompt architecture problem.

Currently the coder has a 160-line system prompt that grows with each bug fix. I'm considering:
A) Layered prompts: base (tools/workflow) + mission-context (goal, workspace state) + friction (recent failures + evolution learnings), each managed separately
B) A meta-prompt that selects which rules are relevant to the current task
C) Moving rules out of prompts entirely — into tool-gated checks (the system blocks bad patterns before the LLM sees them)

Which approach, or what combination? How would you implement the meta-prompt specifically?

### Answer
Use **C + A**, not B as the primary mechanism.

**Opinionated answer:**  
If a rule can be enforced mechanically, **do not ask the LLM to remember it**. Prompts are for judgment; gates are for invariants. Meta-prompting is useful only for selecting **advisory heuristics**, not safety-critical or environment-critical rules.

## What to do

### 1) Put hard rules into tool-gated checks
Anything deterministic goes here:

- file must exist before read
- command must respect timeout wrapper
- no raw subprocess if your sandbox wrapper exists
- no writing outside workspace
- dependency/import precheck
- block known-bad API patterns when detectable
- require output artifact paths for benchmark/report tasks

These should be enforced in the tool layer, with structured errors like:
```json
{
  "error_type": "preflight_failed",
  "reason": "missing_file",
  "details": {"path": "data/train.csv"},
  "suggested_actions": ["list workspace files", "search for alternate dataset path"]
}
```

This shrinks prompt size and improves reliability immediately.

---

### 2) Use layered prompts for non-enforceable context
Your prompt stack should be:

#### Base prompt: stable, short
Only include:
- role
- available tools
- required reasoning schema
- success criteria
- “never claim success without tool evidence”

Keep this under ~40 lines.

#### Mission layer: dynamic state
- goal
- current task
- known workspace files
- budget/time remaining
- prior artifacts
- current hypotheses

#### Friction layer: tiny, recent, typed
Not prose. Use max 3–5 items:
```json
[
  {"type":"env", "trigger":"transformers args", "lesson":"inspect installed signature before coding"},
  {"type":"runtime", "trigger":"long training", "lesson":"run smoke test on 1 batch first"}
]
```

No long “lessons learned” dump.

---

### 3) Use a meta-prompt only as a **rule selector**
Do not let it generate arbitrary policy. Let it choose from a rule library.

## How to implement the meta-prompt

Create a structured rule catalog:
```json
[
  {
    "id": "python.import_precheck",
    "when": ["python_execution", "new_library"],
    "instruction": "Before writing full code, run a 1-line import check for new libraries."
  },
  {
    "id": "ml.api_signature_check",
    "when": ["transformers", "trainer_args"],
    "instruction": "Inspect installed API signature before using Trainer arguments."
  },
  {
    "id": "long_job_smoketest",
    "when": ["training", "benchmarking"],
    "instruction": "Run a minimal smoke test before full execution."
  }
]
```

Then run a **selector call** before the worker acts.

### Selector input
- task description
- planned action
- tool to be used
- detected libs/files
- recent failures
- environment facts

### Selector prompt
Use something like:

> You are a policy selector.  
> Choose at most 5 rules from the catalog that are directly relevant to the next action.  
> Prefer rules that reduce execution risk, wasted runtime, or repeated known failures.  
> Do not invent rules. Return JSON only:
> - selected_rule_ids
> - why_each_applies
> - omitted_high-risk_checks

### Output
```json
{
  "selected_rule_ids": [
    "python.import_precheck",
    "ml.api_signature_check",
    "long_job_smoketest"
  ],
  "why_each_applies": {
    "python.import_precheck": "Task uses sentence_transformers not yet verified",
    "ml.api_signature_check": "Plan includes TrainingArguments fields",
    "long_job_smoketest": "Task proposes model training"
  },
  "omitted_high_risk_checks": []
}
```

Then inject only the selected instructions into the worker prompt.

---

## Final architecture
- **C for hard constraints**
- **A for context separation**
- **B only as a narrow retrieval/selection step**

If you use B to compensate for missing gates, you’ll rebuild the same brittle system with extra latency.

---
## Turn 3

### Question
The daemon system I studied has a structured QUESTION phase between thinking and acting:
- "Am I choosing this because it's easy or because it matters?"
- "What's ONE harder thing I could do instead?"

I tried "inner monologue" in Opus (letting workers reflect before executing). A/B test showed it HURTS — Grade D with monologue vs Grade B without. It wasted tool-calling turns on reflection instead of execution.

But the daemon's approach works because its questions are STRUCTURED and SHORT, not open-ended reflection.

How do I add judgment to Opus workers without wasting their limited turns (10 tool calls max)? The key constraint: every turn spent on reflection is a turn not spent on actual work.

### Answer
Don’t add “reflection turns.” Add **judgment fields inside the same turn that already leads to action**.

Your A/B failed because you paid for free-form introspection. That’s dead weight. The daemon works because its QUESTION phase is a **decision gate**, not a diary.

## What to do

### 1) Collapse judgment into a 15-second structured pre-action check
Before every tool call, require the worker to emit a tiny schema:

```json
{
  "action": "run_python",
  "purpose": "benchmark baseline model",
  "expected_evidence": "metrics.json and stdout with accuracy",
  "risk": "dependency_missing|timeout|wrong_file|none",
  "why_this_not_easier": "produces benchmark evidence, not just more literature notes",
  "harder_better_option": "full ablation study",
  "why_not_now": "insufficient validated baseline yet"
}
```

Then immediately execute the tool.

This is **not another turn**. It’s the assistant message that triggers the tool call.

---

### 2) Only ask judgment questions at high-leverage moments
Do not do this before every trivial action. Gate it by event type:

Require the schema only when:
- starting a new subtask
- running code
- installing dependencies
- launching long jobs
- after a failed command
- before declaring task complete

Skip it for:
- `ls`
- reading a file
- viewing stdout
- small follow-up edits

That preserves turns.

---

### 3) Replace open reflection with 3 forced-choice checks
Free text causes drift. Use compact classifications.

#### A. Value check
“Is this action evidence-producing?”
- yes: experiment / benchmark / artifact / verification
- no: setup / exploration / formatting

If “no” for 2 consecutive substantive actions, supervisor flags coasting.

#### B. Risk check
“What is most likely to fail?”
- env
- dependency
- timeout
- assumption
- none

This primes diagnosis without rambling.

#### C. Alternative check
“What stronger action exists?”
- none
- bigger experiment
- better baseline
- direct verification
- decomposition first

If stronger action exists but is deferred, require one short reason.

---

## Implementation pattern

### Worker output format
```json
{
  "decision": {
    "value_type": "artifact",
    "primary_risk": "timeout",
    "stronger_alternative": "better_baseline",
    "defer_reason": "need smoke test first"
  },
  "tool_call": {
    "tool": "python",
    "args": {"code": "..."}
  }
}
```

### Tool wrapper
- accepts `decision` + `tool_call`
- logs decision for supervisor
- rejects missing decision on gated actions

---

## Add anti-coasting without extra turns
Compute it externally from traces:

Flag if last 2–3 actions:
- produced no new file, metric, or verified claim
- repeated search/read actions
- expanded notes without narrowing uncertainty

Then inject one line into next prompt:
> Warning: recent actions produced no evidence. Next step must create or verify an artifact.

No extra reflection turn needed.

---

## Hard truth
You don’t have a “thinking deficit.” You have a **control bandwidth** problem.  
So don’t buy judgment with extra turns. **Embed judgment into action selection**, and enforce it only where mistakes are expensive.

---
## Turn 4

### Question
Current learning system: EvolutionStore saves {"type": "pitfall", "pattern": "use eval_strategy not evaluation_strategy", "confidence": 0.8}. Word-overlap dedup. Injected into planner prompt.

Daemon's friction system: [FRICTION: what failed | what would be better]. Accumulated per-session, injected into next cycle. Friction includes IMAGINED improvements, not just error logs.

The fundamental difference: EvolutionStore records WHAT happened. Friction records WHAT SHOULD CHANGE.

How should I redesign the learning system? Should friction be per-mission (resets each run) or persistent? How do I prevent the friction buffer from becoming another bloated prompt?

### Answer
Redesign it around **control updates**, not memories.

Your current EvolutionStore is basically a bug scrapbook. Useless at runtime because it stores facts without **applicability, causal trigger, or prescribed behavior**.

## What to store
Every learning item should be a **policy delta**:

```json
{
  "scope": "mission|persistent",
  "trigger": {
    "task_type": "model_training",
    "env": "mac",
    "symptom": "bitsandbytes import failure"
  },
  "diagnosis": "package unsupported on this platform",
  "bad_action": "retry same install/import path",
  "better_action": "switch to CPU-compatible alternative or disable quantization",
  "enforcement": "advisory|preflight|hard_block",
  "confidence": 0.92,
  "evidence": ["stderr hash", "successful workaround run id"],
  "ttl": 5
}
```

That’s the unit. Not “what happened,” but **when X happens, stop doing Y, do Z instead**.

---

## Split learning into 2 systems

### 1) Mission Friction Buffer — short-lived, aggressive
Purpose: adapt within the current run.

Store only:
- repeated failure patterns
- current environment constraints
- anti-coasting warnings
- task-sizing lessons
- strategy shifts

Example:
```json
[
  {
    "trigger":"training jobs exceed timeout",
    "better_action":"run smoke test on subset before full training",
    "ttl":2
  },
  {
    "trigger":"paper search produced no benchmarkable baseline",
    "better_action":"switch from literature expansion to direct baseline implementation",
    "ttl":1
  }
]
```

**This should reset every mission.**  
Why? Most friction is local: current repo, current environment, current dead ends. Persisting it blindly will poison future runs.

### 2) Persistent Policy Store — conservative, promoted learnings only
Only promote friction into persistent memory if:
- it recurs across missions, or
- it has strong causal evidence and a verified fix

Promotion rule:
- seen in ≥3 missions with same trigger, or
- one-shot but high-confidence environment invariant

Examples worth persisting:
- API deprecations
- platform incompatibilities
- known timeout patterns
- required preflight checks

---

## How to prevent prompt bloat
Do **not** inject raw friction history.

Use a **friction compiler** each cycle:
1. Read all mission friction + relevant persistent policies
2. Rank by:
   - relevance to current action
   - recency
   - severity
   - confidence
3. Emit max **3 active control directives**

Output format:
```json
{
  "active_directives": [
    "Before training, run a smoke test on 1 batch.",
    "Do not retry dependency installs unchanged on macOS.",
    "Next step must produce evidence, not more exploration."
  ]
}
```

That’s what goes into prompt/tool wrapper. Not the full store.

---

## Critical design rule
Friction should mostly affect **execution policy**, not planning prose.

Best use:
- preflight checks
- retry suppression
- task decomposition thresholds
- anti-coasting triggers
- tool gating

Worst use:
- dumping “lessons learned” into planner context

---

## Hard truth
If you persist imagined improvements too easily, you’ll create superstition memory.  
So:

- **Mission friction:** can include hypotheses and imagined better actions
- **Persistent store:** only verified policy deltas

That’s the right split:
- per-mission = adaptive and speculative
- persistent = sparse and earned

If you want, I can give you the exact schema and promotion pipeline for replacing EvolutionStore.

---
## Turn 5

### Question
Final question. If you were rebuilding this from the current codebase (not from scratch), what's the minimal set of changes that would take it from "sometimes Grade B, often Grade D" to "consistently Grade B+, occasionally Grade A"?

Give me a prioritized implementation plan — what to build first, second, third. Each item should be a concrete code change, not a principle.

### Answer
Here’s the **minimum viable rebuild**. Three changes, in this order. Don’t touch anything else first.

---

## 1) Add a mandatory `DecisionEnvelope` around risky tool calls
**Why first:** This fixes blind execution, bad retries, and wasted turns without changing your whole architecture.

### Code change
Create a typed wrapper used by coder/reviewer before:
- python execution
- shell/command execution
- dependency install
- long benchmark/train jobs
- task completion

```python
@dataclass
class DecisionEnvelope:
    purpose: str
    expected_evidence: str
    value_type: Literal["artifact","verification","setup","exploration"]
    primary_risk: Literal["env","dependency","timeout","assumption","none"]
    stronger_alternative: Literal["none","better_baseline","bigger_experiment","decompose_first","direct_verification"]
    defer_reason: str | None
```

### Enforcement
- Tool router rejects risky calls if envelope missing.
- Log envelope with every tool result.
- Add a tiny prompt template requiring JSON + tool call in same assistant turn.

### Immediate effect
You get structured judgment **without extra turns**.

---

## 2) Build a `FailureAnalyzer` and route all failures through it
**Why second:** Your supervisor is dumb on failure. This is the fastest path to fewer Grade D collapses.

### Code change
After any nonzero exit / exception / verifier mismatch, call:

```python
@dataclass
class FailureAnalysis:
    failure_class: Literal["env_bug","task_too_big","wrong_approach","dependency","bad_assumption"]
    root_cause: str
    evidence: list[str]
    next_action: Literal["retry_modified","decompose","switch_worker","patch_env","abort_branch"]
    patch: dict | None
```

Implementation:
- Feed task, command/code, stderr/stdout tail, envelope, workspace state to MiniMax
- Return structured JSON only
- Supervisor must consume `next_action`; ban raw “retry same task”

### Add one hard rule
If same command/code hash failed once, exact retry is blocked unless analyzer says why it will differ.

### Immediate effect
You convert failures into adaptations instead of loops.

---

## 3) Replace prompt bloat with `PolicySelector + ToolGuards`
**Why third:** This stabilizes behavior and stops the prompt from rotting further.

### Code change A: ToolGuards
Implement deterministic preflight checks in tool layer:
- file existence
- workspace path safety
- timeout wrapper required
- dependency import smoke check
- block raw subprocess if managed runner exists

Return structured preflight errors.

### Code change B: PolicySelector
Move bug rules into a JSON catalog. Before risky actions, run a cheap selector that picks max 3 relevant rules based on:
- tool
- task type
- libs mentioned
- env
- recent failures

Inject only selected rules into worker prompt.

### Immediate effect
Prompt drops from 160 lines to ~40 stable lines + tiny dynamic policy slice.

---

# Exact rollout order

### Week 1: `DecisionEnvelope`
- add schema
- update worker prompt/output parser
- enforce on risky tools
- log to traces

### Week 2: `FailureAnalyzer`
- implement analyzer call
- wire supervisor transitions
- block unchanged retries
- add task decomposition action

### Week 3: `ToolGuards`
- file/dependency/timeout/subprocess guards
- structured preflight errors

### Week 4: `PolicySelector`
- migrate top 20 recurring rules from prompt into catalog
- selector returns top 3 only
- shrink base prompt aggressively

---

# What not to do yet
- don’t rebuild memory first
- don’t add more reviewers/judges
- don’t add free-form reflection
- don’t tune scoring

Those won’t move B/D reliability nearly as much.

If you implement only these 3 changes well, you’ll get the jump you want.

