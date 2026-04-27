# CHOICES.md — Source of Plan

All project choices in priority order. Higher choices constrain lower ones.
Use `/choose-wisely` to add, change, remove, or reorder choices (triggers cascading review).
Use `/choose-wisely audit` to check for contradictions and structural issues.

## Rules

- **Position = Priority**: higher constrains lower, no exceptions
- **Gravity rule**: changing a choice triggers three checks — break support? section coherence? impact below? — then cascades recursively until no new impacts are found
- **Section order is fixed**: Mission > User Experiences > Features > Operations > Data > Architecture > Technology > Implementation
- **Within sections**: items ordered by priority (higher constrains lower)
- **Supports line required**: every choice (except top of stack) lists the IDs it directly supports. This is the dependency graph.
- **Architecture is tool-agnostic**: Architecture describes patterns and structure. Technology names the tools that implement those patterns.
- **No status values**: git diff is the change record. The committed file is always the current truth.
- **Inheritance**: subdirectory CHOICES.md files inherit from nearest ancestor. Parent choices prepend as higher priority. Each file must be internally consistent with its full ancestry.

### Choice Entry Format

```
### X-0001: Title of choice
Supports: X-0000, Y-0000

One to two lines of rationale. Not a spec. Just why this choice was made and what it means.
```

- ID format: `PREFIX-NNNN` where prefix indicates section and NNNN is a globally unique 4-digit number, sequential from creation. IDs are never reused after deletion.
- Section prefixes: `M-` (Mission), `UX-` (User Experiences), `F-` (Features), `O-` (Operations), `D-` (Data), `A-` (Architecture), `T-` (Technology), `I-` (Implementation)
- ID numbers do not reflect file order or priority — position in the file determines priority
- `Supports:` line lists IDs of higher-priority choices this choice directly supports
- No status annotations — the committed file is always "current truth"

---

## Mission

### M-0001: Evidence-Based Reasoning Framework
Supports: (top of stack)

Force LLMs to validate assumptions, fact-check sources, and explore/prove/disprove ideas before responding. Track what needs more evidence or confidence. Recursively break down knowledge, questions, and work into verifiable claims. The framework produces reliable, trustworthy responses — even small reasoning models perform extremely well when forced to reason with evidence.

### M-0002: Minimize Hallucinations via Verified Claims
Supports: M-0001

Treat all knowledge as provisional claims with confidence scores — never as facts. Force models to cite sources, track uncertainty, and revise beliefs when evidence changes. Hallucinations become detectable confidence gaps.

CHALLENGED by TruthfulQA (-13pp): Full decomposition increased false confidence rather than reducing hallucinations. Claim-based reasoning with extensive decomposition may amplify model's tendency to construct plausible-sounding but incorrect justifications. Lightweight verification (cot_lite) or direct prompting performs better on truthfulness tasks. Mission remains valid but implementation approach needs refinement — verification must be evidence-based, not just structured elaboration.

### M-0003: Handle Long Context and Complex Tasks
Supports: M-0001

Break complex problems into claim graphs that persist across sessions. Enable models to work on tasks too large for a single context window by traversing, validating, and synthesizing interconnected claims.

### M-0004: Verify Assumptions Explicitly
Supports: M-0001, M-0002

Surface implicit assumptions as explicit claims. Counter-claims are implicit LLM behavior — LLM may create alternative claims (course corrections or opposites to disprove) when questioning validity. Must be fallacy-aware. Reasoning chain auditable and correctable.

### M-0005: Privacy-First with Cloud Option
Supports: M-0001

Support local-first operation (Ollama, LM Studio) for complete data privacy. Cloud providers available for users who prioritize performance. The framework works with any OpenAI-compatible backend.

### M-0006: Show Your Work for User Trust
Supports: M-0001, M-0002

Increase user trust through transparent reasoning. Display breakdown of supporting claims, confidence scores, tool calls, and primary sources. Users can audit any conclusion back to its evidence chain.

### M-0007: Host Conjecture as LLM Endpoint
Supports: M-0001, M-0005

Conjecture is a middle-layer LLM provider that hosts as localhost or VPS. Enhances queries with claim context, routes to gpt-oss-20B by default. Sessions store claims in local DB; LLM can elevate claims to project/team scope. This is the primary deployment model.

---

## User Experiences

### UX-0001: CLI as Primary Interface
Supports: M-0001

Power users interact via CLI with rich terminal output. Commands are discoverable (`--help`), composable, and scriptable. The CLI never contains business logic — it only formats and displays.

### UX-0002: Progressive Disclosure of Complexity
Supports: M-0001, UX-0001

Basic commands (`create`, `search`, `stats`) work immediately. Advanced features (batch operations, backend selection, web research) are available but not required. Users grow into the system.

### UX-0003: Auto-Detection Over Configuration
Supports: M-0005, UX-0002

System automatically detects available LLM backends (local first, then cloud). Users can override with `--backend` flag, but sensible defaults should "just work" on first run.

### UX-0004: Multi-Interface Access (CLI, TUI, Web, MCP)
Supports: M-0001

Users access the framework via CLI, TUI, web interface, or MCP. All testing and validation occurs through CLI as the most direct raw interface. Other interfaces delegate to the same Endpoint layer.

### UX-0005: Claim Provenance via Chain/Tree View
Supports: M-0006, UX-0001

Every claim displays its provenance — the chain or tree of supporting claims. Users trace any assertion back through its evidence hierarchy to primary sources.

### UX-0006: Live Reasoning Breakdown for Active Prompts
Supports: M-0006, UX-0005

Users can inspect the breakdown and supporting claims for any past or current prompt the framework is processing. Enables real-time transparency into model reasoning.

### UX-0007: Claim Visualization UI
Supports: M-0006, UX-0005

Web/TUI interface for visualizing the support tree of claims, reasoning chains, and primary sources. Users see how conclusions connect to evidence. Complements MCP delivery — transparency surface for users who want to audit reasoning.

### UX-0008: Chat-First Interaction
Supports: M-0001, UX-0001

Users interact with Conjecture as an LLM provider via chat, not direct claim CRUD. Claims are internal infrastructure. Users make statements or ask questions; the system manages claims transparently.

---

## Features

### F-0001: Claim Management Core
Supports: M-0002, M-0003

Create, retrieve, search, and analyze claims. Each claim has: id, content, confidence (0.0-1.0), state, type, tags, and bidirectional relationships (supers/subs).

### F-0002: Confidence Scoring
Supports: M-0002, F-0001

Every claim carries a confidence score reflecting uncertainty. High-confidence claims (>0.95) are not "facts" — they are well-supported conjectures still subject to revision.

### F-0003: Semantic Search
Supports: M-0003, F-0001

Find relevant claims using natural language queries. Vector embeddings enable similarity search beyond keyword matching. Results ranked by relevance and confidence.

### F-0004: Multi-Provider LLM Support
Supports: M-0005, F-0001

Support multiple LLM providers with automatic fallback. Local providers (Ollama, LM Studio) for privacy; cloud providers (Chutes.ai, OpenRouter, OpenAI, Anthropic) for power.

### F-0005: Web Research Integration
Supports: M-0004, F-0001

Automatically gather evidence from the web to support or challenge claims. Web search results become claims with their own confidence scores and source attribution.

### F-0006: Domain-Adaptive Reasoning
Supports: M-0001, F-0001

Problem-type detection (mathematical, logical, multi-step) triggers specialized prompts. Domain-adaptive prompts demonstrated 100% improvement in mathematical reasoning (Cycle 1).

### F-0007: Nine Claim Type Categories
Supports: M-0002, F-0001

Claims support types: IMPRESSION, ASSUMPTION, OBSERVATION, CONJECTURE, CONCEPT, EXAMPLE, GOAL, REFERENCE, ASSERTION. Each reflects different epistemological status. Claims can have multiple types.

### F-0008: Claim Filtering with Range Constraints
Supports: F-0001, F-0003

Queries filter by tags, confidence (min/max), state, type, scope, date range, and limit (1-1000). Enables flexible but bounded querying with validated ranges.

---

## Operations

### O-0001: Code-Test-Commit Workflow
Supports: M-0001

One small change at a time. Test thoroughly before commit. Update RESULTS.md with metrics. Justify any repo size increase. Quality gates block regressions.

### O-0002: Skeptical Validation Thresholds
Supports: M-0004, O-0001

Claimed improvements require minimum 2-4% measured improvement to be accepted. Prevents metric gaming and false positives. 62% of improvement cycles passed validation (8/13).

### O-0003: Real Testing Over Mocks
Supports: M-0004, O-0001

Mocking is banned. Tests must validate real behavior with real components. Integration tests catch issues that unit tests with mocks would miss.

### O-0004: Graceful Degradation on Provider Failure
Supports: M-0005, F-0004

Cloud providers are optional. When any LLM provider fails, the system automatically falls back to next available provider. Error classification (network/timeout/rate-limit/auth) determines retry behavior.

### O-0005: Exponential Backoff with Circuit Breaker
Supports: O-0004, F-0004

LLM operations use exponential backoff (10s-10min range) with error-type-specific multipliers. Circuit breaker pattern prevents cascading failures during provider outages.

### O-0006: DeepEval Benchmark Suite (OSS Models)
Supports: M-0001, M-0007, O-0002

Three DeepEval benchmarks for OSS models people can run locally: GSM8K (grade school math), MathQA (math reasoning), HellaSwag (commonsense). Target model: openai/gpt-oss-20b. Proven: GSM8K +40pp (16.7% → 56.7%) demonstrates Conjecture's value for math reasoning. In benchmark suite runs, use 1 persistent session for claim accumulation across test cases.

### O-0009: Task-type routing is REQUIRED for production
Supports: M-0002, A-0015

Decomposition helps reasoning (BBH +9pp, Synthetic +18pp) but HURTS recall/commonsense (MMLU -17pp, TruthfulQA -13pp). Production deployment MUST classify queries and route: three-prompt for hard reasoning tasks, cot_lite for recall/commonsense. See O-0002 validation notes for evidence.

### O-0010: Model capability threshold for three-prompt
Supports: M-0001, A-0015

Three-prompt architecture requires 70B+ models. 8B models show -32pp regression vs direct prompting. Optimization via context reduction or iteration limiting is insufficient. Direct prompting recommended for <32B models. Validate new models before deploying three-prompt.
Supports: D-0010, D-0001

GC runs when claim count exceeds threshold. Removes claims that are both clean (not dirty) and low-confidence. Prevents unbounded growth while preserving valuable knowledge. GC does not run during active reasoning.

### O-0008: Benchmark Validation Requirements
Supports: M-0001, O-0006

Run minimum 10 benchmarks with 40+ samples each. Conjecture must perform >= Direct on ALL benchmarks (no regressions). Must show +20pp improvement on at least 5 benchmarks. This validates the framework's value proposition — claim enhancement must never hurt performance, and must meaningfully improve reasoning on challenging tasks.

VALIDATED with caveats (7/10 benchmarks complete, 100 samples each):
- ✅ Hard reasoning (BBH +9pp, Synthetic +18pp)
- ❌ Recall/commonsense (MMLU -17pp, TruthfulQA -13pp, HellaSwag -10pp)
- ≈ High-baseline (GSM8K +1pp, ARC -1pp — saturated)

**Task-type routing REQUIRED** — decomposition helps reasoning but hurts recall/commonsense. Production deployment must classify queries and route appropriately. Lightweight alternatives (cot_lite +2pp) viable for recall tasks. Original "no regressions" rule holds ONLY with routing. Complete validation requires 3 more benchmarks (DROP, MATH, HumanEval) plus multi-model testing.

Multi-model validation (2026-03-07) revealed model-size dependency: three-prompt architecture fails catastrophically on small models (8B: -32pp BBH, p<0.001) but succeeds on large models (670B: +10pp BBH, p=0.018).

A-0015 re-validation (2026-03-08, Phase 4): Hypothesis DISPROVED. Delegated retrieval (44%) vs no-retrieval (40%) showed only +4pp improvement (p=0.685, NOT significant). 8B models still regress -28pp vs direct (p=0.0046, SIGNIFICANT). The three-prompt architecture has an inherent capability threshold — small models lack the meta-reasoning, confidence calibration, and multi-prompt context management required.

Tier 1 8B optimization attempts (2026-03-08, n=50 BBH):
- Context limit (5 claims): 48% vs 40% baseline (+8pp, p=0.42 NOT significant) — context size NOT the issue
- Single-step forcing (max_iterations=1): 58% vs 40% baseline (+18pp, p=0.072 marginal) — iteration overhead contributes but insufficient
- Three-model ensemble: BLOCKED by model availability (infrastructure limitation)

**Architectural constraint validated:** Three-prompt requires 70B+ models. 8B optimization via context reduction, iteration limiting, or ensemble voting does not restore viability. Direct prompting recommended for <32B models (72-90% accuracy vs 40-58% three-prompt).

---

## Data

### D-0001: Universal Claim Model
Supports: M-0002, F-0001

Single data model across all layers: id, content, confidence, state, type, tags, supers, subs. No complex derived models. Simplicity over expressiveness.

### D-0002: Evaluation Priority Tuple
Supports: D-0001, M-0002

Claims are selected for evaluation by [dirty, confidence, root_similarity] tuple. Root similarity is vector distance to the root context (full conversation). Dirty claims with low confidence and high relevance evaluated first. No discrete state machine — just continuous priority scoring.

### D-0003: Bidirectional Relationships
Supports: D-0001, F-0001

Claims link via `supers` (claims this provides evidence FOR, toward root) and `subs` (claims that provide evidence FOR this). Naming reflects decomposition: break down into subs, build up to supers. Enables graph traversal for context building.

### D-0004: Tag Lifecycle Management
Supports: D-0001, F-0003

Tags are LLM-generated, not user-assigned. Split trigger: tag >20% usage. Process: sample ≤100 claims → LLM suggests ≤8 replacement tags → batch claims (20) → LLM assigns replacement per claim. JSON in/out. If total >500 → merge similar. Max 20 per claim.

### D-0005: Four-Level Scope Model
Supports: M-0005, D-0001

Claims support four scopes: WORKSPACE (directory-only), USER, PROJECT, TEAM, GLOBAL. No session scope — all claims persist. LLM decides scope assignment. Scope determines visibility and sharing boundaries.

### D-0006: Dirty Flag for Re-Evaluation Queue
Supports: D-0001, D-0002

Claims track dirty state with reasons. When claim changes, mark all `supers` dirty — unidirectional cascade toward root context. Never cascade to `subs`. Clean = no subs changed, re-eval would be no-op. Evaluation loop only processes dirty claims.

### D-0007: Acyclic Graph Enforcement
Supports: D-0003

Claims form directed acyclic graph (DAG). Relationship manager detects cycles via traversal. Prevents A→B→A relationships. Essential for context building without infinite loops.

### D-0008: Relationship as First-Class Object
Supports: D-0003, D-0001

Relationships have source_id, target_id, type, confidence (0.0-1.0), metadata dict, timestamp. Enables rich relationship semantics beyond simple edges.

### D-0009: Root Context as Claim
Supports: D-0001, M-0003, A-0009

Root context = entire conversation (user + framework messages) stored as a single claim. Context decomposed into supporting claims (questions, assumptions, conjectures) via LLM. Root similarity measures claim relevance to this conversation claim. Workspace-scoped: persists within the workspace directory.

### D-0010: No Claim Deletion by Users
Supports: D-0001, D-0006, UX-0008

Users cannot delete claims directly. Instead, users mark claims dirty or drop confidence via chat statements. The system re-evaluates dirty claims. Prevents knowledge loss while allowing correction.

---

## Architecture

### A-0001: 4-Layer Architecture
Supports: M-0001, D-0001

Strict separation: Presentation (CLI) → Endpoint (public API) → Process (intelligence) → Data (storage). Each layer has single responsibility. No layer skipping.

### A-0002: Presentation Layer is Dumb
Supports: A-0001, UX-0001

`src/cli/` contains NO business logic. It instantiates ConjectureEndpoint, calls methods, and formats output. All intelligence lives in Process layer.

### A-0003: ConjectureEndpoint as Public API
Supports: A-0001

Single entry point for all external consumers. Three core methods: `create_claim()`, `get_claim()`, `evaluate()`. Orchestrates Data and Process layers.

### A-0004: Process Layer Owns Intelligence
Supports: A-0001, F-0006

Context Builder assembles claim graphs. LLM Processor interprets context and generates insights. Instruction identification via tag hints + LLM validation.

### A-0005: Context Building Strategy
Supports: A-0004, D-0003, D-0007

Build context by: 1) Traverse `supers` to root (100% inclusion), 2) Traverse `subs` to depth 2, 3) Semantic fill if token budget permits. Relies on acyclic graph property to prevent infinite traversal.

### A-0006: Provider Pattern for LLM Backends
Supports: A-0001, F-0004

Pluggable providers implement common interface. Auto-detection selects best available. LLMBridge provides unified access. Fallback on provider failure.

### A-0007: Standardized API Response Wrapper
Supports: A-0003, UX-0001

All endpoints return standardized wrapper with success, data, message, errors fields. Custom JSON encoder handles datetime to ISO format. Enables consistent client handling.

### A-0008: Batch Processing with ClaimBatch Model
Supports: A-0004, D-0001

Batch operations use ClaimBatch (claims list + batch_id + timestamp). BatchResult aggregates individual results with success rate calculation. Enables efficient bulk processing.

### A-0009: Input Decomposition via LLM
Supports: A-0004, M-0002, M-0003

The Process Layer treats all input as compound. Prompts are decomposed into constituent claims (questions, assertions, references, context) using LLM analysis before reasoning. This is not a separate step — decomposition is how the framework thinks.

### A-0010: LLM Operates via Claim Tools
Supports: A-0004, M-0002, M-0006, F-0001, D-0004

The LLM is given tools to CRUD claims, respond to user, and invoke other skills. Responses aren't raw text — they're structured claim operations. This makes all LLM reasoning traceable through the claim graph. Tag maintenance (split/merge) triggers programmatically on each CRUD operation.

### A-0011: Cascading Evaluation on Upstream Changes
Supports: A-0004, D-0006, M-0004

When claim changes, mark all `supers` dirty. Unidirectional cascade toward root context. Never mark `subs` dirty. Ensures parent conclusions re-evaluated when evidence changes.

### A-0012: LLM-Driven Halt or Explore
Supports: A-0004, M-0002, M-0006, D-0009

LLM decides whether to halt and respond OR explore further by creating new claims to question or investigate. Not a system-imposed threshold. If LLM is unsatisfied with confidence or evidence, it creates new claims rather than forcing a response.

### A-0013: MCP Delivery Model
Supports: A-0001, M-0001, UX-0004

Expose Conjecture as an MCP server with tools: `build_context(query) → context_blob`, `upsert_claim(claim, confidence, super_ids, sub_ids)`, `explore_next() → claim`, `get_claim_support(claim_or_query) → sub-claims`. Any MCP-compatible client (Claude Desktop, Cursor, custom) can use Conjecture as a reasoning backend.

### A-0014: Streaming Evaluation State
Supports: A-0004, UX-0006

Evaluation state is observable in real-time. Current claims being evaluated, their confidence, and reasoning steps are exposed via streaming API or polling endpoint. Enables live reasoning breakdown for active prompts.

### A-0015: Delegated Tool Calling for Knowledge Retrieval
Supports: M-0007, A-0010, M-0001

Small models can reason but lack embedded knowledge. The LLM endpoint delegates knowledge retrieval tool calls to the calling system rather than executing them directly. When the model needs external evidence (web search, business DB, simulation), it emits structured tool-call requests. The caller performs the tool call and appends results to the next prompt's context as evidence claims. The core loop is: retrieve → decompose to claims → reason with evidence. This is essential for small models (<32B) which showed -32pp regression when forced to reason without retrieved evidence (multi-model validation, 2026-03-07). The endpoint does not perform tool calls — it requests them and continues reasoning when results are appended.

### A-0016: Goldilocks Principle for Tiny Models (1-2B Parameters) [HYPOTHESIS]
Supports: A-0015, A-0004, M-0001

STATISTICAL CAVEAT: Exploratory findings from small samples (n=10-20). Only 1/7 key results achieved statistical significance (p<0.05). All claimed positive improvements (word count, claim count, task effects) have p>0.10 and wide confidence intervals. The ONE validated finding is NEGATIVE: ultra-concise claims harm commonsense reasoning (HellaSwag -40pp, p=0.004). Requires n≥100 validation before production claims. See STATISTICAL_REALITY_CHECK.md for full analysis. Treat as testable hypothesis, not validated architecture.

EXPLORATORY FINDINGS (not statistically validated): Tiny models (1-2B parameters) may have fixed cognitive capacity limiting claim processing. Tentative optimal claim count is 1-3 claims maximum (p=0.299, n=10, NOT significant). Small samples suggest >3 claims may cause overload but this needs validation. Claim content should be task-specific: reasoning tasks may benefit from abstract principles ("use transitivity"), calculation tasks may benefit from format guidance ("show work, answer as ####"). Ultra-concise claims (~5 words) showed +25pp effect vs 15-word claims but NOT statistically significant (p=0.102, 95% CI: [-5pp, +55pp]). VALIDATED NEGATIVE: Commonsense tasks (HellaSwag) showed -40pp regression with ultra-concise claims (p=0.004, highly significant). Single direct prompt pattern suggested across small samples but needs validation. Exploratory results on LFM-2.5-1.2B: BBH 90%→100% (p=0.299), MMLU 10%→20% (observed but not tested), GSM8K 60%→70% (observed but not tested). All require n≥100 replication before deployment (2026-03-08 statistical analysis).

---

## Technology

### T-0001: Python 3.8+
Supports: A-0001

Python for rapid development and rich AI/ML ecosystem. Async/await for LLM interactions. Type hints throughout for IDE support and documentation.

### T-0002: Pydantic for Data Validation
Supports: T-0001, D-0001

All data structures use Pydantic models. Automatic validation, serialization, and documentation. Pydantic v2 for performance.

### T-0003: SQLite for Claim Storage
Supports: D-0001, A-0001

SQLite as primary persistent storage. Simple, embedded, no server required. aiosqlite for async access. Parameterized queries for SQL injection protection.

### T-0004: FAISS+SQLite for Vector Search
Supports: F-0003, D-0001

FAISS with SQLite backing for vector embeddings and semantic search. ChromaDB tested and rejected (slow, heavy dependencies). Do not revisit ChromaDB — it is deprecated.

### T-0005: Typer + Rich for CLI
Supports: UX-0001, T-0001

Typer for CLI framework with automatic help generation. Rich for beautiful terminal output, progress indicators, and emoji support.

### T-0006: Local LLM Providers
Supports: M-0005, F-0004

Ollama (localhost:11434) and LM Studio (localhost:1234) for local inference. No API keys required. Complete data privacy.

### T-0007: Custom LLM Endpoints
Supports: M-0005, F-0004

OpenAI/Anthropic-compatible endpoints (Chutes.ai, OpenRouter, etc.) configured via JSON. Enables self-hosted or alternative providers.

### T-0008: Claude Agent SDK (Primary)
Supports: M-0001, F-0004, A-0010

Claude Agent SDK with Claude Max for primary inference. Default model: Haiku 4.5 (cost-effective). SDK handles authentication and secrets. Enables tool use and structured claim operations.

---

## Implementation

### I-0001: Hierarchical Configuration
Supports: A-0001, T-0001

Priority: Workspace config (`.conjecture/config.json`) → User config (`~/.conjecture/config.json`) → Default config. JSON format. Claude Agent SDK handles Claude secrets. Custom OpenAI/Anthropic-compatible endpoints configured via JSON.

### I-0002: Async by Default
Supports: T-0001, F-0004

All LLM calls are async. Context collection optimized for minimal latency. Results cached where appropriate.

### I-0003: Full Type Annotations
Supports: T-0001, T-0002

Every function has type hints. Enables IDE completion, static analysis, and serves as documentation. Pydantic models enforce runtime validation.

### I-0004: Cross-Platform Support
Supports: UX-0001, T-0001

Works on Windows, macOS, and Linux. UTF-8 encoding for emoji/Unicode. Platform-specific scripts (.bat/.sh) for common operations.

### I-0005: Test Coverage Minimum 85%
Supports: O-0001, O-0003

Unit, integration, performance, and security tests. pytest with markers for categorization. Coverage tracking with baseline comparison. No commit if coverage drops.

### I-0006: Security by Default
Supports: M-0005, T-0003

Parameterized queries prevent SQL injection. Input validation via Pydantic. No secrets in repository. API keys in user config only.

### I-0007: Test Categorization via pytest Markers
Supports: I-0005, O-0001

Tests use markers (unit, integration, performance, slow, asyncio, models, sqlite, chroma) for selective execution. 300-second timeout. asyncio_mode=auto for event loop handling.

### I-0008: Structured Logging with Module Loggers
Supports: O-0001, I-0002

All modules use `get_logger(__name__)` for module-scoped logging. Consistent format: timestamp, module, level, message. LLM error handler tracks operation stats.
