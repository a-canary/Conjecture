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

### O-0006: DeepEval Benchmark Suite (Small Models)
Supports: M-0001, O-0002

Three DeepEval benchmarks: DROP (hard math), ARC (science), BIG-Bench Hard (logic). Test on 8B-class models (llama3.1-8b, Qwen2.5-7B) where Conjecture adds value. R&D confirmed: strong models (DeepSeek-V3) hit 100% baseline — no room to improve.

### O-0007: Threshold-Based Garbage Collection
Supports: D-0010, D-0001

GC runs when claim count exceeds threshold. Removes claims that are both clean (not dirty) and low-confidence. Prevents unbounded growth while preserving valuable knowledge. GC does not run during active reasoning.

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
