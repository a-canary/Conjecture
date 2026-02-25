# Memory

## Current State
<!-- One paragraph: where are we? What's in flight? -->
Gap analysis ~45% complete. 367 tests, 359 pass, 8 xfailed. Coverage 22.00%. All 4 GAPs addressed. Process Layer models at 97.92% coverage. Data Layer functional. Continuing toward 25% coverage milestone. 53 commits ahead (SSH blocked).

## Recent Sessions
<!-- Outcome-tagged log. Most recent first. Max 10 entries. -->
<!-- Format: - YYYY-MM-DD: OUTCOME — summary -->
- 2026-02-25: GATES_MET — Added 25 tests for process/models.py (0% → 97.92%). Overall coverage 19% → 22%. 359 pass, 8 xfailed. Gap analysis ~45%.
- 2026-02-25: GATES_MET — Created ConjectureProcessingInterface, bridged DataManager→OptimizedSQLiteManager. endpoint_app now uses real implementation. 334 pass, 8 xfailed. Gap analysis ~40% complete.
- 2026-02-25: GATES_MET — Fixed GAP-4: Added FastAPI to requirements, fixed ProcessingInterface→SimpleProcessingInterface. 334 pass, 8 xfailed (3 more tests unblocked).
- 2026-02-25: GATES_MET — Implemented SQLite persistence (GAP-2). OptimizedSQLiteManager now functional with async CRUD, batch ops, dirty queries. E2E tests updated and passing. 334 pass, 8 xfailed.
- 2026-02-25: GATES_MET — Created repositories.py (GAP-1). ClaimRepository, RepositoryFactory unblock Process Layer. Context builder and dynamic priming engine now import.
- 2026-02-25: RESEARCH_COMPLETE — Gap analysis: ~20% of CHOICES.md implemented. Identified 4 critical gaps. Data Layer ~70% complete, Process Layer ~10%, Endpoint Layer ~5%.
- 2026-02-25: WORK_DISPATCHED — Hypothesis validation infra (#153): added retry logic to gpt_oss_integration.py, improved \boxed{} extraction in external_benchmarks.py. 3 of 6 remaining items done.
- 2026-02-25: GATES_MET — Backlog cleanup: marked 6 items resolved (#102, #103, #105, #106, #111, #112). Removed 3 TODOs from src/. 331 tests pass, 0 collection errors.
- 2026-02-25: GATES_MET — Cycle complete. LanceDB removed (43 skipped→0). 342 tests, 331 pass, 11 xfail. Coverage 18.36%. optimized_sqlite_manager extended with batch methods.
- 2026-02-25: GATES_MET — Fixed import chain in agent_harness.py (DataManager from support_systems, not non-existent data_manager). All 385 tests now collect and pass.
- 2026-02-24: WORK_DISPATCHED — Conjecture harness created (experiments/conjecture_harness.py). Claim tracking, reasoning steps, halt conditions working. ARC data loaded.
- 2026-02-24: GATES_MET — Test suite fixed after supers/subs rename. 287 passed, 8 xfailed. 7 test files updated. All imports working.
- 2026-02-24: WORK_DISPATCHED — Phase 3 framework created. experiments/arc_agi2_benchmark.py compares bare Haiku vs Haiku+Conjecture. Sample tasks included.
- 2026-02-24: GATES_MET — Phase 1+2 verified. 55 tests pass (8 model, 47 ops). Added file_utils.py stub. Dashboard updated.
- 2026-02-24: WORK_DISPATCHED — Phase 2 in progress. Added anthropic SDK, implemented AnthropicProcessor with claude-3-5-haiku-latest default. Updated providers.json.
- 2026-02-24: GATES_MET — Phase 1 complete. Renamed supported_by→subs, supports→supers in 6 core/agent files + 3 specs. Fixed cascade direction. All imports verified.
- 2026-02-24: WORK_DISPATCHED — Doc cleanup: root 36→6, specs 13→6, docs 30→8, README 606→100 lines. Archived stale analysis/swebench/emoji docs.
- 2026-02-24: PLAN_CREATED — Gap analysis fixes complete. Added T-0008 (Claude Agent SDK), O-0006 (ARC-AGI-2 benchmark). 5-scope model. supers/subs naming.
- 2026-02-24: USER_INPUT — Root context refined: single claim (not multiple), session-scoped, no eval limit. Verified dirty_flag.py matches cascading behavior.
- 2026-02-24: USER_INPUT — Investigated D-0004 tag lifecycle; updated to LLM-generated tags (not user-assigned), programmatic maintenance on CRUD
- 2026-02-24: PLAN_CREATED — CHOICES.md init complete (59 choices), added A-0009 through A-0012 for evaluation model, halt conditions, cascading

## Learnings
<!-- Distilled lessons. Pruned when stale. -->
- **Tags are LLM-generated**: Users don't create tags directly. LLM creates tags when creating claims. Split: >20% usage → sample 100 → LLM suggests ≤8 replacements → batch 20 → LLM assigns per claim. Merge: >500 total.
- **Root context = single claim**: Full conversation stored as one claim, decomposed into supporting claims. Halt when root is clean + LLM satisfied. No fixed eval limit.
- **Dirty cascades upward only**: When claim changes, mark all `.supers` dirty. Never mark `.subs`. Unidirectional toward root context.
- **Relationship naming**: `supers` = claims this provides evidence FOR (toward root). `subs` = claims that provide evidence FOR this (children). Reflects decomposition model.
- **Counter-claims are implicit**: LLM naturally creates alternatives when questioning validity. No formal system needed — just fallacy awareness.
- **5-scope model**: SESSION, WORKSPACE, USER (LLM assigns) → TEAM, PUBLIC (LLM suggests, requires approval).
- **Claude Agent SDK**: Primary LLM provider. Handles secrets. Default: Haiku 4.5. Custom endpoints via JSON config.

## Known Issues
<!-- Recurring failures, provider errors, environment quirks -->
- ~~**Rename needed**: `supports` → `supers`, `supported_by` → `subs`~~ **FIXED 2026-02-24**
- ~~**dirty_flag.py line 98**: bidirectional cascade~~ **FIXED 2026-02-24** — Now unidirectional to supers only
- ~~**GAP-1: Missing repositories.py**~~ **FIXED 2026-02-25** — Created with ClaimRepository, RepositoryFactory
- ~~**GAP-2: Database stubs**~~ **FIXED 2026-02-25** — OptimizedSQLiteManager fully implemented with async CRUD
- **GAP-3: Process layer 0% coverage** — ProcessLLMProcessor and ProcessContextBuilder import correctly but have no test coverage
- ~~**GAP-4: FastAPI missing**~~ **FIXED 2026-02-25** — Added fastapi/uvicorn to requirements, fixed SimpleProcessingInterface usage
- **Git push blocked**: SSH key not configured — requires user to configure SSH keys or use HTTPS
- **Python venv required**: Use `/workspace/.venv/bin/python` for testing
