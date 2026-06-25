# NEXT.md — conjecture

Project is **feature-complete**. All 50+ CHOICES.md items implemented. 1042 tests pass, 18 skipped (OpenRouter API integration tests, appropriately skipped).

## Resolved

- **Hygiene: archive/ + .archive/ retired** — **✅ COMPLETE** — committed (this PR). Moved both subtrees to `~/trash/1782403344_conjecture-archive*/` (377 + 4 files, ~102KB deletions). `docs/index.md` cleaned, `pytest.ini` dead config removed, 1075 tests still pass.
- **A-0016 STATISTICAL_REALITY_CHECK.md** — **✅ COMPLETE** — committed 461f227. Restored from archive, resolving broken CHOICES.md reference.
- **O-0009 Task-type routing** — **✅ COMPLETE** — committed 1c3fc23. QueryType enum + classify_query() (90%+ accuracy on 21-query held-out set, 27 tests), integrated into evaluate() with RECALL fast-path (no decomposition), query_type surfaced in response.
- **R&D-FACT-CHECK: fact_checking_pipeline tests** — **✅ COMPLETE** — committed 12426fc. Added test suite for R&D artifact `src/core/fact_checking_pipeline.py` from 2026-05-04 sprint. 24 tests covering SelfConsistencyChecker, VectorSearchVerifier, CascadeInvalidator, FactCheckingPipeline. Fix: VectorSearchVerifier now returns SKIPPED when vector_store is None. Full suite: 1075 passing.

## Deferred

### O-0008: DROP / MATH / HumanEval benchmarks
Generated: 2026-04-29
Source: CHOICES.md O-0008
Reason: Evaluation runs, not code. Outside sprint scope per Director.
Unblock: Director or user direction to run benchmarks

## Out of Scope — needs Director review

*(none — all CHOICES items implemented)*

---

*Historical completions (pre-2026-04-29):*
- UX-0007 Phase 3 (TUI Interactive Browser) — **✅ COMPLETE** — committed 27f790d
- All 50+ CHOICES.md items — **✅ COMPLETE** — committed 408a947
