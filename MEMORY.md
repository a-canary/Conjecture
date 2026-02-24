# Memory

## Current State
<!-- One paragraph: where are we? What's in flight? -->
CHOICES.md initialized with 59 choices across 8 sections. Recent work refined tag lifecycle (D-0004) to specify LLM-generated tags with nightly condensation process. Architecture choices clarified: input decomposition, claim tools, cascading evaluation, halt conditions. Ready for implementation planning.

## Recent Sessions
<!-- Outcome-tagged log. Most recent first. Max 10 entries. -->
<!-- Format: - YYYY-MM-DD: OUTCOME — summary -->
- 2026-02-24: USER_INPUT — Root context refined: single claim (not multiple), session-scoped, no eval limit. Verified dirty_flag.py matches cascading behavior.
- 2026-02-24: USER_INPUT — Investigated D-0004 tag lifecycle; updated to LLM-generated tags (not user-assigned), programmatic maintenance on CRUD
- 2026-02-24: PLAN_CREATED — CHOICES.md init complete (59 choices), added A-0009 through A-0012 for evaluation model, halt conditions, cascading

## Learnings
<!-- Distilled lessons. Pruned when stale. -->
- **Tags are LLM-generated**: Users don't create tags directly. LLM creates tags when creating claims. Tag maintenance (split >10% usage, merge >500 total) triggers programmatically on each CRUD operation.
- **Root context = single claim**: Full conversation stored as one claim, decomposed into supporting claims. Halt when root is clean + LLM satisfied. No fixed eval limit.
- **Dirty cascades upward**: When supporting claim changes, parent claims marked dirty. Cascades toward root context. Clean = re-eval would be no-op.

## Known Issues
<!-- Recurring failures, provider errors, environment quirks -->
