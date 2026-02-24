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
- **Tags are LLM-generated**: Users don't create tags directly. LLM creates tags when creating claims. Split: >20% usage → sample 100 → LLM suggests ≤8 replacements → batch 20 → LLM assigns per claim. Merge: >500 total.
- **Root context = single claim**: Full conversation stored as one claim, decomposed into supporting claims. Halt when root is clean + LLM satisfied. No fixed eval limit.
- **Dirty cascades upward only**: When claim changes, mark claims in `.supports` dirty (parents). Never mark `.supported_by` (children). Unidirectional toward root context.
- **Counter-claims are implicit**: LLM naturally creates alternatives when questioning validity. No formal system needed — just fallacy awareness.

## Known Issues
<!-- Recurring failures, provider errors, environment quirks -->
- **dirty_flag.py line 98**: `_cascade_dirty_flags()` cascades bidirectionally (`supported_by | supports`). Should only cascade to `supports` (unidirectional upward).
