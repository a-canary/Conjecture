# Memory

## Current State
<!-- One paragraph: where are we? What's in flight? -->
CHOICES.md initialized with 59 choices across 8 sections. Recent work refined tag lifecycle (D-0004) to specify LLM-generated tags with nightly condensation process. Architecture choices clarified: input decomposition, claim tools, cascading evaluation, halt conditions. Ready for implementation planning.

## Recent Sessions
<!-- Outcome-tagged log. Most recent first. Max 10 entries. -->
<!-- Format: - YYYY-MM-DD: OUTCOME — summary -->
- 2026-02-24: USER_INPUT — Investigated D-0004 tag lifecycle; updated to LLM-generated tags (not user-assigned), nightly condensation (>10% usage splits, >500 merges), A-0010 now supports D-0004
- 2026-02-24: PLAN_CREATED — CHOICES.md init complete (59 choices), added A-0009 through A-0012 for evaluation model, halt conditions, cascading

## Learnings
<!-- Distilled lessons. Pruned when stale. -->
- **Tags are LLM-generated**: Users don't create tags directly. LLM creates tags when creating claims, sees all existing tags in prompt, can reuse or create new. Nightly condensation keeps tags useful.

## Known Issues
<!-- Recurring failures, provider errors, environment quirks -->
