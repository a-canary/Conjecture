# Memory

## Current State
<!-- One paragraph: where are we? What's in flight? -->
CHOICES.md finalized (~63 choices). Docs cleaned up: essential files at root (6), streamlined specs (6), trimmed docs (8). Claude Agent SDK + ARC-AGI-2 benchmarking direction set. Ready for implementation: code rename (supers/subs), SDK integration, benchmark framework.

## Recent Sessions
<!-- Outcome-tagged log. Most recent first. Max 10 entries. -->
<!-- Format: - YYYY-MM-DD: OUTCOME — summary -->
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
- **Rename needed**: `supports` → `supers`, `supported_by` → `subs` in models.py, dirty_flag.py, claim_operations.py, repositories, and all references.
- **dirty_flag.py line 98**: `_cascade_dirty_flags()` cascades bidirectionally. Should only cascade to `supers` (unidirectional upward).
