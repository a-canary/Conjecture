# Plan

## Goal
Implement Conjecture evaluation system with Claude Agent SDK integration and ARC-AGI-2 benchmark validation.

## Phase 1: Code Refactor ✓
### Steps
- [x] 1.1 Rename `supports`→`supers`, `supported_by`→`subs` in src/data/models.py
- [x] 1.2 Fix dirty_flag.py cascade direction (unidirectional to supers only)
- [x] 1.3 Complete rename in remaining core files (relationship_manager, claim_operations, support_relationship_manager)
- [x] 1.4 Update agent layer files (llm_inference, data_flow)
- [x] 1.5 Update specs (architecture.md, requirements.md, README.md)
### Gates (all must pass to complete phase)
- [x] Gate: All `supported_by`/`supports` references replaced with `subs`/`supers` in core files
- [x] Gate: No import errors on core modules
- [x] Gate: Core unit tests pass (55 tests: 8 model, 47 operations)

## Phase 2: Claude Agent SDK Integration ✓
### Steps
- [x] 2.1 Add anthropic SDK dependency to pyproject.toml
- [x] 2.2 Implement AnthropicProcessor with proper error handling
- [x] 2.3 Configure claude-3-5-haiku-latest as default model
- [x] 2.4 Test SDK imports and processor instantiation
- [x] 2.5 Update provider config (providers.json with primary flag)
- [ ] 2.6 Test with live API (requires ANTHROPIC_API_KEY)
### Gates
- [x] Gate: AnthropicProcessor imports and initializes
- [ ] Gate: SDK provider responds to test prompts (needs API key)
- [x] Gate: Existing provider configs still work as fallback

## Phase 3: ARC-AGI-2 Benchmark Framework
### Steps
- [x] 3.1 Create benchmark runner (experiments/arc_agi2_benchmark.py)
- [x] 3.2 Implement bare Haiku baseline measurement
- [x] 3.3 Implement Haiku+Conjecture measurement (enhanced prompting placeholder)
- [x] 3.4 Create comparison report generator and metrics
- [x] 3.5 Download real ARC-AGI-2 task data (1000 training tasks)
- [x] 3.6 Integrate full Conjecture harness (claims, dirty flags, cascade)
- [ ] 3.7 Run full benchmark with API key
### Gates
- [x] Gate: Benchmark framework runs with sample tasks
- [ ] Gate: Baseline scores recorded for 10+ real ARC-AGI-2 tasks
- [ ] Gate: Comparative metrics show measurable improvement

## Current Phase: 3
## Status: in-progress
## Notes: Framework complete. Need real ARC data and API key for full validation.
