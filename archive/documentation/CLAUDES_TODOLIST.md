# Conjecture Project - Development Review & Todo List

**Generated**: November 29, 2025  
**Purpose**: Identify cleanup opportunities, design clarifications needed, and incomplete code sections

---

## Table of Contents
1. [Files to Delete](#1-files-to-delete)
2. [Design Issues Requiring Clarification](#2-design-issues-requiring-clarification)
3. [Incomplete Code Sections](#3-incomplete-code-sections)
4. [Test File Consolidation](#4-test-file-consolidation)
5. [Documentation Cleanup](#5-documentation-cleanup)
6. [Recommended Actions](#6-recommended-actions)

---

## 1. Files to Delete

### 1.1 Broken/Backup Files (IMMEDIATE DELETE)

These files are vestiges of debugging/fixing cycles and add no value:

| File | Reason |
|------|--------|
| `src/cli/backends/cloud_broken.py` | Broken version - use `cloud.py` or `cloud_fixed.py` |
| `src/cli/backends/hybrid_broken.py` | Broken version - use `hybrid.py` or `hybrid_fixed.py` |
| `src/cli/backends/local_broken.py` | Broken version - use `local.py` or `local_fixed.py` |
| `src/cli/base_cli_broken.py` | Broken version - use `base_cli.py` |
| `src/cli/base_cli_test.py` | Test file in wrong location |
| `src/config/old_config.py` | Deprecated configuration |
| `src/core/models_backup.py` | Backup file - original exists |

### 1.2 Redirect/Stub CLI Files (DELETE AFTER VERIFICATION)

These files only display redirect messages and serve no functional purpose:

| File | Redirects To | Action |
|------|--------------|--------|
| `src/simple_cli.py` | `src/cli/modular_cli.py` | Delete - just prints redirect notice |
| `src/full_cli.py` | `src/cli/modular_cli.py` | Delete - just prints redirect notice |
| `src/enhanced_cli.py` | `src/cli/modular_cli.py` | Delete - just prints redirect notice |
| `src/local_cli.py` | `src/cli/modular_cli.py` | Delete - just prints redirect notice |

**Note**: Verify no external scripts reference these before deletion.

### 1.3 Fixed Files to Rename/Consolidate

The `*_fixed.py` files should replace their base versions:

| Current File | Action |
|--------------|--------|
| `src/cli/backends/cloud_fixed.py` | Compare with `cloud.py`, keep better version as `cloud.py` |
| `src/cli/backends/hybrid_fixed.py` | Compare with `hybrid.py`, keep better version as `hybrid.py` |
| `src/cli/backends/local_fixed.py` | Compare with `local.py`, keep better version as `local.py` |

**Decision Needed**: The only difference between `cloud.py` and `cloud_fixed.py` is the import path:
- `cloud.py`: `from ..base_cli import BaseCLI`
- `cloud_fixed.py`: `from .base_cli import BaseCLI`

Determine which import is correct and consolidate.

### 1.4 Root-Level Test Files (MOVE TO `tests/`)

These 19 test files at root level should be moved to `tests/` or deleted:

| File | Recommendation |
|------|----------------|
| `test_context_builder.py` | Move to `tests/` |
| `test_core_direct.py` | Delete (ad-hoc testing) |
| `test_core_tools_complete.py` | Keep as `tests/test_core_tools.py` |
| `test_core_tools_integration.py` | Merge into `tests/test_core_tools.py` |
| `test_core_tools_simple.py` | Delete (subset of complete) |
| `test_core_tools_working.py` | Delete (duplicate) |
| `test_design_only.py` | Delete (design exploration, not tests) |
| `test_direct_backend.py` | Delete (manual testing script) |
| `test_dirty_propagation.py` | Merge into `tests/test_dirty_flag.py` |
| `test_emoji.py` | Delete (24 lines, subset of comprehensive) |
| `test_emoji_comprehensive.py` | Keep as `tests/test_emoji.py` |
| `test_emoji_integration.py` | Delete (27 lines, minimal) |
| `test_emoji_research.py` | Delete (research code, not tests) |
| `test_emoji_simple.py` | Delete (subset of comprehensive) |
| `test_prompt_command.py` | Move to `tests/` |
| `test_tag_architecture.py` | Move to `tests/` |
| `test_verbose.py` | Delete (manual testing) |
| `test_workspace.py` | Delete (one-off testing) |

### 1.5 Duplicate Test Directories

The project has tests in THREE locations:
- `test/` (30 files)
- `tests/` (55+ files)
- Root directory (19 files)

**Recommendation**: Consolidate ALL tests into `tests/` directory.

---

## 2. Design Issues Requiring Clarification

### 2.1 Architecture Decisions Needed

#### Q1: Which CLI Backend Import Pattern is Correct?

**Issue**: Inconsistent import paths in backend files.

```python
# cloud.py uses:
from ..base_cli import BaseCLI

# cloud_fixed.py uses:
from .base_cli import BaseCLI
```

**Question for User**: Which import structure is intended? This affects module organization.

#### Q2: Should Deprecated CLI Files Be Kept for Backward Compatibility?

**Current State**: `simple_cli.py`, `full_cli.py`, `enhanced_cli.py`, `local_cli.py` all just print redirect notices.

**Options**:
1. **Delete them** - Clean codebase, users must update scripts
2. **Make them actually redirect** - Import and call `modular_cli.main()`
3. **Keep as warnings** - Current behavior

**Question for User**: What is the expected backward compatibility period?

#### Q3: Multiple Configuration Adapters - Which is Primary?

**Issue**: `src/config/adapters/` has 5 adapter files:
- `base_adapter.py`
- `individual_env_adapter.py`
- `simple_provider_adapter.py`
- `simple_validator_adapter.py`
- `unified_provider_adapter.py`

Plus validators:
- `individual_env_validator.py`
- `simple_provider_validator.py`
- `simple_validator.py`
- `unified_provider_validator.py`

**Question for User**: 
- Which adapter/validator is the current recommended approach?
- Should deprecated adapters be removed or kept for migration?

#### Q4: Duplicate Tools Directories

**Issue**: Two `tools/` directories exist:
- `tools/` (root) - Contains claim_tools, file_tools, interaction_tools, web tools
- `src/tools/` - Contains registry, ingest_examples

**Question for User**: What is the intended relationship between these directories?

#### Q5: Multiple Emoji/Symbol Utilities

**Issue**: `src/utils/` has 4 files for emoji/symbol handling:
- `emoji_support.py`
- `rich_emoji_support.py`
- `terminal_emoji.py`
- `symbols.py`

**Question for User**: Can these be consolidated into a single module?

### 2.2 Specification vs Implementation Gaps

Based on `specs/project.md` and `specs/requirements.md`:

| Specification | Implementation Status | Clarification Needed |
|---------------|----------------------|----------------------|
| Async Claim Evaluation Service | Partial (`src/processing/async_eval.py`) | Is this the intended implementation? |
| Claim Scopes (Global, Team, Project, User, Session) | Unknown | Are scopes fully implemented? |
| Dirty Flag System | Implemented (`src/core/dirty_flag.py`) | Complete or needs work? |
| Event System | Unknown | Where is the event emission implemented? |
| FileLock Tool | Referenced in spec | Is this implemented? |
| ClaimSupport Tool | Referenced in spec | Is this implemented? |
| Session Management | Referenced in spec | Implementation location? |
| TUI Interface | Design only (`src/interfaces/tui_design.md`) | Timeline for implementation? |
| GUI Interface | Design only (`src/interfaces/simple_gui.py`, `simple_tui.py`) | Are these functional? |
| MCP Interface | Referenced in spec | Implementation status? |

### 2.3 Provider Strategy Questions

**Issue**: Multiple LLM provider integrations exist in `src/processing/llm/`:
- `anthropic_integration.py`
- `chutes_integration.py`
- `cohere_integration.py`
- `gemini_integration.py`
- `google_integration.py`
- `groq_integration.py`
- `openai_integration.py`
- `openrouter_integration.py`
- `lm_studio_adapter.py`
- `local_providers_adapter.py`

**Questions**:
1. Are all these actively maintained?
2. Which providers are considered primary vs experimental?
3. The strategic doc mentions "Chutes.ai response format adaptation needed" - is this still an issue?

---

## 3. Incomplete Code Sections

### 3.1 Interface Implementations

| File | Status | What's Needed |
|------|--------|---------------|
| `src/interfaces/simple_gui.py` | Unknown | Needs review - is this functional? |
| `src/interfaces/simple_tui.py` | Unknown | Needs review - is this functional? |
| `src/interfaces/llm_interface.py` | Unknown | Needs review |
| `src/interfaces/tui_design.md` | Design only | Implementation pending |

### 3.2 Processing Components

| File | Component | Status |
|------|-----------|--------|
| `src/processing/async_eval.py` | Async Claim Evaluation | Needs verification against spec |
| `src/processing/tool_creator.py` | Dynamic tool creation | Unknown completeness |
| `src/processing/exploration_engine.py` | Knowledge exploration | Unknown completeness |
| `src/processing/example_generator.py` | Example generation | Unknown completeness |

### 3.3 Agent Harness

The `src/processing/agent_harness/` directory exists but unclear if complete:
- `error_handler.py`
- `models.py`
- `session_manager.py`
- `state_tracker.py`
- `workflow_engine.py`

**Question**: Is the agent harness functional or a work-in-progress?

### 3.4 Context System

| File | Purpose | Status |
|------|---------|--------|
| `src/context/complete_context_builder.py` | Context building from claims | Needs verification |
| `src/processing/context_collector.py` | Context collection | Needs verification |
| `src/processing/support_systems/context_builder.py` | Support context | Needs verification |

**Note**: Multiple context-related files may have overlapping responsibilities.

### 3.5 Skills System

The `skills/` directory contains Python files that appear to be skill definitions:
- `coding_principles.py`
- `research_coding_projects.py`
- `skill_creation.py`
- `tool_creation.py`

**Question**: Are these integrated into the system or standalone examples?

---

## 4. Test File Consolidation

### 4.1 Duplicate Test Files to Merge

| Test Area | Files to Consolidate | Keep |
|-----------|---------------------|------|
| Data Layer | `test/test_data_layer.py`, `tests/test_data_layer.py`, `tests/test_data_layer_complete.py` | `tests/test_data_layer.py` (merge all) |
| Dirty Flag | `tests/test_dirty_flag.py`, `tests/test_dirty_flag_integration.py`, `test/test_dirty_flag_standalone.py`, `test_dirty_propagation.py` | `tests/test_dirty_flag.py` (merge all) |
| Emoji | 5 files at root | `tests/test_emoji.py` (keep comprehensive) |
| Setup Wizard | `tests/test_setup_wizard.py`, `test/test_setup_wizard_simple.py` | `tests/test_setup_wizard.py` |
| SQLite Manager | `tests/test_sqlite_manager.py`, `tests/test_sqlite_manager_comprehensive.py` | Keep comprehensive |
| LLM Providers | 4 files across directories | `tests/test_llm_providers.py` (consolidate) |
| Core Tools | 4 files at root | `tests/test_core_tools.py` (consolidate) |

### 4.2 Test Files to Delete

| File | Reason |
|------|--------|
| `test/test_sample.py` | Only contains `assert True == True` |
| `tests/test_simple_functionality_fixed.py` | Duplicate with minor differences |
| `tests/test_tool_validator_simple.py` | Subset of standalone |
| Root emoji files (4) | Keep only comprehensive |
| Root core tools files (3) | Keep only complete |

### 4.3 Test Files to Archive

| Directory/File | Reason |
|----------------|--------|
| `tests/phase3/` | Historical phase tests - archive if phases complete |
| `tests/phase4/` | Historical phase tests - archive if phases complete |
| `tests/refined_architecture/` | Architecture evolution tests |
| `tests/test_rust_minesweeper.py` | Use-case test, may be historical |

---

## 5. Documentation Cleanup

### 5.1 Root-Level Markdown Files to Archive

| File | Recommendation |
|------|----------------|
| `CORE_TOOLS_IMPLEMENTATION_SUMMARY.md` | Move to `docs/implementation/` or `archive/` |
| `EMOJI_IMPLEMENTATION_SUMMARY.md` | Move to `docs/implementation/` or `archive/` |
| `EMOJI_PACKAGE_RESEARCH_SUMMARY.md` | Move to `archive/` (research complete) |
| `EMOJI_USAGE.md` | Keep (referenced in README) or move to `docs/` |
| `PROMPT_IMPLEMENTATION_PROGRESS.md` | Move to `docs/implementation/` or `archive/` |
| `PROMPT_IMPLEMENTATION_SUMMARY.md` | Move to `docs/implementation/` or `archive/` |
| `RECOMMENDATIONS_AND_ANALYSIS.md` | Move to `docs/` or `archive/` |
| `SIMPLIFICATION_SUMMARY.md` | Keep (referenced in README) or move to `docs/` |

### 5.2 Demo Files Review

The `demo/` directory has 15 files. Consider:
- Moving useful demos to `examples/`
- Archiving or deleting outdated demos
- Consolidating overlapping demos

### 5.3 README References to Fix

The README references:
- `src/engine.py` - **Does not exist**
- `src/tools.py` - **Does not exist**
- `src/config/config.py` - Exists

**Action**: Update README to reflect actual file structure.

---

## 6. Recommended Actions

### Priority 1: Immediate Cleanup (Low Risk)

```
[ ] Delete all *_broken.py files
[ ] Delete models_backup.py
[ ] Delete old_config.py
[ ] Delete base_cli_test.py
[ ] Consolidate *_fixed.py files (after determining correct import pattern)
```

### Priority 2: Test Consolidation (Medium Effort)

```
[ ] Move all root test_*.py files to tests/
[ ] Merge duplicate test files
[ ] Delete test/test_sample.py
[ ] Consolidate test/ into tests/ directory
```

### Priority 3: Design Decisions Required

```
[ ] Decide: Keep or delete redirect CLI files?
[ ] Decide: Primary configuration adapter?
[ ] Decide: Consolidate tools/ directories?
[ ] Decide: Consolidate emoji utilities?
```

### Priority 4: Code Completion Verification

```
[ ] Verify async_eval.py completeness vs spec
[ ] Verify interface implementations (GUI, TUI)
[ ] Verify agent harness completeness
[ ] Verify skills system integration
```

### Priority 5: Documentation Update

```
[ ] Update README with correct file paths
[ ] Archive implementation summary files
[ ] Consolidate or archive demo files
```

---

## Summary Statistics

| Category | Count | Action |
|----------|-------|--------|
| Files to delete (broken/backup) | 7 | Immediate |
| Files to delete (redirects) | 4 | After verification |
| Root test files to move | 19 | Move to tests/ |
| Duplicate test files to merge | ~15 | Consolidate |
| Design questions to resolve | 5 | User decision |
| Implementation areas to verify | 4 | Code review |
| Doc files to archive | 8 | Low priority |

**Estimated Cleanup Impact**: 
- ~30 fewer files at root level
- ~20 fewer duplicate test files
- Cleaner, more navigable codebase
- Reduced maintenance burden

---

## Notes for User

1. **Before deleting redirect CLIs**: Check if any scripts, documentation, or users depend on the old entry points.

2. **Import path decision**: The `cloud.py` vs `cloud_fixed.py` difference is a single import line. Test which works with your module structure.

3. **Test consolidation**: Consider running all tests before and after consolidation to ensure no coverage is lost.

4. **Archive vs Delete**: When in doubt, archive to `archive/` rather than delete. The archive already has good organization.

5. **Specification alignment**: The `specs/project.md` describes a sophisticated system. Verify how much is implemented vs planned.
