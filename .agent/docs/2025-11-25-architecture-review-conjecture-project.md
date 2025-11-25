## Honest Architecture Review: Conjecture

### Strengths

1. **Good Domain Modeling**: The `Claim` model with Pydantic is well-designed with proper validation, state management, and dirty flag tracking. The enum-based states (`ClaimState`, `ClaimType`) provide type safety.

2. **LLM Abstraction**: The `LLMBridge` pattern provides a clean interface between the application and providers. The fallback mechanism is thoughtful.

3. **Modular CLI**: The backend-agnostic CLI design with pluggable backends (`auto`, `local`, `cloud`, `hybrid`) is a good separation of concerns.

---

### Critical Issues

**1. Massive Code Duplication / Multiple Implementations**
- There are **5+ CLI files**: `cli.py`, `simple_cli.py`, `local_cli.py`, `enhanced_cli.py`, `full_cli.py`, plus `src/cli/modular_cli.py`
- There are **two `Conjecture` classes**: one in `src/conjecture.py` (sync), one in `src/core.py` (async) with overlapping functionality
- Multiple model files: `models.py`, `basic_models.py`, `models_backup.py`
- This creates confusion about which is canonical and increases maintenance burden

**2. Inconsistent Async/Sync Design**
- `src/conjecture.py` is synchronous, `src/core.py` is async
- Some components mix sync/async patterns inconsistently
- The `_explore_with_llm` method is sync but could block on network I/O

**3. Broken Imports & Dead Code**
- `src/core.py` imports from non-existent paths (`processing.llm_bridge`, `processing.context`, `data.data_manager`)
- Many files reference modules that have been deleted (see git status: deleted `src/engine.py`, `src/tools.py`, `src/data.py`)
- Config has undefined attributes (`exploration_batch_size`, `embedding_model`, `llm_model`, `llm_api_url`)

**4. Over-Engineered LLM Layer**
- 12+ LLM integration files for individual providers when a single adapter pattern would suffice
- `LLMManager` duplicates functionality of `LLMBridge`
- Provider initialization logic is scattered across multiple files

**5. Config Chaos**
- 7 config files in `src/config/` plus 4 adapter files
- `simple_config.py` references attributes it doesn't define (`embedding_model`, `llm_model`)
- `to_dict()` method uses undefined properties
- Monkey-patching methods onto `Config` class is an anti-pattern

**6. No Clear Data Layer**
- README mentions SQLite and ChromaDB but the actual data layer implementation is unclear
- `data/data_manager.py` appears to be missing or unreferenced
- Embedding storage strategy is undefined

---

### Actionable Recommendations

**Immediate (Fix Broken Code)**
1. Delete redundant CLI files - keep only `src/cli/modular_cli.py`
2. Fix or remove `src/core.py` (it imports deleted modules)
3. Fix `Config.to_dict()` to only reference existing attributes
4. Clean up git working tree - commit or discard the 50+ uncommitted changes

**Short-Term (Architecture Cleanup)**
1. Choose ONE `Conjecture` class - recommend async version in `src/conjecture.py` with proper async/await
2. Consolidate LLM providers into a single adapter with a provider registry pattern (you already have this partly in `llm_manager.py`)
3. Create a single `Config` class with proper attribute definitions and remove monkey-patching
4. Implement a clear repository pattern for data access

**Medium-Term (Design Improvements)**
1. Define clear module boundaries:
   - `core/` - domain models only
   - `services/` - business logic (exploration, claim management)
   - `adapters/` - external integrations (LLM, storage)
   - `cli/` - one CLI implementation
2. Add dependency injection for testability
3. Create integration tests that verify actual LLM/storage interactions

**Testing Gap**
- 40+ test files but unclear test coverage
- No visible CI/CD configuration
- Tests may be testing deleted/deprecated code paths

---

### Summary

The project has good foundational ideas (claim-based reasoning, LLM abstraction, modular backends) but suffers from **accumulated technical debt** from multiple iterations and incomplete refactors. The immediate priority should be **consolidation** - pick canonical implementations and delete duplicates, then fix broken imports before adding new features.