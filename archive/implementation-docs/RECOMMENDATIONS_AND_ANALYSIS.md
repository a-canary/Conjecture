# Conjecture Project Analysis and Recommendations

## 1. Executive Summary
The 'Conjecture' project is an ambitious AI reasoning system designed to build a structured knowledge graph of "Claims" and leverage them for evidence-based reasoning. The long-term vision includes dynamic tool creation, complex claim scoping, and asynchronous evaluation. However, the current implementation is a "Phase 1" prototype that relies on manual tool registration and a simplified data model.

Recent work has focused on bridging this gap by refactoring the core data models to be more flexible (replacing rigid Enums with Tags) and treating tool usage examples as first-class data (Claims) rather than code comments.

## 2. Architecture Gap Analysis

| Feature | Intended Design (Vision) | Current Implementation (Reality) | Status |
| :--- | :--- | :--- | :--- |
| **Knowledge Model** | Unified "Claim" system for facts, skills, tools, and examples. | `Claim` model existed but used rigid `ClaimType` enum. | **Refactored** (Moved to Tags) |
| **Tooling** | Dynamic Python files created by LLM on the fly. | Manual `@register_tool` decorator in `registry.py`. | **Gap** (High Priority) |
| **Tool Examples** | "Sample-Claims" stored in DB, retrievable for context. | Hardcoded strings in Python `examples()` functions. | **In Progress** (Ingestion script ready) |
| **Context** | Intelligent retrieval of Skills/Samples based on task. | Basic vector search or manual context building. | **Gap** |
| **Scoping** | Session/User/Project/Global scopes for knowledge sharing. | Not implemented. | **Future** |

## 3. Current Progress & Status

### ✅ Completed Actions
*   **Data Model Flexibilty:** Removed the restrictive `ClaimType` enum from `src/core/models.py` and all dependent files. The system now uses a flexible `tags` system (e.g., `tags=["tool", "webSearch", "skill"]`), allowing for arbitrary categorization without code changes.
*   **Codebase Refactoring:** Updated `src/conjecture.py`, `src/processing/async_eval.py`, `src/data/data_manager.py`, `src/data/repositories.py`, and `src/processing/tool_manager.py` to support the tag-based architecture.
*   **Tool Ingestion Utility:** Created `src/tools/ingest_examples.py`. This script uses introspection to extract `examples()` from existing tools and converts them into standard Claims, effectively unlocking this "hidden knowledge" for the LLM.

### 🚧 In Progress / Pending
*   **Ingestion Execution:** The `ingest_examples.py` script is ready to run but was paused to resolve import errors (now fixed).
*   **Test Suite Updates:** Existing tests rely on `ClaimType` and need to be updated to assert on `tags` instead.

## 4. Recommendations & Roadmap

### Immediate Priority: Consolidate the "Universal Claim" Model
1.  **Finalize Ingestion:** Run `src/tools/ingest_examples.py` to populate the database with tool usage examples.
2.  **Fix Tests:** Update the test suite to reflect the removal of `ClaimType`.
3.  **Verify Data Layer:** Ensure `DataManager` correctly indexes and retrieves claims based on tags.

### Next Steps: Dynamic Tooling (Phase 2)
The most significant architectural improvement is enabling the "Dynamic Tooling" vision.

4.  **File-Based Tool Loading:** Refactor `ToolRegistry` to scan a `tools/` directory for `.py` files dynamically, rather than relying on import-time decorators. This allows tools to be added/modified at runtime.
5.  **Tool Creator Agent:** Implement the `ToolCreator` logic (partially drafted in `tool_manager.py`) to allow the LLM to write valid Python code into the `tools/` directory.
6.  **Skill Generation:** When a new tool is created, automatically generate a "Skill-Claim" (how-to guide) and "Sample-Claim" (example usage) so the system knows how to use it immediately.

### Future: Context & Evaluation (Phase 3)
7.  **Context Collector:** Enhance the retrieval logic to specifically look for "Skill-Claims" and "Sample-Claims" when the LLM is attempting a task, building a "Just-In-Time" manual for the agent.
8.  **Async Evaluation:** Fully enable the background service that continuously verifies claims and updates confidence scores.

## 5. Conclusion
The project is moving in the right direction. By flattening the data model (everything is a Claim with tags) and ingesting tool examples as data, we have laid the foundation for the self-improving system described in the design docs. The next critical leap is the **Dynamic Tooling** system, which will transition Conjecture from a static tool user to a tool creator.
