# Dead Code Archive

**Archived**: 2026-03-01
**Reason**: Code cleanup - modules with broken imports or no active usage

## broken_imports/

Modules that cannot be imported due to missing dependencies (`src.processing.tool_registry` was deleted).

| File | Lines | Issue |
|------|-------|-------|
| agent_coordination.py | 335 | Imports non-existent tool_registry |
| data_flow.py | 334 | Imports non-existent tool_registry |
| llm_inference.py | 452 | Imports non-existent tool_registry |
| claim_operations.py | 337 | Only imported by broken modules |

**Note**: These files referenced a tool execution framework that was removed in commits:
- `651f0d4` ("Deep clean all subfolders")
- `59d433c` ("Aggressive cleanup - delete slop")

## unused_modules/

Modules that compile but are not imported anywhere in active code.

| File | Lines | Issue |
|------|-------|-------|
| relationship_manager.py | 383 | Not imported anywhere |
| support_relationship_manager.py | 496 | Not imported anywhere |

**Note**: These were superseded by the `dirty_flag.py` system.

## Recovery

To restore any module:
```bash
mv archive/dead_code/[category]/[file].py src/[original_path]/
```

Then fix any import errors and update `__init__.py` files as needed.
