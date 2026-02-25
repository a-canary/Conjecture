# Project Analysis

## Metrics (2026-02-25)

| Metric | Value |
|--------|-------|
| src_lines | 25,762 |
| test_lines | 9,095 |
| src_files | 86 |
| test_files | 18 |
| tests_collected | 385 |
| tests_passed | 331 |
| tests_skipped | 43 |
| tests_xfailed | 8 |
| tests_errors | 3 |
| test_pass_rate | 85.97% |
| code_coverage | 18.31% |
| deprecation_warnings | 60 |
| commits_ahead_origin | 28 |

## Summary

Test infrastructure restored after creating stub modules (`lancedb_adapter.py`, `data_manager.py`, `optimized_sqlite_manager.py`) enabling collection of all 385 tests. Coverage at 18.31% exceeds 15% target from backlog item 152. Main concern is 28 unpushed commits and 3 test errors from unimplemented `OptimizedSQLiteManager`.

## Improvements This Cycle

- Fixed 5 test collection errors by creating minimal stubs for missing modules
- All 385 tests now collect (up from 295 with 5 errors)
- 331 tests passing with 85.97% pass rate

## Concerns

- `datetime.utcnow()` deprecation warnings (60 total)
- 3 e2e tests error due to unimplemented `OptimizedSQLiteManager.initialize()`
- 28 commits not pushed to origin
