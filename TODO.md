# TODO.md - Conjecture Project Tasks

**Intent**: Concise inventory of task concepts for project development and maintenance.

## High Priority Tasks

### Testing & Quality
- [x] Fix async test failures (pytest-asyncio configured properly) - Cycle 29
- [x] Add missing batch operations to OptimizedSQLiteManager - Cycle 29
- [x] Fix ID utilities tests (14 failures) - Cycle 3
- [x] Fix monitoring utilities tests - Cycle 3
- [x] Achieve 100% test pass rate for core+utility tests - Cycle 3
- [x] Create comprehensive test suite for claim_operations.py (97.48% coverage) - Cycle 4
- [ ] Improve test coverage from 7.33% to at least 50% (currently 7.33%, next target: 10%)
- [ ] Create test suites for relationship_manager.py, support_relationship_manager.py, dirty_flag.py
- [ ] Fix timeout issues in e2e tests requiring real LLM connections
- [ ] Resolve remaining Pydantic deprecation warnings (external dependencies)

### Code Quality
- [ ] Clean up unparsable Python files causing coverage warnings
- [ ] Remove dead code and deprecated files
- [ ] Standardize import statements across modules
- [ ] Add type hints to core modules

### Documentation
- [ ] Update README.md with current project status
- [ ] Document API endpoints and interfaces
- [ ] Create developer setup guide
- [ ] Add inline documentation for complex algorithms

### Performance
- [ ] Optimize database query performance
- [ ] Improve LLM response caching
- [ ] Reduce memory usage in context building
- [ ] Optimize import loading times

### Infrastructure
- [ ] Set up CI/CD pipeline
- [ ] Configure automated testing
- [ ] Implement logging and monitoring
- [ ] Set up development environment containers

## Medium Priority Tasks

### Features
- [ ] Enhance claim validation logic
- [ ] Improve context optimization algorithms
- [ ] Add support for additional LLM providers
- [ ] Implement batch processing capabilities

### Integration
- [ ] Improve external API integrations
- [ ] Add webhook support
- [ ] Implement export/import functionality
- [ ] Add plugin system for extensions

## Low Priority Tasks

### Maintenance
- [ ] Archive old benchmark results
- [ ] Clean up temporary files and caches
- [ ] Update dependencies to latest versions
- [ ] Refactor legacy code modules

### Research
- [ ] Evaluate new LLM models
- [ ] Research vector database alternatives
- [ ] Investigate performance profiling tools
- [ ] Explore deployment options

## Completed Tasks

### Cycle 4 (2025-12-17) - Coverage Improvement Sprint
- [x] Created comprehensive test suite for src/core/claim_operations.py
- [x] Achieved 97.48% coverage for claim_operations.py module
- [x] Improved overall coverage from 7.15% to 7.33% (+0.18%)
- [x] Added 47 tests covering all claim manipulation functions
- [x] Identified broken code path in update_claim_with_dirty_propagation()
- [x] Documented coverage gaps: relationship_manager, support_relationship_manager, dirty_flag

### Cycle 3 (2025-12-17) - Utility Test Fixes
- [x] Fixed 14 failing utility tests (ID utilities + monitoring utilities)
- [x] Achieved 100% pass rate for all core+utility tests (131+47 = 178 tests)
- [x] Updated ID format tests for new c{timestamp}_{uuid} format
- [x] Fixed PerformanceMonitor API method calls (start_timer, end_timer, etc.)
- [x] Fixed logger tests to use setup_logger instead of get_logger

### Cycle 29 (2025-12-17)
- [x] Added batch_create_claims method to OptimizedSQLiteManager
- [x] Added batch_update_claims method to OptimizedSQLiteManager
- [x] Fixed test_batch_processing_workflow test
- [x] Achieved 117/131 core unit tests passing (89.3% pass rate)
- [x] Fixed database batch operations for claim lifecycle

### Cycle 24 (2025-12-14)
- [x] Add Pydantic v2 compatibility with protected_namespaces configuration
- [x] Update 9 model classes with proper Pydantic v2 configuration
- [x] Address deprecation warnings about model_ namespace conflicts

### Cycle 23 (2025-12-14)
- [x] Fix database column mismatch resolution
- [x] Resolve SQLite INSERT statement placeholder count issues
- [x] Restore basic claim creation functionality

## Known Issues

- Async tests failing due to missing pytest-asyncio plugin
- Low test coverage (7.66%) needs improvement
- Some Python files have parsing issues affecting coverage
- Pydantic warnings persist in some external modules
- Benchmark results directory needs organization

## Next Steps

1. Fix async test issues to improve test reliability
2. Increase test coverage for better code quality assurance
3. Clean up code parsing warnings
4. Establish regular maintenance cycles
5. Improve documentation for new contributors