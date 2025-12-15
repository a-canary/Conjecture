# TODO.md - Conjecture Project Tasks

**Intent**: Concise inventory of task concepts for project development and maintenance.

## High Priority Tasks

### Testing & Quality
- [ ] Fix async test failures (need pytest-asyncio plugin)
- [ ] Improve test coverage from 7.66% to at least 50%
- [ ] Fix code parsing issues in coverage reports
- [ ] Resolve remaining Pydantic deprecation warnings

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