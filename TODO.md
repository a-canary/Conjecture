# TODO.md - Future Development Cycles

**File Intent**: Concise inventory of future cycles' concepts and expectations for iterative development.

## Rules and Expectations

- One concept per cycle
- Focus on single, testable hypotheses
- Keep descriptions short and actionable
- Prioritize by impact and feasibility
- Update after each cycle completion

## Item Template

```
### [PRIORITY] Cycle Name
**Hypothesis**: [Brief testable statement]
**Target**: [Specific, measurable goal]
**Approach**: [Implementation strategy in 1-2 sentences]
**Success Criteria**: [How we measure success]
**Estimated Effort**: [Low/Medium/High]
```

## Future Cycles

### [HIGH] Process-Presentation Layers API
**Hypothesis**: Clean separation between business logic and UI layers improves maintainability and testability
**Target**: Documented async API with clear layer boundaries
**Approach**: Refactor existing code to separate concerns and define async interfaces
**Success Criteria**: API documentation complete, layer separation achieved, existing functionality preserved
**Estimated Effort**: High

### [HIGH] EndPoint App Development
**Hypothesis**: Simple, transparent testing app enables rapid validation and debugging
**Target**: Functional EndPoint app for testing and validation
**Approach**: Build lightweight application with clear interfaces to core systems
**Success Criteria**: EndPoint app can test all major functions, provides clear feedback
**Estimated Effort**: High

### [HIGH] End-to-End Testing with Endpoint App
**Hypothesis**: Comprehensive E2E testing validates system reliability and performance
**Target**: Complete test suite covering ConjectureDB priming, recursive claims, and batch processing
**Approach**: Develop test scenarios using EndPoint app for all core workflows
**Success Criteria**: 95% test coverage, all critical paths validated, benchmark baseline established
**Estimated Effort**: High

### [HIGH] Test Suite Error Resolution
**Hypothesis**: Systematic fixes for remaining test errors will restore full test functionality
**Target**: All 1,317+ tests collecting and running without critical errors
**Approach**: Fix missing fixtures, import issues, and async context problems
**Success Criteria**: 95%+ test collection success rate, core test suites passing
**Estimated Effort**: Medium

### [HIGH] Cycle Regression Prevention
**Hypothesis**: Automated regression testing prevents issue recurrence across development cycles
**Target**: Zero regression issues in new development cycles
**Approach**: Implement comprehensive regression test suite with automated execution
**Success Criteria**: All known issues have regression tests, automated testing passes for 3 consecutive cycles
**Estimated Effort**: Medium

### [HIGH] Claim Replacement Tool
**Hypothesis**: Automated claim replacement preserves relations while transforming seeking-info to found-info claims
**Target**: Core tool supporting task→deliverable, query→response, question→answer, hypothesis→outcome transformations
**Approach**: Implement claim replacement logic with relation preservation and transformation mapping, similar to creating new claim and forcing merge with given claim to retain relations, OR by updating claim content/tags and marking dirty
**Success Criteria**: Tool handles all transformation types, relations preserved, 100% test coverage, supports both merge and dirty-update approaches
**Estimated Effort**: High

### [HIGH] File Indentation Fix Tool
**Hypothesis**: Automated indentation fixing improves code consistency and readability
**Target**: Tool for Python, JSON, MD files with configurable styles based on file type detection
**Approach**: Implement file type detection and style-based indentation correction with support for multiple file formats
**Success Criteria**: Supports 3+ file types, configurable styles, zero formatting errors, automatic file type detection
**Estimated Effort**: High

### [HIGH] LanceDB Integration and Project Simplification
**Hypothesis**: LanceDB provides superior vector storage and simplifies the project architecture compared to ChromaDB
**Target**: Evaluate LanceDB benefits and implement if advantageous for project simplification
**Approach**: Research LanceDB capabilities, benchmark against ChromaDB, and assess migration feasibility
**Success Criteria**: Clear benefit analysis documented, migration path defined if beneficial, performance improvements measured
**Estimated Effort**: Medium

### [MEDIUM] ConjectureDB Knowledge Foundation
**Hypothesis**: Extended ConjectureDB priming with robust knowledge foundation improves reasoning quality
**Target**: 25% improvement in reasoning accuracy through enhanced knowledge base
**Approach**: Systematically expand ConjectureDB with domain-specific foundational claims
**Success Criteria**: Measurable quality improvement across 3+ test domains, foundation claims validated
**Estimated Effort**: Medium

### [MEDIUM] Context Building Optimization
**Hypothesis**: "Use in every context" claims and deeper recursion improve reasoning depth
**Target**: 30% improvement in reasoning depth and context utilization
**Approach**: Implement context prioritization system and enhance recursion depth controls
**Success Criteria**: Context usage metrics show improvement, recursive reasoning depth increased
**Estimated Effort**: Medium

### [MEDIUM] Context Generation Enhancement
**Hypothesis**: Optimized context generation improves reasoning quality and efficiency
**Target**: 20% improvement in reasoning quality with maintained response times
**Approach**: Refine context building algorithms and implement smart context pruning
**Success Criteria**: Quality metrics improve, response times remain within 10% of baseline
**Estimated Effort**: Medium

### [MEDIUM] Process Layer Improvements
**Hypothesis**: Enhanced process layer architecture improves system modularity and extensibility
**Target**: Cleaner process layer with improved separation of concerns
**Approach**: Refactor process layer components and define clear interfaces
**Success Criteria**: Process layer components modularized, interfaces documented, existing functionality preserved
**Estimated Effort**: Medium

### [MEDIUM] Tool Calling Framework Testing
**Hypothesis**: Comprehensive validation ensures reliable tool calling system operation
**Target**: Full test coverage with robust error handling for tool calling framework
**Approach**: Develop comprehensive test suite covering all tool calling scenarios and edge cases, including integration tests for all core tools
**Success Criteria**: 95%+ test coverage, all error scenarios handled, automated validation passes, all core tools tested
**Estimated Effort**: Medium

### [MEDIUM] Claim Merging Logic Enhancement
**Hypothesis**: Improved claim merging with conflict resolution reduces duplication by 80%
**Target**: Enhanced merging algorithm preserving relations and resolving conflicts
**Approach**: Implement conflict detection, resolution strategies, and relation preservation, with testing and tweaking of merging logic
**Success Criteria**: 80% reduction in duplicates, relations preserved, conflict resolution accurate, comprehensive test coverage
**Estimated Effort**: Medium

### [MEDIUM] Adversarial Claim Generation
**Hypothesis**: Counter-claim generation for eval claims improves reasoning robustness
**Target**: System generating adversarial claims for >40% confidence eval claims
**Approach**: Implement evaluation technique to generate claims that disprove current eval claim when expected to have >40% confidence
**Success Criteria**: Generates counter-claims for all eval claims >40% confidence, improves reasoning quality, systematic disproval capability
**Estimated Effort**: Medium

### [LOW] CLI Revamp and Testing
**Hypothesis**: Modernized CLI improves user experience and reliability
**Target**: Updated CLI with comprehensive test coverage
**Approach**: Refactor CLI using modern patterns and add comprehensive tests
**Success Criteria**: CLI functionality complete, test coverage >90%, user feedback positive
**Estimated Effort**: Medium

### [LOW] WebUI Development
**Hypothesis**: Web interface expands accessibility and usability
**Target**: Functional WebUI for core Conjecture operations
**Approach**: Develop responsive web interface using modern framework
**Success Criteria**: WebUI supports core operations, cross-browser compatible, user-tested
**Estimated Effort**: High

### [LOW] TUI Revamp
**Hypothesis**: Enhanced terminal UI improves power-user experience
**Target**: Modernized TUI with improved navigation and features
**Approach**: Refactor TUI using modern terminal UI library
**Success Criteria**: TUI supports all CLI operations, improved UX, keyboard shortcuts documented
**Estimated Effort**: Medium

### [HIGH] 4-Layer Migration Phase 1: Standards
**Hypothesis**: Formalizing the 4-layer architecture documentation creates a single source of truth for refactoring
**Target**: All specs and README aligned with specs/architecture.md
**Approach**: Update docs to remove legacy "Enhanced" claims and point to new spec
**Success Criteria**: No contradictory architecture claims in documentation
**Estimated Effort**: Low

### [HIGH] 4-Layer Migration Phase 2: Core Refactoring
**Hypothesis**: Splitting the God Class into Endpoint and Process layers improves modularity
**Target**: Functioning ConjectureEndpoint and ProcessLayer separated from legacy code
**Approach**: Create src/process and src/endpoint, migrate logic incrementally
**Success Criteria**: CLI calls ConjectureEndpoint successfully without src/conjecture.py
**Estimated Effort**: High

### [HIGH] 4-Layer Migration Phase 3: Cleanup
**Hypothesis**: Deleting legacy code and tests prevents regression and confusion
**Target**: Zero references to src/conjecture.py or inflated tests
**Approach**: Delete legacy files and fix remaining imports
**Success Criteria**: Clean build with only 4-layer architecture files
**Estimated Effort**: Medium