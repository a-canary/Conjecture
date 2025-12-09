# TODO.md - Future Development Cycles

**Intent**: Concise inventory of future cycles' concepts and expectations for iterative development.

---

## ✅ COMPLETED

### [CRITICAL] Critical Syntax Error Resolution
**Status**: ✅ Completed (be82f3d - Fix critical syntax errors blocking static analysis tools)
**Hypothesis**: Fixing syntax errors in critical files will unblock static analysis and enable proper code evaluation
**Target**: Zero syntax errors blocking analysis tools
**Approach**: Fix syntax errors in 5 identified critical files, validate with static analysis tools
**Success Criteria**: All syntax errors resolved, static analysis tools run without blocking errors

### [HIGH] 4-Layer Migration Phase 4: Cleanup
**Status**: ✅ Completed (8f3df23 - Validate and commit Processing Interface implementation)
**Hypothesis**: Deleting legacy code and tests prevents regression and confusion
**Target**: Zero references to src/conjecture.py or inflated tests
**Approach**: Delete legacy files and fix remaining imports
**Success Criteria**: Clean build with only 4-layer architecture files

---

### [HIGH] Process-Presentation Layers API
**Hypothesis**: Clean separation between business logic and UI layers improves maintainability and testability
**Target**: Documented async API with clear layer boundaries
**Approach**: Refactor existing code to separate concerns and define async interfaces
**Success Criteria**: API documentation complete, layer separation achieved, existing functionality preserved

### [HIGH] EndPoint App Development
**Hypothesis**: Simple, transparent testing app enables rapid validation and debugging
**Target**: Functional EndPoint app for testing and validation
**Approach**: Build lightweight application with clear interfaces to core systems
**Success Criteria**: EndPoint app can test all major functions, provides clear feedback

### [HIGH] End-to-End Testing with Endpoint App
**Hypothesis**: Comprehensive E2E testing validates system reliability and performance
**Target**: Complete test suite covering ConjectureDB priming, recursive claims, and batch processing
**Approach**: Develop test scenarios using EndPoint app for all core workflows
**Success Criteria**: 95% test coverage, all critical paths validated, benchmark baseline established

### [HIGH] Test Suite Error Resolution
**Hypothesis**: Systematic fixes for remaining test errors will restore full test functionality
**Target**: All 761+ tests collecting and running without critical errors
**Approach**: Fix missing fixtures, import issues, and async context problems
**Success Criteria**: 95%+ test collection success rate, core test suites passing

### [HIGH] Cycle Regression Prevention
**Hypothesis**: Automated regression testing prevents issue recurrence across development cycles
**Target**: Zero regression issues in new development cycles
**Approach**: Implement comprehensive regression test suite with automated execution
**Success Criteria**: All known issues have regression tests, automated testing passes for 3 consecutive cycles

### [HIGH] Claim Replacement Tool
**Hypothesis**: Automated claim replacement preserves relations while transforming seeking-info to found-info claims
**Target**: Core tool supporting task→deliverable, query→response, question→answer, hypothesis→outcome transformations
**Approach**: Implement claim replacement logic with relation preservation and transformation mapping, similar to creating new claim and forcing merge with given claim to retain relations, OR by updating claim content/tags and marking dirty
**Success Criteria**: Tool handles all transformation types, relations preserved, 100% test coverage, supports both merge and dirty-update approaches

### [HIGH] File Indentation Fix Tool
**Hypothesis**: Automated indentation fixing improves code consistency and readability
**Target**: Tool for Python, JSON, MD files with configurable styles based on file type detection
**Approach**: Implement file type detection and style-based indentation correction with support for multiple file formats
**Success Criteria**: Supports 3+ file types, configurable styles, zero formatting errors, automatic file type detection

### [HIGH] LanceDB Integration and Project Simplification
**Hypothesis**: LanceDB provides superior vector storage and simplifies the project architecture compared to ChromaDB
**Target**: Evaluate LanceDB benefits and implement if advantageous for project simplification
**Approach**: Research LanceDB capabilities, benchmark against ChromaDB, and assess migration feasibility
**Success Criteria**: Clear benefit analysis documented, migration path defined if beneficial, performance improvements measured

### [MEDIUM] ConjectureDB Knowledge Foundation
**Hypothesis**: Extended ConjectureDB priming with robust knowledge foundation improves reasoning quality
**Target**: 25% improvement in reasoning accuracy through enhanced knowledge base
**Approach**: Systematically expand ConjectureDB with domain-specific foundational claims
**Success Criteria**: Measurable quality improvement across 3+ test domains, foundation claims validated

### [MEDIUM] Context Building Optimization
**Hypothesis**: "Use in every context" claims and deeper recursion improve reasoning depth
**Target**: 30% improvement in reasoning depth and context utilization
**Approach**: Implement context prioritization system and enhance recursion depth controls
**Success Criteria**: Context usage metrics show improvement, recursive reasoning depth increased

### [MEDIUM] Context Generation Enhancement
**Hypothesis**: Optimized context generation improves reasoning quality and efficiency
**Target**: 20% improvement in reasoning quality with maintained response times
**Approach**: Refine context building algorithms and implement smart context pruning
**Success Criteria**: Quality metrics improve, response times remain within 10% of baseline

### [MEDIUM] Process Layer Improvements
**Hypothesis**: Enhanced process layer architecture improves system modularity and extensibility
**Target**: Cleaner process layer with improved separation of concerns
**Approach**: Refactor process layer components and define clear interfaces
**Success Criteria**: Process layer components modularized, interfaces documented, existing functionality preserved

### [MEDIUM] Tool Calling Framework Testing
**Hypothesis**: Comprehensive validation ensures reliable tool calling system operation
**Target**: Full test coverage with robust error handling for tool calling framework
**Approach**: Develop comprehensive test suite covering all tool calling scenarios and edge cases, including integration tests for all core tools
**Success Criteria**: 95%+ test coverage, all error scenarios handled, automated validation passes, all core tools tested

### [MEDIUM] Claim Merging Logic Enhancement
**Hypothesis**: Improved claim merging with conflict resolution reduces duplication by 80%
**Target**: Enhanced merging algorithm preserving relations and resolving conflicts
**Approach**: Implement conflict detection, resolution strategies, and relation preservation, with testing and tweaking of merging logic
**Success Criteria**: 80% reduction in duplicates, relations preserved, conflict resolution accurate, comprehensive test coverage

### [MEDIUM] Adversarial Claim Generation
**Hypothesis**: Counter-claim generation for eval claims improves reasoning robustness
**Target**: System generating adversarial claims for >40% confidence eval claims
**Approach**: Implement evaluation technique to generate claims that disprove current eval claim when expected to have >40% confidence
**Success Criteria**: Generates counter-claims for all eval claims >40% confidence, improves reasoning quality, systematic disproval capability

### [MEDIUM] Refactor to Remove ClaimType and Use Only Tags
**Hypothesis**: Removing ClaimType enum and using only tags will simplify the data model and improve flexibility
**Target**: Eliminate ClaimType from the codebase and migrate all functionality to tag-based classification
**Approach**: Analyze current ClaimType usage, implement tag-based replacements, update data models and processing logic, migrate existing claims
**Success Criteria**: ClaimType enum removed, all functionality preserved through tags, migration script for existing data, 100% test coverage

### [LOW] CLI Revamp and Testing
**Hypothesis**: Modernized CLI improves user experience and reliability
**Target**: Updated CLI with comprehensive test coverage
**Approach**: Refactor CLI using modern patterns and add comprehensive tests
**Success Criteria**: CLI functionality complete, test coverage >90%, user feedback positive

### [LOW] WebUI Development
**Hypothesis**: Web interface expands accessibility and usability
**Target**: Functional WebUI for core Conjecture operations
**Approach**: Develop responsive web interface using modern framework
**Success Criteria**: WebUI supports core operations, cross-browser compatible, user-tested

### [LOW] TUI Revamp
**Hypothesis**: Enhanced terminal UI improves power-user experience
**Target**: Modernized TUI with improved navigation and features
**Approach**: Refactor TUI using modern terminal UI library
**Success Criteria**: TUI supports all CLI operations, improved UX, keyboard shortcuts documented

### [HIGH] 4-Layer Migration Phase 4: Cleanup
**Hypothesis**: Deleting legacy code and tests prevents regression and confusion
**Target**: Zero references to src/conjecture.py or inflated tests
**Approach**: Delete legacy files and fix remaining imports
**Success Criteria**: Clean build with only 4-layer architecture files

### [HIGH] Process Layer Enhancement - Tool Integration
**Hypothesis**: Integrating tool calling capabilities into Process Layer will enable complex claim evaluation workflows
**Target**: Process Layer can invoke tools during claim evaluation and instruction processing
**Approach**: Implement tool registry integration in ProcessLLMProcessor, add tool execution context
**Success Criteria**: Tools can be called from Process Layer, tool results incorporated in claim evaluation

### [HIGH] Process Layer Enhancement - Advanced Context Building
**Hypothesis**: Enhanced context building in Process Layer will improve reasoning quality and efficiency
**Target**: 25% improvement in context relevance and 15% reduction in processing time
**Approach**: Implement smart context prioritization, claim relevance scoring, and context compression
**Success Criteria**: Context relevance metrics improve, processing time reduced, no quality loss



### [MEDIUM] Process Layer Enhancement - Performance Optimization
**Hypothesis**: Optimizing Process Layer performance will improve overall system responsiveness
**Target**: 20% reduction in claim processing time with maintained quality
**Approach**: Implement async optimizations, caching, and batch processing
**Success Criteria**: Processing time reduced, quality maintained, no regressions

### [CRITICAL] Async Test Configuration Fix
**Status**: ✅ Completed (2025-12-09 - Fixed async test configuration in quick_discovery_test.py)
**Hypothesis**: Fixing async test configuration in quick_discovery_test.py will restore full test suite functionality
**Target**: All async tests properly configured with pytest-asyncio markers
**Approach**: Add @pytest.mark.asyncio decorator to async test functions
**Success Criteria**: Async tests collect and run without configuration errors

### [HIGH] Orphaned Module Import Cleanup
**Status**: ✅ Completed (2025-12-09 - Fixed all critical import errors blocking test suite)
**Hypothesis**: Removing orphaned module imports will eliminate test failures and improve system stability
**Target**: Zero import errors from deprecated modules
**Approach**: Remove imports for `local.unified_manager`, `cli.backends.hybrid_backend`, `src.config.adapters` and update dependent code
**Success Criteria**: All import errors resolved, tests pass, system stability improved



### [MEDIUM] Unicode Encoding Security Fix
**Hypothesis**: Resolving Unicode encoding issues will enable comprehensive security scanning
**Target**: Security scanning tools run without Unicode-related failures
**Approach**: Fix encoding issues in security scanning configuration and test files
**Success Criteria**: Security analysis completes successfully, no Unicode-related errors

### [MEDIUM] Test Infrastructure Stabilization
**Status**: ✅ Completed (2025-12-09 - Fixed 3 critical test infrastructure issues)
**Hypothesis**: Targeted fixes for critical test infrastructure issues will restore test suite functionality from 41.1% to 80%+ pass rate
**Target**: 80%+ test pass rate through infrastructure fixes
**Approach**: Fix async test configuration, Claim validation, and DeepEval API compatibility
**Success Criteria**: Test infrastructure stabilized, core functionality tests passing

### [MEDIUM] Endpoint Provider Management Test Fixes
**Hypothesis**: Fixing endpoint provider management test failures will restore API functionality testing
**Target**: All provider management tests passing with proper error handling
**Approach**: Investigate and fix test_set_strategy_missing_strategy and related test failures
**Success Criteria**: Provider management endpoint tests fully functional

### [LOW] Systematic Dead Code Removal
**Hypothesis**: Systematic removal of 87% orphaned code will significantly improve project maintainability and performance
**Target**: Remove 321 orphaned files while preserving core functionality
**Approach**: Execute dead code removal tools after test suite stabilization, validate functionality after removal
**Success Criteria**: 87% code reduction achieved, core functionality preserved, tests pass

### [HIGH] Batch Evaluation with Dependency Management
**Hypothesis**: Intelligent batching can evaluate dependent claims simultaneously while maintaining correct evaluation order
**Target**: Batch evaluation system that handles "A supports B" relationships efficiently
**Approach**: Implement dependency-aware batching that marks target claims clean before context building and supporting claims dirty on updates
**Success Criteria**: 30% improvement in evaluation throughput, correct dependency handling maintained, comprehensive test coverage

### [HIGH] Tool Calling State Management System
**Hypothesis**: Adding a "pending-tool-response" state prevents premature re-evaluation of claims awaiting tool responses
**Target**: Three-state claim system (clean/dirty/pending-tool-response) with proper state transitions
**Approach**: Extend ClaimState enum, implement state transition logic, add timeout handling for tool responses
**Success Criteria**: No premature evaluations, proper timeout handling, state transitions validated, 100% test coverage

### [HIGH] Parallel Tool Execution Framework
**Hypothesis**: Intelligent parallel execution of independent tool calls improves performance while maintaining dependency order
**Target**: Framework that can execute 3 web searches in parallel then summarize results
**Approach**: Implement dependency graph analysis, parallel execution queue, and result aggregation system
**Success Criteria**: Parallel execution for independent tools, proper sequencing for dependent operations, 40% performance improvement

### [MEDIUM] Tool Response Priority Queue System
**Hypothesis**: Prioritizing tool call responses over dirty claim evaluations improves system responsiveness
**Target**: Priority queue system that gives precedence to tool response processing
**Approach**: Implement dual-queue system with priority weighting, tool response buffering, and integrated claim re-evaluation
**Success Criteria**: Tool responses processed immediately, dirty claims handled in background, no resource conflicts

### [MEDIUM] Investigation: Current Claim State Implementation
**Hypothesis**: Understanding the current ClaimState implementation will inform the design of the new pending-tool-response state
**Target**: Complete analysis of existing claim state management and transition patterns
**Approach**: Examine src/core/models.py ClaimState enum, analyze state transition logic in processing layer, document current patterns
**Success Criteria**: Documentation of current implementation, identification of extension points, risk assessment for state addition

### [MEDIUM] Investigation: Tool Calling Integration Points
**Hypothesis**: Mapping tool calling integration points will reveal where state management and priority queuing should be implemented
**Target**: Complete map of tool calling flow through the 4-layer architecture
**Approach**: Trace tool calls from CLI through Endpoint to Process layer, identify buffering and queuing mechanisms, document integration points
**Success Criteria**: Flow diagram of tool calling, identified integration points for state management, priority queue placement recommendations

### [LOW] Investigation: Dependency Graph Analysis Requirements
**Hypothesis**: Understanding dependency analysis requirements will inform the batching system design
**Target**: Requirements analysis for dependency graph traversal and batching algorithms
**Approach**: Examine existing claim relationship structures, analyze batching scenarios, document algorithm requirements
**Success Criteria**: Requirements document for dependency analysis, batching algorithm specifications, edge case identification

### [LOW] Prototype: Tool Call State Transition Test
**Hypothesis**: A simple prototype will validate the three-state claim system approach
**Target**: Working prototype demonstrating clean→dirty→pending-tool-response→clean transitions
**Approach**: Create minimal test case showing state transitions during tool call execution, validate timeout handling
**Success Criteria**: Prototype code demonstrating state transitions, test cases covering all transitions, timeout validation