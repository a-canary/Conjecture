# TODO.md - Systematic Improvement Cycles

## **Current Focus: 100-Cycle Systematic Improvement**

### **Cycle 1 - Domain-Adaptive System Prompt [PROVEN ✓]**
**Hypothesis**: Problem type detection + specialized prompts will improve accuracy by matching reasoning approach to problem domain
**Target**: +15% accuracy on math problems, +10% on logic problems, reduce latency gap
**Result**: 100% improvement (exceeded 15% target by 85%)
**Files**: `src/agent/prompt_system.py` (updated), `src/benchmarking/improvement_cycle_agent.py`
**Status**: SUCCESS - Committed as f81324f
**Learning**: Domain-adaptive prompts significantly improve problem-solving accuracy

### **Cycle 2 - Enhanced Context Integration [PROVEN ✓]**
**Hypothesis**: Problem-type-specific context engineering (formulas, patterns, templates) will add +10% accuracy
**Target**: Additional +10% accuracy, better multi-step reasoning, mathematical scaffolding
**Result**: SUCCESS - Context integration implemented and validated
**Files**: `src/agent/prompt_system.py` (enhanced with `_get_context_for_problem_type`), `src/benchmarking/improvement_cycle_agent.py`
**Status**: SUCCESS - Committed successfully
**Learning**: Context scaffolding provides structured guidance for different problem domains

### **Cycle 3 - Self-Verification Enhancement [IMPLEMENTING]**
**Hypothesis**: Self-verification mechanisms will detect and correct errors, improving reliability by 70% error detection rate
**Target**: 70% error detection rate, 10-15% accuracy improvement, reduced user corrections
**Files**: `src/agent/prompt_system.py` (enhanced with verification), `src/benchmarking/improvement_cycle_agent.py`
**Status**: Implementation phase - adding self-verification and error correction
**Approach**: Build on Cycles 1-2 with reliability enhancement through self-checking

### **High Priority Concepts [UNTESTED]**
- Context optimization for mathematical reasoning
- Logic-focused prompt engineering
- Hybrid adaptive strategies
- Chain-of-thought structure enhancement

### **Proven Concepts [COMPLETED]**
- Database reset utility for clean benchmarking
- Fast prototype testing framework
- Baseline performance measurement
- Domain-adaptive system prompts (Cycle 1)
- Enhanced context integration (Cycle 2)

### **Failed Concepts [NONE YET]**
- *Will be populated as cycles progress*

---

## Legacy Items
**Hypothesis**: Context engineering and prompt refinement will boost GraniteTiny+Conjecture performance on SWEBench to ≥70%.
**Target**: ≥70% accuracy on SWEBench; comparable improvements on AIME2025 and LiveCodeBench v6.
**Approach**: 
- Set up SWEBench harness for GraniteTiny model.
- Run baseline evaluation.
- [ ] Set up SWEBench harness for GraniteTiny model
- [ ] Run baseline evaluation and record metrics
- [ ] Analyze failure modes and identify context engineering opportunities
- [ ] Refine prompts and context selection strategies
- [ ] Implement agent harness improvements (e.g., tool integration, state management)
- [ ] Re‑evaluate on SWEBench, aim for ≥70% accuracy
- [ ] Run secondary benchmarks (AIME2025, LiveCodeBench v6) and compare progress
- [ ] Document results and update RESULTS.md
- Iteratively refine prompts, context selection, and agent harness.
- Track metrics on AIME2025 and LiveCodeBench v6 as secondary benchmarks.
**Success Criteria**: Achieve ≥70% on SWEBench and maintain/improve scores on the other benchmarks.



**Intent**: Concise inventory of future cycles' concepts and expectations for iterative development. Completed cycles are moved to RESULTS.md

---

---
### [COMPLETED] Fix DataConfig Import Path Issues
**Hypothesis**: Correcting BatchResult import path will resolve test collection errors and improve test suite stability
**Target**: Zero import-related collection errors
**Approach**: Fixed BatchResult import in src/data/data_manager.py to import from src.core.common_results instead of src.data.models
**Success Criteria**: All tests collect successfully without import errors, test infrastructure stability improved

### [HIGH] Add Missing Test Fixtures
**Hypothesis**: Adding the missing `sample_claim_data` fixture to conftest.py will resolve test collection errors
**Target**: Zero fixture-related collection errors
**Approach**: Define `sample_claim_data` fixture in tests/conftest.py with proper claim structure
**Success Criteria**: All tests requiring the fixture collect successfully

### [MEDIUM] Update EmbeddingService Tests
**Hypothesis**: Updating EmbeddingService test interface will match the current API implementation
**Target**: Zero EmbeddingService interface mismatch errors
**Approach**: Review current EmbeddingService implementation and update test expectations
**Success Criteria**: All EmbeddingService tests pass with current API

### [MEDIUM] Sync RoutingStrategy Enum Values
**Hypothesis**: Adding the missing `LEAST_LOADED` value to RoutingStrategy enum will resolve test failures
**Target**: Zero RoutingStrategy enum mismatch errors
**Approach**: Update RoutingStrategy enum in src/core/models.py to include LEAST_LOADED value
**Success Criteria**: All routing strategy tests pass with complete enum values

### [MEDIUM] Fix Claim Field Validation
**Hypothesis**: Adding missing `confidence` field validation to Claim creation tests will resolve test failures
**Target**: Zero Claim field validation errors
**Approach**: Update Claim creation tests to include required confidence field with valid values
**Success Criteria**: All Claim creation tests pass with proper field validation

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

### [HIGH] DataConfig Model Fix
**Hypothesis**: Fixing the DataConfig model will resolve validation errors affecting 38 tests
**Target**: Zero DataConfig-related test failures
**Approach**: Update DataConfig model in src/config/unified_config.py to include missing attributes and proper validation
**Success Criteria**: All DataConfig tests pass, model validation errors eliminated

### [HIGH] Missing Imports and Model Classes
**Hypothesis**: Adding missing imports and model classes will resolve import errors across the test suite
**Target**: Zero import-related test failures
**Approach**: Identify and add missing imports, create missing model classes, update import statements
**Success Criteria**: All tests run without import errors, missing classes properly defined

### [HIGH] Pydantic v2 Migration
**Hypothesis**: Migrating from deprecated Pydantic v1 methods to v2 will resolve compatibility issues
**Target**: Zero Pydantic deprecation warnings or errors
**Approach**: Replace deprecated Pydantic v1 methods with v2 equivalents across the codebase
**Success Criteria**: All Pydantic-related tests pass, no deprecation warnings, v2 compatibility achieved

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

### [MEDIUM] Unicode Encoding Security Fix
**Hypothesis**: Resolving Unicode encoding issues will enable comprehensive security scanning
**Target**: Security scanning tools run without Unicode-related failures
**Approach**: Fix encoding issues in security scanning configuration and test files
**Success Criteria**: Security analysis completes successfully, no Unicode-related errors

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

### [MEDIUM] Extend ANALYSIS.md with Warning Metrics
**Hypothesis**: Adding warning count metrics to ANALYSIS.md will provide better visibility into code quality issues
**Target**: Warning metrics included in ANALYSIS.md with trend tracking
**Approach**: Analyze current warning generation, add warning count tracking to analysis metrics
**Success Criteria**: Warning metrics tracked, trends visible, baseline established

### [MEDIUM] Extend ANALYSIS.md with External Dependencies Metrics
**Hypothesis**: Tracking external dependencies in ANALYSIS.md will improve project maintainability awareness
**Target**: External dependency count and health metrics included in ANALYSIS.md
**Approach**: Inventory external dependencies, implement dependency tracking, add to analysis metrics
**Success Criteria**: Dependencies cataloged, health metrics tracked, security vulnerabilities monitored

### [MEDIUM] Extend ANALYSIS.md with Unknown Values Tracking
**Hypothesis**: Tracking '?' values and unknown data points in ANALYSIS.md will highlight areas needing investigation
**Target**: Unknown value metrics and investigation backlog included in ANALYSIS.md
**Approach**: Identify sources of '?' values, implement tracking system, add to analysis metrics
**Success Criteria**: Unknown values quantified, trends tracked, investigation priorities established

### [MEDIUM] Temporal Claim Management for Agent Workflows
**Hypothesis**: Adding temporal awareness to claims will improve the accuracy of time-sensitive information during agent workflows
**Target**: Lightweight system for reflecting the temporal nature of claims about project status, research, and other time-sensitive information
**Approach**: Experiment with temporal features including timestamps in context, temporal priority in claim merging, and flags for time-sensitive claims
**Success Criteria**: Timestamp system implemented, temporal claim merging working, time-sensitive claim flagging functional, measurable improvement in temporal accuracy during agent workflows
