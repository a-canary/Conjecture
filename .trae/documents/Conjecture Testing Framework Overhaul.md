# Conjecture Testing Framework Overhaul Plan (No Mocking)

## Project Analysis Summary
**Conjecture** is an AI-powered evidence-based reasoning system with the following core components:
- **Core Models**: Claim-based reasoning system with Pydantic models (`src/core/models.py`)
- **Claim Operations**: Pure functions for claim manipulation (`src/core/claim_operations.py`)
- **Configuration**: Unified config system with Pydantic settings (`src/config/unified_config.py`)
- **Processing**: LLM bridge and unified processing layer (`src/processing/unified_bridge.py`)
- **Main Interface**: Main Conjecture class (`src/conjecture.py`)

## Current Test Situation
- **200+ existing test files** scattered across multiple directories
- Tests located in `tests/`, `research/`, `src/testing_optimization/`, and root level
- Many outdated, duplicate, or broken tests
- Comprehensive pytest configuration already in place

## Implementation Plan

### Phase 1: Clean Slate (Delete All Tests)
1. **Remove all test files** including:
   - `tests/test_*.py` (all test files)
   - `research/test_*.py` 
   - `src/testing_optimization/test_*.py`
   - Root level `test_*.py` files
   - All `__pycache__` directories

### Phase 2: Core Component Testing (20 Simple & Fast Unit Tests)
Create focused unit tests using **real components** (no mocking):

**Models & Data Layer (8 tests):**
1. Claim model validation and creation
2. Claim state transitions and dirty flag operations
3. Claim relationship integrity
4. Claim filtering and search operations
5. Claim confidence calculations
6. Pydantic model validation edge cases
7. Claim batch operations
8. Data model serialization/deserialization

**Configuration System (4 tests):**
9. Unified config initialization and defaults
10. Configuration validation and error handling
11. Provider configuration management
12. Settings model validation

**Processing Layer (5 tests):**
13. LLM bridge request/response handling (with real local providers)
14. Claim processing operations
15. Context building and optimization
16. Tool creation and execution
17. Error handling in processing pipeline

**Utilities & Helpers (3 tests):**
18. ID generation and validation
19. Retry utilities and timeout handling
20. Logging and monitoring utilities

### Phase 3: End-to-End Testing (3 E2E Tests)
Create comprehensive integration tests with **real components**:

1. **Complete Claim Lifecycle Test**
   - Create claim → Process → Evaluate → Update relationships
   - Test dirty flag propagation and re-evaluation
   - Use real configuration and processing components

2. **Multi-Claim Reasoning Test**
   - Create claim network with relationships
   - Test batch processing and confidence propagation
   - Validate hierarchical reasoning with real data layer

3. **Configuration-Driven Processing Test**
   - Test with different provider configurations (local providers)
   - Validate error handling and fallback mechanisms
   - Test performance under different settings with real components

### Phase 4: Test Execution & Error Resolution
1. **Run test suite** with coverage reporting
2. **Fix any import or dependency issues**
3. **Resolve test failures** through iterative debugging
4. **Validate coverage targets** (aiming for 80%+ core coverage)

## Technical Approach (No Mocking)
- **Use pytest** with existing configuration
- **Real components only** - no mocking of dependencies
- **Local/test configurations** to avoid external dependencies
- **In-memory databases** where possible for speed
- **Real LLM providers** configured for local testing
- **Focus on fast execution** (target < 2 seconds per test)
- **Comprehensive fixtures** for common test scenarios

## Expected Outcomes
- **Clean, maintainable test suite** with 23 focused tests
- **Fast execution** for rapid development feedback
- **High coverage** of critical code paths
- **Reliable E2E tests** with real component integration
- **Foundation for future test expansion**
- **No mocking complexity** - tests reflect real system behavior

This plan will transform the current bloated test suite into a focused, fast, and reliable testing framework that uses real components to validate Conjecture's evidence-based reasoning system.