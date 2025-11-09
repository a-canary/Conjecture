# Skill-Based Agency Test Suite

Comprehensive test suite for the Phase 2 skill-based agency implementation, covering all components with unit tests, integration tests, security tests, performance tests, and edge case handling.

## Test Structure

```
tests/skill_agency/
├── conftest.py                 # Test fixtures and shared setup
├── pytest.ini                 # Pytest configuration
├── requirements.txt            # Test dependencies
├── run_tests.py               # Test runner script
├── README.md                   # This file
├── test_skill_models.py       # Skill model unit tests
├── test_skill_manager.py      # Skill manager unit tests
├── test_response_parser.py    # Response parser unit tests
├── test_tool_executor.py      # Tool executor unit tests
├── test_example_generator.py  # Example generator unit tests
├── test_integration.py        # End-to-end integration tests
├── test_security.py          # Security and validation tests
├── test_performance.py       # Performance benchmarks
├── test_edge_cases.py        # Edge case and error handling tests
```

## Test Categories

### 🧪 Unit Tests
- **test_skill_models.py**: Tests for all data models (SkillClaim, ExecutionResult, etc.)
- **test_skill_manager.py**: Tests for skill management and execution
- **test_response_parser.py**: Tests for LLM response parsing (XML, JSON, Markdown)
- **test_tool_executor.py**: Tests for safe code execution and security validation
- **test_example_generator.py**: Tests for automatic example generation

### 🔗 Integration Tests
- **test_integration.py**: End-to-end workflows testing component interactions
- Complete execution pipelines from LLM response to skill execution
- Tool discovery and example generation workflows
- Error recovery and resilience testing

### 🔒 Security Tests
- **test_security.py**: Security validation for code execution and input validation
- Code injection prevention tests
- Malicious input handling
- Resource limit enforcement
- XSS and injection attack prevention

### ⚡ Performance Tests
- **test_performance.py**: Performance benchmarks and regression tests
- Response parsing speed tests
- Execution time benchmarks
- Memory usage validation
- Concurrent execution performance

### 🎯 Edge Case Tests
- **test_edge_cases.py**: Error handling and edge case testing
- Malformed input handling
- Resource exhaustion scenarios
- Unicode and special character handling
- Concurrent stress testing

## Running Tests

### Quick Start
```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Runner Script
```bash
# Run all tests
python run_tests.py

# Run specific category
python run_tests.py -c unit
python run_tests.py -c integration
python run_tests.py -c security
python run_tests.py -c performance
python run_tests.py -c edge_case

# Run with coverage and HTML report
python run_tests.py --coverage --html

# Run smoke tests only
python run_tests.py --smoke

# Run in parallel (without coverage)
python run_tests.py --parallel

# Stop on first failure
python run_tests.py --failfast

# Comprehensive category-by-category run
python run_tests.py --comprehensive
```

### Pytest Commands
```bash
# Run unit tests only
pytest -m unit

# Run integration tests only
pytest -m integration

# Run security tests with verbose output
pytest -m security -vv

# Run performance tests with timeout
pytest -m performance --timeout=600

# Run specific test file
pytest test_skill_manager.py -v

# Run specific test method
pytest test_skill_manager.py::TestSkillManager::test_register_skill_claim_success -v

# Run with coverage
pytest --cov=src.processing --cov=src.core.skill_models --cov-report=term-missing

# Run with HTML coverage report
pytest --cov=src --cov-report=html --cov-report=xml
```

## Test Fixtures

The test suite provides comprehensive fixtures in `conftest.py`:

### Mock Objects
- `data_manager_mock`: Mocked DataManager for testing
- `mock_async_functions`: Mock built-in skill functions
- `skill_manager`, `response_parser`, `tool_executor`, `example_generator`: Component instances

### Test Data
- `sample_skill_claim`: Valid skill claim for testing
- `sample_execution_result`: Successful execution result
- `sample_tool_call`: Tool call for testing
- `xml_response_samples`: Various XML response formats
- `json_response_samples`: Various JSON response formats
- `markdown_response_samples`: Various Markdown response formats
- `code_execution_samples`: Safe and dangerous code snippets

### Configuration
- `execution_limits`: Custom resource limits for testing
- `performance_benchmarks`: Performance thresholds
- `error_scenarios`: Common error messages

## Test Coverage

### Success Criteria
- ✅ **>95% code coverage** for all components
- ✅ **All security validations pass**
- ✅ **Performance benchmarks met** (<100ms execution, <10ms parsing)
- ✅ **Comprehensive error handling**
- ✅ **Integration workflows functional**

### Coverage Targets
- `src/core/skill_models.py`: 100% coverage
- `src/processing/skill_manager.py`: >95% coverage
- `src/processing/response_parser.py`: >95% coverage
- `src/processing/tool_executor.py`: >95% coverage
- `src/processing/example_generator.py`: >95% coverage

## Performance Benchmarks

### Response Parsing
- XML parsing: <10ms per response
- JSON parsing: <10ms per response
- Markdown parsing: <10ms per response

### Skill Execution
- Execution time: <100ms per skill
- Memory usage: <50MB per execution
- Security validation: <5ms per code block

### System Performance
- End-to-end workflow: <150ms total
- Memory efficiency: <20MB increase for 50 skills + 20 executions
- Concurrent execution: Efficient scaling with load

## Security Validation

### Code Execution Security
- ❌ **Blocked**: `eval()`, `exec()`, `__import__()`, `open()`, `file()`
- ❌ **Blocked**: Dangerous modules (`os`, `sys`, `subprocess`, `socket`, etc.)
- ✅ **Allowed**: Safe modules (`math`, `json`, `datetime`, `collections`, etc.)

### Input Validation
- XML injection prevention
- JSON bomb protection
- Command injection blocking
- XSS prevention

### Resource Limits
- Execution timeout enforcement
- Memory usage limits
- Output size restrictions
- Network access blocking

## Continuous Integration

### GitHub Actions Integration
```yaml
- name: Run Unit Tests
  run: |
    python run_tests.py -c unit --failfast

- name: Run Integration Tests
  run: |
    python run_tests.py -c integration

- name: Run Security Tests
  run: |
    python run_tests.py -c security

- name: Run Performance Tests
  run: |
    python run_tests.py -c performance --timeout=600

- name: Generate Coverage Report
  run: |
    python run_tests.py --coverage --html
```

### Quality Gates
- All tests must pass
- Coverage >95% required
- No security regressions
- Performance benchmarks maintained
- No timeout failures

## Debugging Tests

### Running with Debug Output
```bash
# Verbose output with logs
pytest -v -s --log-cli-level=DEBUG

# Run specific failing test with full output
pytest test_skill_manager.py::failing_test -vv -s --tb=long

# Stop at first failure with traceback
pytest --failfast --tb=long
```

### Common Issues

1. **Import Errors**: Ensure `src/` is in Python path
2. **Async Test Errors**: Use `pytest.mark.asyncio` or `pytest-asyncio`
3. **Timeout Errors**: Increase timeout with `--timeout` flag
4. **Coverage Issues**: Exclude test files from coverage calculation

### Test Isolation
- Each test has isolated fixtures
- Database interactions are mocked
- File system access is restricted
- Network calls are prevented

## Contributing

### Adding New Tests
1. Follow existing naming conventions (`test_*` methods)
2. Use appropriate markers (`@pytest.mark.unit`, etc.)
3. Add comprehensive edge cases
4. Include performance benchmarks for new functionality
5. Add security tests for input validation

### Test Standards
- All async tests must use `@pytest.mark.asyncio`
- Tests should be deterministic (no external dependencies)
- Mock external systems (database, network)
- Include both positive and negative test cases
- Verify error conditions and edge cases

## License

This test suite is part of the Conjecture project and follows the same license terms.