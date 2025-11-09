# Conjecture Data Layer Test Suite

This comprehensive test suite provides complete coverage for the Conjecture data layer implementation, ensuring robustness, performance, and reliability.

## 📋 Test Overview

The test suite is organized into multiple categories covering all aspects of the data layer:

### 🏗️ **Core Components**
- **Model Tests** (`test_models.py`) - Pydantic model validation and data integrity
- **SQLite Manager Tests** (`test_sqlite_manager.py`) - CRUD operations and relationships
- **Chroma Manager Tests** (`test_chroma_manager.py`) - Vector operations and similarity search
- **Embedding Service Tests** (`test_embedding_service.py`) - Text embedding generation
- **DataManager Integration** (`test_data_manager_integration.py`) - End-to-end workflows

### 🚀 **Performance & Quality**
- **Performance Tests** (`test_performance.py`) - Latency, scalability, and throughput benchmarks
- **Error Handling** (`test_error_handling.py`) - Edge cases, validation, and failure scenarios
- **Test Utilities** (`test_utilities.py`) - Data generators, scenarios, and assertions

## 🎯 Test Categories

### Unit Tests
- Individual component testing
- Fast execution (<1 second)
- Mock dependencies for isolation
- 100% functional coverage

### Integration Tests  
- Component interaction testing
- End-to-end workflow validation
- Medium execution time (<10 seconds)
- Real database operations

### Performance Tests
- Benchmark compliance checks
- Latency requirements validation
- Scalability testing up to 10,000+ claims
- Resource usage monitoring

### Error Handling Tests
- Input validation edge cases
- Database error scenarios
- Concurrent operation conflicts
- Recovery and resilience testing

## 📊 Performance Requirements

The test suite validates these performance benchmarks:

| Operation | Target | Measured By |
|-----------|--------|-------------|
| Simple claim retrieval | <10ms | `test_performance.py` |
| Similarity search | <100ms | `test_chroma_manager.py` |
| Batch creation (100 claims) | <1s | `test_data_manager_integration.py` |
| Filter queries | <50ms | `test_sqlite_manager.py` |
| Concurrency throughput | 20+ ops/sec | `test_performance.py` |
| Memory per claim | <0.1MB | `test_performance.py` |

## 🧪 Running Tests

### Basic Execution
```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py

# Run specific test class
pytest tests/test_models.py::TestClaimModel

# Run specific test method
pytest tests/test_models.py::TestClaimModel::test_valid_claim_creation
```

### Test Categories
```bash
# Run only unit tests
pytest tests/ -m unit

# Run only integration tests
pytest tests/ -m integration

# Run only performance tests
pytest tests/ -m performance

# Run only error handling tests
pytest tests/ -m error_handling

# Skip slow tests
pytest tests/ -m "not slow"
```

### Coverage Reporting
```bash
# Run tests with coverage
pytest tests/ --cov=src/data --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=src/data --cov-report=html

# Coverage with minimum threshold
pytest tests/ --cov=src/data --cov-fail-under=80
```

### Parallel Execution
```bash
# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Run with specific number of workers
pytest tests/ -n 4
```

### Performance Benchmarking
```bash
# Run performance tests with benchmarks
pytest tests/ -m performance --benchmark-only

# Save benchmark results
pytest tests/ -m performance --benchmark-autosave

# Compare with previous benchmarks
pytest tests/ -m performance --benchmark-compare
```

## 🔧 Configuration

### pytest.ini
The primary configuration file contains:
- Test discovery patterns
- Markers for categorization
- Async configuration
- Performance settings
- Coverage requirements

### conftest.py
Centralized fixtures provide:
- Database setup/teardown
- Mock embeddings service
- Sample data generation
- Performance measurement utilities
- Test isolation

## 📝 Test Markers

### Category Markers
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Component integration tests
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.asyncio` - Async test functions

### Component Markers
- `@pytest.mark.models` - Model validation tests
- `@pytest.mark.sqlite` - SQLite manager tests
- `@pytest.mark.chroma` - Chroma manager tests
- `@pytest.mark.embeddings` - Embedding service tests
- `@pytest.mark.data_manager` - DataManager tests

### Quality Markers
- `@pytest.mark.error_handling` - Error scenarios
- `@pytest.mark.edge_case` - Boundary conditions
- `@pytest.mark.security` - Security validation

## 📈 Test Data

### Sample Data Structure
```python
# Basic claim
{
    "id": "c0000001",
    "content": "The Earth orbits around the Sun",
    "confidence": 0.95,
    "tags": ["astronomy", "space"],
    "created_by": "scientist",
    "dirty": False
}
```

### Test Domains
- **Science** - Astronomy, physics, chemistry, biology
- **Technology** - AI, software, hardware, computing
- **Health** - Medicine, clinical research, public health
- **Environment** - Climate change, ecology, conservation
- **Social** - Policy, economics, social sciences

### Data Generators
- `TestDataGenerators` class for realistic claim generation
- Domain-specific content templates
- Configurable complexity levels
- Deterministic seeding for reproducible tests

## 🔍 Test Structure

### Test Class Organization
```python
class TestComponentName:
    """Test suite for specific component."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_specific_functionality(self):
        """Test specific functionality with descriptive name."""
        # Arrange
        # Act  
        # Assert
        pass
```

### Naming Conventions
- Test classes: `Test[ComponentName]`
- Test methods: `test_[functionality]_[scenario]`
- Fixtures: `[resource_name]` or `[component]_manager`
- Descriptive docstrings for all tests

### Assertion Guidelines
- Use clear, specific assertion messages
- Test both positive and negative cases
- Validate all public interface contracts
- Check error types and messages for failures

## 🚨 Common Issues

### Import Path Issues
If you encounter import errors:
```bash
# Add source directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or run from project root
python -m pytest tests/
```

### Database Cleanup
Tests use in-memory databases and temporary directories. If tests fail during cleanup:
```bash
# Remove temporary test files
rm -rf ./test_*
rm -rf ./data/test_*

# Clear ChromaDB caches
rm -rf ./chroma_*/logs/
```

### Embedding Model Issues
For real embedding tests (optional):
```bash
# Install sentence-transformers for real embeddings
pip install sentence-transformers torch

# Tests will use mock embeddings if not installed
```

### Async Test Issues
Ensure proper async configuration:
```bash
# Use pytest-asyncio plugin
pip install pytest-asyncio

# Async mode should be auto-configured in pytest.ini
```

## 📊 Coverage Metrics

Current test coverage targets:
- **Overall Coverage**: ≥80%
- **Critical Path Coverage**: ≥95%
- **Error Handling Coverage**: ≥90%
- **Performance Test Coverage**: 100%

### Coverage Reports
```bash
# View detailed coverage report
open htmlcov/index.html

# Check coverage by component
pytest tests/ --cov=src/data --cov-report=term --cov-report=html
```

## 🔧 Development Workflow

### Adding New Tests
1. Identify appropriate test category
2. Follow naming conventions
3. Use existing fixtures where possible
4. Add comprehensive assertions
5. Update documentation

### Test-Driven Development
1. Write failing test first
2. Implement minimal functionality
3. Refactor while maintaining green tests
4. Add additional test cases

### Continuous Integration
```yaml
# Example CI configuration
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r tests/requirements.txt
      - name: Run tests
        run: pytest tests/ --cov=src/data
```

## 📚 Best Practices

### Test Design
- **Arrange, Act, Assert** pattern
- **Single Responsibility** per test
- **Descriptive** test names and docstrings
- **Independent** test execution
- **Deterministic** test results

### Test Data Management
- Use **fixtures** for common setup
- **Factory patterns** for data generation
- **Isolated** test environments
- **Cleanup** in teardown phases

### Performance Testing
- **Baseline** measurements
- **Statistical** significance testing
- **Resource** monitoring
- **Regression** detection

### Error Testing
- **Boundary** conditions
- **Invalid** inputs
- **Resource** exhaustion
- **Concurrent** conflicts

## 🎯 Success Criteria

The test suite meets success criteria when:

### ✅ Functional Requirements
- All CRUD operations tested
- Error scenarios covered
- Edge cases validated
- Integration flows verified

### ✅ Performance Requirements  
- Latency benchmarks met
- Scalability targets achieved
- Resource usage controlled
- Concurrency handled properly

### ✅ Quality Requirements
- Coverage thresholds met
- Test reliability high
- Documentation complete
- Maintenance feasible

### ✅ Development Requirements
- Fast feedback loop
- Clear failure diagnostics
- Local execution possible
- CI integration ready

## 📞 Support

For test-related issues:
1. Check this README first
2. Review specific test file documentation
3. Examine fixture implementations
4. Check error messages and logs
5. Review pytest configuration

### Test Suite Statistics
- **Total Test Files**: 9
- **Test Classes**: 25+
- **Test Methods**: 200+
- **Performance Benchmarks**: 15+
- **Error Scenarios**: 50+
- **Test Coverage**: Target 80%+

This comprehensive test suite ensures the Conjecture data layer meets the highest standards of quality, performance, and reliability.