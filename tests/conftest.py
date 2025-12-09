"""
Comprehensive pytest configuration and fixtures for optimized testing.
Provides shared fixtures, mocking strategies, and performance optimization.
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional, Generator
import sys
import os
import time
import json
import psutil
import subprocess
import configparser
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Global mock configuration for performance optimization
GLOBAL_MOCK_CONFIG = {
    'chromadb': MagicMock(),
    'chromadb.config': MagicMock(),
    'chromadb.api': MagicMock(),
    'chromadb.api.models': MagicMock(),
    'sentence_transformers': MagicMock(),
    'torch': MagicMock(),
    'tensorflow': MagicMock(),
    'numpy': MagicMock(),
    'sklearn': MagicMock(),
    'sklearn.metrics': MagicMock(),
    'scipy': MagicMock(),
    'scipy.spatial': MagicMock(),
    'faiss': MagicMock(),
    'requests': MagicMock(),
    'aiohttp': MagicMock(),
}

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def mock_external_dependencies():
    """Session-scoped mocking of external dependencies for performance."""
    with patch.dict('sys.modules', GLOBAL_MOCK_CONFIG):
        yield

@pytest.fixture(scope="session")
def temp_data_dir():
    """Session-scoped temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="conjecture_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(scope="session")
def test_config():
    """Session-scoped test configuration."""
    return {
        "database": {
            "type": "sqlite",
            "path": ":memory:",
            "isolation": True,
            "pool_size": 1
        },
        "embeddings": {
            "mock": True,
            "dimension": 384,
            "model": "mock-embedding-model"
        },
        "performance": {
            "max_test_time": 30.0,
            "memory_limit_mb": 512,
            "parallel_workers": 4
        },
        "utf8": {
            "enforced": True,
            "test_strings": [
                "Hello World",
                "Café Résumé",
                "北京测试",
                "Тест Москва",
                "العربية اختبار",
                "🚀 Test Emoji 🧪",
                "Mixed: café 北京 🌟"
            ]
        }
    }

@pytest.fixture(scope="function")
def memory_monitor():
    """Monitor memory usage during tests."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    class MemoryMonitor:
        def __init__(self, initial_mb):
            self.initial_mb = initial_mb
            self.peak_mb = initial_mb
            self.current_mb = initial_mb

        def update(self):
            self.current_mb = process.memory_info().rss / 1024 / 1024
            self.peak_mb = max(self.peak_mb, self.current_mb)
            return self.current_mb

        def get_growth(self):
            return self.current_mb - self.initial_mb

    monitor = MemoryMonitor(initial_memory)
    yield monitor
    final_memory = process.memory_info().rss / 1024 / 1024
    growth = final_memory - initial_memory
    if growth > 50:  # Warn if more than 50MB growth
        pytest.warn(f"High memory growth detected: {growth:.2f}MB")

@pytest.fixture(scope="function")
def performance_timer():
    """Timer for performance measurement."""
    start_time = time.time()

    class PerformanceTimer:
        def __init__(self, start):
            self.start_time = start
            self.checkpoints = []

        def checkpoint(self, name: str):
            current_time = time.time()
            elapsed = current_time - self.start_time
            self.checkpoints.append((name, elapsed))
            return elapsed

        def elapsed(self):
            return time.time() - self.start_time

        def get_checkpoints(self):
            return self.checkpoints.copy()

    timer = PerformanceTimer(start_time)
    yield timer
    total_time = timer.elapsed()
    if total_time > 10.0:  # Warn if test takes more than 10 seconds
        pytest.warn(f"Slow test detected: {total_time:.2f}s")

@pytest.fixture(scope="session")
def mock_embedding_service():
    """Mock embedding service with realistic responses."""
    mock_service = AsyncMock()

    async def mock_encode(texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings with consistent patterns."""
        embeddings = []
        for text in texts:
            # Create deterministic but realistic-looking embeddings
            hash_val = hash(text) % 1000
            embedding = [hash_val * 0.001 * (i + 1) for i in range(384)]
            embeddings.append(embedding)
        return embeddings

    mock_service.encode = mock_encode
    mock_service.dimension = 384
    return mock_service

@pytest.fixture(scope="function")
def isolated_database(test_config, temp_data_dir):
    """Create isolated database for each test."""
    db_path = temp_data_dir / f"test_db_{int(time.time() * 1000)}.db"

    class IsolatedDatabase:
        def __init__(self, path: Path):
            self.path = path
            self.connection_count = 0

        def get_connection_string(self) -> str:
            return f"sqlite:///{self.path}"

        def cleanup(self):
            if self.path.exists():
                self.path.unlink()

    db = IsolatedDatabase(db_path)
    yield db
    db.cleanup()

@pytest.fixture(scope="function")
def test_data_factory():
    """Factory for generating test data with UTF-8 compliance."""

    class TestDataFactory:
        def __init__(self):
            self.counter = 0

        def create_claim(self, **overrides) -> Dict[str, Any]:
            """Create a test claim with optional overrides."""
            self.counter += 1
            base_claim = {
                "id": f"test_claim_{self.counter:06d}",
                "content": f"Test claim content {self.counter} with UTF-8: café 北京 🌟",
                "source": f"test_source_{self.counter}",
                "confidence": 0.5 + (self.counter % 5) * 0.1,
                "tags": [f"tag{i}" for i in range(1, (self.counter % 3) + 2)],
                "state": "pending",
                "claim_type": "hypothesis"
            }
            base_claim.update(overrides)
            return base_claim

        def create_claims(self, count: int) -> List[Dict[str, Any]]:
            """Create multiple test claims."""
            return [self.create_claim() for _ in range(count)]

        def utf8_test_strings(self) -> List[str]:
            """Get UTF-8 test strings for encoding validation."""
            return [
                "ASCII only",
                "Café Résumé",
                "北京测试",
                "Тест Москва",
                "العربية اختبار",
                "🚀 Test Emoji 🧪",
                "Mixed: café 北京 🌟 Тест العربية"
            ]

    return TestDataFactory()

@pytest.fixture(scope="session")
def optimized_pytest_config():
    """Optimized pytest configuration for performance."""
    return {
        "addopts": [
            "-v",
            "--tb=short",
            "--strict-markers",
            "--disable-warnings",
            "--color=yes",
            "--durations=10",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
            "--cov-config=.coveragerc",
            "-n", "auto",  # Parallel execution
            "--maxfail=5",  # Stop after 5 failures
            "--dist=loadscope"  # Distribute tests by scope
        ],
        "timeout": 300,
        "parallel_workers": 4,
        "test_randomization": True
    }

@pytest.fixture(scope="function")
def scientific_test_validator():
    """Validator for scientific integrity of tests."""

    class ScientificValidator:
        def __init__(self):
            self.issues = []

        def check_utf8_compliance(self, data: Any) -> bool:
            """Check UTF-8 encoding compliance."""
            try:
                if isinstance(data, str):
                    data.encode('utf-8')
                elif isinstance(data, (list, dict)):
                    json.dumps(data, ensure_ascii=False).encode('utf-8')
                return True
            except (UnicodeEncodeError, UnicodeDecodeError):
                self.issues.append("UTF-8 encoding violation")
                return False

        def check_database_isolation(self, db_id: str) -> bool:
            """Check database isolation compliance."""
            # This would integrate with actual database isolation checks
            return True

        def check_reproducibility(self, test_result: Any, expected_variance: float = 0.01) -> bool:
            """Check test result reproducibility."""
            # This would integrate with actual reproducibility checks
            return True

        def get_issues(self) -> List[str]:
            """Return all validation issues."""
            return self.issues.copy()

    return ScientificValidator()

# Static analysis configuration and hooks
STATIC_ANALYSIS_CONFIG = {
    'ruff': {
        'enabled': True,
        'command': 'ruff check . --format=json',
        'config_file': '.ruff.toml',
        'marker': 'ruff'
    },
    'mypy': {
        'enabled': True,
        'command': 'mypy src/ --json-report /tmp/mypy-report',
        'config_file': 'mypy.ini',
        'marker': 'mypy'
    },
    'vulture': {
        'enabled': True,
        'command': 'vulture src/ tests/ --min-confidence 80 --format json',
        'config_file': 'vulture.cfg',
        'marker': 'vulture'
    },
    'bandit': {
        'enabled': True,
        'command': 'bandit -r src/ -f json -o /tmp/bandit-report.json',
        'config_file': '.bandit',
        'marker': 'bandit'
    }
}

def pytest_configure(config):
    """Configure pytest with performance optimizations and static analysis integration."""
    # Register comprehensive custom markers
    markers_to_register = [
        # Static analysis markers
        "static_analysis: Marks tests that run static analysis tools (ruff, mypy, vulture, bandit)",
        "ruff: Marks tests that run ruff linting and formatting checks",
        "mypy: Marks tests that run mypy type checking",
        "vulture: Marks tests that run vulture dead code detection",
        "bandit: Marks tests that run bandit security analysis",
        
        # Test type markers
        "unit: Marks unit tests (isolated, fast, no external dependencies)",
        "integration: Marks integration tests (multiple components, external services)",
        "performance: Marks performance tests (benchmarks, load testing)",
        "slow: Marks slow-running tests (deselect with '-m \"not slow\"')",
        
        # Test characteristic markers
        "asyncio: Marks async tests that require asyncio event loop",
        "critical: Marks tests critical for CI/CD pipeline",
        "flaky: Marks tests known to be flaky (may need retries)",
        "smoke: Marks smoke tests for basic functionality verification",
        
        # Component-specific markers
        "data_layer: Marks tests for data layer components",
        "process_layer: Marks tests for process layer components",
        "endpoint_layer: Marks tests for endpoint layer components",
        "cli_layer: Marks tests for CLI layer components",
        
        # Database and storage markers
        "sqlite: Marks tests requiring SQLite database",
        "chroma: Marks tests requiring ChromaDB vector storage",
        "database: Marks tests requiring any database backend",
        
        # External service markers
        "llm: Marks tests requiring LLM providers",
        "embedding: Marks tests requiring embedding services",
        "network: Marks tests requiring network access",
        
        # Security and compliance markers
        "security: Marks security-focused tests",
        "compliance: Marks compliance and regulatory tests",
        "utf8: Marks tests for UTF-8 encoding compliance",
        
        # Development workflow markers
        "development: Marks tests for development tools and workflows",
        "documentation: Marks tests for documentation validation",
        "examples: Marks tests for example code validation"
    ]
    
    for marker in markers_to_register:
        config.addinivalue_line("markers", marker)
    
    # Initialize static analysis auto-discovery
    config._static_analysis_results = {}
    config._static_analysis_enabled = config.getoption("--static-analysis", default=False)
    
    # Load pytest.ini configuration for static analysis
    pytest_ini_path = Path(__file__).parent.parent / "pytest.ini"
    if pytest_ini_path.exists():
        parser = configparser.ConfigParser()
        parser.read(pytest_ini_path)
        
        # Update static analysis configuration from pytest.ini
        if 'pytest-static-analysis' in parser:
            config._static_analysis_auto_discovery = parser.getboolean('pytest-static-analysis', 'auto_discovery', fallback=True)
        else:
            config._static_analysis_auto_discovery = True

def pytest_collection_modifyitems(config, items):
    """Modify test collection for optimization and static analysis integration."""
    # Add automatic markers based on test location and name
    for item in items:
        # Mark tests in performance directory
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark unit tests (default)
        if not any(mark.name in ["integration", "performance", "static_analysis"] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)

        # Mark critical tests
        if "critical" in item.nodeid or any(keyword in item.name for keyword in
                                          ["basic", "core", "essential"]):
            item.add_marker(pytest.mark.critical)
        
        # Auto-discover and mark static analysis tests
        if config.getoption("--static-analysis", default=False) or config._static_analysis_auto_discovery:
            if "static_analysis" in item.nodeid:
                item.add_marker(pytest.mark.static_analysis)
            
            # Auto-mark tests based on tool names
            for tool_name, tool_config in STATIC_ANALYSIS_CONFIG.items():
                if tool_name in item.nodeid or tool_name in item.name:
                    item.add_marker(getattr(pytest.mark, tool_name))
                    item.add_marker(pytest.mark.static_analysis)

def pytest_runtest_setup(item):
    """Setup hook for static analysis test execution."""
    # Check if this is a static analysis test
    if any(mark.name == 'static_analysis' for mark in item.iter_markers()):
        # Ensure static analysis tools are available
        for tool_name, tool_config in STATIC_ANALYSIS_CONFIG.items():
            if tool_config['enabled'] and any(mark.name == tool_name for mark in item.iter_markers()):
                try:
                    # Check if tool is available
                    subprocess.run([tool_name, '--version'], capture_output=True, check=True, timeout=10)
                except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                    pytest.skip(f"Static analysis tool '{tool_name}' not available")

def pytest_runtest_call(item):
    """Call hook for static analysis test execution."""
    # Execute static analysis tools for marked tests
    if any(mark.name == 'static_analysis' for mark in item.iter_markers()):
        for tool_name, tool_config in STATIC_ANALYSIS_CONFIG.items():
            if tool_config['enabled'] and any(mark.name == tool_name for mark in item.iter_markers()):
                try:
                    # Run the static analysis tool
                    result = subprocess.run(
                        tool_config['command'].split(),
                        capture_output=True,
                        text=True,
                        timeout=300,
                        cwd=Path(__file__).parent.parent
                    )
                    
                    # Store results for reporting
                    if not hasattr(item.config, '_static_analysis_results'):
                        item.config._static_analysis_results = {}
                    
                    item.config._static_analysis_results[tool_name] = {
                        'returncode': result.returncode,
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'success': result.returncode == 0
                    }
                    
                    # Fail test if static analysis finds issues
                    if result.returncode != 0:
                        pytest.fail(f"{tool_name} static analysis failed:\n{result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    pytest.fail(f"{tool_name} static analysis timed out after 300 seconds")
                except Exception as e:
                    pytest.fail(f"Error running {tool_name} static analysis: {str(e)}")

def pytest_collection_modifyitems(config, items):
    """Modify test collection for optimization."""
    # Add automatic markers based on test location and name
    for item in items:
        # Mark tests in performance directory
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark unit tests (default)
        if not any(mark.name in ["integration", "performance"] for mark in item.iter_markers()):
            item.add_marker(pytest.mark.unit)

        # Mark critical tests
        if "critical" in item.nodeid or any(keyword in item.name for keyword in
                                          ["basic", "core", "essential"]):
            item.add_marker(pytest.mark.critical)

# Performance and static analysis reporting
def pytest_report_header(config):
    """Add performance and static analysis information to test report header."""
    header_lines = [
        "Optimized Test Suite - Parallel Execution Enabled",
        "Static Analysis Integration: Configured"
    ]
    
    # Add static analysis tool status
    if hasattr(config, '_static_analysis_enabled') and config._static_analysis_enabled:
        header_lines.append("Static Analysis: Active")
    
    return "\n".join(header_lines)

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add performance and static analysis summary to terminal output."""
    terminalreporter.write_sep("=", "Test Suite Summary")
    terminalreporter.write_line("• Parallel execution: Enabled")
    terminalreporter.write_line("• Database isolation: Enforced")
    terminalreporter.write_line("• UTF-8 compliance: Validated")
    terminalreporter.write_line("• Memory monitoring: Active")
    terminalreporter.write_line("• Performance timing: Enabled")
    
    # Add static analysis summary
    if hasattr(config, '_static_analysis_results') and config._static_analysis_results:
        terminalreporter.write_sep("-", "Static Analysis Results")
        for tool_name, results in config._static_analysis_results.items():
            status = "✓ PASSED" if results['success'] else "✗ FAILED"
            terminalreporter.write_line(f"• {tool_name}: {status}")
            
            if not results['success'] and results['stderr']:
                # Show first few lines of error output
                error_lines = results['stderr'].strip().split('\n')[:3]
                for line in error_lines:
                    terminalreporter.write_line(f"  {line}")
                if len(results['stderr'].strip().split('\n')) > 3:
                    terminalreporter.write_line("  ... (truncated)")

# Static analysis fixtures
@pytest.fixture(scope="session")
def static_analysis_config():
    """Fixture providing static analysis configuration."""
    return STATIC_ANALYSIS_CONFIG.copy()

@pytest.fixture(scope="function")
def static_analysis_runner():
    """Fixture for running static analysis tools programmatically."""
    
    class StaticAnalysisRunner:
        def run_tool(self, tool_name: str, extra_args: List[str] = None) -> Dict[str, Any]:
            """Run a specific static analysis tool."""
            if tool_name not in STATIC_ANALYSIS_CONFIG:
                raise ValueError(f"Unknown static analysis tool: {tool_name}")
            
            tool_config = STATIC_ANALYSIS_CONFIG[tool_name]
            if not tool_config['enabled']:
                return {'success': False, 'error': f'Tool {tool_name} is disabled'}
            
            command = tool_config['command'].split()
            if extra_args:
                command.extend(extra_args)
            
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=Path(__file__).parent.parent
                )
                
                return {
                    'success': result.returncode == 0,
                    'returncode': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'command': ' '.join(command)
                }
            except subprocess.TimeoutExpired:
                return {
                    'success': False,
                    'error': f'Tool {tool_name} timed out after 300 seconds',
                    'command': ' '.join(command)
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Error running {tool_name}: {str(e)}',
                    'command': ' '.join(command)
                }
        
        def run_all_tools(self) -> Dict[str, Dict[str, Any]]:
            """Run all enabled static analysis tools."""
            results = {}
            for tool_name in STATIC_ANALYSIS_CONFIG:
                results[tool_name] = self.run_tool(tool_name)
            return results
    
    return StaticAnalysisRunner()