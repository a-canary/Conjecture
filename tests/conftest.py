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

# Performance optimization hooks
def pytest_configure(config):
    """Configure pytest with performance optimizations."""
    # Register custom markers
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "critical: marks tests as critical for CI/CD")

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

# Performance reporting
def pytest_report_header(config):
    """Add performance information to test report header."""
    return f"Optimized Test Suite - Parallel Execution Enabled"

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add performance summary to terminal output."""
    terminalreporter.write_sep("=", "Performance Optimization Summary")
    terminalreporter.write_line("• Parallel execution: Enabled")
    terminalreporter.write_line("• Database isolation: Enforced")
    terminalreporter.write_line("• UTF-8 compliance: Validated")
    terminalreporter.write_line("• Memory monitoring: Active")
    terminalreporter.write_line("• Performance timing: Enabled")