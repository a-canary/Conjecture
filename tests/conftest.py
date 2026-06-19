# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration: marker registration, auto-marking, and plugins.

Split from a previously-monolithic conftest.py. Three sibling plugins
provide the heavier integrations:

  - ``conftest_duration`` — per-test duration tracking and report
  - ``conftest_static``   — optional ruff/mypy/vulture/bandit execution
  - ``conftest_mocks``    — localhost network mocks + asyncio.sleep cap

This file owns the cross-cutting concerns that don't fit any single
plugin: marker registration (so ``--strict-markers`` doesn't reject
``@pytest.mark.unit`` etc.), test auto-marking based on nodeid/name
(unit / integration / performance / critical), and the openrouter
fixture import for ``tests/test_openrouter_benchmark.py``.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports (kept for parity with previous conftest)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


pytest_plugins = [
    "conftest_duration",
    "conftest_static",
    "conftest_mocks",
]


def pytest_configure(config):
    """Register the comprehensive marker catalog.

    The full list is registered here (not in pytest.ini) so adding a
    new marker does not require a config edit and so the strict
    ``--strict-markers`` flag in pytest.ini accepts any of them.
    """
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
        "cli_layer: Marks tests for the CLI layer components",
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
        "documentation: Marks documentation validation tests",
        "examples: Marks tests for example code validation",
        # Additional markers
        "models: Marks tests for Pydantic model validation and behavior",
        "error_handling: Marks tests for error handling and edge cases",
        "test_marker_fix: Test marker to verify configuration is working",
        # Benchmark markers
        "benchmark: Marks benchmark tests requiring LLM API access",
        "openrouter: Marks tests using OpenRouter API",
    ]
    for marker in markers_to_register:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests by nodeid/name (unit / integration / performance / critical).

    Static-analysis auto-marking lives in ``conftest_static.py`` and
    runs as a separate hook on the same items; both hooks compose
    additively.
    """
    for item in items:
        if "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)

        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        if not any(
            mark.name in ["integration", "performance", "static_analysis"]
            for mark in item.iter_markers()
        ):
            item.add_marker(pytest.mark.unit)

        if "critical" in item.nodeid or any(
            keyword in item.name for keyword in ["basic", "core", "essential"]
        ):
            item.add_marker(pytest.mark.critical)


# --- GLM46 judge config fallback ---
#
# ``TestEnhancedGLM46Judge`` defines its own ``judge_config`` fixture
# on the class, but ``TestJudgePerformance`` is a sibling class and
# falls back to the module-level conftest fixture. Kept here so that
# test keeps working without reaching into the test file.
@pytest.fixture(scope="function")
def judge_config():
    """Judge configuration for GLM46 judge tests."""
    return {"key": "test_api_key", "url": "http://test-api.com", "model": "glm-4.6"}


# --- OpenRouter Free Model Fixtures ---
# Re-exported here for tests/test_openrouter_benchmark.py. The
# fixtures live in tests/fixtures/openrouter_free.py and provide
# real-API test infrastructure (gated by --benchmark marker / env).
try:
    from tests.fixtures.openrouter_free import (  # noqa: F401
        openrouter_api_key,
        openrouter_config,
        openrouter_client,
        free_models,
        gpt_oss_20b,
        nemotron_30b,
        benchmark_prompt_factory,
    )
except ImportError:
    # Fixtures not available - tests will skip gracefully
    pass
