"""Pytest plugin: localhost-mock patches and global sleep cap.

Extracted from conftest.py. Two side effects that were originally
inline at the bottom of conftest.py:

1. ``fast_localhost_mocks`` autouse fixture — patches
   ``aiohttp.ClientSession.get`` and ``post`` so localhost network
   calls return instantly instead of hitting a real (and absent) LLM
   endpoint. Skipped for tests marked ``benchmark`` or ``network``.

2. Module-level monkey-patch of ``asyncio.sleep`` to cap any sleep
   at 0.1s. This is a hammer that makes the test suite complete
   quickly but also masks bugs in code that depends on real wall
   time. Kept for behavior parity with the previous conftest; flag
   in PR body as a follow-up.

The mock provider fixtures (mock_ollama_response, mock_lmstudio_response,
mock_providers_config, fast_test_config) are no longer consumed by
any test in the current suite, but they are exported here so callers
that used to find them via the conftest import path can keep
working. Each carries a deprecation note in its docstring.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest


# ---- fast localhost mocks (autouse, behaviorally significant) ----


def create_fast_response_mock() -> AsyncMock:
    """Build a one-shot aiohttp response mock that returns instantly."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {"response": "Fast mock response"}
    mock_response.text.return_value = "Fast mock text"
    return mock_response


@pytest.fixture(autouse=True)
def fast_localhost_mocks(request):
    """Auto-applied fixture to mock all localhost connections.

    Skipped for tests marked with 'benchmark' or 'network' so they
    can hit the real endpoint.
    """
    for mark in request.node.iter_markers():
        if mark.name in ("benchmark", "network"):
            yield
            return

    with (
        patch("aiohttp.ClientSession.get") as mock_get,
        patch("aiohttp.ClientSession.post") as mock_post,
    ):
        mock_get.return_value.__aenter__.return_value = create_fast_response_mock()
        mock_post.return_value.__aenter__.return_value = create_fast_response_mock()
        yield


# ---- global asyncio.sleep cap ----
_original_sleep = asyncio.sleep


async def fast_sleep(duration: float) -> None:
    """Cap any asyncio.sleep at 0.1s for fast tests."""
    return await _original_sleep(min(duration, 0.1))


asyncio.sleep = fast_sleep


# ---- mock provider fixtures (kept for backward compatibility) ----
#
# None of these are currently consumed by tests/test_*.py, but they
# were public on the conftest namespace. Moved here so a stale
# ``from conftest import mock_ollama_response`` style import doesn't
# fail; flag the deprecation in the PR body and remove in a
# follow-up slice if no consumer re-emerges.


@pytest.fixture
def mock_ollama_response():
    """DEPRECATED: no current consumer. Returns an Ollama-shaped mock."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {
        "model": "llama2",
        "created_at": "2024-01-01T00:00:00Z",
        "response": "Mock LLM response for testing",
    }
    return mock_response


@pytest.fixture
def mock_lmstudio_response():
    """DEPRECATED: no current consumer. Returns an LM Studio-shaped mock."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {
        "model": "granite-7b",
        "choices": [{"message": {"content": "Mock LM Studio response"}}],
    }
    return mock_response


@pytest.fixture
def mock_providers_config() -> List[Dict[str, Any]]:
    """DEPRECATED: no current consumer. Returns a mock providers list."""
    return [
        {
            "name": "mock-ollama",
            "url": "http://mock-localhost:11434",
            "model": "llama2",
            "api_key": "mock-key",
        },
        {
            "name": "mock-lmstudio",
            "url": "http://mock-localhost:1234",
            "model": "granite-7b",
            "api_key": "mock-key",
        },
    ]


@pytest.fixture
def fast_test_config() -> Dict[str, Any]:
    """DEPRECATED: no current consumer. Returns a fast-test config dict."""
    return {
        "processing": {
            "confidence_threshold": 0.85,
            "max_context_size": 1000,
            "batch_size": 2,
            "timeout": 1,
            "retry_delay": 0.1,
            "max_retries": 1,
        },
        "database": {
            "database_path": ":memory:",
            "cache_size": 100,
            "connection_timeout": 1,
        },
        "providers": [
            {
                "name": "mock-provider",
                "url": "http://mock-localhost:9999",
                "model": "mock-model",
                "timeout": 1,
                "enabled": True,
            }
        ],
        "debug": False,
        "monitoring": {"enable_performance_tracking": False},
    }
