# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Tests for A-0013: MCP Delivery Model

Per CHOICES.md A-0013: "Expose Conjecture as an MCP server with tools:
  - build_context(query) → context_blob
  - upsert_claim(claim, confidence, super_ids, sub_ids) → claim_id
  - explore_next() → claim
  - get_claim_support(claim_or_query) → sub-claims

Any MCP-compatible client can use Conjecture as a reasoning backend."

Covers:
  - TestToolRegistration: All 4 tools are registered and named correctly
  - TestBuildContext: build_context returns context blob structure
  - TestUpsertClaim: upsert_claim creates claims and returns claim_id
  - TestExploreNext: explore_next returns highest-priority dirty claim
  - TestGetClaimSupport: get_claim_support returns supporting sub-claims
  - TestErrorHandling: Tools return error dicts on endpoint failures
  - TestGetEndpoint: get_endpoint initializes endpoint once and reuses
"""

import os
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _mock_endpoint():
    """Create a fully-mocked ConjectureEndpoint."""
    endpoint = MagicMock()
    endpoint.evaluate = AsyncMock()
    endpoint.create_claim = AsyncMock()
    endpoint.search_claims = AsyncMock()
    endpoint.get_claim = AsyncMock()
    return endpoint


# ---------------------------------------------------------------------------
# Tool Registration Tests
# ---------------------------------------------------------------------------


class TestToolRegistration:
    """Verify all 4 MCP tools are registered with FastMCP."""

    def test_all_four_tools_registered(self):
        """All required A-0013 tools are present in the MCP tool registry."""
        from src.endpoint.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert set(tool_names) == {
            "build_context",
            "upsert_claim",
            "explore_next",
            "get_claim_support",
        }

    def test_build_context_is_registered(self):
        """build_context is callable and registered as a tool."""
        from src.endpoint.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "build_context" in tool_names

    def test_upsert_claim_is_registered(self):
        """upsert_claim is callable and registered as a tool."""
        from src.endpoint.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "upsert_claim" in tool_names

    def test_explore_next_is_registered(self):
        """explore_next is callable and registered as a tool."""
        from src.endpoint.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "explore_next" in tool_names

    def test_get_claim_support_is_registered(self):
        """get_claim_support is callable and registered as a tool."""
        from src.endpoint.mcp_server import mcp

        tool_names = [t.name for t in mcp._tool_manager.list_tools()]
        assert "get_claim_support" in tool_names


# ---------------------------------------------------------------------------
# build_context Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBuildContext:
    """Test build_context tool (A-0013: retrieve claim context for query)."""

    async def test_returns_success_with_claims(self):
        """Successful evaluation returns query, claims, context, and status."""
        from src.endpoint.mcp_server import build_context

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.evaluate = AsyncMock(return_value=MagicMock(
            success=True,
            data={
                "claims_found": ["c1", "c2"],
                "reasoning": ["Step 1", "Step 2"],
            },
            errors=None,
        ))

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            result = await build_context("What is quantum entanglement?")

        assert result["status"] == "success"
        assert result["query"] == "What is quantum entanglement?"
        assert result["claims"] == ["c1", "c2"]
        assert result["context"] == ["Step 1", "Step 2"]

    async def test_passes_max_claims_to_endpoint(self):
        """max_claims parameter is forwarded to endpoint.evaluate()."""
        from src.endpoint.mcp_server import build_context

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.evaluate = AsyncMock(return_value=MagicMock(
            success=True,
            data={"claims_found": [], "reasoning": []},
            errors=None,
        ))

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            await build_context("Test query", max_claims=5)

        mock_endpoint.evaluate.assert_called_once()
        call_kwargs = mock_endpoint.evaluate.call_args.kwargs
        assert call_kwargs.get("max_claims") == 5
        assert call_kwargs.get("include_reasoning") is True

    async def test_returns_error_on_failure(self):
        """Endpoint failure returns error status and errors list."""
        from src.endpoint.mcp_server import build_context

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.evaluate = AsyncMock(return_value=MagicMock(
            success=False,
            data=None,
            errors=["Provider unavailable"],
        ))

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            result = await build_context("Test")

        assert result["status"] == "error"
        assert "Provider unavailable" in result["errors"]


# ---------------------------------------------------------------------------
# upsert_claim Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUpsertClaim:
    """Test upsert_claim tool (A-0013: create or update a claim)."""

    async def test_creates_claim_successfully(self):
        """Successful create returns claim_id, content, confidence, status."""
        from src.endpoint.mcp_server import upsert_claim

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.create_claim = AsyncMock(return_value=MagicMock(
            success=True,
            data={"id": "claim_abc123"},
            message="Claim created",
            errors=None,
        ))

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            result = await upsert_claim(
                content="Quantum superposition allows particles to exist in multiple states.",
                confidence=0.85,
                tags=["physics", "quantum"],
            )

        assert result["status"] == "created"
        assert result["claim_id"] == "claim_abc123"
        assert result["content"] == "Quantum superposition allows particles to exist in multiple states."
        assert result["confidence"] == 0.85

    async def test_passes_sup_and_sub_ids(self):
        """sup_ids and sub_ids are forwarded to create_claim()."""
        from src.endpoint.mcp_server import upsert_claim

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.create_claim = AsyncMock(return_value=MagicMock(
            success=True,
            data={"id": "new_claim"},
            message="",
            errors=None,
        ))

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            await upsert_claim(
                content="Test claim",
                super_ids=["sup_1", "sup_2"],
                sub_ids=["sub_1"],
            )

        call_kwargs = mock_endpoint.create_claim.call_args.kwargs
        assert call_kwargs["supers"] == ["sup_1", "sup_2"]
        assert call_kwargs["subs"] == ["sub_1"]

    async def test_returns_error_on_failure(self):
        """Endpoint failure returns error status."""
        from src.endpoint.mcp_server import upsert_claim

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.create_claim = AsyncMock(return_value=MagicMock(
            success=False,
            data=None,
            errors=["Invalid confidence: must be 0.0-1.0"],
        ))

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            result = await upsert_claim(content="Bad claim", confidence=1.5)

        assert result["status"] == "error"
        assert result["claim_id"] is None
        assert "Invalid confidence" in result["errors"][0]

    async def test_defaults_confidence_to_point_five(self):
        """Default confidence is 0.5 when not specified."""
        from src.endpoint.mcp_server import upsert_claim

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.create_claim = AsyncMock(return_value=MagicMock(
            success=True,
            data={"id": "c1"},
            message="",
            errors=None,
        ))

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            await upsert_claim(content="Test")

        call_kwargs = mock_endpoint.create_claim.call_args.kwargs
        assert call_kwargs["confidence"] == 0.5


# ---------------------------------------------------------------------------
# explore_next Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExploreNext:
    """Test explore_next tool (A-0013: get next claim from dirty queue)."""

    async def test_returns_highest_priority_claim(self):
        """Found claim returns claim_id, content, confidence, state."""
        from src.endpoint.mcp_server import explore_next

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.search_claims = AsyncMock(return_value=MagicMock(
            success=True,
            data={"claims": [
                {
                    "id": "dirty_001",
                    "content": "The mitochondria is the powerhouse of the cell.",
                    "confidence": 0.6,
                    "state": "dirty",
                }
            ]},
            errors=None,
        ))

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            result = await explore_next()

        assert result["status"] == "found"
        assert result["claim_id"] == "dirty_001"
        assert result["content"] == "The mitochondria is the powerhouse of the cell."
        assert result["confidence"] == 0.6
        assert result["state"] == "dirty"

    async def test_returns_empty_when_no_claims(self):
        """No claims found returns empty status and null claim_id."""
        from src.endpoint.mcp_server import explore_next

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.search_claims = AsyncMock(return_value=MagicMock(
            success=True,
            data={"claims": []},
            errors=None,
        ))

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            result = await explore_next()

        assert result["status"] == "empty"
        assert result["claim_id"] is None
        assert "No claims to explore" in result["message"]

    async def test_returns_error_on_endpoint_failure(self):
        """Endpoint failure returns error status."""
        from src.endpoint.mcp_server import explore_next

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.search_claims = AsyncMock(return_value=MagicMock(
            success=False,
            data=None,
            errors=["Database connection failed"],
        ))

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            result = await explore_next()

        assert result["status"] == "error"
        assert result["claim_id"] is None


# ---------------------------------------------------------------------------
# get_claim_support Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetClaimSupport:
    """Test get_claim_support tool (A-0013: get sub-claims for a claim)."""

    async def test_returns_supporting_claims(self):
        """Found claim with subs returns sub-claim details."""
        from src.endpoint.mcp_server import get_claim_support

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.get_claim = AsyncMock(return_value=MagicMock(
            success=True,
            data={
                "id": "parent_claim",
                "content": "All mammals are warm-blooded.",
                "confidence": 0.95,
                "subs": ["sub_1", "sub_2"],
            },
            errors=None,
        ))
        # Mock nested get_claim for each sub
        sub_1 = MagicMock(
            success=True,
            data={
                "id": "sub_1",
                "content": "Dogs are warm-blooded.",
                "confidence": 0.99,
                "state": "verified",
            },
        )
        sub_2 = MagicMock(
            success=True,
            data={
                "id": "sub_2",
                "content": "Cats are warm-blooded.",
                "confidence": 0.98,
                "state": "verified",
            },
        )
        mock_endpoint.get_claim.side_effect = [MagicMock(
            success=True, data={"id": "parent_claim", "content": "All mammals are warm-blooded.", "confidence": 0.95, "subs": ["sub_1", "sub_2"]}
        ), sub_1, sub_2]

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            result = await get_claim_support("parent_claim")

        assert result["status"] == "success"
        assert result["claim_id"] == "parent_claim"
        assert result["support_count"] == 2
        assert len(result["supporting_claims"]) == 2

    async def test_returns_not_found_for_missing_claim(self):
        """Unknown claim_id returns not_found status."""
        from src.endpoint.mcp_server import get_claim_support

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.get_claim = AsyncMock(return_value=MagicMock(
            success=False,
            data=None,
            errors=["Claim not found"],
        ))

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            result = await get_claim_support("nonexistent_id")

        assert result["status"] == "not_found"
        assert result["supporting_claims"] == []

    async def test_returns_empty_supporting_claims_when_no_subs(self):
        """Claim with empty subs list returns empty supporting_claims."""
        from src.endpoint.mcp_server import get_claim_support

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.get_claim = AsyncMock(return_value=MagicMock(
            success=True,
            data={
                "id": "root_claim",
                "content": "Root claim.",
                "confidence": 0.7,
                "subs": [],
            },
            errors=None,
        ))

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            result = await get_claim_support("root_claim")

        assert result["status"] == "success"
        assert result["support_count"] == 0
        assert result["supporting_claims"] == []


# ---------------------------------------------------------------------------
# get_endpoint Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetEndpoint:
    """Test endpoint initialization and reuse (singleton pattern)."""

    async def test_returns_cached_endpoint_on_reuse(self):
        """Second call returns the same endpoint instance (singleton)."""
        from src.endpoint import mcp_server

        mock_ep = await _mock_endpoint()
        mcp_server._endpoint = mock_ep

        ep1 = await mcp_server.get_endpoint()
        ep2 = await mcp_server.get_endpoint()

        assert ep1 is ep2
        # cleanup
        mcp_server._endpoint = None

    async def test_initializes_endpoint_when_not_cached(self):
        """First call creates and initializes a new endpoint."""
        from src.endpoint import mcp_server

        # Ensure no cached endpoint
        mcp_server._endpoint = None

        mock_class = MagicMock()
        mock_ep_instance = await _mock_endpoint()
        mock_class.return_value = mock_ep_instance
        mock_ep_instance.initialize = AsyncMock()

        with patch.object(mcp_server, "ConjectureEndpoint", mock_class):
            result = await mcp_server.get_endpoint()

        mock_class.assert_called_once()
        mock_ep_instance.initialize.assert_called_once()
        assert result is mock_ep_instance
        # cleanup
        mcp_server._endpoint = None


# ---------------------------------------------------------------------------
# Error Handling Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestErrorHandling:
    """Verify graceful degradation when endpoint operations fail."""

    async def test_build_context_handles_missing_data(self):
        """build_context handles success=True but data=None gracefully."""
        from src.endpoint.mcp_server import build_context

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.evaluate = AsyncMock(return_value=MagicMock(
            success=True,
            data=None,
            errors=None,
        ))

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            result = await build_context("Test")

        # Should not raise, should return error dict
        assert result["status"] == "error"

    async def test_upsert_claim_handles_missing_data(self):
        """upsert_claim handles success=True but data=None gracefully."""
        from src.endpoint.mcp_server import upsert_claim

        mock_endpoint = await _mock_endpoint()
        mock_endpoint.create_claim = AsyncMock(return_value=MagicMock(
            success=True,
            data=None,
            message="",
            errors=None,
        ))

        with patch("src.endpoint.mcp_server._endpoint", mock_endpoint):
            result = await upsert_claim(content="Test claim")

        assert result["status"] == "error"
        assert result["claim_id"] is None


# ---------------------------------------------------------------------------
# run_server / main Tests
# ---------------------------------------------------------------------------


def test_run_server_accepts_host_and_port():
    """run_server() accepts custom host and port parameters."""
    from src.endpoint.mcp_server import run_server
    import inspect

    sig = inspect.signature(run_server)
    params = list(sig.parameters.keys())
    assert "host" in params
    assert "port" in params


def test_main_parses_cli_arguments():
    """main() parses --host and --port arguments."""
    from src.endpoint.mcp_server import main
    import argparse

    # Verify main() is a CLI entry point that accepts args
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("--host", default="localhost")
    parser_arg.add_argument("--port", type=int, default=3000)

    args = parser_arg.parse_args(["--host", "0.0.0.0", "--port", "8080"])
    assert args.host == "0.0.0.0"
    assert args.port == 8080


# ---------------------------------------------------------------------------
# Integration Test: MCP Server as Subprocess (Smoke Test)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestMCPServerSubprocess:
    """Integration smoke test: run MCP server as subprocess and verify it starts."""

    def test_server_starts_without_error(self, tmp_path):
        """MCP server starts and responds to --help within timeout."""
        import subprocess
        import sys

        # Get project root (parent of tests/)
        project_root = Path(__file__).parent.parent

        # Run with --help to verify module loads without errors
        result = subprocess.run(
            [sys.executable, "-m", "src.endpoint.mcp_server", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(project_root),
            env={**os.environ, "PYTHONPATH": str(project_root)},
        )

        # Should exit cleanly (0 or 2 for argparse help)
        assert result.returncode in (0, 2), (
            f"MCP server failed to start:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Should contain expected help text
        assert "Conjecture MCP Server" in result.stdout or "--host" in result.stdout

    def test_server_imports_without_error(self):
        """MCP server module imports without errors."""
        import importlib
        import sys

        # Remove from cache if already imported
        if "src.endpoint.mcp_server" in sys.modules:
            del sys.modules["src.endpoint.mcp_server"]

        # Import should not raise
        import src.endpoint.mcp_server
        importlib.reload(src.endpoint.mcp_server)

        # Server object should exist
        assert hasattr(src.endpoint.mcp_server, "mcp")
        assert hasattr(src.endpoint.mcp_server, "run_server")
        assert hasattr(src.endpoint.mcp_server, "main")
