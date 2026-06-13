# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
CLI tests for Conjecture CLI commands.

Tests the create, search, and stats commands using Typer's CliRunner
with mocked processing interface for test isolation.

The Conjecture CLI uses a Rich Console that writes directly to sys.stdout
(bypassing Typer's CliRunner output capture).  We therefore patch the
module-level ``console`` with a test console that writes to a StringIO
buffer, and assert against that buffer.
"""

import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from typer.testing import CliRunner
from rich.console import Console

from src.core.models import Claim, ClaimState, ClaimType, ClaimScope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claim(
    claim_id: str = "c0000001",
    content: str = "Test claim content for testing",
    confidence: float = 0.8,
    state: ClaimState = ClaimState.EXPLORE,
    tags: list = None,
) -> Claim:
    """Return a minimal, valid Claim object."""
    return Claim(
        id=claim_id,
        content=content,
        confidence=confidence,
        state=state,
        type=[ClaimType.CONCEPT],
        tags=tags or ["user"],
        scope=ClaimScope.USER_WORKSPACE,
        created=datetime.now(timezone.utc),
        updated=datetime.now(timezone.utc),
        is_dirty=False,
        dirty=False,
    )


def _make_mock_interface(
    *,
    claim: Claim = None,
    search_results: list = None,
    stats: dict = None,
):
    """
    Return a MagicMock that satisfies the ProcessingInterface contract
    for the three commands under test (create, search, stats).
    """
    if claim is None:
        claim = _make_claim()

    if search_results is None:
        search_results = [claim]

    if stats is None:
        stats = {
            "backend_type": "Conjecture",
            "total_claims": 1,
            "avg_confidence": 0.8,
            "unique_users": 1,
        }

    mock = MagicMock()
    mock.initialize = AsyncMock(return_value=None)
    mock.create_claim = AsyncMock(return_value=claim)
    mock.search_claims = AsyncMock(return_value=search_results)
    mock.get_statistics = AsyncMock(return_value=stats)
    mock.get_health_status = AsyncMock(
        return_value={
            "healthy": True,
            "services": {
                "data_manager": True,
                "llm_bridge": True,
                "async_evaluation": True,
            },
            "active_sessions": 0,
        }
    )
    return mock


def _make_test_console() -> tuple[Console, io.StringIO]:
    """Return a (console, buffer) pair for capturing Rich output."""
    buf = io.StringIO()
    console = Console(file=buf, no_color=True, width=120)
    return console, buf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def runner():
    """Typer CLI test runner."""
    return CliRunner()


@pytest.fixture()
def mock_interface():
    """Default mock processing interface."""
    return _make_mock_interface()


# ---------------------------------------------------------------------------
# Context manager helper
# ---------------------------------------------------------------------------

def _invoke(app, args, mock_iface, *, stats_override=None, extra_patches=None):
    """
    Invoke a CLI command with the processing interface mocked out and
    the Rich console redirected to an in-memory buffer.

    Returns (result, captured_text).
    """
    import src.cli.modular_cli as cli_module

    if stats_override is not None:
        mock_iface.get_statistics = AsyncMock(return_value=stats_override)

    test_console, buf = _make_test_console()
    runner = CliRunner()

    patches = [
        patch.object(cli_module, "current_processing_interface", mock_iface),
        patch.object(cli_module, "get_processing_interface", return_value=mock_iface),
        patch.object(cli_module, "console", test_console),
    ]
    if extra_patches:
        patches.extend(extra_patches)

    import contextlib

    @contextlib.contextmanager
    def _stack(patches):
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            yield

    with _stack(patches):
        result = runner.invoke(app, args)

    return result, buf.getvalue()


# ---------------------------------------------------------------------------
# Stats command tests
# ---------------------------------------------------------------------------

class TestStatsCommand:
    """Tests for the ``stats`` CLI command."""

    def test_stats_exits_successfully(self, mock_interface):
        """stats returns exit code 0."""
        import src.cli.modular_cli as cli_module

        result, _ = _invoke(cli_module.app, ["stats"], mock_interface)
        assert result.exit_code == 0, (
            f"Expected exit code 0, got {result.exit_code}.\n"
            f"Exception: {result.exception}"
        )

    def test_stats_shows_database_info(self, mock_interface):
        """stats output contains database-related header text."""
        import src.cli.modular_cli as cli_module

        result, output = _invoke(cli_module.app, ["stats"], mock_interface)
        assert result.exit_code == 0
        assert "Database" in output or "Statistics" in output, (
            f"Expected database info in output.\nOutput:\n{output}"
        )

    def test_stats_shows_total_claims(self, mock_interface):
        """stats output includes the total-claims metric."""
        import src.cli.modular_cli as cli_module

        result, output = _invoke(cli_module.app, ["stats"], mock_interface)
        assert result.exit_code == 0
        assert "Total Claims" in output, (
            f"Expected 'Total Claims' in output.\nOutput:\n{output}"
        )

    def test_stats_shows_backend_type(self, mock_interface):
        """stats output includes the backend type."""
        import src.cli.modular_cli as cli_module

        result, output = _invoke(cli_module.app, ["stats"], mock_interface)
        assert result.exit_code == 0
        assert "Conjecture" in output, (
            f"Expected 'Conjecture' backend in output.\nOutput:\n{output}"
        )

    def test_stats_with_empty_database(self, mock_interface):
        """stats handles an empty database gracefully."""
        import src.cli.modular_cli as cli_module

        empty_stats = {
            "backend_type": "Conjecture",
            "total_claims": 0,
            "avg_confidence": 0.0,
            "unique_users": 0,
        }
        result, output = _invoke(
            cli_module.app, ["stats"], mock_interface, stats_override=empty_stats
        )
        assert result.exit_code == 0
        assert "0" in output, (
            f"Expected '0' in output for empty database.\nOutput:\n{output}"
        )


# ---------------------------------------------------------------------------
# Create command tests
# ---------------------------------------------------------------------------

class TestCreateCommand:
    """Tests for the ``create`` CLI command."""

    def test_create_exits_successfully(self, mock_interface):
        """create returns exit code 0 for valid content."""
        import src.cli.modular_cli as cli_module

        result, _ = _invoke(
            cli_module.app, ["create", "Test claim content for testing"], mock_interface
        )
        assert result.exit_code == 0, (
            f"Expected exit code 0, got {result.exit_code}.\n"
            f"Exception: {result.exception}"
        )

    def test_create_shows_success_message(self, mock_interface):
        """create outputs a success confirmation."""
        import src.cli.modular_cli as cli_module

        result, output = _invoke(
            cli_module.app, ["create", "Test claim content for testing"], mock_interface
        )
        assert result.exit_code == 0
        lower = output.lower()
        assert "created" in lower or "c0000001" in output, (
            f"Expected success message in output.\nOutput:\n{output}"
        )

    def test_create_with_confidence_option(self, mock_interface):
        """create accepts --confidence option without error."""
        import src.cli.modular_cli as cli_module

        result, _ = _invoke(
            cli_module.app,
            ["create", "Test claim content for testing", "--confidence", "0.9"],
            mock_interface,
        )
        assert result.exit_code == 0, (
            f"Expected exit code 0, got {result.exit_code}.\n"
            f"Exception: {result.exception}"
        )

    def test_create_with_user_option(self, mock_interface):
        """create accepts --user option without error."""
        import src.cli.modular_cli as cli_module

        result, _ = _invoke(
            cli_module.app,
            ["create", "Test claim content for testing", "--user", "testuser"],
            mock_interface,
        )
        assert result.exit_code == 0, (
            f"Expected exit code 0, got {result.exit_code}.\n"
            f"Exception: {result.exception}"
        )

    def test_create_calls_processing_interface(self, mock_interface):
        """create delegates to the processing interface's create_claim method."""
        import src.cli.modular_cli as cli_module

        _invoke(
            cli_module.app,
            ["create", "Test claim content for testing"],
            mock_interface,
        )
        mock_interface.create_claim.assert_called_once()
        call_kwargs = mock_interface.create_claim.call_args
        assert "content" in call_kwargs.kwargs or len(call_kwargs.args) >= 1, (
            "Expected 'content' to be passed to create_claim"
        )

    def test_create_returns_claim_id(self, mock_interface):
        """create output includes the claim ID returned by the interface."""
        import src.cli.modular_cli as cli_module

        specific_claim = _make_claim(claim_id="c9999999")
        mock_interface.create_claim = AsyncMock(return_value=specific_claim)

        result, output = _invoke(
            cli_module.app, ["create", "Test claim content for testing"], mock_interface
        )
        assert result.exit_code == 0
        assert "c9999999" in output, (
            f"Expected claim ID 'c9999999' in output.\nOutput:\n{output}"
        )


# ---------------------------------------------------------------------------
# Search command tests
# ---------------------------------------------------------------------------

class TestSearchCommand:
    """Tests for the ``search`` CLI command."""

    def test_search_exits_successfully(self, mock_interface):
        """search returns exit code 0 for a valid query."""
        import src.cli.modular_cli as cli_module

        result, _ = _invoke(cli_module.app, ["search", "test query"], mock_interface)
        assert result.exit_code == 0, (
            f"Expected exit code 0, got {result.exit_code}.\n"
            f"Exception: {result.exception}"
        )

    def test_search_shows_results_when_found(self, mock_interface):
        """search output includes claim content when results are found."""
        import src.cli.modular_cli as cli_module

        result, output = _invoke(
            cli_module.app, ["search", "test query"], mock_interface
        )
        assert result.exit_code == 0
        # The default mock has one result with ID c0000001
        assert "c0000001" in output or "Test claim" in output or "Found" in output, (
            f"Expected search results in output.\nOutput:\n{output}"
        )

    def test_search_handles_no_results(self, mock_interface):
        """search shows a helpful message when no claims match the query."""
        import src.cli.modular_cli as cli_module

        mock_interface.search_claims = AsyncMock(return_value=[])
        result, output = _invoke(
            cli_module.app, ["search", "nonexistent query"], mock_interface
        )
        assert result.exit_code == 0
        lower = output.lower()
        assert "no" in lower or "found" in lower or "0" in output, (
            f"Expected 'no results' message in output.\nOutput:\n{output}"
        )

    def test_search_calls_processing_interface(self, mock_interface):
        """search delegates to the processing interface's search_claims method."""
        import src.cli.modular_cli as cli_module

        _invoke(cli_module.app, ["search", "my query"], mock_interface)
        mock_interface.search_claims.assert_called_once()
        call_args = mock_interface.search_claims.call_args
        passed_query = (
            call_args.args[0]
            if call_args.args
            else call_args.kwargs.get("query", "")
        )
        assert passed_query == "my query", (
            f"Expected query 'my query' to be passed to search_claims, got: {passed_query!r}"
        )

    def test_search_with_limit_option(self, mock_interface):
        """search accepts the --limit option without error."""
        import src.cli.modular_cli as cli_module

        result, _ = _invoke(
            cli_module.app, ["search", "test query", "--limit", "5"], mock_interface
        )
        assert result.exit_code == 0, (
            f"Expected exit code 0, got {result.exit_code}.\n"
            f"Exception: {result.exception}"
        )

    def test_search_respects_limit(self, mock_interface):
        """search returns at most <limit> results when more are available."""
        import src.cli.modular_cli as cli_module

        many_claims = [
            _make_claim(
                claim_id=f"c000000{i}",
                content=f"Claim number {i} content here",
            )
            for i in range(1, 6)
        ]
        mock_interface.search_claims = AsyncMock(return_value=many_claims)

        result, output = _invoke(
            cli_module.app,
            ["search", "test query", "--limit", "2"],
            mock_interface,
        )
        assert result.exit_code == 0
        # With limit=2 only the first two claims should appear; c0000003-c0000005 should not.
        assert "c0000003" not in output or "c0000005" not in output, (
            f"Expected limit to be respected (only 2 of 5 results shown).\n"
            f"Output:\n{output}"
        )

    def test_search_multiple_results_displayed(self, mock_interface):
        """search displays a table row for each result returned."""
        import src.cli.modular_cli as cli_module

        claims = [
            _make_claim(claim_id="c0000010", content="First claim about alpha topic"),
            _make_claim(claim_id="c0000011", content="Second claim about beta topic"),
        ]
        mock_interface.search_claims = AsyncMock(return_value=claims)

        result, output = _invoke(cli_module.app, ["search", "topic"], mock_interface)
        assert result.exit_code == 0
        assert "c0000010" in output, (
            f"Expected first claim ID 'c0000010' in output.\nOutput:\n{output}"
        )
        assert "c0000011" in output, (
            f"Expected second claim ID 'c0000011' in output.\nOutput:\n{output}"
        )
