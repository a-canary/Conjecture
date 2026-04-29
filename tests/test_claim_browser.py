"""
Tests for Claim Browser TUI (UX-0007 Phase 3)

Tests interactive TUI browser with keyboard navigation, search/filter,
and claim expansion.  Uses mocked ProcessingInterface for isolation.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from src.core.models import Claim, ClaimState, ClaimType, ClaimScope
from src.cli.claim_browser import (
    ClaimBrowser,
    BrowserState,
    ClaimTreeNode,
    browse_claims,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_claim(
    claim_id: str = "c0000001",
    content: str = "Test claim content",
    confidence: float = 0.8,
    subs: list = None,
    supers: list = None,
    state: ClaimState = ClaimState.EXPLORE,
    types: list = None,
) -> Claim:
    """Return a minimal Claim for testing."""
    return Claim(
        id=claim_id,
        content=content,
        confidence=confidence,
        state=state,
        type=types or [ClaimType.CONCEPT],
        tags=["test"],
        scope=ClaimScope.USER_WORKSPACE,
        subs=subs or [],
        supers=supers or [],
        created=datetime.now(timezone.utc),
        updated=datetime.now(timezone.utc),
        is_dirty=False,
        dirty=False,
    )


class TestClaimTreeNode:
    """Tests for ClaimTreeNode dataclass."""

    def test_creation(self):
        claim = _make_claim("c0000001", "Root claim", 0.9)
        node = ClaimTreeNode(claim=claim, children=[], depth=0, expanded=True)
        assert node.claim.id == "c0000001"
        assert node.depth == 0
        assert node.expanded is True
        assert node.children == []

    def test_to_dict(self):
        child_claim = _make_claim("c0000002", "Child claim", 0.7)
        child_node = ClaimTreeNode(
            claim=child_claim, children=[], depth=1, expanded=False
        )
        root_claim = _make_claim("c0000001", "Root claim", 0.9, subs=["c0000002"])
        root = ClaimTreeNode(claim=root_claim, children=[child_node], depth=0, expanded=True)

        d = root.to_dict()
        assert d["id"] == "c0000001"
        assert d["children"][0]["id"] == "c0000002"
        assert d["children"][0]["depth"] == 1


class TestBrowserState:
    """Tests for BrowserState enum."""

    def test_all_states_defined(self):
        assert BrowserState.QUIT.value == "quit"
        assert BrowserState.RUNNING.value == "running"
        assert BrowserState.SEARCHING.value == "searching"


class TestClaimBrowser:
    """Tests for ClaimBrowser class."""

    @pytest.fixture
    def sample_claims(self):
        """Build a small sample claim tree."""
        root = _make_claim("c0000001", "Root hypothesis", 0.9, subs=["c0000002", "c0000003"])
        child2 = _make_claim("c0000002", "Supporting evidence A", 0.85, supers=["c0000001"], subs=["c0000004"])
        child3 = _make_claim("c0000003", "Supporting evidence B", 0.7, supers=["c0000001"])
        child4 = _make_claim("c0000004", "Deep evidence", 0.6, supers=["c0000002"])
        return {
            "c0000001": root,
            "c0000002": child2,
            "c0000003": child3,
            "c0000004": child4,
        }

    @pytest.fixture
    def mock_interface(self, sample_claims):
        """Mock ProcessingInterface that resolves claims synchronously."""
        interface = MagicMock()
        # Use a proper lambda so dict.get returns None for unknown keys
        interface.get_claim = MagicMock(
            side_effect=lambda cid: sample_claims.get(cid)
        )
        interface.search_claims = AsyncMock(
            return_value=[sample_claims["c0000001"], sample_claims["c0000002"], sample_claims["c0000003"]]
        )
        interface.get_statistics = AsyncMock(return_value={
            "total_claims": 4,
            "avg_confidence": 0.7875,
        })
        return interface

    def _make_browser(self, mock_interface, sample_claims, root_id="c0000001"):
        """Create a browser, patching _fetch_claim to bypass async loop."""
        from unittest.mock import patch
        def fetch(self, cid):
            return sample_claims.get(cid)
        with patch.object(ClaimBrowser, "_fetch_claim", fetch):
            b = ClaimBrowser(mock_interface, root_id=root_id)
        return b

    def test_init_builds_tree(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        assert browser.root_id == "c0000001"
        assert browser.current_index == 0
        assert browser.state == BrowserState.RUNNING

    def test_init_handles_missing_root(self, sample_claims):
        interface = MagicMock()
        interface.get_claim = MagicMock(return_value=None)
        def fetch(self, cid):
            return None
        with patch.object(ClaimBrowser, "_fetch_claim", fetch):
            browser = ClaimBrowser(interface, root_id="nonexistent")
        assert browser.root_node is None

    def test_flatten_visible_nodes(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        nodes = browser._flatten_visible()
        # Root is collapsed by default → only root visible
        assert len(nodes) == 1
        assert nodes[0].claim.id == "c0000001"

    def test_flatten_expands_children(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        # Expand the root
        browser._toggle_expand(browser.root_node)
        nodes = browser._flatten_visible()
        # Root + child2 + child3 (+ child4 if child2 also expanded)
        assert len(nodes) == 3  # root expanded, both children visible

    def test_toggle_expand(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        root = browser.root_node
        assert root.expanded is False
        browser._toggle_expand(root)
        assert root.expanded is True
        browser._toggle_expand(root)
        assert root.expanded is False

    def test_move_cursor_down(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        browser._toggle_expand(browser.root_node)  # expand root
        initial = browser.current_index
        browser._move_down()
        assert browser.current_index == initial + 1

    def test_move_cursor_up(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        browser._toggle_expand(browser.root_node)
        browser._move_down()
        browser._move_down()
        initial = browser.current_index
        browser._move_up()
        assert browser.current_index == initial - 1

    def test_cursor_at_top_stays(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        assert browser.current_index == 0
        browser._move_up()
        assert browser.current_index == 0  # should not go negative

    def test_cursor_at_bottom_stays(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        browser._toggle_expand(browser.root_node)
        # Move to end
        nodes = browser._flatten_visible()
        browser.current_index = len(nodes) - 1
        last_idx = browser.current_index
        browser._move_down()
        assert browser.current_index == last_idx  # should not go past end

    def test_enter_key_toggles_expand(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        nodes = browser._flatten_visible()
        browser.current_index = 0
        node = nodes[browser.current_index]
        browser._on_enter(node)
        assert node.expanded is True

    def test_quit_sets_state(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        browser._on_quit()
        assert browser.state == BrowserState.QUIT

    def test_search_mode(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        browser._on_search()
        assert browser.state == BrowserState.SEARCHING
        assert browser.search_query == ""

    def test_exit_search_restores_running(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        browser._on_search()
        browser._on_search_exit()
        assert browser.state == BrowserState.RUNNING

    def test_search_updates_results(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        browser._on_search()
        browser.search_query = "evidence"
        browser._update_search_results()
        # search_claims was called with "evidence"
        mock_interface.search_claims.assert_called_once_with("evidence")

    def test_search_exit_resets_results(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        browser._on_search()
        browser.search_results = [_make_claim()]
        browser._on_search_exit()
        assert browser.search_results == []

    def test_confidence_color_high(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        assert browser._confidence_color(0.95) == "green"

    def test_confidence_color_medium(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        assert browser._confidence_color(0.7) == "yellow"

    def test_confidence_color_low(self, mock_interface, sample_claims):
        browser = self._make_browser(mock_interface, sample_claims, "c0000001")
        # 0.3 < 0.4 threshold → red
        assert browser._confidence_color(0.3) == "red"


@pytest.fixture
def sample_claims():
    """Build a small sample claim tree (module-scoped for reuse)."""
    root = _make_claim("c0000001", "Root hypothesis", 0.9, subs=["c0000002", "c0000003"])
    child2 = _make_claim("c0000002", "Supporting evidence A", 0.85, supers=["c0000001"], subs=["c0000004"])
    child3 = _make_claim("c0000003", "Supporting evidence B", 0.7, supers=["c0000001"])
    child4 = _make_claim("c0000004", "Sub-claim for A", 0.6, supers=["c0000002"])
    return {
        "c0000001": root,
        "c0000002": child2,
        "c0000003": child3,
        "c0000004": child4,
    }


@pytest.fixture
def mock_interface(sample_claims):
    """Mock ProcessingInterface that resolves claims synchronously."""
    interface = MagicMock()
    interface.get_claim = MagicMock(
        side_effect=lambda cid: sample_claims.get(cid)
    )
    interface.search_claims = AsyncMock(
        return_value=[sample_claims["c0000001"], sample_claims["c0000002"], sample_claims["c0000003"]]
    )
    interface.get_statistics = AsyncMock(return_value={
        "total_claims": 4,
        "avg_confidence": 0.7875,
    })
    return interface


class TestBrowseClaimsFunction:
    """Tests for the browse_claims factory function."""

    def test_creates_browser_instance(self, mock_interface, sample_claims):
        def fetch(self, cid):
            return sample_claims.get(cid)
        with patch.object(ClaimBrowser, "_fetch_claim", fetch):
            browser = browse_claims(mock_interface, "c0000001")
        assert isinstance(browser, ClaimBrowser)
        assert browser.root_id == "c0000001"


class TestBrowseCLICommand:
    """Tests for the `conjecture browse` CLI command integration (UX-0007 Phase 3)."""

    def _make_mock_interface(self, sample_claims):
        interface = MagicMock()
        interface.get_claim = MagicMock(
            side_effect=lambda cid: sample_claims.get(cid)
        )
        interface.search_claims = AsyncMock(
            return_value=[sample_claims["c0000001"], sample_claims["c0000002"]]
        )
        return interface

    def test_browse_command_registered_in_app(self):
        """Verify `browse` is a registered Typer command on the CLI app — check --help works."""
        from src.cli.modular_cli import app
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["browse", "--help"])
        assert result.exit_code == 0, (
            f"`browse --help` failed with exit code {result.exit_code}. "
            f"stderr: {result.stderr[:200] if result.stderr else 'none'}"
        )
        assert "interactive" in result.output.lower() or "browse" in result.output.lower()

    def test_browse_command_instantiates_browser(self, sample_claims):
        """`browse` command creates a ClaimBrowser and calls run_interactive."""
        mock_interface = self._make_mock_interface(sample_claims)

        def fetch(self, cid):
            return sample_claims.get(cid)

        with patch.object(ClaimBrowser, "_fetch_claim", fetch):
            # Patch run_interactive at the source so TTY never runs
            with patch("src.cli.claim_browser.ClaimBrowser.run_interactive", return_value=None):
                with patch("src.cli.modular_cli.current_processing_interface", mock_interface):
                    with patch(
                        "src.cli.modular_cli.get_processing_interface",
                        return_value=mock_interface,
                    ):
                        from src.cli.modular_cli import app
                        from typer.testing import CliRunner

                        runner = CliRunner()
                        result = runner.invoke(app, ["browse", "c0000001", "--max-depth", "3"])
                        # Should exit cleanly (run_interactive is patched to no-op)
                        assert result.exit_code == 0, (
                            f"Unexpected exit code {result.exit_code}. "
                            f"stdout: {result.output[:200]}"
                        )

    def test_browse_command_renders_initial_tree(self, sample_claims):
        """`browse` command renders the initial tree state without crashing."""
        mock_interface = self._make_mock_interface(sample_claims)

        def fetch(self, cid):
            return sample_claims.get(cid)

        with patch.object(ClaimBrowser, "_fetch_claim", fetch):
            with patch("src.cli.claim_browser.ClaimBrowser.run_interactive", return_value=None):
                with patch("src.cli.modular_cli.current_processing_interface", mock_interface):
                    with patch(
                        "src.cli.modular_cli.get_processing_interface",
                        return_value=mock_interface,
                    ):
                        from src.cli.modular_cli import app
                        from typer.testing import CliRunner

                        runner = CliRunner()
                        result = runner.invoke(app, ["browse", "c0000001"])
                        # Should exit cleanly after run_interactive returns
                        assert result.exit_code == 0, (
                            f"browse exited with {result.exit_code}. "
                            f"output: {result.output[:300]}"
                        )
