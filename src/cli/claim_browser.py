# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Claim Browser TUI (UX-0007 Phase 3)

Interactive TUI for browsing claim trees with keyboard navigation.
Supports: j/k (up/down), Enter (expand/collapse), q (quit), / (search).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

from src.core.models import Claim


class BrowserState(Enum):
    """Browser session states."""
    RUNNING = "running"
    SEARCHING = "searching"
    QUIT = "quit"


@dataclass
class ClaimTreeNode:
    """A node in the browsable claim tree."""
    claim: Claim
    children: List["ClaimTreeNode"] = field(default_factory=list)
    depth: int = 0
    expanded: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.claim.id,
            "content": self.claim.content,
            "confidence": self.claim.confidence,
            "depth": self.depth,
            "expanded": self.expanded,
            "children": [c.to_dict() for c in self.children],
        }


class ClaimBrowser:
    """
    Interactive claim tree browser.

    Keyboard controls:
        j / ↓  : Move cursor down
        k / ↑  : Move cursor up
        l / →  : Expand current node (show children)
        h / ←  : Collapse current node (hide children)
        Enter  : Toggle expand/collapse of current node
        /      : Enter search mode
        Esc    : Exit search / cancel
        q      : Quit browser
    """

    def __init__(
        self,
        processing_interface,
        root_id: str,
        max_depth: int = 5,
    ):
        self.interface = processing_interface
        self.root_id = root_id
        self.max_depth = max_depth
        self.state = BrowserState.RUNNING
        self.current_index = 0
        self.search_query = ""
        self.search_results: List[Claim] = []
        self._all_nodes: List[ClaimTreeNode] = []
        self.root_node: Optional[ClaimTreeNode] = None

        self._build_tree()

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------

    def _build_tree(self) -> None:
        """Build the claim tree starting from root_id."""
        root_claim = self._fetch_claim(self.root_id)
        if root_claim is None:
            return
        self.root_node = self._build_node(root_claim, depth=0)
        self._refresh_flat()

    def _build_node(self, claim: Claim, depth: int) -> ClaimTreeNode:
        """Recursively build a tree node with its children."""
        if depth >= self.max_depth:
            return ClaimTreeNode(claim=claim, depth=depth, expanded=False)

        children = []
        for sub_id in (claim.subs or []):
            child_claim = self._fetch_claim(sub_id)
            if child_claim:
                children.append(self._build_node(child_claim, depth + 1))

        return ClaimTreeNode(
            claim=claim,
            children=children,
            depth=depth,
            expanded=False,
        )

    def _fetch_claim(self, claim_id: str) -> Optional[Claim]:
        """Fetch a single claim from the interface. Override in tests."""
        return self._sync_fetch(claim_id)

    def _sync_fetch(self, claim_id: str) -> Optional[Claim]:
        """Synchronous claim fetch using thread pool."""
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.interface.get_claim(claim_id))
            finally:
                loop.close()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Flat view management
    # ------------------------------------------------------------------

    def _flatten_visible(self) -> List[ClaimTreeNode]:
        """Return currently visible nodes as a flat list."""
        if self.root_node is None:
            return []
        result = []
        self._flatten_recursive(self.root_node, result)
        return result

    def _flatten_recursive(
        self, node: ClaimTreeNode, out: List[ClaimTreeNode]
    ) -> None:
        out.append(node)
        if node.expanded:
            for child in node.children:
                self._flatten_recursive(child, out)

    def _refresh_flat(self) -> None:
        """Rebuild the flat visible list and clamp cursor."""
        self._all_nodes = self._flatten_visible()
        if self.current_index >= len(self._all_nodes):
            self.current_index = max(0, len(self._all_nodes) - 1)

    # ------------------------------------------------------------------
    # Cursor movement
    # ------------------------------------------------------------------

    def _move_down(self) -> None:
        if self.current_index < len(self._all_nodes) - 1:
            self.current_index += 1

    def _move_up(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1

    # ------------------------------------------------------------------
    # Node interaction
    # ------------------------------------------------------------------

    def _toggle_expand(self, node: ClaimTreeNode) -> None:
        if node.children:
            node.expanded = not node.expanded
            self._refresh_flat()

    def _on_enter(self, node: ClaimTreeNode) -> None:
        """Handle Enter key — toggle current node."""
        self._toggle_expand(node)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _on_search(self) -> None:
        """Enter search mode."""
        self.state = BrowserState.SEARCHING
        self.search_query = ""

    def _on_search_exit(self) -> None:
        """Exit search mode, restore running."""
        self.state = BrowserState.RUNNING
        self.search_query = ""
        self.search_results = []

    def _update_search_results(self) -> None:
        """Run semantic search and cache results."""
        if not self.search_query.strip():
            self.search_results = []
            return
        import asyncio
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.search_results = (
                loop.run_until_complete(
                    self.interface.search_claims(self.search_query)
                )
                or []
            )
            loop.close()
        except Exception:
            self.search_results = []

    # ------------------------------------------------------------------
    # Quit
    # ------------------------------------------------------------------

    def _on_quit(self) -> None:
        self.state = BrowserState.QUIT

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _confidence_color(self, confidence: float) -> str:
        if confidence >= 0.8:
            return "green"
        elif confidence >= 0.6:
            return "yellow"
        elif confidence >= 0.4:
            return "orange"
        return "red"

    def _format_node_line(
        self, node: ClaimTreeNode, is_selected: bool
    ) -> str:
        """Format a single line for a visible node."""
        marker = "▶ " if is_selected else "  "
        conf = node.claim.confidence
        color = self._confidence_color(conf)
        conf_marker = "●" if conf >= 0.8 else "○"
        indent = "  " * node.depth
        pipe = "│  " if node.depth > 0 else ""
        content = node.claim.content[:70]
        if len(node.claim.content) > 70:
            content += "…"

        exp_icon = "▼" if node.expanded else "▶" if node.children else " "
        return (
            f"{marker}{indent}{pipe}{exp_icon} "
            f"[{color}]{node.claim.id}[/{color}] "
            f"[{color}]{conf_marker}[/{color}] "
            f"{content}"
        )

    def _get_key(self) -> str:
        """Read a single keypress from the terminal. Cross-platform."""
        import sys
        try:
            # Unix
            import tty
            import termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    seq = sys.stdin.read(2)
                    return ch + seq
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            # Windows fallback
            try:
                import msvcrt
                ch = msvcrt.getch()
                if ch in (b"\x00", b"\xe0"):
                    return ch + msvcrt.getch()
                return ch.decode("utf-8", errors="replace")
            except Exception:
                return "q"

    def run_interactive(self) -> None:
        """Run the interactive browser loop.

        Reads single keystrokes and updates browser state.
        Handles: j/k (navigate), l/Enter (expand), h (collapse),
        / (search), q (quit), Escape (cancel), g/G (top/bottom).
        """
        import sys
        CLEAR_SCREEN = "\033[2J\033[H"
        HIDE_CURSOR = "\033[?25l"
        SHOW_CURSOR = "\033[?25h"

        try:
            sys.stdout.write(HIDE_CURSOR)
            sys.stdout.flush()
            while self.state != BrowserState.QUIT:
                lines = self.render()
                output = CLEAR_SCREEN + "\n".join(lines) + "\n"
                sys.stdout.write(output)
                sys.stdout.flush()
                key = self._get_key()
                if self.state == BrowserState.SEARCHING:
                    self._handle_search_key(key)
                else:
                    self._handle_tree_key(key)
        finally:
            sys.stdout.write(SHOW_CURSOR + CLEAR_SCREEN)
            sys.stdout.flush()

    def _handle_tree_key(self, key: str) -> None:
        """Handle a keypress in tree/running state."""
        if key in ("j", "\x1b[B") or key == b"\x00\x50":
            self._move_down()
        elif key in ("k", "\x1b[A") or key == b"\x00\x48":
            self._move_up()
        elif key in ("l", "\x1b[C", " ", "\r", "\n"):
            if key in ("\r", "\n"):
                node = self._all_nodes[self.current_index]
                self._toggle_expand(node)
            else:
                node = self._all_nodes[self.current_index]
                self._toggle_expand(node)
        elif key == "h" or key == "\x1b[D":
            node = self._all_nodes[self.current_index]
            if node.expanded:
                node.expanded = False
                self._refresh_flat()
        elif key == "/":
            self._on_search()
        elif key in ("q", "Q"):
            self._on_quit()
        elif key == "g":
            self.current_index = 0
        elif key == "G":
            self.current_index = max(0, len(self._all_nodes) - 1)

    def _handle_search_key(self, key: str) -> None:
        """Handle a keypress in search mode."""
        if key == "\x1b":
            self._on_search_exit()
        elif key == "\r" or key == "\n":
            self._update_search_results()
        elif key == "\x7f":
            self.search_query = self.search_query[:-1]
        elif len(key) == 1 and key.isprintable():
            self.search_query += key

    def render(self) -> List[str]:
        """Render all visible lines for the current browser state."""
        lines = []

        if self.state == BrowserState.SEARCHING:
            lines.append("")
            lines.append(f"  [/] Search: {self.search_query}|")
            lines.append("")
            if self.search_results:
                for i, claim in enumerate(self.search_results[:20]):
                    color = self._confidence_color(claim.confidence)
                    marker = "▶" if i == 0 else " "
                    content = claim.content[:72]
                    if len(claim.content) > 72:
                        content += "…"
                    lines.append(
                        f"  {marker} [{color}]{claim.id}[/{color}] "
                        f"[{color}]{claim.confidence:.2f}[/{color}] {content}"
                    )
            else:
                lines.append("  (no results — type query and press Enter to search)")
            return lines

        # Normal tree view
        lines.append("")
        lines.append(f"  [dim]Claim Tree: {self.root_id}[/dim]")
        lines.append(
            "  [dim]j/k: navigate  l/Enter: expand  h: collapse  /: search  q: quit[/dim]"
        )
        lines.append("")

        if not self._all_nodes:
            lines.append("  (no claims to display)")
            return lines

        for i, node in enumerate(self._all_nodes):
            is_selected = (i == self.current_index)
            lines.append(self._format_node_line(node, is_selected))

        return lines


def browse_claims(
    processing_interface,
    root_id: str,
    max_depth: int = 5,
) -> ClaimBrowser:
    """Factory: create and return a ClaimBrowser instance."""
    return ClaimBrowser(
        processing_interface=processing_interface,
        root_id=root_id,
        max_depth=max_depth,
    )
