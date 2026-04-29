"""
Tests for Claim Visualization Utilities (UX-0007)

Tests tree building, trace construction, and graph generation.
"""

import pytest
from datetime import datetime, timezone
from src.data.models import Claim, ClaimState, ClaimType, ClaimScope
from src.utils.visualization import (
    build_claim_tree,
    build_claim_trace,
    build_claim_graph,
    ClaimNode,
    ClaimTrace,
    ClaimGraph,
    confidence_color,
    format_claim_summary,
    render_tree_ascii,
)


def make_claim(
    claim_id: str,
    content: str = "Test claim",
    confidence: float = 0.8,
    subs: list = None,
    supers: list = None,
) -> Claim:
    """Helper to create test claims."""
    return Claim(
        id=claim_id,
        content=content,
        confidence=confidence,
        state=ClaimState.EXPLORE,
        type=[ClaimType.CONCEPT],
        scope=ClaimScope.USER_WORKSPACE,
        subs=subs or [],
        supers=supers or [],
        created=datetime.now(timezone.utc),
    )


class TestConfidenceColor:
    """Tests for confidence_color function."""

    def test_high_confidence_green(self):
        assert confidence_color(0.9) == "green"
        assert confidence_color(0.95) == "green"
        assert confidence_color(1.0) == "green"

    def test_medium_high_confidence_yellow(self):
        assert confidence_color(0.7) == "yellow"
        assert confidence_color(0.8) == "yellow"

    def test_medium_confidence_orange(self):
        assert confidence_color(0.5) == "orange"
        assert confidence_color(0.6) == "orange"

    def test_low_confidence_red(self):
        assert confidence_color(0.3) == "red"
        assert confidence_color(0.4) == "red"
        assert confidence_color(0.0) == "red"


class TestFormatClaimSummary:
    """Tests for format_claim_summary function."""

    def test_short_content(self):
        claim = make_claim("c00000001", "Short text")
        result = format_claim_summary(claim, max_content=20)
        assert "c00000001" in result
        assert "Short text" in result

    def test_long_content_truncated(self):
        long_text = "A" * 100
        claim = make_claim("c00000001", long_text)
        result = format_claim_summary(claim, max_content=60)
        assert "c00000001" in result
        assert "..." in result
        assert len(result) <= 100  # ID + 60 chars + ellipsis

    def test_max_content_exact(self):
        claim = make_claim("c00000001", "ExactlySixtyCharactersOfTextHereXXXXX")
        result = format_claim_summary(claim, max_content=60)
        assert "..." not in result


class TestBuildClaimTree:
    """Tests for build_claim_tree function."""

    def test_single_claim_no_children(self):
        """A claim with no subs should return just that node."""
        claim = make_claim("c00000001")
        
        def get_claim(cid):
            raise ValueError("Not found")
        
        tree = build_claim_tree(claim, get_claim, max_depth=3)
        
        assert tree.claim.id == "c00000001"
        assert tree.depth == 0
        assert len(tree.children) == 0

    def test_tree_with_children(self):
        """A claim with subs should include them."""
        parent = make_claim("c00000001", subs=["c00000002", "c00000003"])
        child1 = make_claim("c00000002")
        child2 = make_claim("c00000003")
        
        def get_claim(cid):
            claims = {"c00000001": parent, "c00000002": child1, "c00000003": child2}
            if cid not in claims:
                raise ValueError("Not found")
            return claims[cid]
        
        tree = build_claim_tree(parent, get_claim, max_depth=3)
        
        assert tree.claim.id == "c00000001"
        assert len(tree.children) == 2
        assert tree.children[0].claim.id == "c00000002"
        assert tree.children[1].claim.id == "c00000003"

    def test_depth_limit(self):
        """Tree should respect max_depth."""
        parent = make_claim("c00000001", subs=["c00000002"])
        child = make_claim("c00000002", subs=["c00000003"])
        grandchild = make_claim("c00000003")
        
        def get_claim(cid):
            claims = {
                "c00000001": parent,
                "c00000002": child,
                "c00000003": grandchild,
            }
            if cid not in claims:
                raise ValueError("Not found")
            return claims[cid]
        
        # max_depth=1 should only include direct children
        tree = build_claim_tree(parent, get_claim, max_depth=1)
        
        assert tree.claim.id == "c00000001"
        assert len(tree.children) == 1
        assert tree.children[0].claim.id == "c00000002"
        # Grandchild should NOT be included due to depth limit
        assert len(tree.children[0].children) == 0

    def test_cycle_detection(self):
        """Should not infinite loop on cycles."""
        # Create a cycle: c1 -> c2 -> c1
        parent = make_claim("c00000001", subs=["c00000002"])
        child = make_claim("c00000002", subs=["c00000001"])
        
        def get_claim(cid):
            claims = {"c00000001": parent, "c00000002": child}
            if cid not in claims:
                raise ValueError("Not found")
            return claims[cid]
        
        tree = build_claim_tree(parent, get_claim, max_depth=10)
        
        # Should complete without infinite loop
        assert tree.claim.id == "c00000001"
        # The child should be included once
        assert len(tree.children) == 1

    def test_confidence_filter(self):
        """Should filter out low confidence claims."""
        parent = make_claim("c00000001", confidence=0.9, subs=["c00000002", "c00000003"])
        low_conf = make_claim("c00000002", confidence=0.3)
        high_conf = make_claim("c00000003", confidence=0.8)
        
        def get_claim(cid):
            claims = {"c00000001": parent, "c00000002": low_conf, "c00000003": high_conf}
            if cid not in claims:
                raise ValueError("Not found")
            return claims[cid]
        
        tree = build_claim_tree(parent, get_claim, min_confidence=0.5)
        
        assert len(tree.children) == 1
        assert tree.children[0].claim.id == "c00000003"

    def test_to_dict_serialization(self):
        """ClaimNode should serialize to dict correctly."""
        claim = make_claim("c00000001")
        node = ClaimNode(claim=claim, children=[], depth=0)
        
        d = node.to_dict()
        
        assert d["id"] == "c00000001"
        assert d["confidence"] == 0.8
        assert d["depth"] == 0
        assert d["children"] == []


class TestBuildClaimTrace:
    """Tests for build_claim_trace function."""

    def test_root_claim(self):
        """A root claim (no supers) should return just itself."""
        claim = make_claim("c00000001")
        
        def get_claim(cid):
            raise ValueError("Not found")
        
        trace = build_claim_trace(claim, get_claim)
        
        assert len(trace.nodes) == 1
        assert trace.nodes[0].claim.id == "c00000001"
        assert trace.path_ids == ["c00000001"]

    def test_trace_to_root(self):
        """Should traverse supers to find root."""
        root = make_claim("c00000001")
        middle = make_claim("c00000002", supers=["c00000001"])
        leaf = make_claim("c00000003", supers=["c00000002"])
        
        def get_claim(cid):
            claims = {"c00000001": root, "c00000002": middle, "c00000003": leaf}
            if cid not in claims:
                raise ValueError("Not found")
            return claims[cid]
        
        trace = build_claim_trace(leaf, get_claim)
        
        assert len(trace.nodes) == 3
        assert trace.path_ids == ["c00000001", "c00000002", "c00000003"]

    def test_cycle_detection(self):
        """Should handle cycles gracefully."""
        parent = make_claim("c00000001", supers=["c00000002"])
        child = make_claim("c00000002", supers=["c00000001"])
        
        def get_claim(cid):
            claims = {"c00000001": parent, "c00000002": child}
            if cid not in claims:
                raise ValueError("Not found")
            return claims[cid]
        
        trace = build_claim_trace(parent, get_claim)
        
        # Should complete without infinite loop
        assert len(trace.nodes) >= 1

    def test_to_dict_serialization(self):
        """ClaimTrace should serialize to dict correctly."""
        claim = make_claim("c00000001")
        node = ClaimNode(claim=claim, children=[], depth=0)
        trace = ClaimTrace(nodes=[node], path_ids=["c00000001"])
        
        d = trace.to_dict()
        
        assert len(d["path"]) == 1
        assert d["path_ids"] == ["c00000001"]
        assert d["length"] == 1


class TestBuildClaimGraph:
    """Tests for build_claim_graph function."""

    def test_single_claim(self):
        """A single claim should create a graph with one node."""
        claim = make_claim("c00000001")
        
        def get_claim(cid):
            if cid == "c00000001":
                return claim
            raise ValueError("Not found")
        
        graph = build_claim_graph(["c00000001"], get_claim)
        
        assert "c00000001" in graph.nodes
        assert len(graph.edges["c00000001"]) == 0
        assert "c00000001" in graph.root_ids

    def test_multiple_claims(self):
        """Multiple related claims should create edges."""
        parent = make_claim("c00000001", subs=["c00000002"])
        child = make_claim("c00000002", supers=["c00000001"])
        
        def get_claim(cid):
            claims = {"c00000001": parent, "c00000002": child}
            if cid not in claims:
                raise ValueError("Not found")
            return claims[cid]
        
        graph = build_claim_graph(["c00000001"], get_claim)
        
        assert "c00000001" in graph.nodes
        assert "c00000002" in graph.nodes
        assert "c00000002" in graph.edges["c00000001"]

    def test_to_dict_for_visualization(self):
        """ClaimGraph should produce D3.js-compatible format."""
        claim = make_claim("c00000001")
        
        def get_claim(cid):
            if cid == "c00000001":
                return claim
            raise ValueError("Not found")
        
        graph = build_claim_graph(["c00000001"], get_claim)
        d = graph.to_dict()
        
        assert "nodes" in d
        assert "edges" in d
        assert "root_ids" in d
        assert len(d["nodes"]) == 1
        assert len(d["edges"]) == 0


class TestRenderTreeAscii:
    """Tests for render_tree_ascii function."""

    def test_single_node(self):
        """A single node should render correctly."""
        claim = make_claim("c00000001", content="Test claim content")
        node = ClaimNode(claim=claim, children=[], depth=0)
        
        result = render_tree_ascii(node)
        
        assert "c00000001" in result
        assert "Test claim" in result

    def test_nested_nodes(self):
        """Nested nodes should show proper indentation."""
        child_claim = make_claim("c00000002", content="Child claim here")
        child_node = ClaimNode(claim=child_claim, children=[], depth=1)
        
        parent_claim = make_claim("c00000001", content="Parent claim here", subs=["c00000002"])
        parent_node = ClaimNode(claim=parent_claim, children=[child_node], depth=0)
        
        result = render_tree_ascii(parent_node)
        
        assert "c00000001" in result
        assert "c00000002" in result
        # Should have indentation
        assert "\n" in result
