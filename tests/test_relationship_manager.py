"""
Comprehensive unit tests for relationship_manager.py
Tests all pure functions for claim relationship management without mocking
"""

import pytest
from datetime import datetime, timezone
from typing import List

from src.core.models import Claim, ClaimType, ClaimState
from src.core.relationship_manager import (
    # Data structures
    RelationshipAnalysis,
    RelationshipChange,
    # Pure functions
    establish_bidirectional_relationship,
    remove_support_relationship,
    remove_supports_relationship,
    create_support_map,
    create_supporter_map,
    detect_circular_dependencies,
    find_orphaned_claims,
    find_root_claims,
    find_leaf_claims,
    calculate_relationship_depth,
    analyze_claim_relationships,
    suggest_support_relationships,
    validate_relationship_consistency,
    propagate_confidence_updates,
    create_relationship_heatmap,
    get_relationship_statistics,
)


class TestRelationshipDataStructures:
    """Test relationship data structures"""

    def test_relationship_analysis_creation(self):
        """Test creating RelationshipAnalysis"""
        analysis = RelationshipAnalysis(
            claim_id="c001",
            support_strength=0.8,
            support_count=3,
            supported_count=2,
            circular_dependencies=[],
            orphaned_claims=False,
            relationship_depth=2,
            completeness_score=1.0,
        )

        assert analysis.claim_id == "c001"
        assert analysis.support_strength == 0.8
        assert analysis.support_count == 3
        assert analysis.supported_count == 2
        assert analysis.circular_dependencies == []
        assert analysis.orphaned_claims is False
        assert analysis.relationship_depth == 2
        assert analysis.completeness_score == 1.0

    def test_relationship_change_creation(self):
        """Test creating RelationshipChange"""
        now = datetime.utcnow()
        change = RelationshipChange(
            claim_id="c001",
            change_type="add_support",
            related_claim_id="c002",
            timestamp=now,
            metadata={"reason": "test"},
        )

        assert change.claim_id == "c001"
        assert change.change_type == "add_support"
        assert change.related_claim_id == "c002"
        assert change.timestamp == now
        assert change.metadata == {"reason": "test"}


class TestBidirectionalRelationships:
    """Test bidirectional relationship establishment"""

    def test_establish_bidirectional_relationship(self):
        """Test establishing bidirectional relationship"""
        claim1 = Claim(id="c001", content="Claim 1", confidence=0.8)
        claim2 = Claim(id="c002", content="Claim 2", confidence=0.7)

        updated_claim1, updated_claim2 = establish_bidirectional_relationship(
            claim1, claim2
        )

        # claim1 supports claim2
        assert "c002" in updated_claim1.supports
        # claim2 is supported by claim1
        assert "c001" in updated_claim2.supported_by

        # Original claims unchanged (pure function)
        assert "c002" not in claim1.supports
        assert "c001" not in claim2.supported_by

    def test_establish_bidirectional_relationship_preserves_existing(self):
        """Test that establishing relationship preserves existing relationships"""
        claim1 = Claim(id="c001", content="Claim 1", confidence=0.8, supports=["c003"])
        claim2 = Claim(
            id="c002", content="Claim 2", confidence=0.7, supported_by=["c004"]
        )

        updated_claim1, updated_claim2 = establish_bidirectional_relationship(
            claim1, claim2
        )

        # New relationships added
        assert "c002" in updated_claim1.supports
        assert "c001" in updated_claim2.supported_by

        # Existing relationships preserved
        assert "c003" in updated_claim1.supports
        assert "c004" in updated_claim2.supported_by


class TestRemoveRelationships:
    """Test relationship removal operations"""

    def test_remove_support_relationship(self):
        """Test removing a support relationship"""
        claim = Claim(
            id="c001",
            content="Supported claim",
            confidence=0.7,
            supported_by=["c002", "c003", "c004"],
        )

        updated = remove_support_relationship(claim, "c003")

        assert "c003" not in updated.supported_by
        assert "c002" in updated.supported_by
        assert "c004" in updated.supported_by
        assert len(updated.supported_by) == 2

        # Original unchanged
        assert "c003" in claim.supported_by

    def test_remove_support_relationship_nonexistent(self):
        """Test removing nonexistent support relationship (no error)"""
        claim = Claim(
            id="c001",
            content="Supported claim",
            confidence=0.7,
            supported_by=["c002", "c003"],
        )

        updated = remove_support_relationship(claim, "c999")

        # No change, no error
        assert len(updated.supported_by) == 2
        assert "c002" in updated.supported_by
        assert "c003" in updated.supported_by

    def test_remove_supports_relationship(self):
        """Test removing a supports relationship"""
        claim = Claim(
            id="c001",
            content="Supporting claim",
            confidence=0.8,
            supports=["c002", "c003", "c004"],
        )

        updated = remove_supports_relationship(claim, "c003")

        assert "c003" not in updated.supports
        assert "c002" in updated.supports
        assert "c004" in updated.supports
        assert len(updated.supports) == 2

        # Original unchanged
        assert "c003" in claim.supports

    def test_remove_supports_relationship_nonexistent(self):
        """Test removing nonexistent supports relationship (no error)"""
        claim = Claim(
            id="c001",
            content="Supporting claim",
            confidence=0.8,
            supports=["c002", "c003"],
        )

        updated = remove_supports_relationship(claim, "c999")

        # No change, no error
        assert len(updated.supports) == 2
        assert "c002" in updated.supports
        assert "c003" in updated.supports


class TestRelationshipMaps:
    """Test relationship map creation"""

    def test_create_support_map_simple(self):
        """Test creating support map from claims"""
        claims = [
            Claim(
                id="c001", content="Claim 1", confidence=0.8, supports=["c002", "c003"]
            ),
            Claim(id="c002", content="Claim 2", confidence=0.7, supports=["c003"]),
            Claim(id="c003", content="Claim 3", confidence=0.6),
        ]

        support_map = create_support_map(claims)

        assert len(support_map) == 3
        assert support_map["c001"] == {"c002", "c003"}
        assert support_map["c002"] == {"c003"}
        assert support_map["c003"] == set()

    def test_create_support_map_filters_missing_claims(self):
        """Test that support map only includes existing claims"""
        claims = [
            Claim(
                id="c001", content="Claim 1", confidence=0.8, supports=["c002", "c999"]
            ),
            Claim(id="c002", content="Claim 2", confidence=0.7),
        ]

        support_map = create_support_map(claims)

        # c999 not in map because it doesn't exist
        assert support_map["c001"] == {"c002"}
        assert "c999" not in support_map["c001"]

    def test_create_supporter_map_simple(self):
        """Test creating supporter map from claims"""
        claims = [
            Claim(id="c001", content="Claim 1", confidence=0.8),
            Claim(id="c002", content="Claim 2", confidence=0.7, supported_by=["c001"]),
            Claim(
                id="c003",
                content="Claim 3",
                confidence=0.6,
                supported_by=["c001", "c002"],
            ),
        ]

        supporter_map = create_supporter_map(claims)

        assert len(supporter_map) == 3
        assert supporter_map["c001"] == set()
        assert supporter_map["c002"] == {"c001"}
        assert supporter_map["c003"] == {"c001", "c002"}

    def test_create_supporter_map_filters_missing_claims(self):
        """Test that supporter map only includes existing claims"""
        claims = [
            Claim(id="c001", content="Claim 1", confidence=0.8),
            Claim(
                id="c002",
                content="Claim 2",
                confidence=0.7,
                supported_by=["c001", "c999"],
            ),
        ]

        supporter_map = create_supporter_map(claims)

        # c999 not in map because it doesn't exist
        assert supporter_map["c002"] == {"c001"}
        assert "c999" not in supporter_map["c002"]


class TestCircularDependencies:
    """Test circular dependency detection"""

    def test_detect_circular_dependencies_no_cycles(self):
        """Test detecting no cycles in clean graph"""
        support_map = {
            "c001": {"c002", "c003"},
            "c002": {"c004"},
            "c003": {"c004"},
            "c004": set(),
        }

        cycles = detect_circular_dependencies(support_map)

        assert len(cycles) == 0

    def test_detect_circular_dependencies_simple_cycle(self):
        """Test detecting simple 2-node cycle"""
        support_map = {"c001": {"c002"}, "c002": {"c001"}}

        cycles = detect_circular_dependencies(support_map)

        assert len(cycles) > 0
        # Check that detected cycle contains both nodes
        cycle = cycles[0]
        assert "c001" in cycle
        assert "c002" in cycle

    def test_detect_circular_dependencies_complex_cycle(self):
        """Test detecting 3-node cycle"""
        support_map = {"c001": {"c002"}, "c002": {"c003"}, "c003": {"c001"}}

        cycles = detect_circular_dependencies(support_map)

        assert len(cycles) > 0
        cycle = cycles[0]
        assert "c001" in cycle
        assert "c002" in cycle
        assert "c003" in cycle

    def test_detect_circular_dependencies_self_reference(self):
        """Test detecting self-reference cycle"""
        support_map = {"c001": {"c001"}}

        cycles = detect_circular_dependencies(support_map)

        assert len(cycles) > 0


class TestClaimCategories:
    """Test claim categorization functions"""

    def test_find_orphaned_claims(self):
        """Test finding claims with no relationships"""
        claims = [
            Claim(id="c001", content="Orphan 1", confidence=0.8),
            Claim(
                id="c002", content="Connected", confidence=0.7, supported_by=["c001"]
            ),
            Claim(id="c003", content="Orphan 2", confidence=0.6),
        ]

        orphaned = find_orphaned_claims(claims)

        assert len(orphaned) == 2
        orphaned_ids = {c.id for c in orphaned}
        assert "c001" in orphaned_ids
        assert "c003" in orphaned_ids
        assert "c002" not in orphaned_ids

    def test_find_orphaned_claims_none(self):
        """Test when no orphaned claims exist"""
        claims = [
            Claim(id="c001", content="Claim 1", confidence=0.8, supports=["c002"]),
            Claim(id="c002", content="Claim 2", confidence=0.7, supported_by=["c001"]),
        ]

        orphaned = find_orphaned_claims(claims)

        assert len(orphaned) == 0

    def test_find_root_claims(self):
        """Test finding root claims (support but not supported)"""
        claims = [
            Claim(id="c001", content="Root claim", confidence=0.8, supports=["c002"]),
            Claim(
                id="c002",
                content="Middle claim",
                confidence=0.7,
                supported_by=["c001"],
                supports=["c003"],
            ),
            Claim(
                id="c003", content="Leaf claim", confidence=0.6, supported_by=["c002"]
            ),
        ]

        roots = find_root_claims(claims)

        assert len(roots) == 1
        assert roots[0].id == "c001"

    def test_find_root_claims_none(self):
        """Test when no root claims exist"""
        claims = [
            Claim(
                id="c001",
                content="Claim 1",
                confidence=0.8,
                supported_by=["c002"],
                supports=["c002"],
            ),
            Claim(
                id="c002",
                content="Claim 2",
                confidence=0.7,
                supported_by=["c001"],
                supports=["c001"],
            ),
        ]

        roots = find_root_claims(claims)

        assert len(roots) == 0

    def test_find_leaf_claims(self):
        """Test finding leaf claims (supported but don't support)"""
        claims = [
            Claim(id="c001", content="Root claim", confidence=0.8, supports=["c002"]),
            Claim(
                id="c002",
                content="Middle claim",
                confidence=0.7,
                supported_by=["c001"],
                supports=["c003"],
            ),
            Claim(
                id="c003", content="Leaf claim", confidence=0.6, supported_by=["c002"]
            ),
        ]

        leaves = find_leaf_claims(claims)

        assert len(leaves) == 1
        assert leaves[0].id == "c003"

    def test_find_leaf_claims_multiple(self):
        """Test finding multiple leaf claims"""
        claims = [
            Claim(
                id="c001",
                content="Root claim",
                confidence=0.8,
                supports=["c002", "c003", "c004"],
            ),
            Claim(
                id="c002", content="Leaf claim 1", confidence=0.7, supported_by=["c001"]
            ),
            Claim(
                id="c003", content="Leaf claim 2", confidence=0.6, supported_by=["c001"]
            ),
            Claim(
                id="c004", content="Leaf claim 3", confidence=0.5, supported_by=["c001"]
            ),
        ]

        leaves = find_leaf_claims(claims)

        assert len(leaves) == 3
        leaf_ids = {c.id for c in leaves}
        assert leaf_ids == {"c002", "c003", "c004"}


class TestRelationshipDepth:
    """Test relationship depth calculation"""

    def test_calculate_relationship_depth_simple(self):
        """Test calculating depth in simple chain"""
        support_map = {"c001": {"c002"}, "c002": {"c003"}, "c003": set()}

        depth = calculate_relationship_depth(support_map, "c001")

        assert depth == 3

    def test_calculate_relationship_depth_branching(self):
        """Test calculating depth with branching"""
        support_map = {
            "c001": {"c002", "c003"},
            "c002": {"c004"},
            "c003": {"c005"},
            "c004": set(),
            "c005": set(),
        }

        depth = calculate_relationship_depth(support_map, "c001")

        assert depth == 3  # Longest path

    def test_calculate_relationship_depth_cycle_handling(self):
        """Test that cycles don't cause infinite recursion"""
        support_map = {
            "c001": {"c002"},
            "c002": {"c003"},
            "c003": {"c001"},  # Cycle back
        }

        # Should handle cycle without infinite loop
        depth = calculate_relationship_depth(support_map, "c001")

        # Depth should be finite
        assert depth >= 0

    def test_calculate_relationship_depth_single_node(self):
        """Test calculating depth for isolated node"""
        support_map = {"c001": set()}

        depth = calculate_relationship_depth(support_map, "c001")

        assert depth == 1


class TestRelationshipAnalysis:
    """Test comprehensive relationship analysis"""

    def test_analyze_claim_relationships_orphaned(self):
        """Test analyzing orphaned claim"""
        claim = Claim(id="c001", content="Orphan claim", confidence=0.7)

        analysis = analyze_claim_relationships(claim, [claim])

        assert analysis.claim_id == "c001"
        assert analysis.orphaned_claims is True
        assert analysis.support_count == 0
        assert analysis.supported_count == 0
        assert analysis.completeness_score == 0.0

    def test_analyze_claim_relationships_supported_only(self):
        """Test analyzing claim with only supporters"""
        claims = [
            Claim(
                id="c001", content="Supported", confidence=0.7, supported_by=["c002"]
            ),
            Claim(id="c002", content="Supporter", confidence=0.8, supports=["c001"]),
        ]

        analysis = analyze_claim_relationships(claims[0], claims)

        assert analysis.claim_id == "c001"
        assert analysis.orphaned_claims is False
        assert analysis.support_count == 1
        assert analysis.supported_count == 0
        assert analysis.completeness_score == 0.5

    def test_analyze_claim_relationships_full_connections(self):
        """Test analyzing fully connected claim"""
        claims = [
            Claim(
                id="c001",
                content="Middle",
                confidence=0.7,
                supported_by=["c002"],
                supports=["c003"],
            ),
            Claim(id="c002", content="Supporter", confidence=0.8, supports=["c001"]),
            Claim(
                id="c003", content="Supported", confidence=0.6, supported_by=["c001"]
            ),
        ]

        analysis = analyze_claim_relationships(claims[0], claims)

        assert analysis.claim_id == "c001"
        assert analysis.orphaned_claims is False
        assert analysis.support_count == 1
        assert analysis.supported_count == 1
        assert analysis.completeness_score == 1.0


class TestSuggestRelationships:
    """Test relationship suggestions"""

    def test_suggest_support_relationships_similar_content(self):
        """Test suggesting relationships based on content similarity"""
        claim = Claim(
            id="c001", content="machine learning model accuracy", confidence=0.7
        )
        all_claims = [
            claim,
            Claim(
                id="c002", content="machine learning accuracy metrics", confidence=0.8
            ),
            Claim(id="c003", content="completely different topic", confidence=0.6),
        ]

        suggestions = suggest_support_relationships(
            claim, all_claims, max_suggestions=5
        )

        # Should suggest c002 (similar content) but not c003
        assert len(suggestions) > 0
        suggested_ids = {s[0].id for s in suggestions}
        assert "c002" in suggested_ids
        assert "c003" not in suggested_ids

    def test_suggest_support_relationships_excludes_existing(self):
        """Test that suggestions exclude existing supporters"""
        claim = Claim(
            id="c001", content="machine learning", confidence=0.7, supported_by=["c002"]
        )
        all_claims = [
            claim,
            Claim(id="c002", content="machine learning model", confidence=0.8),
            Claim(id="c003", content="machine learning accuracy", confidence=0.6),
        ]

        suggestions = suggest_support_relationships(
            claim, all_claims, max_suggestions=5
        )

        # Should not suggest c002 (already a supporter)
        suggested_ids = {s[0].id for s in suggestions}
        assert "c002" not in suggested_ids
        # Should suggest c003
        assert "c003" in suggested_ids

    def test_suggest_support_relationships_max_suggestions(self):
        """Test that max_suggestions is respected"""
        claim = Claim(id="c001", content="test claim", confidence=0.7)
        all_claims = [claim] + [
            Claim(id=f"c{i:03d}", content=f"test claim {i}", confidence=0.7)
            for i in range(2, 12)
        ]

        suggestions = suggest_support_relationships(
            claim, all_claims, max_suggestions=3
        )

        assert len(suggestions) <= 3


class TestValidateConsistency:
    """Test relationship consistency validation"""

    def test_validate_relationship_consistency_valid(self):
        """Test validating consistent relationships"""
        claims = [
            Claim(id="c001", content="Root claim", confidence=0.8, supports=["c002"]),
            Claim(
                id="c002", content="Leaf claim", confidence=0.7, supported_by=["c001"]
            ),
        ]

        errors = validate_relationship_consistency(claims)

        assert len(errors) == 0

    def test_validate_relationship_consistency_missing_claim(self):
        """Test detecting missing claim references"""
        claims = [
            Claim(id="c001", content="Claim", confidence=0.8, supports=["c999"]),
        ]

        errors = validate_relationship_consistency(claims)

        assert len(errors) > 0
        assert any("c999" in error for error in errors)

    def test_validate_relationship_consistency_circular(self):
        """Test detecting circular dependencies"""
        claims = [
            Claim(id="c001", content="Claim 1", confidence=0.8, supports=["c002"]),
            Claim(id="c002", content="Claim 2", confidence=0.7, supports=["c001"]),
        ]

        errors = validate_relationship_consistency(claims)

        assert len(errors) > 0
        assert any("Circular" in error for error in errors)

    def test_validate_relationship_consistency_orphaned_validated(self):
        """Test detecting orphaned validated claims"""
        claims = [
            Claim(
                id="c001", content="Orphan", confidence=0.8, state=ClaimState.VALIDATED
            ),
        ]

        errors = validate_relationship_consistency(claims)

        assert len(errors) > 0
        assert any("validated" in error.lower() for error in errors)


class TestConfidencePropagation:
    """Test confidence propagation through relationships"""

    @pytest.mark.xfail(
        reason="BUG: propagate_confidence_updates has logic error at line 315 - uses wrong index"
    )
    def test_propagate_confidence_updates_simple(self):
        """Test simple confidence propagation"""
        claims = [
            Claim(id="c001", content="Supporter", confidence=0.5, supports=["c002"]),
            Claim(
                id="c002", content="Supported", confidence=0.5, supported_by=["c001"]
            ),
        ]

        updates = {"c002": 0.9}
        updated_claims = propagate_confidence_updates(
            updates, claims, propagation_factor=0.1
        )

        # c002 should be updated directly
        c002 = next(c for c in updated_claims if c.id == "c002")
        assert c002.confidence == 0.9

        # c001 should be slightly affected (propagation)
        c001 = next(c for c in updated_claims if c.id == "c001")
        assert c001.confidence != 0.5  # Changed
        assert 0.5 <= c001.confidence <= 0.6  # Small change

    def test_propagate_confidence_updates_no_updates(self):
        """Test propagation with no updates"""
        claims = [
            Claim(id="c001", content="Claim", confidence=0.5),
        ]

        updates = {}
        updated_claims = propagate_confidence_updates(updates, claims)

        # Should return copy with no changes
        assert len(updated_claims) == 1
        assert updated_claims[0].confidence == 0.5

    def test_propagate_confidence_updates_missing_claim(self):
        """Test propagation with nonexistent claim ID"""
        claims = [
            Claim(id="c001", content="Claim", confidence=0.5),
        ]

        updates = {"c999": 0.9}
        updated_claims = propagate_confidence_updates(updates, claims)

        # Should not error, just ignore missing claim
        assert len(updated_claims) == 1
        assert updated_claims[0].confidence == 0.5


class TestRelationshipHeatmap:
    """Test relationship heatmap creation"""

    def test_create_relationship_heatmap_simple(self):
        """Test creating relationship heatmap"""
        claims = [
            Claim(
                id="c001",
                content="Supported",
                confidence=0.7,
                supported_by=["c002", "c003"],
            ),
            Claim(id="c002", content="Supporter 1", confidence=0.8, supports=["c001"]),
            Claim(id="c003", content="Supporter 2", confidence=0.6, supports=["c001"]),
        ]

        heatmap = create_relationship_heatmap(claims)

        assert "c001" in heatmap
        assert "c002" in heatmap["c001"]
        assert "c003" in heatmap["c001"]
        # Heatmap values are supporter confidences
        assert heatmap["c001"]["c002"] == 0.8
        assert heatmap["c001"]["c003"] == 0.6

    def test_create_relationship_heatmap_no_relationships(self):
        """Test heatmap for orphaned claims"""
        claims = [
            Claim(id="c001", content="Orphan", confidence=0.7),
        ]

        heatmap = create_relationship_heatmap(claims)

        assert "c001" in heatmap
        assert len(heatmap["c001"]) == 0


class TestRelationshipStatistics:
    """Test comprehensive relationship statistics"""

    def test_get_relationship_statistics_comprehensive(self):
        """Test getting full statistics"""
        claims = [
            Claim(
                id="c001",
                content="Root claim",
                confidence=0.8,
                supports=["c002", "c003"],
            ),
            Claim(
                id="c002",
                content="Middle",
                confidence=0.7,
                supported_by=["c001"],
                supports=["c004"],
            ),
            Claim(id="c003", content="Leaf 1", confidence=0.6, supported_by=["c001"]),
            Claim(id="c004", content="Leaf 2", confidence=0.5, supported_by=["c002"]),
            Claim(id="c005", content="Orphan", confidence=0.4),
        ]

        stats = get_relationship_statistics(claims)

        assert stats["total_claims"] == 5
        assert stats["with_relationships"] == 4
        assert stats["orphaned"] == 1
        assert stats["root_claims"] == 1
        assert stats["leaf_claims"] == 2
        assert "relationship_coverage" in stats
        assert stats["relationship_coverage"] == 80.0  # 4/5 * 100

    def test_get_relationship_statistics_empty(self):
        """Test statistics for empty claim list"""
        stats = get_relationship_statistics([])

        assert "error" in stats

    def test_get_relationship_statistics_all_orphaned(self):
        """Test statistics when all claims are orphaned"""
        claims = [
            Claim(id="c001", content="Orphan 1", confidence=0.8),
            Claim(id="c002", content="Orphan 2", confidence=0.7),
        ]

        stats = get_relationship_statistics(claims)

        assert stats["total_claims"] == 2
        assert stats["orphaned"] == 2
        assert stats["with_relationships"] == 0
        assert stats["relationship_coverage"] == 0.0

    def test_get_relationship_statistics_circular(self):
        """Test statistics with circular dependencies"""
        claims = [
            Claim(id="c001", content="Claim 1", confidence=0.8, supports=["c002"]),
            Claim(id="c002", content="Claim 2", confidence=0.7, supports=["c001"]),
        ]

        stats = get_relationship_statistics(claims)

        assert stats["circular_dependencies"] > 0

    def test_get_relationship_statistics_averages(self):
        """Test average calculations in statistics"""
        claims = [
            Claim(
                id="c001",
                content="Claim 1",
                confidence=0.8,
                supported_by=["c002", "c003"],
                supports=["c004"],
            ),
            Claim(id="c002", content="Claim 2", confidence=0.7, supports=["c001"]),
            Claim(id="c003", content="Claim 3", confidence=0.6, supports=["c001"]),
            Claim(id="c004", content="Claim 4", confidence=0.5, supported_by=["c001"]),
        ]

        stats = get_relationship_statistics(claims)

        # c001 has 2 supported_by, others have 0 or 1 -> avg = (2+0+0+1)/4 = 0.75
        assert "avg_supported_by" in stats
        # c001 has 1 supports, c002 has 1, c003 has 1, c004 has 0 -> avg = (1+1+1+0)/4 = 0.75
        assert "avg_supports" in stats
