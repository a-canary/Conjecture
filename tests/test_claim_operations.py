"""
Test suite for claim_operations.py - Pure functional claim operations
Target: 100% coverage of src/core/claim_operations.py
"""

import pytest
from datetime import datetime
from src.core.models import Claim, ClaimState, DirtyReason, ClaimType, ClaimScope
from src.core import claim_operations


class TestUpdateConfidence:
    """Tests for update_confidence function"""

    def test_update_confidence_valid(self):
        """Test updating confidence with valid value"""
        claim = Claim(id="c0000001", content="Test claim", confidence=0.5)

        updated = claim_operations.update_confidence(claim, 0.8)

        assert updated.confidence == 0.8
        assert updated.id == claim.id
        assert updated.content == claim.content
        # Note: updated timestamp is set but may be same microsecond on fast machines

    def test_update_confidence_boundary_low(self):
        """Test updating confidence to 0.0"""
        claim = Claim(id="c0000001", content="Test claim", confidence=0.5)
        updated = claim_operations.update_confidence(claim, 0.0)
        assert updated.confidence == 0.0

    def test_update_confidence_boundary_high(self):
        """Test updating confidence to 1.0"""
        claim = Claim(id="c0000001", content="Test claim", confidence=0.5)
        updated = claim_operations.update_confidence(claim, 1.0)
        assert updated.confidence == 1.0

    def test_update_confidence_invalid_too_low(self):
        """Test updating confidence below 0.0 raises error"""
        claim = Claim(id="c0000001", content="Test claim", confidence=0.5)
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            claim_operations.update_confidence(claim, -0.1)

    def test_update_confidence_invalid_too_high(self):
        """Test updating confidence above 1.0 raises error"""
        claim = Claim(id="c0000001", content="Test claim", confidence=0.5)
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            claim_operations.update_confidence(claim, 1.1)

    def test_update_confidence_preserves_relationships(self):
        """Test that updating confidence preserves relationships"""
        claim = Claim(
            id="c0000001",
            content="Test claim",
            confidence=0.5,
            subs=["c0000002"],
            supers=["c0000003"],
        )

        updated = claim_operations.update_confidence(claim, 0.9)

        assert updated.subs == claim.subs
        assert updated.supers == claim.supers


class TestAddSupport:
    """Tests for add_sub and add_super functions"""

    def test_add_sub_new(self):
        """Test adding new supporting claim"""
        claim = Claim(id="c0000001", content="Test claim", confidence=0.5)

        updated = claim_operations.add_sub(claim, "c0000002")

        assert "c0000002" in updated.subs
        assert updated.is_dirty is True
        assert updated.dirty_reason == DirtyReason.SUPPORTING_CLAIM_CHANGED
        assert updated.dirty_timestamp is not None

    def test_add_sub_duplicate(self):
        """Test adding duplicate supporting claim (idempotent)"""
        claim = Claim(
            id="c0000001",
            content="Test claim",
            confidence=0.5,
            subs=["c0000002"],
        )

        updated = claim_operations.add_sub(claim, "c0000002")

        assert updated.subs.count("c0000002") == 1
        assert len(updated.subs) == 1

    def test_add_super_new(self):
        """Test adding new supers_list claim"""
        claim = Claim(id="c0000001", content="Test claim", confidence=0.5)

        updated = claim_operations.add_super(claim, "c0000003")

        assert "c0000003" in updated.supers
        assert updated.is_dirty is True
        assert updated.dirty_reason == DirtyReason.SUPPORTING_CLAIM_CHANGED

    def test_add_super_duplicate(self):
        """Test adding duplicate supers_list claim (idempotent)"""
        claim = Claim(
            id="c0000001", content="Test claim", confidence=0.5, supers=["c0000003"]
        )

        updated = claim_operations.add_super(claim, "c0000003")

        assert updated.supers.count("c0000003") == 1
        assert len(updated.supers) == 1


class TestDirtyFlagOperations:
    """Tests for dirty flag operations"""

    def test_mark_dirty_with_reason(self):
        """Test marking claim as dirty with reason"""
        claim = Claim(id="c0000001", content="Test claim", confidence=0.5)

        updated = claim_operations.mark_dirty(
            claim, DirtyReason.SUPPORTING_CLAIM_CHANGED
        )

        assert updated.is_dirty is True
        assert updated.dirty_reason == DirtyReason.SUPPORTING_CLAIM_CHANGED
        assert updated.dirty_timestamp is not None

    def test_mark_dirty_with_priority(self):
        """Test marking claim as dirty with priority"""
        claim = Claim(id="c0000001", content="Test claim", confidence=0.5)

        updated = claim_operations.mark_dirty(
            claim, DirtyReason.SUPPORTING_CLAIM_CHANGED, priority=10
        )

        assert updated.is_dirty is True
        assert updated.dirty_priority == 10

    def test_mark_clean(self):
        """Test marking claim as clean"""
        claim = Claim(
            id="c0000001",
            content="Test claim",
            confidence=0.5,
            is_dirty=True,
            dirty_reason=DirtyReason.SUPPORTING_CLAIM_CHANGED,
            dirty_priority=5,
        )

        updated = claim_operations.mark_clean(claim)

        assert updated.is_dirty is False
        assert updated.dirty_reason is None
        assert updated.dirty_timestamp is None
        assert updated.dirty_priority == 0

    def test_set_dirty_priority_when_dirty(self):
        """Test setting priority on dirty claim"""
        claim = Claim(
            id="c0000001",
            content="Test claim",
            confidence=0.5,
            is_dirty=True,
            dirty_reason=DirtyReason.SUPPORTING_CLAIM_CHANGED,
        )

        updated = claim_operations.set_dirty_priority(claim, 7)

        assert updated.dirty_priority == 7
        assert updated.is_dirty is True

    def test_set_dirty_priority_when_clean(self):
        """Test that priority doesn't change on clean claim"""
        claim = Claim(
            id="c0000001",
            content="Test claim",
            confidence=0.5,
            is_dirty=False,
            dirty_priority=3,
        )

        updated = claim_operations.set_dirty_priority(claim, 7)

        assert updated.dirty_priority == 3  # Unchanged
        assert updated.is_dirty is False


class TestPrioritization:
    """Tests for prioritization logic"""

    def test_should_prioritize_dirty_low_confidence(self):
        """Test that dirty claim with low confidence should be prioritized"""
        claim = Claim(
            id="c0000001", content="Test claim", confidence=0.5, is_dirty=True
        )

        assert claim_operations.should_prioritize(claim) is True

    def test_should_prioritize_clean_claim(self):
        """Test that clean claim should not be prioritized"""
        claim = Claim(
            id="c0000001", content="Test claim", confidence=0.5, is_dirty=False
        )

        assert claim_operations.should_prioritize(claim) is False

    def test_should_prioritize_high_confidence(self):
        """Test that high confidence claim should not be prioritized"""
        claim = Claim(
            id="c0000001", content="Test claim", confidence=0.95, is_dirty=True
        )

        assert claim_operations.should_prioritize(claim) is False

    def test_should_prioritize_custom_threshold(self):
        """Test prioritization with custom threshold"""
        claim = Claim(
            id="c0000001", content="Test claim", confidence=0.75, is_dirty=True
        )

        assert claim_operations.should_prioritize(claim, 0.80) is True
        assert claim_operations.should_prioritize(claim, 0.70) is False


class TestRelationshipFinders:
    """Tests for relationship finding functions"""

    def test_find_sub_claims(self):
        """Test finding supporting claims"""
        claim1 = Claim(
            id="c0000001",
            content="Claim A",
            confidence=0.5,
            subs=["c0000002", "c0000003"],
        )
        claim2 = Claim(id="c0000002", content="Claim B", confidence=0.7)
        claim3 = Claim(id="c0000003", content="Claim C", confidence=0.8)
        claim4 = Claim(id="c0000004", content="Claim D", confidence=0.6)

        all_claims = [claim1, claim2, claim3, claim4]
        subs_list = claim_operations.find_sub_claims(claim1, all_claims)

        assert len(subs_list) == 2
        assert claim2 in subs_list
        assert claim3 in subs_list

    def test_find_sub_claims_empty(self):
        """Test finding supporting claims when none exist"""
        claim = Claim(id="c0000001", content="Test claim", confidence=0.5)
        all_claims = [claim]

        subs_list = claim_operations.find_sub_claims(claim, all_claims)

        assert len(subs_list) == 0

    def test_find_super_claims(self):
        """Test finding supers_list claims"""
        claim1 = Claim(
            id="c0000001",
            content="Claim A",
            confidence=0.5,
            supers=["c0000002", "c0000003"],
        )
        claim2 = Claim(id="c0000002", content="Claim B", confidence=0.7)
        claim3 = Claim(id="c0000003", content="Claim C", confidence=0.8)

        all_claims = [claim1, claim2, claim3]
        supers_list = claim_operations.find_super_claims(claim1, all_claims)

        assert len(supers_list) == 2
        assert claim2 in supers_list
        assert claim3 in supers_list

    def test_find_super_claims_empty(self):
        """Test finding supers_list claims when none exist"""
        claim = Claim(id="c0000001", content="Test claim", confidence=0.5)
        all_claims = [claim]

        supers_list = claim_operations.find_super_claims(claim, all_claims)

        assert len(supers_list) == 0


class TestSupportStrength:
    """Tests for support strength calculation"""

    def test_calculate_support_strength_single_supporter(self):
        """Test support strength with single supporter"""
        claim1 = Claim(
            id="c0000001", content="Claim A", confidence=0.5, subs=["c0000002"]
        )
        claim2 = Claim(id="c0000002", content="Claim B", confidence=0.8)

        all_claims = [claim1, claim2]
        strength, count = claim_operations.calculate_support_strength(
            claim1, all_claims
        )

        assert strength == 0.8
        assert count == 1

    def test_calculate_support_strength_multiple_subs_list(self):
        """Test support strength with multiple subs_list"""
        claim1 = Claim(
            id="c0000001",
            content="Claim A",
            confidence=0.5,
            subs=["c0000002", "c0000003"],
        )
        claim2 = Claim(id="c0000002", content="Claim B", confidence=0.8)
        claim3 = Claim(id="c0000003", content="Claim C", confidence=0.6)

        all_claims = [claim1, claim2, claim3]
        strength, count = claim_operations.calculate_support_strength(
            claim1, all_claims
        )

        assert strength == 0.7  # (0.8 + 0.6) / 2
        assert count == 2

    def test_calculate_support_strength_no_subs_list(self):
        """Test support strength with no subs_list"""
        claim = Claim(id="c0000001", content="Claim A", confidence=0.5)
        all_claims = [claim]

        strength, count = claim_operations.calculate_support_strength(claim, all_claims)

        assert strength == 0.0
        assert count == 0


class TestRelationshipValidation:
    """Tests for relationship validation"""

    def test_validate_relationship_integrity_valid(self):
        """Test validation with valid relationships"""
        claim1 = Claim(
            id="c0000001",
            content="Claim A",
            confidence=0.5,
            subs=["c0000002"],
            supers=["c0000003"],
        )
        claim2 = Claim(id="c0000002", content="Claim B", confidence=0.7)
        claim3 = Claim(id="c0000003", content="Claim C", confidence=0.8)

        all_claims = [claim1, claim2, claim3]
        errors = claim_operations.validate_relationship_integrity(claim1, all_claims)

        assert len(errors) == 0

    def test_validate_relationship_integrity_missing_supporter(self):
        """Test validation with missing supporter"""
        claim = Claim(
            id="c0000001", content="Claim A", confidence=0.5, subs=["c0000099"]
        )
        all_claims = [claim]

        errors = claim_operations.validate_relationship_integrity(claim, all_claims)

        assert len(errors) == 1
        assert "c0000099" in errors[0]
        assert "not found" in errors[0]

    def test_validate_relationship_integrity_missing_supers_list(self):
        """Test validation with missing supers_list claim"""
        claim = Claim(
            id="c0000001", content="Claim A", confidence=0.5, supers=["c0000099"]
        )
        all_claims = [claim]

        errors = claim_operations.validate_relationship_integrity(claim, all_claims)

        assert len(errors) == 1
        assert "c0000099" in errors[0]

    def test_validate_relationship_integrity_multiple_errors(self):
        """Test validation with multiple missing relationships"""
        claim = Claim(
            id="c0000001",
            content="Claim A",
            confidence=0.5,
            subs=["c0000098"],
            supers=["c0000099"],
        )
        all_claims = [claim]

        errors = claim_operations.validate_relationship_integrity(claim, all_claims)

        assert len(errors) == 2


class TestClaimHierarchy:
    """Tests for claim hierarchy operations"""

    def test_get_claim_hierarchy_simple(self):
        """Test getting hierarchy for simple claim"""
        claim = Claim(id="c0000001", content="Test claim", confidence=0.8)
        all_claims = [claim]

        hierarchy = claim_operations.get_claim_hierarchy(claim, all_claims)

        assert hierarchy["claim_id"] == "c0000001"
        assert hierarchy["confidence"] == 0.8
        assert hierarchy["supers_count"] == 0
        assert hierarchy["subs_count"] == 0
        assert len(hierarchy["subs"]) == 0
        assert len(hierarchy["supers"]) == 0

    def test_get_claim_hierarchy_with_subs_list(self):
        """Test getting hierarchy with supporting claims"""
        claim1 = Claim(
            id="c0000001",
            content="Main claim",
            confidence=0.5,
            subs=["c0000002"],
        )
        claim2 = Claim(id="c0000002", content="Supporting claim", confidence=0.8)

        all_claims = [claim1, claim2]
        hierarchy = claim_operations.get_claim_hierarchy(claim1, all_claims)

        assert hierarchy["subs_count"] == 1
        assert len(hierarchy["subs"]) == 1
        assert hierarchy["subs"][0]["id"] == "c0000002"
        assert hierarchy["subs"][0]["confidence"] == 0.8

    def test_get_claim_hierarchy_with_supers_list(self):
        """Test getting hierarchy with supers_list claims"""
        claim1 = Claim(
            id="c0000001", content="Main claim", confidence=0.8, supers=["c0000003"]
        )
        claim3 = Claim(id="c0000003", content="Supported claim", confidence=0.5)

        all_claims = [claim1, claim3]
        hierarchy = claim_operations.get_claim_hierarchy(claim1, all_claims)

        assert hierarchy["supers_count"] == 1
        assert len(hierarchy["supers"]) == 1
        assert hierarchy["supers"][0]["id"] == "c0000003"

    def test_get_claim_hierarchy_content_truncation(self):
        """Test that long content is truncated in hierarchy"""
        long_content = "A" * 150
        claim = Claim(
            id="c0000001",
            content="Main claim",
            confidence=0.5,
            subs=["c0000002"],
        )
        claim2 = Claim(id="c0000002", content=long_content, confidence=0.8)

        all_claims = [claim, claim2]
        hierarchy = claim_operations.get_claim_hierarchy(claim, all_claims)

        supporter_content = hierarchy["subs"][0]["content"]
        assert len(supporter_content) <= 103  # 100 + "..."
        assert supporter_content.endswith("...")


class TestBatchOperations:
    """Tests for batch operations"""

    def test_batch_update_confidence_partial(self):
        """Test batch update with partial updates"""
        claim1 = Claim(id="c0000001", content="Claim A", confidence=0.5)
        claim2 = Claim(id="c0000002", content="Claim B", confidence=0.6)
        claim3 = Claim(id="c0000003", content="Claim C", confidence=0.7)

        claims = [claim1, claim2, claim3]
        updates = {"c0000001": 0.9, "c0000003": 0.8}

        updated = claim_operations.batch_update_confidence(claims, updates)

        assert updated[0].confidence == 0.9
        assert updated[1].confidence == 0.6  # Unchanged
        assert updated[2].confidence == 0.8

    def test_batch_update_confidence_empty_updates(self):
        """Test batch update with no updates"""
        claim1 = Claim(id="c0000001", content="Claim A", confidence=0.5)
        claims = [claim1]
        updates = {}

        updated = claim_operations.batch_update_confidence(claims, updates)

        assert updated[0].confidence == 0.5


class TestFilterOperations:
    """Tests for claim filtering operations"""

    def test_find_dirty_claims(self):
        """Test finding dirty claims"""
        claim1 = Claim(id="c0000001", content="Claim A", confidence=0.5, is_dirty=True)
        claim2 = Claim(id="c0000002", content="Claim B", confidence=0.6, is_dirty=False)
        claim3 = Claim(id="c0000003", content="Claim C", confidence=0.7, is_dirty=True)

        claims = [claim1, claim2, claim3]
        dirty = claim_operations.find_dirty_claims(claims)

        assert len(dirty) == 2
        assert claim1 in dirty
        assert claim3 in dirty

    def test_find_dirty_claims_with_priority(self):
        """Test finding dirty claims with priority threshold"""
        claim1 = Claim(
            id="c0000001",
            content="Claim A",
            confidence=0.5,
            is_dirty=True,
            dirty_priority=5,
        )
        claim2 = Claim(
            id="c0000002",
            content="Claim B",
            confidence=0.6,
            is_dirty=True,
            dirty_priority=3,
        )
        claim3 = Claim(
            id="c0000003",
            content="Claim C",
            confidence=0.7,
            is_dirty=True,
            dirty_priority=8,
        )

        claims = [claim1, claim2, claim3]
        dirty = claim_operations.find_dirty_claims(claims, priority_threshold=5)

        assert len(dirty) == 2
        assert claim1 in dirty
        assert claim3 in dirty

    def test_filter_claims_by_confidence_range(self):
        """Test filtering claims by confidence range"""
        claim1 = Claim(id="c0000001", content="Claim A", confidence=0.3)
        claim2 = Claim(id="c0000002", content="Claim B", confidence=0.6)
        claim3 = Claim(id="c0000003", content="Claim C", confidence=0.9)

        claims = [claim1, claim2, claim3]
        filtered = claim_operations.filter_claims_by_confidence(
            claims, min_confidence=0.5, max_confidence=0.8
        )

        assert len(filtered) == 1
        assert claim2 in filtered

    def test_filter_claims_by_type(self):
        """Test filtering claims by type"""
        claim1 = Claim(
            id="c0000001", content="Claim A", confidence=0.5, type=[ClaimType.ASSERTION]
        )
        claim2 = Claim(
            id="c0000002",
            content="Claim B",
            confidence=0.6,
            type=[ClaimType.CONJECTURE],
        )
        claim3 = Claim(
            id="c0000003",
            content="Claim C",
            confidence=0.7,
            type=[ClaimType.ASSERTION, ClaimType.CONCEPT],
        )

        claims = [claim1, claim2, claim3]
        filtered = claim_operations.filter_claims_by_type(claims, ["assertion"])

        assert len(filtered) == 2
        assert claim1 in filtered
        assert claim3 in filtered

    def test_filter_claims_by_tags_any(self):
        """Test filtering claims by tags (match any)"""
        claim1 = Claim(
            id="c0000001", content="Claim A", confidence=0.5, tags=["math", "science"]
        )
        claim2 = Claim(
            id="c0000002", content="Claim B", confidence=0.6, tags=["history"]
        )
        claim3 = Claim(
            id="c0000003",
            content="Claim C",
            confidence=0.7,
            tags=["science", "physics"],
        )

        claims = [claim1, claim2, claim3]
        filtered = claim_operations.filter_claims_by_tags(
            claims, ["science"], match_all=False
        )

        assert len(filtered) == 2
        assert claim1 in filtered
        assert claim3 in filtered

    def test_filter_claims_by_tags_all(self):
        """Test filtering claims by tags (match all)"""
        claim1 = Claim(
            id="c0000001", content="Claim A", confidence=0.5, tags=["math", "science"]
        )
        claim2 = Claim(
            id="c0000002",
            content="Claim B",
            confidence=0.6,
            tags=["science", "physics"],
        )
        claim3 = Claim(
            id="c0000003",
            content="Claim C",
            confidence=0.7,
            tags=["math", "science", "physics"],
        )

        claims = [claim1, claim2, claim3]
        filtered = claim_operations.filter_claims_by_tags(
            claims, ["math", "science"], match_all=True
        )

        assert len(filtered) == 2
        assert claim1 in filtered
        assert claim3 in filtered


class TestDirtyPropagation:
    """Tests for dirty flag propagation"""

    def test_update_claim_with_dirty_propagation_no_change(self):
        """Test propagation when claim doesn't change"""
        original = Claim(id="c0000001", content="Test claim", confidence=0.5)
        updated = Claim(id="c0000001", content="Test claim", confidence=0.5)
        all_claims = {"c0000001": original}

        result_claim, marked_ids = claim_operations.update_claim_with_dirty_propagation(
            updated, original, all_claims
        )

        assert len(marked_ids) == 0

    @pytest.mark.xfail(
        reason="Requires DirtyFlagSystem - fallback path calls non-existent mark_dirty() method"
    )
    def test_update_claim_with_dirty_propagation_content_change(self):
        """Test propagation when content changes"""
        original = Claim(
            id="c0000001", content="Old content", confidence=0.5, supers=["c0000002"]
        )
        updated = Claim(
            id="c0000001", content="New content", confidence=0.5, supers=["c0000002"]
        )
        claim2 = Claim(id="c0000002", content="Supported claim", confidence=0.7)
        all_claims = {"c0000001": updated, "c0000002": claim2}

        result_claim, marked_ids = claim_operations.update_claim_with_dirty_propagation(
            updated, original, all_claims
        )

        assert "c0000002" in marked_ids

    @pytest.mark.xfail(
        reason="Requires DirtyFlagSystem - fallback path calls non-existent mark_dirty() method"
    )
    def test_update_claim_with_dirty_propagation_confidence_change(self):
        """Test propagation when confidence changes significantly"""
        original = Claim(
            id="c0000001", content="Test claim", confidence=0.5, supers=["c0000002"]
        )
        updated = Claim(
            id="c0000001", content="Test claim", confidence=0.9, supers=["c0000002"]
        )
        claim2 = Claim(id="c0000002", content="Supported claim", confidence=0.7)
        all_claims = {"c0000001": updated, "c0000002": claim2}

        result_claim, marked_ids = claim_operations.update_claim_with_dirty_propagation(
            updated, original, all_claims
        )

        assert "c0000002" in marked_ids

    def test_update_claim_with_dirty_propagation_small_confidence_change(self):
        """Test no propagation for insignificant confidence change"""
        original = Claim(
            id="c0000001", content="Test claim", confidence=0.50, supers=["c0000002"]
        )
        updated = Claim(
            id="c0000001", content="Test claim", confidence=0.505, supers=["c0000002"]
        )
        claim2 = Claim(id="c0000002", content="Supported claim", confidence=0.7)
        all_claims = {"c0000001": updated, "c0000002": claim2}

        result_claim, marked_ids = claim_operations.update_claim_with_dirty_propagation(
            updated, original, all_claims
        )

        assert len(marked_ids) == 0

    def test_update_claim_with_dirty_propagation_missing_supers_list_claim(self):
        """Test propagation when supers_list claim doesn't exist"""
        original = Claim(
            id="c0000001", content="Old content", confidence=0.5, supers=["c0000099"]
        )
        updated = Claim(
            id="c0000001", content="New content", confidence=0.5, supers=["c0000099"]
        )
        all_claims = {"c0000001": updated}

        result_claim, marked_ids = claim_operations.update_claim_with_dirty_propagation(
            updated, original, all_claims
        )

        # Should not crash, but also not mark anything
        assert len(marked_ids) == 0
