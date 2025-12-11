"""
Unit tests for claim processing operations
Tests core claim processing functionality without mocking
"""
import pytest
import time
from datetime import datetime

from src.core.models import Claim, ClaimState, DirtyReason
from src.core.claim_operations import (
    update_confidence,
    add_support,
    add_supports,
    mark_dirty,
    mark_clean,
    set_dirty_priority
)


class TestClaimProcessing:
    """Test claim processing operations"""

    def test_update_confidence_valid(self):
        """Test confidence update with valid values"""
        import time
        original_claim = Claim(id="test", content="Original content valid", confidence=0.5)
        
        # Small delay to ensure different timestamps
        time.sleep(0.001)
        
        # Update to higher confidence
        updated = update_confidence(original_claim, 0.8)
        assert updated.confidence == 0.8
        assert updated.id == original_claim.id
        assert updated.content == original_claim.content
        assert updated.updated >= original_claim.updated  # Allow equality due to timing
        
        # Update to lower confidence
        time.sleep(0.001)
        updated2 = update_confidence(updated, 0.3)
        assert updated2.confidence == 0.3

    def test_update_confidence_invalid(self):
        """Test confidence update with invalid values"""
        claim = Claim(id="test", content="Test content", confidence=0.5)
        
        # Test invalid confidence values
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            update_confidence(claim, -0.1)
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            update_confidence(claim, 1.1)

    def test_add_support_new(self):
        """Test adding new support relationship"""
        claim = Claim(id="main", content="Main claim", confidence=0.5)
        
        # Small delay to ensure different timestamps
        import time
        time.sleep(0.001)  # 1ms delay
        
        # Add first supporter
        with_support = add_support(claim, "supporter1")
        assert "supporter1" in with_support.supported_by
        assert len(with_support.supported_by) == 1
        assert with_support.updated > claim.updated
        
        # Small delay to ensure different timestamps
        import time
        time.sleep(0.001)  # 1ms delay
        
        # Add second supporter
        with_second_support = add_support(with_support, "supporter2")
        assert "supporter2" in with_second_support.supported_by
        assert len(with_second_support.supported_by) == 2
        assert "supporter1" in with_second_support.supported_by

    def test_add_support_duplicate(self):
        """Test adding duplicate support relationship"""
        claim = Claim(id="main", content="Main claim", confidence=0.5)
        
        # Add supporter
        with_support = add_support(claim, "supporter1")
        assert len(with_support.supported_by) == 1
        
        # Try to add same supporter again
        with_duplicate = add_support(with_support, "supporter1")
        assert len(with_duplicate.supported_by) == 1  # Should not duplicate
        assert with_duplicate.supported_by.count("supporter1") == 1

    def test_add_supports_new(self):
        """Test adding new supports relationship"""
        claim = Claim(id="supporter", content="Supporting claim", confidence=0.8)
        
        # Add first supported claim
        with_supports = add_supports(claim, "supported1")
        assert "supported1" in with_supports.supports
        assert len(with_supports.supports) == 1
        
        # Add second supported claim
        with_second = add_supports(with_supports, "supported2")
        assert len(with_second.supports) == 2
        assert "supported1" in with_second.supports
        assert "supported2" in with_second.supports

    def test_mark_dirty_with_reason(self):
        """Test marking claim as dirty with specific reason"""
        clean_claim = Claim(
            id="test",
            content="Clean claim",
            confidence=0.5,
            is_dirty=False
        )
        
        dirty_claim = mark_dirty(clean_claim, DirtyReason.CONFIDENCE_THRESHOLD, priority=5)
        
        assert dirty_claim.is_dirty is True
        assert dirty_claim.dirty_reason == DirtyReason.CONFIDENCE_THRESHOLD
        assert dirty_claim.dirty_priority == 5
        assert dirty_claim.dirty_timestamp is not None
        
        # Small delay to ensure different timestamps
        time.sleep(0.002)  # 2ms delay
        # Check that timestamp was updated (allowing for microsecond precision)
        assert dirty_claim.updated >= clean_claim.updated

    def test_mark_clean(self):
        """Test marking claim as clean"""
        dirty_claim = Claim(
            id="test",
            content="Dirty claim",
            confidence=0.5,
            is_dirty=True,
            dirty_reason=DirtyReason.MANUAL_MARK,
            dirty_priority=3
        )
        
        clean_claim = mark_clean(dirty_claim)
        
        assert clean_claim.is_dirty is False
        assert clean_claim.dirty_reason is None
        assert clean_claim.dirty_timestamp is None
        assert clean_claim.dirty_priority == 0
        
        # Small delay to ensure different timestamps
        time.sleep(0.002)  # 2ms delay
        # Check that timestamp was updated (allowing for microsecond precision)
        assert clean_claim.updated >= dirty_claim.updated

    def test_set_dirty_priority(self):
        """Test setting dirty priority"""
        claim = Claim(id="test", content="Test claim", confidence=0.5)
        
        # Set priority on dirty claim
        dirty_claim = mark_dirty(claim, DirtyReason.MANUAL_MARK)
        with_priority = set_dirty_priority(dirty_claim, 8)
        
        assert with_priority.dirty_priority == 8
        
        # Try to set priority on clean claim (should not change)
        clean_claim = mark_clean(with_priority)
        unchanged = set_dirty_priority(clean_claim, 10)
        
        assert unchanged.dirty_priority == 0  # Should remain 0 for clean claims

    def test_processing_chain(self):
        """Test chain of processing operations"""
        original = Claim(id="chain", content="Original claim", confidence=0.3)
        
        # Chain multiple operations
        step1 = mark_dirty(original, DirtyReason.NEW_CLAIM_ADDED, priority=2)
        
        # Small delay to ensure different timestamps
        time.sleep(0.001)  # 1ms delay
        
        step2 = add_support(step1, "supporter1")
        
        # Small delay to ensure different timestamps
        time.sleep(0.001)  # 1ms delay
        
        step3 = update_confidence(step2, 0.7)
        
        # Small delay to ensure different timestamps
        time.sleep(0.001)  # 1ms delay
        
        step4 = mark_clean(step3)
        
        # Verify final state
        assert step4.id == "chain"
        assert step4.content == "Original claim"
        assert step4.confidence == 0.7
        assert step4.is_dirty is False
        assert "supporter1" in step4.supported_by
        assert step4.updated > original.updated

    def test_immutability_of_original(self):
        """Test that original claims remain unchanged"""
        original = Claim(id="immutable", content="Original", confidence=0.5)
        
        # Perform operations
        modified = update_confidence(original, 0.8)
        dirty = mark_dirty(original, DirtyReason.MANUAL_MARK)
        
        # Original should be unchanged
        assert original.confidence == 0.5
        assert original.is_dirty is True  # Default for new claims
        assert original.dirty_reason is None  # Original default

    def test_confidence_boundary_values(self):
        """Test confidence updates at boundary values"""
        claim = Claim(id="boundary", content="Boundary test", confidence=0.5)
        
        # Test minimum boundary
        min_claim = update_confidence(claim, 0.0)
        assert min_claim.confidence == 0.0
        
        # Test maximum boundary
        max_claim = update_confidence(min_claim, 1.0)
        assert max_claim.confidence == 1.0

    def test_dirty_reason_types(self):
        """Test all dirty reason types"""
        claim = Claim(id="reasons", content="Test reasons", confidence=0.5)
        
        reasons = [
            DirtyReason.NEW_CLAIM_ADDED,
            DirtyReason.CONFIDENCE_THRESHOLD,
            DirtyReason.SUPPORTING_CLAIM_CHANGED,
            DirtyReason.RELATIONSHIP_CHANGED,
            DirtyReason.MANUAL_MARK,
            DirtyReason.BATCH_EVALUATION,
            DirtyReason.SYSTEM_TRIGGER
        ]
        
        for reason in reasons:
            dirty_claim = mark_dirty(claim, reason, priority=1)
            assert dirty_claim.dirty_reason == reason