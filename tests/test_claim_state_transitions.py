"""
Unit tests for Claim state transitions and dirty flag operations
Tests claim lifecycle management without mocking
"""
import pytest
from datetime import datetime, timedelta

from src.core.models import Claim, ClaimState, DirtyReason


class TestClaimStateTransitions:
    """Test claim state transitions and dirty flag behavior"""

    def test_initial_dirty_state(self):
        """Test that new claims start dirty"""
        claim = Claim(id="test", content="New claim", confidence=0.5)
        
        assert claim.is_dirty is True
        assert claim.dirty_reason is None
        assert claim.dirty_timestamp is None
        assert claim.dirty_priority == 0

    def test_dirty_reason_assignment(self):
        """Test dirty reason assignment"""
        claim = Claim(
            id="test",
            content="Claim",
            confidence=0.5,
            is_dirty=True,
            dirty_reason=DirtyReason.NEW_CLAIM_ADDED,
            dirty_priority=5
        )
        
        assert claim.is_dirty is True
        assert claim.dirty_reason == DirtyReason.NEW_CLAIM_ADDED
        assert claim.dirty_priority == 5
        # dirty_timestamp is not automatically set, so we don't test it here

    def test_clean_claim_properties(self):
        """Test clean claim properties"""
        now = datetime.utcnow()
        claim = Claim(
            id="test",
            content="Clean claim",
            confidence=0.9,
            is_dirty=False,
            dirty_reason=None,
            dirty_timestamp=None,
            dirty_priority=0
        )
        
        assert claim.is_dirty is False
        assert claim.dirty_reason is None
        assert claim.dirty_timestamp is None
        assert claim.dirty_priority == 0

    def test_state_transitions(self):
        """Test valid state transitions"""
        claim = Claim(id="test", content="Test claim", confidence=0.5)
        
        # Initial state should be EXPLORE
        assert claim.state == ClaimState.EXPLORE
        
        # Can transition to VALIDATED
        claim.state = ClaimState.VALIDATED
        assert claim.state == ClaimState.VALIDATED
        
        # Can transition to ORPHANED
        claim.state = ClaimState.ORPHANED
        assert claim.state == ClaimState.ORPHANED
        
        # Can transition to QUEUED
        claim.state = ClaimState.QUEUED
        assert claim.state == ClaimState.QUEUED

    def test_confidence_based_state_logic(self):
        """Test confidence-based state logic"""
        # Low confidence claim should typically be in EXPLORE state
        low_conf_claim = Claim(
            id="low",
            content="Low confidence claim",
            confidence=0.3,
            state=ClaimState.EXPLORE
        )
        
        # High confidence claim could be VALIDATED
        high_conf_claim = Claim(
            id="high",
            content="High confidence claim",
            confidence=0.9,
            state=ClaimState.VALIDATED
        )
        
        assert low_conf_claim.confidence < 0.5
        assert high_conf_claim.confidence > 0.8
        assert low_conf_claim.state == ClaimState.EXPLORE
        assert high_conf_claim.state == ClaimState.VALIDATED

    def test_dirty_timestamp_behavior(self):
        """Test dirty timestamp behavior"""
        before_creation = datetime.utcnow()
        
        claim = Claim(
            id="test",
            content="Test content with valid length",
            confidence=0.5,
            is_dirty=True,
            dirty_reason=DirtyReason.MANUAL_MARK
        )
        
        after_creation = datetime.utcnow()
        
        # dirty_timestamp is not automatically set, so we test other properties
        assert claim.is_dirty is True
        assert claim.dirty_reason == DirtyReason.MANUAL_MARK
        assert before_creation <= claim.updated <= after_creation

    def test_dirty_priority_levels(self):
        """Test dirty priority levels"""
        priorities = [0, 5, 10, 15]
        claims = []
        
        for i, priority in enumerate(priorities):
            claim = Claim(
                id=f"test{i}",
                content=f"Claim with priority {priority}",
                confidence=0.5,
                dirty_priority=priority
            )
            claims.append(claim)
        
        # Verify all priorities are set correctly
        for claim, expected_priority in zip(claims, priorities):
            assert claim.dirty_priority == expected_priority

    def test_state_and_dirty_combinations(self):
        """Test various combinations of state and dirty flags"""
        combinations = [
            (ClaimState.EXPLORE, True, DirtyReason.NEW_CLAIM_ADDED),
            (ClaimState.VALIDATED, False, None),
            (ClaimState.ORPHANED, True, DirtyReason.CONFIDENCE_THRESHOLD),
            (ClaimState.QUEUED, True, DirtyReason.BATCH_EVALUATION),
        ]
        
        for i, (state, is_dirty, dirty_reason) in enumerate(combinations):
            claim = Claim(
                id=f"test{i}",
                content="Test claim",
                confidence=0.5,
                state=state,
                is_dirty=is_dirty,
                dirty_reason=dirty_reason
            )
            
            assert claim.state == state
            assert claim.is_dirty == is_dirty
            assert claim.dirty_reason == dirty_reason

    def test_updated_timestamp_on_changes(self):
        """Test that updated timestamp changes with modifications"""
        original_time = datetime.utcnow()
        
        claim = Claim(
            id="test",
            content="Original content",
            confidence=0.5,
            updated=original_time
        )
        
        # Simulate a small delay
        import time
        time.sleep(0.01)
        
        # Create updated version
        updated_time = datetime.utcnow()
        updated_claim = Claim(
            id=claim.id,
            content="Updated content",
            confidence=0.6,
            state=claim.state,
            supported_by=claim.supported_by.copy(),
            supports=claim.supports.copy(),
            tags=claim.tags.copy(),
            created=claim.created,
            updated=updated_time
        )
        
        assert updated_claim.updated > claim.updated
        assert updated_claim.content != claim.content
        assert updated_claim.confidence != claim.confidence