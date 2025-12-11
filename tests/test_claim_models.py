"""
Unit tests for Claim model validation and creation
Tests core Pydantic model functionality without mocking
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from src.core.models import Claim, ClaimState, ClaimType, ClaimScope, DirtyReason


class TestClaimModel:
    """Test Claim model validation and creation"""

    def test_minimal_claim_creation(self):
        """Test creating a claim with minimal required fields"""
        claim = Claim(
            id="test123",
            content="Test claim content",
            confidence=0.5
        )
        
        assert claim.id == "test123"
        assert claim.content == "Test claim content"
        assert claim.confidence == 0.5
        assert claim.state == ClaimState.EXPLORE
        assert claim.supported_by == []
        assert claim.supports == []
        assert claim.tags == []
        assert claim.is_dirty is True  # Default for new claims

    def test_full_claim_creation(self):
        """Test creating a claim with all fields"""
        now = datetime.utcnow()
        claim = Claim(
            id="test456",
            content="Full test claim",
            confidence=0.8,
            state=ClaimState.VALIDATED,
            supported_by=["claim1", "claim2"],
            supports=["claim3"],
            tags=["test", "validation"],
            scope=ClaimScope.PUBLIC,
            created=now,
            updated=now,
            is_dirty=False,
            dirty_reason=DirtyReason.CONFIDENCE_THRESHOLD,
            dirty_priority=5
        )
        
        assert claim.id == "test456"
        assert claim.confidence == 0.8
        assert claim.state == ClaimState.VALIDATED
        assert len(claim.supported_by) == 2
        assert len(claim.supports) == 1
        assert len(claim.tags) == 2
        assert claim.scope == ClaimScope.PUBLIC
        assert claim.is_dirty is False
        assert claim.dirty_reason == DirtyReason.CONFIDENCE_THRESHOLD
        assert claim.dirty_priority == 5

    def test_claim_content_validation(self):
        """Test content validation constraints"""
        # Valid content
        claim = Claim(id="test", content="Valid content", confidence=0.5)
        assert claim.content == "Valid content"
        
        # Too short content (less than 5 characters)
        with pytest.raises(ValidationError):
            Claim(id="test", content="Test", confidence=0.5)
        
        # Too long content
        with pytest.raises(ValidationError):
            Claim(id="test", content="X" * 2001, confidence=0.5)

    def test_confidence_validation(self):
        """Test confidence bounds validation"""
        # Valid confidence values
        claim1 = Claim(id="test1", content="Test content valid", confidence=0.0)
        claim2 = Claim(id="test2", content="Test content valid", confidence=0.5)
        claim3 = Claim(id="test3", content="Test content valid", confidence=1.0)
        
        assert claim1.confidence == 0.0
        assert claim2.confidence == 0.5
        assert claim3.confidence == 1.0
        
        # Invalid confidence values
        with pytest.raises(ValidationError):
            Claim(id="test", content="Test content valid", confidence=-0.1)
        
        with pytest.raises(ValidationError):
            Claim(id="test", content="Test content valid", confidence=1.1)

    def test_tags_validation_and_deduplication(self):
        """Test tags validation and duplicate removal"""
        claim = Claim(
            id="test",
            content="Test claim with valid content length",
            confidence=0.5,
            tags=["tag1", "tag2", "tag1", "tag3", "valid_tag"]
        )
        
        # Should remove duplicates but keep valid non-empty tags
        assert len(claim.tags) == 4  # tag1, tag2, tag3, valid_tag (empty string removed)
        assert "tag1" in claim.tags
        assert "tag2" in claim.tags
        assert "tag3" in claim.tags
        assert "valid_tag" in claim.tags
        assert claim.tags.count("tag1") == 1

    def test_timestamp_validation(self):
        """Test timestamp validation logic"""
        created = datetime.utcnow()
        updated = datetime.utcnow()
        
        # Valid timestamps
        claim = Claim(
            id="test",
            content="Test content with valid length",
            confidence=0.5,
            created=created,
            updated=updated
        )
        assert claim.created == created
        assert claim.updated == updated
        
        # Invalid: updated before created
        invalid_updated = datetime.utcnow().replace(year=2020)
        with pytest.raises(ValidationError):
            Claim(
                id="test",
                content="Test content with valid length",
                confidence=0.5,
                created=created,
                updated=invalid_updated
            )

    def test_claim_format_methods(self):
        """Test claim formatting methods"""
        claim = Claim(
            id="123",
            content="Test claim for formatting",
            confidence=0.75,
            tags=["format", "test"]
        )
        
        # Test context formatting
        context_format = claim.format_for_context()
        expected = "[c123 | Test claim for formatting | / 0.75]"
        assert context_format == expected
        
        # Test output formatting
        output_format = claim.format_for_output()
        assert output_format == expected
        
        # Test LLM analysis formatting
        llm_format = claim.format_for_llm_analysis()
        assert "Claim ID: 123" in llm_format
        assert "Content: Test claim for formatting" in llm_format
        assert "Confidence: 0.75" in llm_format
        assert "Tags: format,test" in llm_format

    def test_claim_hash_and_equality(self):
        """Test claim hashability and equality"""
        claim1 = Claim(id="test1", content="Same content", confidence=0.5)
        claim2 = Claim(id="test1", content="Same content", confidence=0.5)
        claim3 = Claim(id="test2", content="Different content", confidence=0.5)
        
        # Test hash
        assert hash(claim1) == hash(claim2)
        assert hash(claim1) != hash(claim3)
        
        # Test that claims can be used in sets
        claim_set = {claim1, claim2, claim3}
        assert len(claim_set) == 2  # claim1 and claim2 are duplicates