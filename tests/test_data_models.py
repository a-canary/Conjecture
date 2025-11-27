"""
Unit tests for data models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.core.models import (
    Claim,
    Relationship,
    ClaimFilter,
    DataConfig,
    ClaimNotFoundError,
    InvalidClaimError,
    RelationshipError,
    DataLayerError,
    validate_claim_id,
    validate_confidence,
    generate_claim_id,
)


class TestClaim:
    """Test Claim model validation and functionality."""


def test_valid_claim_creation(self):
    """Test creating a valid claim."""
    claim = Claim(
        id="c0000001",
        content="This is a test claim with sufficient length",
        confidence=0.8,
    )

    assert claim.id == "c0000001"
    assert claim.confidence == 0.8
    assert claim.dirty is True  # Default value
    assert claim.tags == []  # Default value
    assert isinstance(claim.created_at, datetime)


def test_invalid_claim_id_format(self):
    """Test claim ID validation."""
    with pytest.raises(ValidationError):
        Claim(
            id="invalid_id",
            content="Test content",
            confidence=0.5,
        )


def test_invalid_confidence_range(self):
    """Test confidence validation."""
    with pytest.raises(ValidationError):
        Claim(
            id="c0000001",
            content="Test content",
            confidence=1.5,  # Invalid confidence
        )

    with pytest.raises(ValidationError):
        Claim(
            id="c0000001",
            content="Test content",
            confidence=-0.1,  # Invalid confidence
        )


def test_content_length_validation(self):
    """Test content length validation."""
    with pytest.raises(ValidationError):
        Claim(
            id="c0000001",
            content="Short",  # Too short
            confidence=0.5,
        )


def test_tags_validation(self):
    """Test tags validation."""
    # Valid tags
    claim = Claim(
        id="c0000001",
        content="Test content with sufficient length",
        confidence=0.5,
        tags=["tag1", "tag2", "tag1"],  # Duplicate should be removed
    )
    assert claim.tags == ["tag1", "tag2"]  # Duplicates removed

    # Invalid tags (empty string)
    with pytest.raises(ValidationError):
        Claim(
            id="c0000001",
            content="Test content with sufficient length",
            confidence=0.5,
            tags=["valid", ""],  # Empty tag
        )


class TestRelationship:
    """Test Relationship model validation."""


def test_valid_relationship(self):
    """Test creating a valid relationship."""
    relationship = Relationship(
        supporter_id="c0000001",
        supported_id="c0000002",
    )

    assert relationship.supporter_id == "c0000001"
    assert relationship.supported_id == "c0000002"
    assert isinstance(relationship.created_at, datetime)


def test_invalid_relationship_type(self):
    """Test that all relationships are 'supports' type by default."""
    # All relationships are now 'supports' by design, no validation needed
    relationship = Relationship(
        supporter_id="c0000001",
        supported_id="c0000002",
    )
    # Relationship is always 'supports' now
    assert relationship.supporter_id == "c0000001"
    assert relationship.supported_id == "c0000002"


def test_self_relationship(self):
    """Test that self-relationships are allowed at model level."""
    # Self-relationships are allowed at model level, business logic should handle validation
    relationship = Relationship(
        supporter_id="c0000001",
        supported_id="c0000001",  # Same ID
    )
    assert relationship.supporter_id == relationship.supported_id


class TestClaimFilter:
    """Test ClaimFilter validation."""

    def test_valid_filter(self):
        """Test creating a valid filter."""
        filter_obj = ClaimFilter(
            tags=["tag1", "tag2"],
            confidence_min=0.3,
            confidence_max=0.8,
            dirty_only=True,
            limit=50,
        )

        assert filter_obj.tags == ["tag1", "tag2"]
        assert filter_obj.confidence_min == 0.3
        assert filter_obj.confidence_max == 0.8
        assert filter_obj.dirty_only is True
        assert filter_obj.limit == 50

    def test_confidence_range_validation(self):
        """Test confidence range validation."""
        with pytest.raises(ValidationError):
            ClaimFilter(
                confidence_min=0.8,
                confidence_max=0.3,  # Less than min
            )

    def test_default_values(self):
        """Test default filter values."""
        filter_obj = ClaimFilter()

        assert filter_obj.tags is None
        assert filter_obj.confidence_min is None
        assert filter_obj.confidence_max is None
        assert filter_obj.dirty_only is None
        assert filter_obj.limit == 100
        assert filter_obj.offset == 0


class TestDataConfig:
    """Test DataConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DataConfig()

        assert config.sqlite_path == "./data/conjecture.db"
        assert config.chroma_path == "./data/vector_db"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.cache_size == 1000
        assert config.cache_ttl == 300
        assert config.batch_size == 100

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DataConfig(
            sqlite_path="/custom/path.db",
            chroma_path="/custom/vector",
            embedding_model="custom-model",
            cache_size=500,
            cache_ttl=600,
            batch_size=50,
        )

        assert config.sqlite_path == "/custom/path.db"
        assert config.chroma_path == "/custom/vector"
        assert config.embedding_model == "custom-model"
        assert config.cache_size == 500
        assert config.cache_ttl == 600
        assert config.batch_size == 50


class TestUtilityFunctions:
    """Test utility functions."""

    def test_validate_claim_id(self):
        """Test claim ID validation function."""
        # Valid IDs
        assert validate_claim_id("c0000001") is True
        assert validate_claim_id("c1234567") is True
        assert validate_claim_id("c9999999") is True

        # Invalid IDs
        assert validate_claim_id("C0000001") is False  # Upper case
        assert validate_claim_id("c000001") is False  # Too short
        assert validate_claim_id("c00000001") is False  # Too long
        assert validate_claim_id("a0000001") is False  # Wrong letter
        assert validate_claim_id("c00000a1") is False  # Contains letter
        assert validate_claim_id("c-000001") is False  # Contains dash

    def test_validate_confidence(self):
        """Test confidence validation function."""
        # Valid values
        assert validate_confidence(0.0) is True
        assert validate_confidence(0.5) is True
        assert validate_confidence(1.0) is True

        # Invalid values
        assert validate_confidence(-0.1) is False
        assert validate_confidence(1.1) is False
        assert validate_confidence(2.0) is False

    def test_generate_claim_id(self):
        """Test claim ID generation."""
        # Test various counters
        assert generate_claim_id(1) == "c0000001"
        assert generate_claim_id(42) == "c0000042"
        assert generate_claim_id(1234567) == "c1234567"
        assert generate_claim_id(9999999) == "c9999999"

        # Test zero padding
        assert len(generate_claim_id(1)) == 8
        assert len(generate_claim_id(9999999)) == 8


class TestExceptions:
    """Test custom exceptions."""

    def test_claim_not_found_error(self):
        """Test ClaimNotFoundError."""
        error = ClaimNotFoundError("Claim c0000001 not found")
        assert str(error) == "Claim c0000001 not found"
        assert isinstance(error, Exception)

    def test_invalid_claim_error(self):
        """Test InvalidClaimError."""
        error = InvalidClaimError("Invalid claim data")
        assert str(error) == "Invalid claim data"
        assert isinstance(error, Exception)

    def test_relationship_error(self):
        """Test RelationshipError."""
        error = RelationshipError("Invalid relationship")
        assert str(error) == "Invalid relationship"
        assert isinstance(error, Exception)

    def test_data_layer_error(self):
        """Test DataLayerError."""
        error = DataLayerError("Data layer operation failed")
        assert str(error) == "Data layer operation failed"
        assert isinstance(error, Exception)
