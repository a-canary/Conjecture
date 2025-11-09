"""
Comprehensive tests for Pydantic models and validation in the Conjecture data layer.
"""
import pytest
from datetime import datetime, timedelta
from typing import List
import re

from src.data.models import (
    Claim, Relationship, ClaimFilter, DataConfig, ProcessingResult, BatchResult,
    ClaimNotFoundError, InvalidClaimError, RelationshipError, DataLayerError,
    validate_claim_id, validate_confidence, generate_claim_id
)


class TestClaimModel:
    """Test the Claim model validation and behavior."""

    @pytest.mark.models
    def test_valid_claim_creation(self, sample_claim_data):
        """Test creating a valid claim."""
        claim = Claim(**sample_claim_data)
        
        assert claim.id == "c0000001"
        assert claim.content == sample_claim_data["content"]
        assert claim.confidence == 0.95
        assert claim.dirty is True
        assert set(claim.tags) == {"astronomy", "science", "physics"}
        assert claim.created_by == "test_user"
        assert isinstance(claim.created_at, datetime)

    @pytest.mark.models
    def test_claim_with_minimal_data(self):
        """Test creating a claim with only required fields."""
        claim = Claim(
            id="c0000002",
            content="This is a minimal valid claim with at least 10 characters.",
            confidence=0.5,
            created_by="user123"
        )
        
        assert claim.id == "c0000002"
        assert claim.dirty is True  # Default value
        assert claim.tags == []     # Default value
        assert claim.updated_at is None  # Optional field

    @pytest.mark.models
    def test_claim_id_validation(self):
        """Test claim ID format validation."""
        # Valid IDs
        valid_ids = ["c0000001", "c1234567", "c9999999"]
        for claim_id in valid_ids:
            claim = Claim(
                id=claim_id,
                content="Valid claim for testing ID validation.",
                confidence=0.7,
                created_by="test_user"
            )
            assert claim.id == claim_id

        # Invalid IDs
        invalid_ids = [
            "C0000001",      # Uppercase C
            "c000001",       # Too short
            "c00000001",     # Too long
            "cABCDEFG",      # Non-numeric
            "00000001",      # Missing prefix
            "c-123456",      # Invalid character
            "",              # Empty string
            "c 1234567",     # Space
            "c123_4567"      # Underscore
        ]
        
        for invalid_id in invalid_ids:
            with pytest.raises(ValueError, match="Claim ID must be in format c#######"):
                Claim(
                    id=invalid_id,
                    content="Valid content but invalid ID.",
                    confidence=0.7,
                    created_by="test_user"
                )

    @pytest.mark.models
    def test_content_validation(self):
        """Test claim content length validation."""
        # Valid content
        valid_contents = [
            "A" * 10,                           # Exactly 10 characters
            "This is a valid claim with sufficient length.",
            "A" * 1000,                         # Long content
            "特殊字符 and 日本語 and 🚀 emojis"   # Special characters
        ]
        
        for content in valid_contents:
            claim = Claim(
                id="c0000001",
                content=content,
                confidence=0.7,
                created_by="test_user"
            )
            assert claim.content == content

        # Invalid content
        invalid_contents = [
            "",               # Empty string
            "A" * 9,          # Too short (9 characters)
            "   ",            # Only whitespace
        ]
        
        for content in invalid_contents:
            with pytest.raises(ValueError, match="ensure this value has at least 10 characters"):
                Claim(
                    id="c0000001",
                    content=content,
                    confidence=0.7,
                    created_by="test_user"
                )

    @pytest.mark.models
    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence values
        valid_confidences = [0.0, 0.1, 0.5, 0.99, 1.0]
        
        for confidence in valid_confidences:
            claim = Claim(
                id="c0000001",
                content="Valid claim for confidence testing.",
                confidence=confidence,
                created_by="test_user"
            )
            assert claim.confidence == confidence

        # Invalid confidence values
        invalid_confidences = [-0.1, -1.0, 1.1, 2.0, 100.0]
        
        for confidence in invalid_confidences:
            with pytest.raises(ValueError, match="ensure this value is less than or equal to 1"):
                Claim(
                    id="c0000001",
                    content="Valid claim but invalid confidence.",
                    confidence=confidence,
                    created_by="test_user"
                )

    @pytest.mark.models
    def test_tags_validation(self):
        """Test tags validation and deduplication."""
        # Valid tags
        valid_tags_cases = [
            ["science", "physics"],
            ["tag1", "tag2", "tag3"],
            ["single_tag"],
            [],                    # Empty list
            ["duplicate", "duplicate", "unique"]  # Duplicates should be removed
        ]
        
        for tags in valid_tags_cases:
            claim = Claim(
                id="c0000001",
                content="Valid claim for tags testing.",
                confidence=0.7,
                tags=tags,
                created_by="test_user"
            )
            # Check duplicates are removed
            assert len(set(claim.tags)) == len(claim.tags)
            assert all(tag.strip() == tag for tag in claim.tags)

        # Invalid tags
        invalid_tags_cases = [
            ["", "valid_tag"],                 # Empty tag
            ["valid_tag", None],               # None value
            ["valid_tag", 123],                # Non-string value
            [""],                              # Only empty string
        ]
        
        for tags in invalid_tags_cases:
            with pytest.raises(ValueError, match="All tags must be non-empty strings"):
                Claim(
                    id="c0000001",
                    content="Valid claim but invalid tags.",
                    confidence=0.7,
                    tags=tags,
                    created_by="test_user"
                )

    @pytest.mark.models
    def test_serialization(self, valid_claim):
        """Test claim serialization to dict and JSON."""
        # Test dict conversion
        claim_dict = valid_claim.dict()
        assert isinstance(claim_dict, dict)
        assert claim_dict["id"] == valid_claim.id
        assert claim_dict["content"] == valid_claim.content

        # Test JSON serialization
        claim_json = valid_claim.json()
        assert isinstance(claim_json, str)
        assert valid_claim.id in claim_json
        assert valid_claim.content in claim_json

    @pytest.mark.models
    def test_timestamp_handling(self, valid_claim):
        """Test timestamp handling and JSON encoding."""
        created_at = valid_claim.created_at
        assert isinstance(created_at, datetime)

        # Test JSON encoding of datetime
        claim_dict = valid_claim.dict()
        assert "created_at" in claim_dict

    @pytest.mark.models
    def test_claim_equality(self):
        """Test claim equality comparison."""
        claim1 = Claim(
            id="c0000001",
            content="Test content for equality.",
            confidence=0.7,
            created_by="user1"
        )
        claim2 = Claim(
            id="c0000001",
            content="Test content for equality.",
            confidence=0.7,
            created_by="user1"
        )
        claim3 = Claim(
            id="c0000002",
            content="Test content for equality.",
            confidence=0.7,
            created_by="user1"
        )

        # Claims with same ID should be equal
        assert claim1 == claim2
        assert claim1 != claim3

    @pytest.mark.models
    def test_claim_hash(self):
        """Test claim hashing for use in sets."""
        claim1 = Claim(
            id="c0000001",
            content="Test content for hashing.",
            confidence=0.7,
            created_by="user1"
        )
        claim2 = Claim(
            id="c0000001",
            content="Test content for hashing.",
            confidence=0.7,
            created_by="user1"
        )

        # Same claims should have same hash
        assert hash(claim1) == hash(claim2)

        # Claims should be hashable (can be used in sets)
        claim_set = {claim1, claim2}
        assert len(claim_set) == 1


class TestRelationshipModel:
    """Test the Relationship model validation and behavior."""

    @pytest.mark.models
    def test_valid_relationship_creation(self, sample_relationship_data):
        """Test creating a valid relationship."""
        relationship = Relationship(**sample_relationship_data)
        
        assert relationship.supporter_id == "c0000001"
        assert relationship.supported_id == "c0000002"
        assert relationship.relationship_type == "supports"
        assert relationship.created_by == "test_user"
        assert isinstance(relationship.created_at, datetime)

    @pytest.mark.models
    def test_relationship_with_minimal_data(self):
        """Test creating a relationship with only required fields."""
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002"
        )
        
        assert relationship.relationship_type == "supports"  # Default
        assert relationship.created_by is None              # Optional
        assert relationship.id is None                      # Optional

    @pytest.mark.models
    def test_relationship_type_validation(self):
        """Test relationship type validation."""
        valid_types = ["supports", "contradicts", "extends", "clarifies"]
        
        for rel_type in valid_types:
            relationship = Relationship(
                supporter_id="c0000001",
                supported_id="c0000002",
                relationship_type=rel_type
            )
            assert relationship.relationship_type == rel_type

        # Invalid relationship types
        invalid_types = ["invalid", "related", "references", "", None]
        
        for rel_type in invalid_types:
            with pytest.raises(ValueError, match="Relationship type must be one of"):
                Relationship(
                    supporter_id="c0000001",
                    supported_id="c0000002",
                    relationship_type=rel_type
                )

    @pytest.mark.models
    def test_relationship_serialization(self, valid_relationship):
        """Test relationship serialization."""
        relationship_dict = valid_relationship.dict()
        assert isinstance(relationship_dict, dict)
        assert "supporter_id" in relationship_dict
        assert "supported_id" in relationship_dict
        assert "relationship_type" in relationship_dict


class TestClaimFilterModel:
    """Test the ClaimFilter model validation and behavior."""

    @pytest.mark.models
    def test_empty_filter(self):
        """Test creating an empty filter."""
        filter_obj = ClaimFilter()
        
        assert filter_obj.tags is None
        assert filter_obj.confidence_min is None
        assert filter_obj.confidence_max is None
        assert filter_obj.limit == 100  # Default
        assert filter_obj.offset == 0   # Default

    @pytest.mark.models
    def test_filter_with_all_fields(self):
        """Test creating a filter with all fields specified."""
        filter_obj = ClaimFilter(
            tags=["science", "physics"],
            confidence_min=0.7,
            confidence_max=0.9,
            dirty_only=True,
            created_by="test_user",
            content_contains="quantum",
            limit=20,
            offset=10
        )
        
        assert filter_obj.tags == ["science", "physics"]
        assert filter_obj.confidence_min == 0.7
        assert filter_obj.confidence_max == 0.9
        assert filter_obj.dirty_only is True
        assert filter_obj.created_by == "test_user"
        assert filter_obj.content_contains == "quantum"
        assert filter_obj.limit == 20
        assert filter_obj.offset == 10

    @pytest.mark.models
    def test_confidence_range_validation(self):
        """Test confidence range validation."""
        # Valid ranges
        valid_ranges = [
            (0.3, 0.7),
            (0.0, 1.0),
            (0.5, 0.5),
            (None, 0.8),
            (0.2, None)
        ]
        
        for min_conf, max_conf in valid_ranges:
            filter_obj = ClaimFilter(
                confidence_min=min_conf,
                confidence_max=max_conf
            )
            assert filter_obj.confidence_min == min_conf
            assert filter_obj.confidence_max == max_conf

        # Invalid ranges (max < min)
        with pytest.raises(ValueError, match="confidence_max must be >= confidence_min"):
            ClaimFilter(confidence_min=0.8, confidence_max=0.3)

        with pytest.raises(ValueError, match="confidence_max must be >= confidence_min"):
            ClaimFilter(confidence_min=0.7, confidence_max=0.6)

    @pytest.mark.models
    def test_filter_bounds_validation(self):
        """Test filter parameter bounds validation."""
        # Valid bounds
        ClaimFilter(limit=1)      # Minimum
        ClaimFilter(limit=1000)   # Maximum
        ClaimFilter(offset=0)     # Minimum

        # Invalid bounds
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 1"):
            ClaimFilter(limit=0)

        with pytest.raises(ValueError, match="ensure this value is less than or equal to 1000"):
            ClaimFilter(limit=1001)

        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 0"):
            ClaimFilter(offset=-1)


class TestDataConfigModel:
    """Test the DataConfig model validation and behavior."""

    @pytest.mark.models
    def test_default_config(self):
        """Test default configuration values."""
        config = DataConfig()
        
        assert config.sqlite_path == "./data/conjecture.db"
        assert config.chroma_path == "./data/vector_db"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.cache_size == 1000
        assert config.cache_ttl == 300
        assert config.batch_size == 100

    @pytest.mark.models
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DataConfig(
            sqlite_path="/custom/path.db",
            chroma_path="/custom/vector",
            embedding_model="custom-model",
            cache_size=2000,
            cache_ttl=600,
            batch_size=50
        )
        
        assert config.sqlite_path == "/custom/path.db"
        assert config.chroma_path == "/custom/vector"
        assert config.embedding_model == "custom-model"
        assert config.cache_size == 2000
        assert config.cache_ttl == 600
        assert config.batch_size == 50

    @pytest.mark.models
    def test_config_bounds_validation(self):
        """Test configuration parameter bounds."""
        # Valid bounds
        DataConfig(cache_size=0)      # Minimum
        DataConfig(cache_size=10000)   # Large value
        DataConfig(cache_ttl=0)        # Minimum
        DataConfig(batch_size=1)       # Minimum
        DataConfig(batch_size=1000)    # Maximum

        # Invalid bounds
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 0"):
            DataConfig(cache_size=-1)

        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 0"):
            DataConfig(cache_ttl=-1)

        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 1"):
            DataConfig(batch_size=0)

        with pytest.raises(ValueError, match="ensure this value is less than or equal to 1000"):
            DataConfig(batch_size=1001)


class TestProcessingResultModel:
    """Test the ProcessingResult model."""

    @pytest.mark.models
    def test_valid_processing_result(self):
        """Test creating a valid processing result."""
        result = ProcessingResult(
            claim_id="c0000001",
            success=True,
            message="Processed successfully",
            updated_confidence=0.85,
            processing_time=0.123,
            metadata={"model": "bert", "version": "1.0"}
        )
        
        assert result.claim_id == "c0000001"
        assert result.success is True
        assert result.message == "Processed successfully"
        assert result.updated_confidence == 0.85
        assert result.processing_time == 0.123
        assert result.metadata == {"model": "bert", "version": "1.0"}

    @pytest.mark.models
    def test_minimal_processing_result(self):
        """Test creating a processing result with minimal data."""
        result = ProcessingResult(
            claim_id="c0000001",
            success=False
        )
        
        assert result.claim_id == "c0000001"
        assert result.success is False
        assert result.message is None
        assert result.updated_confidence is None
        assert result.processing_time is None
        assert result.metadata is None


class TestBatchResultModel:
    """Test the BatchResult model."""

    @pytest.mark.models
    def test_valid_batch_result(self):
        """Test creating a valid batch result."""
        processing_results = [
            ProcessingResult(claim_id="c0000001", success=True),
            ProcessingResult(claim_id="c0000002", success=False, message="Error"),
            ProcessingResult(claim_id="c0000003", success=True)
        ]
        
        batch_result = BatchResult(
            total_items=3,
            successful_items=2,
            failed_items=1,
            results=processing_results,
            errors=["Processing failed for c0000002"]
        )
        
        assert batch_result.total_items == 3
        assert batch_result.successful_items == 2
        assert batch_result.failed_items == 1
        assert len(batch_result.results) == 3
        assert len(batch_result.errors) == 1


class TestUtilityFunctions:
    """Test utility validation and helper functions."""

    @pytest.mark.models
    def test_validate_claim_id(self):
        """Test claim ID validation utility function."""
        # Valid IDs
        assert validate_claim_id("c0000001") is True
        assert validate_claim_id("c1234567") is True
        assert validate_claim_id("c9999999") is True

        # Invalid IDs
        assert validate_claim_id("C0000001") is False
        assert validate_claim_id("c000001") is False
        assert validate_claim_id("c00000001") is False
        assert validate_claim_id("") is False

    @pytest.mark.models
    def test_validate_confidence(self):
        """Test confidence validation utility function."""
        # Valid confidence values
        assert validate_confidence(0.0) is True
        assert validate_confidence(0.5) is True
        assert validate_confidence(1.0) is True

        # Invalid confidence values
        assert validate_confidence(-0.1) is False
        assert validate_confidence(-1.0) is False
        assert validate_confidence(1.1) is False
        assert validate_confidence(2.0) is False

    @pytest.mark.models
    def test_generate_claim_id(self):
        """Test claim ID generation utility function."""
        # Test various counters
        assert generate_claim_id(1) == "c0000001"
        assert generate_claim_id(42) == "c0000042"
        assert generate_claim_id(1234567) == "c1234567"
        assert generate_claim_id(0) == "c0000000"
        assert generate_claim_id(9999999) == "c9999999"

        # Test that it generates correct format
        for i in [0, 1, 10, 100, 1000, 10000, 100000, 1000000]:
            claim_id = generate_claim_id(i)
            assert claim_id.startswith('c')
            assert len(claim_id) == 8
            assert claim_id[1:].isdigit()


class TestCustomExceptions:
    """Test custom exception classes."""

    @pytest.mark.models
    def test_claim_not_found_error(self):
        """Test ClaimNotFoundError exception."""
        error = ClaimNotFoundError("Claim c0000001 not found")
        assert str(error) == "Claim c0000001 not found"
        
        # Test that it's a proper exception
        with pytest.raises(ClaimNotFoundError):
            raise ClaimNotFoundError("Test message")

    @pytest.mark.models
    def test_invalid_claim_error(self):
        """Test InvalidClaimError exception."""
        error = InvalidClaimError("Invalid claim data")
        assert str(error) == "Invalid claim data"
        
        with pytest.raises(InvalidClaimError):
            raise InvalidClaimError("Test")

    @pytest.mark.models
    def test_relationship_error(self):
        """Test RelationshipError exception."""
        error = RelationshipError("Cannot create relationship")
        assert str(error) == "Cannot create relationship"
        
        with pytest.raises(RelationshipError):
            raise RelationshipError("Test")

    @pytest.mark.models
    def test_data_layer_error(self):
        """Test DataLayerError base exception."""
        error = DataLayerError("Database operation failed")
        assert str(error) == "Database operation failed"
        
        with pytest.raises(DataLayerError):
            raise DataLayerError("Test")


class TestModelIntegration:
    """Integration tests for model interactions."""

    @pytest.mark.models
    def test_claim_relationship_combination(self):
        """Test using claims and relationships together."""
        claim1 = Claim(
            id="c0000001",
            content="First claim in relationship test.",
            confidence=0.8,
            created_by="user1"
        )
        
        claim2 = Claim(
            id="c0000002", 
            content="Second claim in relationship test.",
            confidence=0.7,
            created_by="user1"
        )
        
        relationship = Relationship(
            supporter_id=claim1.id,
            supported_id=claim2.id,
            relationship_type="supports",
            created_by="user1"
        )
        
        # Verify relationship links claims correctly
        assert relationship.supporter_id == claim1.id
        assert relationship.supported_id == claim2.id
        assert relationship.supporter_id != relationship.supported_id

    @pytest.mark.models
    def test_batch_result_with_claims(self):
        """Test batch result processing with multiple claims."""
        claim_ids = ["c0000001", "c0000002", "c0000003"]
        results = []
        
        for claim_id in claim_ids:
            result = ProcessingResult(
                claim_id=claim_id,
                success=True,
                updated_confidence=0.8
            )
            results.append(result)
        
        batch_result = BatchResult(
            total_items=len(claim_ids),
            successful_items=3,
            failed_items=0,
            results=results
        )
        
        assert len(batch_result.results) == len(claim_ids)
        assert all(r.success for r in batch_result.results)

    @pytest.mark.models
    def test_filter_with_real_claims(self):
        """Test applying filters to real claim data."""
        claim = Claim(
            id="c0000001",
            content="Scientific claim about physics with high confidence.",
            confidence=0.9,
            tags=["science", "physics"],
            dirty=False,
            created_by="scientist"
        )
        
        # Test various filters
        science_filter = ClaimFilter(tags=["science"])
        high_conf_filter = ClaimFilter(confidence_min=0.8)
        clean_filter = ClaimFilter(dirty_only=False)
        
        # These would typically be used in database queries
        # Here we just test the filter creation
        assert science_filter.tags == ["science"]
        assert high_conf_filter.confidence_min == 0.8
        assert clean_filter.dirty_only is False


# Performance and stress tests for models
class TestModelPerformance:
    """Performance tests for model operations."""

    @pytest.mark.models
    def test_claim_creation_performance(self, benchmark):
        """Benchmark claim creation performance."""
        def create_claim():
            return Claim(
                id="c0000001",
                content="Performance test claim with sufficient content length.",
                confidence=0.7,
                tags=["performance", "test"],
                created_by="perf_user"
            )
        
        result = benchmark(create_claim)
        assert isinstance(result, Claim)

    @pytest.mark.models
    def test_claim_validation_performance(self, claim_generator):
        """Benchmark claim validation performance."""
        claim_data = claim_generator.generate(1)[0]
        
        def validate_and_create():
            return Claim(**claim_data)
        
        # Should be very fast (<1ms)
        result = benchmark(validate_and_create)
        assert isinstance(result, Claim)

    @pytest.mark.models
    def test_serialization_performance(self, valid_claim: Claim):
        """Benchmark claim serialization performance."""
        def serialize_claim():
            return valid_claim.json()
        
        result = benchmark(serialize_claim)
        assert isinstance(result, str)
        assert len(result) > 0


# Edge cases and boundary tests
class TestModelEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.models
    def test_extreme_confidence_values(self):
        """Test extreme confidence values."""
        # Boundary values
        Claim(
            id="c0000001",
            content="Test claim with minimum confidence.",
            confidence=0.0,
            created_by="test_user"
        )
        
        Claim(
            id="c0000002",
            content="Test claim with maximum confidence.",
            confidence=1.0,
            created_by="test_user"
        )

    @pytest.mark.models
    def test_long_content(self):
        """Test claim with very long content."""
        long_content = "A" * 10000  # 10KB of content
        
        claim = Claim(
            id="c0000001",
            content=long_content,
            confidence=0.7,
            created_by="test_user"
        )
        
        assert len(claim.content) == 10000

    @pytest.mark.models
    def test_many_tags(self):
        """Test claim with many tags."""
        many_tags = [f"tag_{i}" for i in range(100)]
        
        claim = Claim(
            id="c0000001",
            content="Test claim with many tags.",
            confidence=0.7,
            tags=many_tags,
            created_by="test_user"
        )
        
        assert len(claim.tags) == 100
        assert len(set(claim.tags)) == 100  # No duplicates

    @pytest.mark.models
    def test_special_characters_in_content(self):
        """Test claim content with various special characters."""
        special_contents = [
            "Claim with unicode: αβγδε 中文 日本語 🚀",
            "Claim with emojis: 🧬 🧪 🔬 📊",
            "Claim with quotes: 'single' and \"double\" quotes",
            "Claim with newlines:\nLine 1\nLine 2\nLine 3",
            "Claim with tabs:\tTabbed\tcontent\t here",
            "Claim with special chars: !@#$%^&*()[]{}|;:',.<>/?",
            "Claim with math: E=mc² and ∫f(x)dx = F(x) + C"
        ]
        
        for content in special_contents:
            claim = Claim(
                id="c0000001",
                content=content,
                confidence=0.7,
                created_by="test_user"
            )
            assert claim.content == content

    @pytest.mark.models
    def test_self_supporting_relationship(self):
        """Test relationship validation for self-supporting claims."""
        # This should be caught at the application level, not model level
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000001",  # Same claim
            relationship_type="supports"
        )
        
        # Model accepts it (validation happens in DataManager)
        assert relationship.supporter_id == relationship.supported_id