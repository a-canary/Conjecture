"""
Comprehensive tests for data layer components (src/data/models.py and src/data/repositories.py)
Focus on improving coverage for exception classes, enums, validators, and repository implementations.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock all external dependencies before importing our modules
with patch.dict('sys.modules', {
    'chromadb': MagicMock(),
    'chromadb.config': MagicMock(),
    'chromadb.api': MagicMock(),
    'chromadb.api.models': MagicMock(),
    'sentence_transformers': MagicMock(),
    'torch': MagicMock(),
    'tensorflow': MagicMock(),
    'numpy': MagicMock(),
    'sklearn': MagicMock(),
    'sklearn.metrics': MagicMock(),
    'scipy': MagicMock(),
    'scipy.spatial': MagicMock(),
}):
    # Now import the modules we want to test
    from src.data.models import (
        DataLayerError, ClaimNotFoundError, InvalidClaimError, 
        RelationshipError, EmbeddingError, ClaimState, ClaimType,
        Claim, validate_claim_id, validate_confidence, generate_claim_id,
        Relationship, ClaimFilter, DataConfig, QueryResult, ProcessingStats,
        ProcessingResult
    )
    from src.data.repositories import (
        ClaimRepository, DataManagerClaimRepository, RepositoryFactory, get_data_manager
    )


class TestDataLayerExceptions:
    """Test all exception classes in the data layer."""
    
    def test_data_layer_error_basic(self):
        """Test DataLayerError basic functionality."""
        error = DataLayerError("Test error")
        assert str(error) == "Test error"
        assert error.args == ("Test error",)
    
    def test_data_layer_error_inheritance(self):
        """Test DataLayerError inherits from Exception."""
        error = DataLayerError("Test")
        assert isinstance(error, Exception)
    
    def test_claim_not_found_error_inheritance(self):
        """Test ClaimNotFoundError inherits from DataLayerError."""
        error = ClaimNotFoundError("Test")
        assert isinstance(error, DataLayerError)
        assert isinstance(error, Exception)
    
    def test_invalid_claim_error_inheritance(self):
        """Test InvalidClaimError inherits from DataLayerError."""
        error = InvalidClaimError("Test")
        assert isinstance(error, DataLayerError)
        assert isinstance(error, Exception)
    
    def test_relationship_error_inheritance(self):
        """Test RelationshipError inherits from DataLayerError."""
        error = RelationshipError("Test")
        assert isinstance(error, DataLayerError)
        assert isinstance(error, Exception)
    
    def test_embedding_error_inheritance(self):
        """Test EmbeddingError inherits from DataLayerError."""
        error = EmbeddingError("Test")
        assert isinstance(error, DataLayerError)
        assert isinstance(error, Exception)


class TestClaimStateEnum:
    """Test ClaimState enum functionality."""
    
    def test_claim_state_values(self):
        """Test all ClaimState enum values."""
        assert ClaimState.EXPLORE.value == "Explore"
        assert ClaimState.VALIDATED.value == "Validated"
        assert ClaimState.ORPHANED.value == "Orphaned"
        assert ClaimState.QUEUED.value == "Queued"
    
    def test_claim_state_creation(self):
        """Test creating ClaimState instances."""
        explore_state = ClaimState("Explore")
        validated_state = ClaimState("Validated")
        orphaned_state = ClaimState("Orphaned")
        queued_state = ClaimState("Queued")
        
        assert explore_state == ClaimState.EXPLORE
        assert validated_state == ClaimState.VALIDATED
        assert orphaned_state == ClaimState.ORPHANED
        assert queued_state == ClaimState.QUEUED
    
    def test_claim_state_iteration(self):
        """Test iterating over ClaimState values."""
        states = list(ClaimState)
        assert len(states) == 4
        assert ClaimState.EXPLORE in states
        assert ClaimState.VALIDATED in states
        assert ClaimState.ORPHANED in states
        assert ClaimState.QUEUED in states
    
    def test_claim_state_comparison(self):
        """Test ClaimState comparison operations."""
        assert ClaimState.EXPLORE != ClaimState.VALIDATED
        assert ClaimState.EXPLORE == ClaimState.EXPLORE


class TestClaimTypeEnum:
    """Test ClaimType enum functionality."""
    
    def test_claim_type_values(self):
        """Test all ClaimType enum values."""
        assert ClaimType.CONCEPT.value == "concept"
        assert ClaimType.REFERENCE.value == "reference"
        assert ClaimType.THESIS.value == "thesis"
        assert ClaimType.SKILL.value == "skill"
        assert ClaimType.EXAMPLE.value == "example"
        assert ClaimType.GOAL.value == "goal"
    
    def test_claim_type_creation(self):
        """Test creating ClaimType instances."""
        concept_type = ClaimType("concept")
        reference_type = ClaimType("reference")
        thesis_type = ClaimType("thesis")
        skill_type = ClaimType("skill")
        example_type = ClaimType("example")
        goal_type = ClaimType("goal")
        
        assert concept_type == ClaimType.CONCEPT
        assert reference_type == ClaimType.REFERENCE
        assert thesis_type == ClaimType.THESIS
        assert skill_type == ClaimType.SKILL
        assert example_type == ClaimType.EXAMPLE
        assert goal_type == ClaimType.GOAL
    
    def test_claim_type_iteration(self):
        """Test iterating over ClaimType values."""
        types = list(ClaimType)
        assert len(types) == 6
        assert ClaimType.CONCEPT in types
        assert ClaimType.REFERENCE in types
        assert ClaimType.THESIS in types
        assert ClaimType.SKILL in types
        assert ClaimType.EXAMPLE in types
        assert ClaimType.GOAL in types
    
    def test_claim_type_comparison(self):
        """Test ClaimType comparison operations."""
        assert ClaimType.CONCEPT != ClaimType.REFERENCE
        assert ClaimType.CONCEPT == ClaimType.CONCEPT


class TestUtilityFunctions:
    """Test utility functions in the data layer."""
    
    def test_validate_claim_id_valid(self):
        """Test validate_claim_id with valid IDs."""
        assert validate_claim_id("c0000001") == True
        assert validate_claim_id("c1234567") == True
        assert validate_claim_id("c0000000") == True
        assert validate_claim_id("c9999999") == True
    
    def test_validate_claim_id_invalid_format(self):
        """Test validate_claim_id with invalid formats."""
        assert validate_claim_id("a0000001") == False
        assert validate_claim_id("00000001") == False
        assert validate_claim_id("c000001") == False  # Too short
        assert validate_claim_id("c00000001") == False  # Too long
    
    def test_validate_claim_id_invalid_characters(self):
        """Test validate_claim_id with invalid characters."""
        assert validate_claim_id("c00000a1") == False
        assert validate_claim_id("cabcdefgh") == False
    
    def test_validate_confidence_valid(self):
        """Test validate_confidence with valid values."""
        assert validate_confidence(0.0) == True
        assert validate_confidence(0.5) == True
        assert validate_confidence(1.0) == True
        assert validate_confidence(0.95) == True
    
    def test_validate_confidence_invalid_low(self):
        """Test validate_confidence with values too low."""
        assert validate_confidence(-0.1) == False
        assert validate_confidence(-1.0) == False
    
    def test_validate_confidence_invalid_high(self):
        """Test validate_confidence with values too high."""
        assert validate_confidence(1.1) == False
        assert validate_confidence(2.0) == False
    
    def test_generate_claim_id(self):
        """Test generate_claim_id function."""
        claim_id = generate_claim_id(1)
        assert isinstance(claim_id, str)
        assert len(claim_id) == 8
        assert claim_id.startswith('c')
        assert claim_id[1:].isdigit()
        assert claim_id == "c0000001"
        
        # Test with different counter
        claim_id_2 = generate_claim_id(123)
        assert claim_id_2 == "c0000123"
    
    def test_generate_claim_id_default(self):
        """Test generate_claim_id with default counter."""
        claim_id = generate_claim_id()
        assert claim_id == "c0000001"


class TestClaimModel:
    """Test Claim model functionality."""
    
    def test_claim_model_minimal_creation(self):
        """Test creating a Claim with minimal required fields."""
        claim = Claim(
            id="c0000001",
            content="This is a test claim with sufficient content length",
            confidence=0.8
        )
        
        assert claim.id == "c0000001"
        assert claim.content == "This is a test claim with sufficient content length"
        assert claim.confidence == 0.8
        assert claim.dirty == True  # Default value
        assert claim.created is not None
        assert claim.updated is None
        assert claim.tags == []
        assert claim.state == ClaimState.EXPLORE  # Default value
        assert claim.type == [ClaimType.CONCEPT]  # Default value
        assert claim.embedding is None
    
    def test_claim_model_full_creation(self):
        """Test creating a Claim with all fields."""
        now = datetime.now()
        claim = Claim(
            id="c0000002",
            content="A comprehensive claim with all fields populated",
            confidence=0.95,
            dirty=False,
            created=now,
            updated=now,
            tags=["test", "comprehensive"],
            state=ClaimState.VALIDATED,
            type=[ClaimType.THESIS, ClaimType.EXAMPLE],
            embedding=[0.1, 0.2, 0.3],
            is_dirty=False,
            dirty_reason="Test reason",
            dirty_timestamp=now,
            dirty_priority=5,
            supported_by=["c0000001"],
            supports=["c0000003"]
        )
        
        assert claim.id == "c0000002"
        assert claim.confidence == 0.95
        assert claim.dirty == False
        assert claim.created == now
        assert claim.updated == now
        assert claim.tags == ["test", "comprehensive"]
        assert claim.state == ClaimState.VALIDATED
        assert claim.type == [ClaimType.THESIS, ClaimType.EXAMPLE]
        assert claim.embedding == [0.1, 0.2, 0.3]
        assert claim.is_dirty == False
        assert claim.dirty_reason == "Test reason"
        assert claim.dirty_timestamp == now
        assert claim.dirty_priority == 5
        assert claim.supported_by == ["c0000001"]
        assert claim.supports == ["c0000003"]
    
    def test_claim_model_invalid_id_format(self):
        """Test Claim model validation for invalid ID format."""
        with pytest.raises(ValueError, match="Claim ID must be in format c########"):
            Claim(
                id="invalid_id",
                content="Test claim content",
                confidence=0.8
            )
    
    def test_claim_model_invalid_confidence_range(self):
        """Test Claim model validation for invalid confidence range."""
        with pytest.raises(ValueError):
            Claim(
                id="c0000001",
                content="Test claim content",
                confidence=1.5  # Too high
            )
        
        with pytest.raises(ValueError):
            Claim(
                id="c0000002",
                content="Test claim content",
                confidence=-0.1  # Too low
            )
    
    def test_claim_model_content_too_short(self):
        """Test Claim model validation for content too short."""
        with pytest.raises(ValueError):
            Claim(
                id="c0000001",
                content="Short",  # Less than 10 characters
                confidence=0.8
            )
    
    def test_claim_model_content_too_long(self):
        """Test Claim model validation for content too long."""
        long_content = "x" * 5001  # More than 5000 characters
        with pytest.raises(ValueError):
            Claim(
                id="c0000001",
                content=long_content,
                confidence=0.8
            )
    
    def test_claim_model_invalid_tags(self):
        """Test Claim model validation for invalid tags."""
        with pytest.raises(ValueError, match="Tags must be non-empty strings"):
            Claim(
                id="c0000001",
                content="Test claim content with sufficient length",
                confidence=0.8,
                tags=["valid", "", "invalid"]  # Empty string tag
            )
    
    def test_claim_model_tags_deduplication(self):
        """Test that duplicate tags are removed."""
        claim = Claim(
            id="c0000001",
            content="Test claim content with sufficient length",
            confidence=0.8,
            tags=["test", "claim", "test", "duplicate", "claim"]  # Duplicates
        )
        
        # Should remove duplicates while preserving order
        assert claim.tags == ["test", "claim", "duplicate"]
    
    def test_claim_model_updated_before_created(self):
        """Test Claim model validation for updated timestamp before created."""
        created = datetime.now()
        updated = datetime(2020, 1, 1)  # Earlier than created
        
        with pytest.raises(ValueError, match="Updated timestamp cannot be before creation timestamp"):
            Claim(
                id="c0000001",
                content="Test claim content with sufficient length",
                confidence=0.8,
                created=created,
                updated=updated
            )
    
    def test_claim_model_datetime_serialization(self):
        """Test datetime serialization."""
        claim = Claim(
            id="c0000001",
            content="Test claim content with sufficient length",
            confidence=0.8
        )
        
        # Test the serializer method
        serialized_created = claim.serialize_datetime(claim.created)
        assert isinstance(serialized_created, str)
        assert serialized_created.endswith('Z') or 'T' in serialized_created
        
        # Test with None
        assert claim.serialize_datetime(None) is None
    
    def test_claim_model_to_dict(self):
        """Test Claim model conversion to dict."""
        now = datetime.now()
        claim = Claim(
            id="c0000001",
            content="Test claim content with sufficient length",
            confidence=0.8,
            tags=["test"],
            created=now,
            updated=now  # Set updated after created to avoid validation error
        )
        
        claim_dict = claim.model_dump(exclude_none=True)
        assert isinstance(claim_dict, dict)
        assert claim_dict["id"] == "c0000001"
        assert claim_dict["content"] == "Test claim content with sufficient length"
        assert claim_dict["confidence"] == 0.8
        assert claim_dict["tags"] == ["test"]
        # Should exclude None values by default
        assert "embedding" not in claim_dict
    
    def test_claim_model_created_at_property(self):
        """Test backward compatibility created_at property."""
        claim = Claim(
            id="c0000001",
            content="Test claim content with sufficient length",
            confidence=0.8
        )
        
        assert claim.created_at == claim.created
    
    def test_claim_model_to_chroma_metadata(self):
        """Test conversion to ChromaDB metadata format."""
        now = datetime.now()
        claim = Claim(
            id="c0000001",
            content="Test claim content with sufficient length",
            confidence=0.8,
            state=ClaimState.VALIDATED,
            type=[ClaimType.CONCEPT, ClaimType.EXAMPLE],
            tags=["test", "metadata"],
            supported_by=["c0000002"],
            supports=["c0000003"],
            created=now,
            updated=now  # Set updated after created to avoid validation error
        )
        
        metadata = claim.to_chroma_metadata()
        
        assert metadata["confidence"] == 0.8
        assert metadata["state"] == "Validated"
        assert metadata["supported_by"] == "c0000002"
        assert metadata["supports"] == "c0000003"
        assert metadata["type"] == "concept,example"
        assert metadata["tags"] == "test,metadata"
        assert "created" in metadata
        assert "updated" in metadata
        assert metadata["dirty"] == True
    
    def test_claim_model_from_chroma_result(self):
        """Test creating claim from ChromaDB result."""
        metadata = {
            "confidence": 0.85,
            "state": "Explore",
            "supported_by": "c0000001,c0000002",
            "supports": "c0000003",
            "type": "concept,example",
            "tags": "test,chroma",
            "created": "2023-01-01T00:00:00",
            "updated": "2023-01-02T00:00:00",
            "dirty": False
        }
        
        claim = Claim.from_chroma_result(
            id="c0000004",
            content="Test content from ChromaDB",
            metadata=metadata
        )
        
        assert claim.id == "c0000004"
        assert claim.content == "Test content from ChromaDB"
        assert claim.confidence == 0.85
        assert claim.state == ClaimState.EXPLORE
        assert claim.supported_by == ["c0000001", "c0000002"]
        assert claim.supports == ["c0000003"]
        assert claim.type == [ClaimType.CONCEPT, ClaimType.EXAMPLE]
        assert claim.tags == ["test", "chroma"]
        assert claim.dirty == False
    
    def test_claim_model_mark_dirty(self):
        """Test marking claim as dirty."""
        claim = Claim(
            id="c0000001",
            content="Test claim content with sufficient length",
            confidence=0.8,
            dirty=False
        )
        
        original_updated = claim.updated
        claim.mark_dirty()
        
        assert claim.dirty == True
        assert claim.updated is not None
        assert claim.updated != original_updated
    
    def test_claim_model_mark_clean(self):
        """Test marking claim as clean."""
        claim = Claim(
            id="c0000001",
            content="Test claim content with sufficient length",
            confidence=0.8,
            dirty=True
        )
        
        original_updated = claim.updated
        claim.mark_clean()
        
        assert claim.dirty == False
        assert claim.updated is not None
        assert claim.updated != original_updated
    
    def test_claim_model_update_confidence(self):
        """Test updating confidence score."""
        claim = Claim(
            id="c0000001",
            content="Test claim content with sufficient length",
            confidence=0.8
        )
        
        original_dirty = claim.dirty
        claim.update_confidence(0.9)
        
        assert claim.confidence == 0.9
        assert claim.dirty == True  # Should be marked dirty after update
        
        # Test invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            claim.update_confidence(1.5)
    
    def test_claim_model_add_support(self):
        """Test adding supporting claims."""
        claim = Claim(
            id="c0000001",
            content="Test claim content with sufficient length",
            confidence=0.8
        )
        
        original_dirty = claim.dirty
        claim.add_support("c0000002")
        
        assert "c0000002" in claim.supported_by
        assert claim.dirty == True
        
        # Test adding duplicate (should not add)
        claim.add_support("c0000002")
        assert claim.supported_by.count("c0000002") == 1
        
        # Test adding empty string (should not add)
        claim.add_support("")
        assert "" not in claim.supported_by
    
    def test_claim_model_add_supports(self):
        """Test adding supported claims."""
        claim = Claim(
            id="c0000001",
            content="Test claim content with sufficient length",
            confidence=0.8
        )
        
        claim.add_supports("c0000003")
        
        assert "c0000003" in claim.supports
        assert claim.dirty == True
    
    def test_claim_model_repr(self):
        """Test Claim model string representation."""
        claim = Claim(
            id="c0000001",
            content="Test claim content with sufficient length",
            confidence=0.85,
            state=ClaimState.VALIDATED
        )
        
        repr_str = repr(claim)
        assert "c0000001" in repr_str
        assert "0.85" in repr_str
        assert "Validated" in repr_str


class TestRelationshipModel:
    """Test Relationship model functionality."""
    
    def test_relationship_model_creation(self):
        """Test creating a Relationship."""
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002"
        )
        
        assert relationship.supporter_id == "c0000001"
        assert relationship.supported_id == "c0000002"
        assert relationship.created is not None
    
    def test_relationship_model_with_timestamp(self):
        """Test creating a Relationship with custom timestamp."""
        now = datetime.now()
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002",
            created=now
        )
        
        assert relationship.created == now
    
    def test_relationship_model_created_at_property(self):
        """Test backward compatibility created_at property."""
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002"
        )
        
        assert relationship.created_at == relationship.created
    
    def test_relationship_model_repr(self):
        """Test Relationship model string representation."""
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002"
        )
        
        repr_str = repr(relationship)
        assert "c0000001" in repr_str
        assert "c0000002" in repr_str
        assert "supports" in repr_str


class TestClaimFilter:
    """Test ClaimFilter model functionality."""
    
    def test_claim_filter_defaults(self):
        """Test ClaimFilter default values."""
        filter_obj = ClaimFilter()
        
        assert filter_obj.tags is None
        assert filter_obj.confidence_min is None
        assert filter_obj.confidence_max is None
        assert filter_obj.dirty_only is None
        assert filter_obj.content_contains is None
        assert filter_obj.limit == 100
        assert filter_obj.offset == 0
        assert filter_obj.created_after is None
        assert filter_obj.created_before is None
        assert filter_obj.states is None
        assert filter_obj.types is None
    
    def test_claim_filter_with_values(self):
        """Test ClaimFilter with specific values."""
        now = datetime.now()
        filter_obj = ClaimFilter(
            tags=["test", "filter"],
            confidence_min=0.5,
            confidence_max=0.9,
            dirty_only=True,
            content_contains="search term",
            limit=50,
            offset=10,
            created_after=now,
            states=[ClaimState.EXPLORE, ClaimState.VALIDATED],
            types=[ClaimType.CONCEPT]
        )
        
        assert filter_obj.tags == ["test", "filter"]
        assert filter_obj.confidence_min == 0.5
        assert filter_obj.confidence_max == 0.9
        assert filter_obj.dirty_only == True
        assert filter_obj.content_contains == "search term"
        assert filter_obj.limit == 50
        assert filter_obj.offset == 10
        assert filter_obj.created_after == now
        assert filter_obj.states == [ClaimState.EXPLORE, ClaimState.VALIDATED]
        assert filter_obj.types == [ClaimType.CONCEPT]
    
    def test_claim_filter_confidence_range_validation(self):
        """Test confidence range validation."""
        # Valid range
        filter_obj = ClaimFilter(confidence_min=0.3, confidence_max=0.7)
        assert filter_obj.confidence_min == 0.3
        assert filter_obj.confidence_max == 0.7
        
        # Invalid range (max < min)
        with pytest.raises(ValueError, match="confidence_max .* must be >= confidence_min"):
            ClaimFilter(confidence_min=0.7, confidence_max=0.3)
    
    def test_claim_filter_limit_validation(self):
        """Test limit validation."""
        # Test default setting
        filter_obj = ClaimFilter(limit=None)
        assert filter_obj.limit == 100
    
    def test_claim_filter_offset_validation(self):
        """Test offset validation."""
        # Test default setting
        filter_obj = ClaimFilter(offset=None)
        assert filter_obj.offset == 0


class TestDataConfig:
    """Test DataConfig model functionality."""
    
    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        
        assert config.sqlite_path == "./data/conjecture.db"
        assert config.chroma_path == "./data/vector_db"
        assert config.chroma_collection == "claims"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.embedding_dim is None
        assert config.cache_size == 1000
        assert config.cache_ttl == 300
        assert config.batch_size == 100
        assert config.max_connections == 10
        assert config.query_timeout == 30
        assert config.use_chroma == True
        assert config.use_embeddings == True
        assert config.auto_sync == True
    
    def test_data_config_custom_values(self):
        """Test DataConfig with custom values."""
        config = DataConfig(
            sqlite_path="/custom/path.db",
            chroma_path="/custom/vector",
            chroma_collection="custom_collection",
            embedding_model="custom-model",
            embedding_dim=768,
            cache_size=2000,
            cache_ttl=600,
            batch_size=50,
            max_connections=20,
            query_timeout=60,
            use_chroma=False,
            use_embeddings=False,
            auto_sync=False
        )
        
        assert config.sqlite_path == "/custom/path.db"
        assert config.chroma_path == "/custom/vector"
        assert config.chroma_collection == "custom_collection"
        assert config.embedding_model == "custom-model"
        assert config.embedding_dim == 768
        assert config.cache_size == 2000
        assert config.cache_ttl == 600
        assert config.batch_size == 50
        assert config.max_connections == 20
        assert config.query_timeout == 60
        assert config.use_chroma == False
        assert config.use_embeddings == False
        assert config.auto_sync == False


class TestQueryResult:
    """Test QueryResult model functionality."""
    
    def test_query_result_defaults(self):
        """Test QueryResult default values."""
        result = QueryResult()
        
        assert result.items == []
        assert result.total_count is None
        assert result.query_time is None
        assert result.has_more == False
        assert len(result) == 0
    
    def test_query_result_with_items(self):
        """Test QueryResult with items."""
        items = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        result = QueryResult(
            items=items,
            total_count=100,
            query_time=0.5,
            has_more=True
        )
        
        assert result.items == items
        assert result.total_count == 100
        assert result.query_time == 0.5
        assert result.has_more == True
        assert len(result) == 3


class TestProcessingStats:
    """Test ProcessingStats model functionality."""
    
    def test_processing_stats_creation(self):
        """Test ProcessingStats creation."""
        start_time = datetime.now()
        stats = ProcessingStats(
            operation="test_operation",
            start_time=start_time
        )
        
        assert stats.operation == "test_operation"
        assert stats.start_time == start_time
        assert stats.end_time is None
        assert stats.items_processed == 0
        assert stats.items_succeeded == 0
        assert stats.items_failed == 0
    
    def test_processing_stats_properties(self):
        """Test ProcessingStats calculated properties."""
        start_time = datetime.now()
        # Ensure end_time is after start_time
        import time
        time.sleep(0.01)  # Small delay to ensure different timestamps
        end_time = datetime.now()
        
        stats = ProcessingStats(
            operation="test_operation",
            start_time=start_time,
            end_time=end_time,
            items_processed=100,
            items_succeeded=90,
            items_failed=10
        )
        
        assert stats.duration is not None
        assert stats.success_rate == 90.0
        assert stats.items_per_second is not None
    
    def test_processing_stats_methods(self):
        """Test ProcessingStats methods."""
        stats = ProcessingStats(
            operation="test_operation",
            start_time=datetime.now()  # Provide required start_time
        )
        
        assert stats.start_time is not None
        
        stats.add_success(5)
        stats.add_failure(2)
        
        assert stats.items_processed == 7
        assert stats.items_succeeded == 5
        assert stats.items_failed == 2
        
        stats.end()
        assert stats.end_time is not None


class TestProcessingResult:
    """Test ProcessingResult model functionality."""
    
    def test_processing_result_minimal(self):
        """Test ProcessingResult with minimal fields."""
        result = ProcessingResult(
            claim_id="c0000001",
            success=True
        )
        
        assert result.claim_id == "c0000001"
        assert result.success == True
        assert result.message is None
        assert result.updated_confidence is None
        assert result.processing_time is None
        assert result.metadata is None
    
    def test_processing_result_full(self):
        """Test ProcessingResult with all fields."""
        result = ProcessingResult(
            claim_id="c0000001",
            success=False,
            message="Processing failed",
            updated_confidence=0.7,
            processing_time=1.5,
            metadata={"error": "details"}
        )
        
        assert result.claim_id == "c0000001"
        assert result.success == False
        assert result.message == "Processing failed"
        assert result.updated_confidence == 0.7
        assert result.processing_time == 1.5
        assert result.metadata == {"error": "details"}


class TestClaimRepository:
    """Test ClaimRepository abstract class."""
    
    def test_claim_repository_is_abstract(self):
        """Test that ClaimRepository cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ClaimRepository()
    
    def test_claim_repository_interface_methods(self):
        """Test that ClaimRepository defines the expected interface."""
        # Check that all required methods are defined
        assert hasattr(ClaimRepository, 'create')
        assert hasattr(ClaimRepository, 'get_by_id')
        assert hasattr(ClaimRepository, 'update')
        assert hasattr(ClaimRepository, 'delete')
        assert hasattr(ClaimRepository, 'search')
        assert hasattr(ClaimRepository, 'list_by_state')


class TestDataManagerClaimRepository:
    """Test DataManagerClaimRepository implementation."""
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock data manager for testing."""
        manager = Mock()
        manager.create_claim = AsyncMock()
        manager.get_claim = AsyncMock()
        manager.update_claim = AsyncMock()
        manager.delete_claim = AsyncMock()
        manager.search_claims = AsyncMock()
        manager.list_claims = AsyncMock()
        return manager
    
    @pytest.fixture
    def repository(self, mock_data_manager):
        """Create a repository instance with mock data manager."""
        return DataManagerClaimRepository(mock_data_manager)
    
    @pytest.mark.asyncio
    async def test_create_claim_success(self, repository, mock_data_manager):
        """Test successful claim creation."""
        # Mock claim data
        claim_data = {
            'content': 'Test claim content with sufficient length',
            'confidence': 0.8,
            'tags': ['test'],
            'state': 'Explore'
        }
        
        # Mock returned claim
        mock_claim = Mock()
        mock_data_manager.create_claim.return_value = mock_claim
        
        result = await repository.create(claim_data)
        
        assert result == mock_claim
        mock_data_manager.create_claim.assert_called_once_with(
            content='Test claim content with sufficient length',
            confidence=0.8,
            tags=['test'],
            state=ClaimState('Explore')
        )
    
    @pytest.mark.asyncio
    async def test_create_claim_with_claim_type_backwards_compatibility(self, repository, mock_data_manager):
        """Test claim creation with backwards compatibility for claim_type."""
        claim_data = {
            'content': 'Test claim content with sufficient length',
            'confidence': 0.8,
            'claim_type': 'concept',
            'tags': ['existing']
        }
        
        mock_claim = Mock()
        mock_data_manager.create_claim.return_value = mock_claim
        
        result = await repository.create(claim_data)
        
        assert result == mock_claim
        # Should add claim_type to tags (only once if concept already present)
        expected_tags = ['existing', 'concept']  # claim_type added once
        mock_data_manager.create_claim.assert_called_once_with(
            content='Test claim content with sufficient length',
            confidence=0.8,
            tags=expected_tags,
            state=ClaimState('Explore')
        )
    
    @pytest.mark.asyncio
    async def test_create_claim_error(self, repository, mock_data_manager):
        """Test claim creation with error."""
        claim_data = {
            'content': 'Test claim content with sufficient length',
            'confidence': 0.8
        }
        
        mock_data_manager.create_claim.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Database error"):
            await repository.create(claim_data)
    
    @pytest.mark.asyncio
    async def test_get_by_id_success(self, repository, mock_data_manager):
        """Test successful claim retrieval by ID."""
        mock_claim = Mock()
        mock_data_manager.get_claim.return_value = mock_claim
        
        result = await repository.get_by_id("c0000001")
        
        assert result == mock_claim
        mock_data_manager.get_claim.assert_called_once_with("c0000001")
    
    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repository, mock_data_manager):
        """Test claim retrieval when claim not found."""
        mock_data_manager.get_claim.side_effect = Exception("Not found")
        
        result = await repository.get_by_id("c0000001")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_claim(self, repository, mock_data_manager):
        """Test claim update."""
        mock_claim = Mock()
        mock_data_manager.update_claim.return_value = mock_claim
        
        updates = {"confidence": 0.9}
        result = await repository.update("c0000001", updates)
        
        assert result == mock_claim
        mock_data_manager.update_claim.assert_called_once_with("c0000001", updates)
    
    @pytest.mark.asyncio
    async def test_delete_claim_success(self, repository, mock_data_manager):
        """Test successful claim deletion."""
        result = await repository.delete("c0000001")
        
        assert result is True
        mock_data_manager.delete_claim.assert_called_once_with("c0000001")
    
    @pytest.mark.asyncio
    async def test_delete_claim_failure(self, repository, mock_data_manager):
        """Test claim deletion when it fails."""
        mock_data_manager.delete_claim.side_effect = Exception("Delete failed")
        
        result = await repository.delete("c0000001")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_search_claims(self, repository, mock_data_manager):
        """Test claim search."""
        mock_result = Mock()
        mock_result.claims = [Mock(), Mock()]
        mock_data_manager.search_claims.return_value = mock_result
        
        result = await repository.search("test query", limit=5)
        
        assert result == mock_result.claims
        mock_data_manager.search_claims.assert_called_once_with("test query", limit=5)
    
    @pytest.mark.asyncio
    async def test_list_by_state(self, repository, mock_data_manager):
        """Test listing claims by state."""
        mock_result = Mock()
        mock_result.claims = [Mock(), Mock()]
        mock_data_manager.list_claims.return_value = mock_result
        
        result = await repository.list_by_state(ClaimState.EXPLORE)
        
        assert result == mock_result.claims
        mock_data_manager.list_claims.assert_called_once_with({"state": ClaimState.EXPLORE})


class TestRepositoryFactory:
    """Test RepositoryFactory functionality."""
    
    def test_create_claim_repository(self):
        """Test creating a claim repository."""
        mock_data_manager = Mock()
        
        repository = RepositoryFactory.create_claim_repository(mock_data_manager)
        
        assert isinstance(repository, DataManagerClaimRepository)
        assert repository.data_manager == mock_data_manager


class TestGetDataManager:
    """Test get_data_manager function."""
    
    def test_get_data_manager_default(self):
        """Test get_data_manager with default parameters."""
        # Just test that the function can be called without errors
        try:
            result = get_data_manager()
            assert result is not None
        except ImportError:
            # If dependencies are missing, that's expected in test environment
            pytest.skip("Data manager dependencies not available")
    
    def test_get_data_manager_with_mock_embeddings(self):
        """Test get_data_manager with mock embeddings."""
        # Just test that the function can be called without errors
        try:
            result = get_data_manager(use_mock_embeddings=True)
            assert result is not None
        except ImportError:
            # If dependencies are missing, that's expected in test environment
            pytest.skip("Data manager dependencies not available")


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""
    
    def test_claim_model_with_boundary_content_lengths(self):
        """Test Claim model with boundary content lengths."""
        # Test exactly 10 characters (minimum)
        claim_min = Claim(
            id="c0000001",
            content="x" * 10,  # Exactly 10 characters
            confidence=0.8
        )
        assert len(claim_min.content) == 10
        
        # Test exactly 5000 characters (maximum)
        claim_max = Claim(
            id="c0000002",
            content="x" * 5000,  # Exactly 5000 characters
            confidence=0.8
        )
        assert len(claim_max.content) == 5000
    
    def test_claim_model_with_boundary_confidence_values(self):
        """Test Claim model with boundary confidence values."""
        # Test minimum boundary
        claim_min = Claim(
            id="c0000001",
            content="Test claim content with sufficient length",
            confidence=0.0
        )
        assert claim_min.confidence == 0.0
        
        # Test maximum boundary
        claim_max = Claim(
            id="c0000002",
            content="Test claim content with sufficient length",
            confidence=1.0
        )
        assert claim_max.confidence == 1.0
    
    def test_claim_filter_with_boundary_values(self):
        """Test ClaimFilter with boundary values."""
        # Test boundary confidence values
        filter_obj = ClaimFilter(
            confidence_min=0.0,
            confidence_max=1.0,
            limit=1,  # Minimum
            offset=0   # Minimum
        )
        
        assert filter_obj.confidence_min == 0.0
        assert filter_obj.confidence_max == 1.0
        assert filter_obj.limit == 1
        assert filter_obj.offset == 0
    
    def test_processing_stats_edge_cases(self):
        """Test ProcessingStats with edge cases."""
        stats = ProcessingStats(
            operation="test",
            start_time=datetime.now(),
            items_processed=0
        )
        
        # Test with zero items processed
        assert stats.success_rate == 0.0
        
        # Test with no end time
        assert stats.duration is None
        assert stats.items_per_second is None
    
    def test_relationship_model_edge_cases(self):
        """Test Relationship model with edge cases."""
        # Test with same supporter and supported ID
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000001"
        )
        
        assert relationship.supporter_id == "c0000001"
        assert relationship.supported_id == "c0000001"
        assert relationship.created is not None