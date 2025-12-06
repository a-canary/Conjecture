#!/usr/bin/env python3
"""
Focused comprehensive tests for data layer components that avoid problematic imports.
Tests src/data/models.py and src/data/repositories.py directly.
"""

import pytest
import asyncio
import tempfile
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock problematic imports before they happen
sys.modules['chromadb'] = Mock()
sys.modules['chromadb.api'] = Mock()
sys.modules['chromadb.api.client'] = Mock()
sys.modules['chromadb.api.models'] = Mock()
sys.modules['chromadb.utils'] = Mock()
sys.modules['chromadb.utils.embedding_functions'] = Mock()

# Now import the actual models we need to test
from src.data.models import (
    DataLayerError,
    ClaimNotFoundError,
    InvalidClaimError,
    RelationshipError,
    EmbeddingError,
    ClaimState,
    ClaimType,
    Claim,
    Relationship,
    ClaimFilter,
    DataConfig,
    QueryResult,
    ProcessingStats,
    ProcessingResult,
    validate_claim_id,
    validate_confidence,
    generate_claim_id,
)

# Import core models for testing
from src.core.models import (
    Claim as CoreClaim,
    ClaimState as CoreClaimState,
    ClaimType as CoreClaimType,
    ClaimScope,
    RelationshipError as CoreRelationshipError,
    DataLayerError as CoreDataLayerError,
    ClaimNotFoundError as CoreClaimNotFoundError,
    InvalidClaimError as CoreInvalidClaimError,
)

# Mock data manager for repository testing
class MockDataManager:
    """Mock data manager for testing repositories"""
    
    def __init__(self):
        self.claims = {}
        self.relationships = {}
    
    async def create_claim(self, **kwargs):
        """Mock create claim"""
        claim_id = kwargs.get('id', f"c{len(self.claims) + 1:08d}")
        claim = CoreClaim(
            id=claim_id,
            content=kwargs.get('content', ''),
            confidence=kwargs.get('confidence', 0.0),
            tags=kwargs.get('tags', []),
            state=kwargs.get('state', CoreClaimState.EXPLORE)
        )
        self.claims[claim_id] = claim
        return claim
    
    async def get_claim(self, claim_id):
        """Mock get claim"""
        return self.claims.get(claim_id)
    
    async def update_claim(self, claim_id, updates):
        """Mock update claim"""
        if claim_id in self.claims:
            claim = self.claims[claim_id]
            for key, value in updates.items():
                setattr(claim, key, value)
            return claim
        return None
    
    async def delete_claim(self, claim_id):
        """Mock delete claim"""
        if claim_id in self.claims:
            del self.claims[claim_id]
            return True
        return False
    
    async def search_claims(self, query, limit=10):
        """Mock search claims"""
        results = []
        for claim in self.claims.values():
            if query.lower() in claim.content.lower():
                results.append(claim)
                if len(results) >= limit:
                    break
        
        mock_result = Mock()
        mock_result.claims = results
        return mock_result
    
    async def list_claims(self, filter_obj):
        """Mock list claims"""
        results = []
        for claim in self.claims.values():
            if 'state' in filter_obj:
                if claim.state.value == filter_obj['state'].value:
                    results.append(claim)
        
        mock_result = Mock()
        mock_result.claims = results
        return mock_result


# Import repositories after mocking
from src.data.repositories import (
    ClaimRepository,
    DataManagerClaimRepository,
    RepositoryFactory,
    get_data_manager,
)


class TestDataLayerExceptions:
    """Test all exception classes in models.py"""
    
    def test_data_layer_error(self):
        """Test DataLayerError base exception"""
        error = DataLayerError("Test data layer error")
        assert str(error) == "Test data layer error"
        assert isinstance(error, Exception)
        assert isinstance(error, DataLayerError)
    
    def test_claim_not_found_error(self):
        """Test ClaimNotFoundError exception"""
        error = ClaimNotFoundError("Claim c0000001 not found")
        assert str(error) == "Claim c0000001 not found"
        assert isinstance(error, Exception)
        assert isinstance(error, DataLayerError)
        assert isinstance(error, ClaimNotFoundError)
    
    def test_invalid_claim_error(self):
        """Test InvalidClaimError exception"""
        error = InvalidClaimError("Invalid claim data provided")
        assert str(error) == "Invalid claim data provided"
        assert isinstance(error, Exception)
        assert isinstance(error, DataLayerError)
        assert isinstance(error, InvalidClaimError)
    
    def test_relationship_error(self):
        """Test RelationshipError exception"""
        error = RelationshipError("Invalid relationship between claims")
        assert str(error) == "Invalid relationship between claims"
        assert isinstance(error, Exception)
        assert isinstance(error, DataLayerError)
        assert isinstance(error, RelationshipError)
    
    def test_embedding_error(self):
        """Test EmbeddingError exception"""
        error = EmbeddingError("Failed to generate embedding")
        assert str(error) == "Failed to generate embedding"
        assert isinstance(error, Exception)
        assert isinstance(error, DataLayerError)
        assert isinstance(error, EmbeddingError)


class TestClaimStateEnum:
    """Test ClaimState enum"""
    
    def test_claim_state_values(self):
        """Test all ClaimState enum values"""
        assert ClaimState.EXPLORE.value == "Explore"
        assert ClaimState.VALIDATED.value == "Validated"
        assert ClaimState.ORPHANED.value == "Orphaned"
        assert ClaimState.QUEUED.value == "Queued"
    
    def test_claim_state_iteration(self):
        """Test iterating over ClaimState enum"""
        states = list(ClaimState)
        assert len(states) == 4
        assert ClaimState.EXPLORE in states
        assert ClaimState.VALIDATED in states
        assert ClaimState.ORPHANED in states
        assert ClaimState.QUEUED in states
    
    def test_claim_state_comparison(self):
        """Test ClaimState enum comparison"""
        assert ClaimState.EXPLORE == ClaimState.EXPLORE
        assert ClaimState.EXPLORE != ClaimState.VALIDATED
        assert ClaimState.EXPLORE == "Explore"
        assert ClaimState.EXPLORE != "Invalid"


class TestClaimTypeEnum:
    """Test ClaimType enum"""
    
    def test_claim_type_values(self):
        """Test all ClaimType enum values"""
        assert ClaimType.CONCEPT.value == "concept"
        assert ClaimType.REFERENCE.value == "reference"
        assert ClaimType.THESIS.value == "thesis"
        assert ClaimType.SKILL.value == "skill"
        assert ClaimType.EXAMPLE.value == "example"
        assert ClaimType.GOAL.value == "goal"
    
    def test_claim_type_iteration(self):
        """Test iterating over ClaimType enum"""
        types = list(ClaimType)
        assert len(types) == 6
        assert ClaimType.CONCEPT in types
        assert ClaimType.REFERENCE in types
        assert ClaimType.THESIS in types
        assert ClaimType.SKILL in types
        assert ClaimType.EXAMPLE in types
        assert ClaimType.GOAL in types
    
    def test_claim_type_comparison(self):
        """Test ClaimType enum comparison"""
        assert ClaimType.CONCEPT == ClaimType.CONCEPT
        assert ClaimType.CONCEPT != ClaimType.REFERENCE
        assert ClaimType.CONCEPT == "concept"
        assert ClaimType.CONCEPT != "invalid"


class TestClaimModel:
    """Test Claim model with all validators and serializers"""
    
    def test_valid_claim_creation(self):
        """Test creating a valid claim with all fields"""
        created = datetime.utcnow()
        updated = datetime.utcnow()
        
        claim = Claim(
            id="c0000001",
            content="This is a test claim with sufficient length for validation",
            confidence=0.85,
            dirty=False,
            created=created,
            updated=updated,
            tags=["test", "validation"],
            state=ClaimState.VALIDATED,
            type=[ClaimType.CONCEPT, ClaimType.EXAMPLE],
            embedding=[0.1, 0.2, 0.3],
            scope=ClaimScope.USER_WORKSPACE,
            is_dirty=False,
            dirty_reason=None,
            dirty_timestamp=None,
            dirty_priority=0,
            supported_by=["c0000002"],
            supports=["c0000003"]
        )
        
        assert claim.id == "c0000001"
        assert claim.content == "This is a test claim with sufficient length for validation"
        assert claim.confidence == 0.85
        assert claim.dirty is False
        assert claim.created == created
        assert claim.updated == updated
        assert claim.tags == ["test", "validation"]
        assert claim.state == ClaimState.VALIDATED
        assert claim.type == [ClaimType.CONCEPT, ClaimType.EXAMPLE]
        assert claim.embedding == [0.1, 0.2, 0.3]
        assert claim.scope == ClaimScope.USER_WORKSPACE
        assert claim.is_dirty is False
        assert claim.dirty_reason is None
        assert claim.dirty_timestamp is None
        assert claim.dirty_priority == 0
        assert claim.supported_by == ["c0000002"]
        assert claim.supports == ["c0000003"]
    
    def test_claim_default_values(self):
        """Test claim creation with default values"""
        claim = Claim(
            id="c0000001",
            content="This is a test claim with sufficient length",
            confidence=0.8
        )
        
        assert claim.dirty is True  # Default
        assert claim.tags == []  # Default
        assert claim.state == ClaimState.EXPLORE  # Default
        assert claim.type == [ClaimType.CONCEPT]  # Default
        assert claim.embedding is None  # Default
        assert claim.scope == ClaimScope.USER_WORKSPACE  # Default
        assert claim.is_dirty is True  # Default
        assert claim.dirty_reason is None  # Default
        assert claim.dirty_timestamp is None  # Default
        assert claim.dirty_priority == 0  # Default
        assert claim.supported_by == []  # Default
        assert claim.supports == []  # Default
        assert isinstance(claim.created, datetime)
        assert claim.updated is None
    
    def test_claim_id_validation(self):
        """Test claim ID format validation"""
        # Valid IDs
        valid_ids = ["c0000001", "c1234567", "c9999999"]
        for claim_id in valid_ids:
            claim = Claim(
                id=claim_id,
                content="Valid claim content for testing",
                confidence=0.5
            )
            assert claim.id == claim_id
        
        # Invalid IDs
        invalid_ids = [
            "C0000001",  # Upper case
            "a0000001",  # Wrong letter
            "c000001",   # Too short
            "c00000001", # Too long
            "c00000a1",  # Contains letter
            "00000001",  # Missing prefix
            "c-000001",  # Contains dash
        ]
        
        for claim_id in invalid_ids:
            with pytest.raises(ValueError, match="Claim ID must be in format c########"):
                Claim(
                    id=claim_id,
                    content="Invalid claim content for testing",
                    confidence=0.5
                )
    
    def test_claim_content_validation(self):
        """Test claim content length validation"""
        # Valid content
        valid_content = "This is a valid claim content with sufficient length"
        claim = Claim(
            id="c0000001",
            content=valid_content,
            confidence=0.5
        )
        assert claim.content == valid_content
        
        # Too short content
        with pytest.raises(ValueError, match="ensure this value has at least 10 characters"):
            Claim(
                id="c0000001",
                content="Short",
                confidence=0.5
            )
        
        # Too long content
        long_content = "x" * 5001
        with pytest.raises(ValueError, match="ensure this value has at most 5000 characters"):
            Claim(
                id="c0000001",
                content=long_content,
                confidence=0.5
            )
    
    def test_claim_confidence_validation(self):
        """Test claim confidence validation"""
        # Valid confidence values
        valid_confidences = [0.0, 0.5, 0.8, 1.0]
        for confidence in valid_confidences:
            claim = Claim(
                id="c0000001",
                content="Valid claim content for testing",
                confidence=confidence
            )
            assert claim.confidence == confidence
        
        # Invalid confidence values
        invalid_confidences = [-0.1, 1.1, 2.0]
        for confidence in invalid_confidences:
            with pytest.raises(ValueError, match="ensure this value is less than or equal to 1"):
                Claim(
                    id="c0000001",
                    content="Valid claim content for testing",
                    confidence=confidence
                )
    
    def test_claim_tags_validation(self):
        """Test claim tags validation"""
        # Valid tags
        valid_tags = ["tag1", "tag2", "tag3"]
        claim = Claim(
            id="c0000001",
            content="Valid claim content for testing",
            confidence=0.5,
            tags=valid_tags
        )
        assert claim.tags == valid_tags
        
        # Tags with duplicates (should be deduplicated)
        claim_with_duplicates = Claim(
            id="c0000002",
            content="Valid claim content for testing",
            confidence=0.5,
            tags=["tag1", "tag2", "tag1", "tag3"]
        )
        assert claim_with_duplicates.tags == ["tag1", "tag2", "tag3"]
        
        # Tags with whitespace (should be stripped)
        claim_with_whitespace = Claim(
            id="c0000003",
            content="Valid claim content for testing",
            confidence=0.5,
            tags=[" tag1 ", "tag2", " tag3 "]
        )
        assert claim_with_whitespace.tags == ["tag1", "tag2", "tag3"]
        
        # Invalid tags (empty string)
        with pytest.raises(ValueError, match="Tags must be non-empty strings"):
            Claim(
                id="c0000004",
                content="Valid claim content for testing",
                confidence=0.5,
                tags=["valid_tag", ""]
            )
        
        # Invalid tags (non-string)
        with pytest.raises(ValueError, match="Tags must be non-empty strings"):
            Claim(
                id="c0000005",
                content="Valid claim content for testing",
                confidence=0.5,
                tags=["valid_tag", 123]
            )
    
    def test_claim_updated_timestamp_validation(self):
        """Test updated timestamp validation"""
        created = datetime.utcnow()
        
        # Valid: updated after created
        updated = created + timedelta(hours=1)
        claim = Claim(
            id="c0000001",
            content="Valid claim content for testing",
            confidence=0.5,
            created=created,
            updated=updated
        )
        assert claim.updated == updated
        
        # Invalid: updated before created
        invalid_updated = created - timedelta(hours=1)
        with pytest.raises(ValueError, match="Updated timestamp cannot be before creation timestamp"):
            Claim(
                id="c0000002",
                content="Valid claim content for testing",
                confidence=0.5,
                created=created,
                updated=invalid_updated
            )
    
    def test_claim_to_dict(self):
        """Test claim to_dict method"""
        claim = Claim(
            id="c0000001",
            content="Test claim content",
            confidence=0.8,
            tags=["test"],
            embedding=[0.1, 0.2, 0.3]
        )
        
        claim_dict = claim.to_dict()
        
        assert claim_dict["id"] == "c0000001"
        assert claim_dict["content"] == "Test claim content"
        assert claim_dict["confidence"] == 0.8
        assert claim_dict["tags"] == ["test"]
        assert claim_dict["embedding"] == [0.1, 0.2, 0.3]
        # Should exclude None values by default
        assert "updated" not in claim_dict
    
    def test_claim_created_at_property(self):
        """Test claim created_at property"""
        created = datetime.utcnow()
        claim = Claim(
            id="c0000001",
            content="Test claim content",
            confidence=0.8,
            created=created
        )
        
        assert claim.created_at == created
    
    def test_claim_to_chroma_metadata(self):
        """Test claim to_chroma_metadata method"""
        created = datetime.utcnow()
        updated = datetime.utcnow()
        
        claim = Claim(
            id="c0000001",
            content="Test claim content",
            confidence=0.8,
            created=created,
            updated=updated,
            tags=["test", "validation"],
            state=ClaimState.VALIDATED,
            supported_by=["c0000002"],
            supports=["c0000003"],
            dirty=False
        )
        
        metadata = claim.to_chroma_metadata()
        
        assert metadata["confidence"] == 0.8
        assert metadata["state"] == "Validated"
        assert metadata["supported_by"] == "c0000002"
        assert metadata["supports"] == "c0000003"
        assert metadata["type"] == "concept"  # Default type
        assert metadata["tags"] == "test,validation"
        assert metadata["created"] == created.isoformat()
        assert metadata["updated"] == updated.isoformat()
        assert metadata["dirty"] is False
    
    def test_claim_from_chroma_result(self):
        """Test claim from_chroma_result class method"""
        created = datetime.utcnow()
        updated = datetime.utcnow()
        
        metadata = {
            "confidence": 0.8,
            "state": "Validated",
            "supported_by": "c0000002",
            "supports": "c0000003",
            "type": "concept,example",
            "tags": "test,validation",
            "created": created.isoformat(),
            "updated": updated.isoformat(),
            "dirty": False
        }
        
        claim = Claim.from_chroma_result(
            id="c0000001",
            content="Test claim content",
            metadata=metadata
        )
        
        assert claim.id == "c0000001"
        assert claim.content == "Test claim content"
        assert claim.confidence == 0.8
        assert claim.state == ClaimState.VALIDATED
        assert claim.supported_by == ["c0000002"]
        assert claim.supports == ["c0000003"]
        assert claim.type == [ClaimType.CONCEPT, ClaimType.EXAMPLE]
        assert claim.tags == ["test", "validation"]
        assert claim.created == created
        assert claim.updated == updated
        assert claim.dirty is False
    
    def test_claim_mark_dirty(self):
        """Test claim mark_dirty method"""
        claim = Claim(
            id="c0000001",
            content="Test claim content",
            confidence=0.8,
            dirty=False
        )
        
        original_updated = claim.updated
        claim.mark_dirty()
        
        assert claim.dirty is True
        assert claim.updated > original_updated
    
    def test_claim_mark_clean(self):
        """Test claim mark_clean method"""
        claim = Claim(
            id="c0000001",
            content="Test claim content",
            confidence=0.8,
            dirty=True
        )
        
        original_updated = claim.updated
        claim.mark_clean()
        
        assert claim.dirty is False
        assert claim.updated > original_updated
    
    def test_claim_update_confidence(self):
        """Test claim update_confidence method"""
        claim = Claim(
            id="c0000001",
            content="Test claim content",
            confidence=0.5,
            dirty=False
        )
        
        # Valid confidence update
        claim.update_confidence(0.9)
        assert claim.confidence == 0.9
        assert claim.dirty is True
        
        # Invalid confidence update
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            claim.update_confidence(1.5)
    
    def test_claim_add_support(self):
        """Test claim add_support method"""
        claim = Claim(
            id="c0000001",
            content="Test claim content",
            confidence=0.8,
            dirty=False
        )
        
        # Add supporting claim
        claim.add_support("c0000002")
        assert "c0000002" in claim.supported_by
        assert claim.dirty is True
        
        # Add duplicate (should not be added)
        original_dirty = claim.dirty
        claim.add_support("c0000002")
        assert claim.supported_by.count("c0000002") == 1
        assert claim.dirty == original_dirty
        
        # Add empty ID (should not be added)
        original_supports = claim.supported_by.copy()
        claim.add_support("")
        assert claim.supported_by == original_supports
    
    def test_claim_add_supports(self):
        """Test claim add_supports method"""
        claim = Claim(
            id="c0000001",
            content="Test claim content",
            confidence=0.8,
            dirty=False
        )
        
        # Add supported claim
        claim.add_supports("c0000002")
        assert "c0000002" in claim.supports
        assert claim.dirty is True
        
        # Add duplicate (should not be added)
        original_dirty = claim.dirty
        claim.add_supports("c0000002")
        assert claim.supports.count("c0000002") == 1
        assert claim.dirty == original_dirty
        
        # Add empty ID (should not be added)
        original_supports = claim.supports.copy()
        claim.add_supports("")
        assert claim.supports == original_supports
    
    def test_claim_repr(self):
        """Test claim __repr__ method"""
        claim = Claim(
            id="c0000001",
            content="Test claim content",
            confidence=0.8,
            state=ClaimState.VALIDATED
        )
        
        repr_str = repr(claim)
        assert "Claim(id=c0000001" in repr_str
        assert "confidence=0.80" in repr_str
        assert "state=Validated" in repr_str


class TestRelationshipModel:
    """Test Relationship model"""
    
    def test_relationship_creation(self):
        """Test creating a valid relationship"""
        created = datetime.utcnow()
        
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002",
            created=created
        )
        
        assert relationship.supporter_id == "c0000001"
        assert relationship.supported_id == "c0000002"
        assert relationship.created == created
    
    def test_relationship_default_values(self):
        """Test relationship creation with default values"""
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002"
        )
        
        assert relationship.supporter_id == "c0000001"
        assert relationship.supported_id == "c0000002"
        assert isinstance(relationship.created, datetime)
    
    def test_relationship_created_at_property(self):
        """Test relationship created_at property"""
        created = datetime.utcnow()
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002",
            created=created
        )
        
        assert relationship.created_at == created
    
    def test_relationship_repr(self):
        """Test relationship __repr__ method"""
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002"
        )
        
        repr_str = repr(relationship)
        assert "Relationship(c0000001 supports c0000002)" in repr_str


class TestClaimFilterModel:
    """Test ClaimFilter model"""
    
    def test_claim_filter_creation(self):
        """Test creating a claim filter with all fields"""
        created_after = datetime.utcnow() - timedelta(days=1)
        created_before = datetime.utcnow()
        
        filter_obj = ClaimFilter(
            tags=["test", "validation"],
            confidence_min=0.3,
            confidence_max=0.8,
            dirty_only=True,
            content_contains="test content",
            limit=50,
            offset=10,
            created_after=created_after,
            created_before=created_before,
            states=[ClaimState.EXPLORE, ClaimState.VALIDATED],
            types=[ClaimType.CONCEPT, ClaimType.EXAMPLE]
        )
        
        assert filter_obj.tags == ["test", "validation"]
        assert filter_obj.confidence_min == 0.3
        assert filter_obj.confidence_max == 0.8
        assert filter_obj.dirty_only is True
        assert filter_obj.content_contains == "test content"
        assert filter_obj.limit == 50
        assert filter_obj.offset == 10
        assert filter_obj.created_after == created_after
        assert filter_obj.created_before == created_before
        assert filter_obj.states == [ClaimState.EXPLORE, ClaimState.VALIDATED]
        assert filter_obj.types == [ClaimType.CONCEPT, ClaimType.EXAMPLE]
    
    def test_claim_filter_defaults(self):
        """Test claim filter default values"""
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
    
    def test_claim_filter_limit_validation(self):
        """Test claim filter limit validation"""
        # Valid limit
        filter_obj = ClaimFilter(limit=50)
        assert filter_obj.limit == 50
        
        # None limit (should default to 100)
        filter_obj = ClaimFilter(limit=None)
        assert filter_obj.limit == 100
        
        # Invalid limit (too small)
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 1"):
            ClaimFilter(limit=0)
        
        # Invalid limit (too large)
        with pytest.raises(ValueError, match="ensure this value is less than or equal to 1000"):
            ClaimFilter(limit=1001)
    
    def test_claim_filter_offset_validation(self):
        """Test claim filter offset validation"""
        # Valid offset
        filter_obj = ClaimFilter(offset=10)
        assert filter_obj.offset == 10
        
        # None offset (should default to 0)
        filter_obj = ClaimFilter(offset=None)
        assert filter_obj.offset == 0
        
        # Invalid offset (negative)
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 0"):
            ClaimFilter(offset=-1)
    
    def test_claim_filter_confidence_range_validation(self):
        """Test claim filter confidence range validation"""
        # Valid range
        filter_obj = ClaimFilter(confidence_min=0.3, confidence_max=0.8)
        assert filter_obj.confidence_min == 0.3
        assert filter_obj.confidence_max == 0.8
        
        # Invalid range (max < min)
        with pytest.raises(ValueError, match="confidence_max must be >= confidence_min"):
            ClaimFilter(confidence_min=0.8, confidence_max=0.3)


class TestDataConfigModel:
    """Test DataConfig model"""
    
    def test_data_config_creation(self):
        """Test creating a data config with all fields"""
        config = DataConfig(
            sqlite_path="/custom/path.db",
            chroma_path="/custom/chroma",
            chroma_collection="custom_collection",
            embedding_model="custom-model",
            embedding_dim=768,
            cache_size=500,
            cache_ttl=600,
            batch_size=50,
            max_connections=5,
            query_timeout=60,
            use_chroma=False,
            use_embeddings=False,
            auto_sync=False
        )
        
        assert config.sqlite_path == "/custom/path.db"
        assert config.chroma_path == "/custom/chroma"
        assert config.chroma_collection == "custom_collection"
        assert config.embedding_model == "custom-model"
        assert config.embedding_dim == 768
        assert config.cache_size == 500
        assert config.cache_ttl == 600
        assert config.batch_size == 50
        assert config.max_connections == 5
        assert config.query_timeout == 60
        assert config.use_chroma is False
        assert config.use_embeddings is False
        assert config.auto_sync is False
    
    def test_data_config_defaults(self):
        """Test data config default values"""
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
        assert config.use_chroma is True
        assert config.use_embeddings is True
        assert config.auto_sync is True


class TestQueryResultModel:
    """Test QueryResult model"""
    
    def test_query_result_creation(self):
        """Test creating a query result"""
        items = [
            {"id": "c0000001", "content": "Test claim 1"},
            {"id": "c0000002", "content": "Test claim 2"}
        ]
        
        result = QueryResult(
            items=items,
            total_count=100,
            query_time=0.05,
            has_more=True
        )
        
        assert result.items == items
        assert result.total_count == 100
        assert result.query_time == 0.05
        assert result.has_more is True
        assert len(result) == 2
    
    def test_query_result_defaults(self):
        """Test query result default values"""
        result = QueryResult()
        
        assert result.items == []
        assert result.total_count is None
        assert result.query_time is None
        assert result.has_more is False
        assert len(result) == 0
    
    def test_query_result_len(self):
        """Test query result __len__ method"""
        items = [{"id": f"c000000{i}"} for i in range(5)]
        result = QueryResult(items=items)
        
        assert len(result) == 5


class TestProcessingStatsModel:
    """Test ProcessingStats model"""
    
    def test_processing_stats_creation(self):
        """Test creating processing stats"""
        start_time = datetime.utcnow()
        end_time = datetime.utcnow() + timedelta(seconds=10)
        
        stats = ProcessingStats(
            operation="test_operation",
            start_time=start_time,
            end_time=end_time,
            items_processed=100,
            items_succeeded=95,
            items_failed=5
        )
        
        assert stats.operation == "test_operation"
        assert stats.start_time == start_time
        assert stats.end_time == end_time
        assert stats.items_processed == 100
        assert stats.items_succeeded == 95
        assert stats.items_failed == 5
    
    def test_processing_stats_duration(self):
        """Test processing stats duration property"""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(seconds=10)
        
        stats = ProcessingStats(
            operation="test_operation",
            start_time=start_time,
            end_time=end_time
        )
        
        assert stats.duration == 10.0
    
    def test_processing_stats_duration_no_end_time(self):
        """Test processing stats duration property with no end time"""
        start_time = datetime.utcnow()
        
        stats = ProcessingStats(
            operation="test_operation",
            start_time=start_time
        )
        
        assert stats.duration is None
    
    def test_processing_stats_success_rate(self):
        """Test processing stats success rate property"""
        stats = ProcessingStats(
            operation="test_operation",
            start_time=datetime.utcnow(),
            items_processed=100,
            items_succeeded=95,
            items_failed=5
        )
        
        assert stats.success_rate == 95.0
    
    def test_processing_stats_success_rate_no_items(self):
        """Test processing stats success rate property with no items"""
        stats = ProcessingStats(
            operation="test_operation",
            start_time=datetime.utcnow()
        )
        
        assert stats.success_rate == 0.0
    
    def test_processing_stats_items_per_second(self):
        """Test processing stats items per second property"""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(seconds=10)
        
        stats = ProcessingStats(
            operation="test_operation",
            start_time=start_time,
            end_time=end_time,
            items_processed=100
        )
        
        assert stats.items_per_second == 10.0
    
    def test_processing_stats_items_per_second_no_duration(self):
        """Test processing stats items per second property with no duration"""
        stats = ProcessingStats(
            operation="test_operation",
            start_time=datetime.utcnow(),
            items_processed=100
        )
        
        assert stats.items_per_second is None
    
    def test_processing_stats_start(self):
        """Test processing stats start method"""
        stats = ProcessingStats(operation="test_operation")
        
        before_start = datetime.utcnow()
        stats.start()
        after_start = datetime.utcnow()
        
        assert before_start <= stats.start_time <= after_start
    
    def test_processing_stats_end(self):
        """Test processing stats end method"""
        stats = ProcessingStats(operation="test_operation")
        
        before_end = datetime.utcnow()
        stats.end()
        after_end = datetime.utcnow()
        
        assert before_end <= stats.end_time <= after_end
    
    def test_processing_stats_add_success(self):
        """Test processing stats add_success method"""
        stats = ProcessingStats(
            operation="test_operation",
            start_time=datetime.utcnow()
        )
        
        stats.add_success(5)
        assert stats.items_processed == 5
        assert stats.items_succeeded == 5
        assert stats.items_failed == 0
        
        stats.add_success(3)
        assert stats.items_processed == 8
        assert stats.items_succeeded == 8
        assert stats.items_failed == 0
    
    def test_processing_stats_add_failure(self):
        """Test processing stats add_failure method"""
        stats = ProcessingStats(
            operation="test_operation",
            start_time=datetime.utcnow()
        )
        
        stats.add_failure(2)
        assert stats.items_processed == 2
        assert stats.items_succeeded == 0
        assert stats.items_failed == 2
        
        stats.add_failure(1)
        assert stats.items_processed == 3
        assert stats.items_succeeded == 0
        assert stats.items_failed == 3


class TestProcessingResultModel:
    """Test ProcessingResult model"""
    
    def test_processing_result_creation(self):
        """Test creating a processing result"""
        result = ProcessingResult(
            claim_id="c0000001",
            success=True,
            message="Processing completed successfully",
            updated_confidence=0.9,
            processing_time=0.05,
            metadata={"test": "data"}
        )
        
        assert result.claim_id == "c0000001"
        assert result.success is True
        assert result.message == "Processing completed successfully"
        assert result.updated_confidence == 0.9
        assert result.processing_time == 0.05
        assert result.metadata == {"test": "data"}
    
    def test_processing_result_defaults(self):
        """Test processing result default values"""
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
    
    def test_processing_result_confidence_validation(self):
        """Test processing result confidence validation"""
        # Valid confidence
        result = ProcessingResult(
            claim_id="c0000001",
            success=True,
            updated_confidence=0.8
        )
        assert result.updated_confidence == 0.8
        
        # Invalid confidence (too high)
        with pytest.raises(ValueError, match="ensure this value is less than or equal to 1"):
            ProcessingResult(
                claim_id="c0000001",
                success=True,
                updated_confidence=1.5
            )
        
        # Invalid confidence (too low)
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 0"):
            ProcessingResult(
                claim_id="c0000001",
                success=True,
                updated_confidence=-0.1
            )
    
    def test_processing_result_processing_time_validation(self):
        """Test processing result processing time validation"""
        # Valid processing time
        result = ProcessingResult(
            claim_id="c0000001",
            success=True,
            processing_time=0.05
        )
        assert result.processing_time == 0.05
        
        # Invalid processing time (negative)
        with pytest.raises(ValueError, match="ensure this value is greater than or equal to 0"):
            ProcessingResult(
                claim_id="c0000001",
                success=True,
                processing_time=-0.1
            )


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_validate_claim_id(self):
        """Test validate_claim_id function"""
        # Valid IDs
        assert validate_claim_id("c0000001") is True
        assert validate_claim_id("c1234567") is True
        assert validate_claim_id("c9999999") is True
        
        # Invalid IDs
        assert validate_claim_id("C0000001") is False  # Upper case
        assert validate_claim_id("c000001") is False   # Too short
        assert validate_claim_id("c00000001") is False  # Too long
        assert validate_claim_id("a0000001") is False   # Wrong letter
        assert validate_claim_id("c00000a1") is False   # Contains letter
        assert validate_claim_id("c-000001") is False   # Contains dash
        assert validate_claim_id("00000001") is False   # Missing prefix
    
    def test_validate_confidence(self):
        """Test validate_confidence function"""
        # Valid values
        assert validate_confidence(0.0) is True
        assert validate_confidence(0.5) is True
        assert validate_confidence(1.0) is True
        
        # Invalid values
        assert validate_confidence(-0.1) is False
        assert validate_confidence(1.1) is False
        assert validate_confidence(2.0) is False
    
    def test_generate_claim_id(self):
        """Test generate_claim_id function"""
        # Test basic generation
        claim_id = generate_claim_id()
        assert len(claim_id) == 8
        assert claim_id.startswith("c")
        assert claim_id[1:].isdigit()
        
        # Test with counter
        claim_id_1 = generate_claim_id(1)
        assert claim_id_1 == "c0000001"
        
        claim_id_42 = generate_claim_id(42)
        assert claim_id_42 == "c0000042"
        
        claim_id_9999999 = generate_claim_id(9999999)
        assert claim_id_9999999 == "c9999999"


class TestClaimRepository:
    """Test ClaimRepository abstract class"""
    
    def test_claim_repository_is_abstract(self):
        """Test that ClaimRepository cannot be instantiated directly"""
        with pytest.raises(TypeError):
            ClaimRepository()
    
    def test_claim_repository_abstract_methods(self):
        """Test that ClaimRepository has all required abstract methods"""
        abstract_methods = ClaimRepository.__abstractmethods__
        expected_methods = {'create', 'get_by_id', 'update', 'delete', 'search', 'list_by_state'}
        
        assert abstract_methods == expected_methods


class TestDataManagerClaimRepository:
    """Test DataManagerClaimRepository implementation"""
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock data manager for testing"""
        return MockDataManager()
    
    @pytest.fixture
    def repository(self, mock_data_manager):
        """Create a repository instance with mock data manager"""
        return DataManagerClaimRepository(mock_data_manager)
    
    @pytest.mark.asyncio
    async def test_create_claim(self, repository, mock_data_manager):
        """Test creating a claim"""
        # Test creation
        claim_data = {
            "content": "Test claim content",
            "confidence": 0.8,
            "tags": ["test"],
            "state": "Explore"
        }
        
        result = await repository.create(claim_data)
        
        assert result is not None
        assert result.content == "Test claim content"
        assert result.confidence == 0.8
        assert result.tags == ["test"]
        assert result.state == CoreClaimState.EXPLORE
    
    @pytest.mark.asyncio
    async def test_create_claim_with_claim_type_backward_compatibility(self, repository, mock_data_manager):
        """Test creating a claim with claim_type for backward compatibility"""
        # Test creation with claim_type
        claim_data = {
            "content": "Test claim content",
            "confidence": 0.8,
            "claim_type": "concept"
        }
        
        result = await repository.create(claim_data)
        
        assert result is not None
        assert result.content == "Test claim content"
        assert result.confidence == 0.8
        # Should have claim_type added to tags
        assert "concept" in result.tags
        assert result.state == CoreClaimState.EXPLORE
    
    @pytest.mark.asyncio
    async def test_get_by_id_success(self, repository, mock_data_manager):
        """Test getting a claim by ID successfully"""
        # First create a claim
        claim = await repository.create({
            "content": "Test claim content",
            "confidence": 0.8
        })
        
        # Test retrieval
        result = await repository.get_by_id(claim.id)
        
        assert result is not None
        assert result.id == claim.id
        assert result.content == claim.content
    
    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repository, mock_data_manager):
        """Test getting a claim by ID when not found"""
        # Test retrieval of non-existent claim
        result = await repository.get_by_id("c9999999")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_claim(self, repository, mock_data_manager):
        """Test updating a claim"""
        # First create a claim
        claim = await repository.create({
            "content": "Test claim content",
            "confidence": 0.8
        })
        
        # Test update
        updates = {"confidence": 0.9, "content": "Updated content"}
        result = await repository.update(claim.id, updates)
        
        assert result is not None
        assert result.confidence == 0.9
        assert result.content == "Updated content"
    
    @pytest.mark.asyncio
    async def test_delete_claim_success(self, repository, mock_data_manager):
        """Test deleting a claim successfully"""
        # First create a claim
        claim = await repository.create({
            "content": "Test claim content",
            "confidence": 0.8
        })
        
        # Test deletion
        result = await repository.delete(claim.id)
        
        assert result is True
        
        # Verify claim is gone
        found = await repository.get_by_id(claim.id)
        assert found is None
    
    @pytest.mark.asyncio
    async def test_delete_claim_not_found(self, repository, mock_data_manager):
        """Test deleting a claim that doesn't exist"""
        # Test deletion of non-existent claim
        result = await repository.delete("c9999999")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_search_claims(self, repository, mock_data_manager):
        """Test searching claims"""
        # Create some claims
        await repository.create({"content": "Test claim about machine learning", "confidence": 0.8})
        await repository.create({"content": "Test claim about data science", "confidence": 0.7})
        await repository.create({"content": "Test claim about AI", "confidence": 0.9})
        
        # Test search
        results = await repository.search("machine learning", limit=5)
        
        assert len(results) == 1
        assert "machine learning" in results[0].content.lower()
    
    @pytest.mark.asyncio
    async def test_list_by_state(self, repository, mock_data_manager):
        """Test listing claims by state"""
        # Create claims with different states
        claim1 = await repository.create({"content": "Test claim 1", "confidence": 0.8})
        claim2 = await repository.create({"content": "Test claim 2", "confidence": 0.7})
        
        # Update one claim to validated state
        await repository.update(claim2.id, {"state": "Validated"})
        
        # Test listing by state
        results = await repository.list_by_state(CoreClaimState.VALIDATED)
        
        assert len(results) == 1
        assert results[0].id == claim2.id
        assert results[0].state == CoreClaimState.VALIDATED


class TestRepositoryFactory:
    """Test RepositoryFactory"""
    
    def test_create_claim_repository(self):
        """Test creating a claim repository"""
        mock_data_manager = MockDataManager()
        
        repository = RepositoryFactory.create_claim_repository(mock_data_manager)
        
        assert isinstance(repository, DataManagerClaimRepository)
        assert repository.data_manager == mock_data_manager


class TestGetDataManager:
    """Test get_data_manager function"""
    
    @patch('src.data.repositories._get_data_manager')
    def test_get_data_manager_default(self, mock_get_dm):
        """Test getting data manager with default parameters"""
        mock_dm = MockDataManager()
        mock_get_dm.return_value = mock_dm
        
        result = get_data_manager()
        
        mock_get_dm.assert_called_once_with(False)
        assert result == mock_dm
    
    @patch('src.data.repositories._get_data_manager')
    def test_get_data_manager_with_mock_embeddings(self, mock_get_dm):
        """Test getting data manager with mock embeddings"""
        mock_dm = MockDataManager()
        mock_get_dm.return_value = mock_dm
        
        result = get_data_manager(use_mock_embeddings=True)
        
        mock_get_dm.assert_called_once_with(True)
        assert result == mock_dm


class TestEdgeCasesAndErrorConditions:
    """Test edge cases and error conditions"""
    
    def test_claim_with_all_optional_fields_none(self):
        """Test claim with all optional fields set to None"""
        claim = Claim(
            id="c0000001",
            content="Test claim content with sufficient length",
            confidence=0.8,
            updated=None,
            embedding=None,
            dirty_reason=None,
            dirty_timestamp=None
        )
        
        assert claim.updated is None
        assert claim.embedding is None
        assert claim.dirty_reason is None
        assert claim.dirty_timestamp is None
    
    def test_claim_with_empty_lists(self):
        """Test claim with empty list fields"""
        claim = Claim(
            id="c0000001",
            content="Test claim content with sufficient length",
            confidence=0.8,
            tags=[],
            type=[],
            supported_by=[],
            supports=[]
        )
        
        assert claim.tags == []
        assert claim.type == []
        assert claim.supported_by == []
        assert claim.supports == []
    
    def test_claim_with_boundary_values(self):
        """Test claim with boundary values"""
        # Minimum content length
        claim_min = Claim(
            id="c0000001",
            content="x" * 10,  # Exactly 10 characters
            confidence=0.0  # Minimum confidence
        )
        assert claim_min.content == "x" * 10
        assert claim_min.confidence == 0.0
        
        # Maximum content length
        claim_max = Claim(
            id="c0000002",
            content="x" * 5000,  # Exactly 5000 characters
            confidence=1.0  # Maximum confidence
        )
        assert claim_max.content == "x" * 5000
        assert claim_max.confidence == 1.0
    
    def test_claim_filter_with_boundary_values(self):
        """Test claim filter with boundary values"""
        # Minimum values
        filter_min = ClaimFilter(
            limit=1,  # Minimum limit
            offset=0   # Minimum offset
        )
        assert filter_min.limit == 1
        assert filter_min.offset == 0
        
        # Maximum values
        filter_max = ClaimFilter(
            limit=1000,  # Maximum limit
            confidence_min=0.0,
            confidence_max=1.0
        )
        assert filter_max.limit == 1000
        assert filter_max.confidence_min == 0.0
        assert filter_max.confidence_max == 1.0
    
    def test_processing_stats_with_zero_values(self):
        """Test processing stats with zero values"""
        stats = ProcessingStats(
            operation="test_operation",
            start_time=datetime.utcnow(),
            items_processed=0,
            items_succeeded=0,
            items_failed=0
        )
        
        assert stats.items_processed == 0
        assert stats.items_succeeded == 0
        assert stats.items_failed == 0
        assert stats.success_rate == 0.0
    
    def test_relationship_with_same_ids(self):
        """Test relationship with same supporter and supported IDs"""
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000001"
        )
        
        assert relationship.supporter_id == "c0000001"
        assert relationship.supported_id == "c0000001"
    
    def test_claim_with_unicode_content(self):
        """Test claim with unicode content"""
        unicode_content = "Test claim with unicode: 🚀 🌟 ñáéíóú"
        claim = Claim(
            id="c0000001",
            content=unicode_content,
            confidence=0.8
        )
        
        assert claim.content == unicode_content
    
    def test_claim_with_special_characters_in_tags(self):
        """Test claim with special characters in tags"""
        special_tags = ["test-tag", "tag_with_underscore", "tag.with.dots"]
        claim = Claim(
            id="c0000001",
            content="Test claim content",
            confidence=0.8,
            tags=special_tags
        )
        
        assert claim.tags == special_tags


if __name__ == "__main__":
    pytest.main([__file__, "-v"])