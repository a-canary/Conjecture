#!/usr/bin/env python3
"""
Simple comprehensive tests for data layer components.
Tests src/data/models.py and src/data/repositories.py by importing them directly.
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

# Mock all problematic imports BEFORE any imports happen
sys.modules['chromadb'] = Mock()
sys.modules['chromadb.api'] = Mock()
sys.modules['chromadb.api.client'] = Mock()
sys.modules['chromadb.api.models'] = Mock()
sys.modules['chromadb.utils'] = Mock()
sys.modules['chromadb.utils.embedding_functions'] = Mock()
sys.modules['sentence_transformers'] = Mock()
sys.modules['transformers'] = Mock()
sys.modules['torch'] = Mock()
sys.modules['tensorflow'] = Mock()

# Import models directly from the file, not through __init__.py
import importlib.util
import importlib.machinery

# Load models module directly
spec = importlib.util.spec_from_file_location("models", os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'models.py'))
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)

# Load repositories module directly
spec_repo = importlib.util.spec_from_file_location("repositories", os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'repositories.py'))
repositories = importlib.util.module_from_spec(spec_repo)
spec_repo.loader.exec_module(repositories)

# Import core models for testing
spec_core = importlib.util.spec_from_file_location("core_models", os.path.join(os.path.dirname(__file__), '..', 'src', 'core', 'models.py'))
core_models = importlib.util.module_from_spec(spec_core)
spec_core.loader.exec_module(core_models)

# Get the classes we need
DataLayerError = models.DataLayerError
ClaimNotFoundError = models.ClaimNotFoundError
InvalidClaimError = models.InvalidClaimError
RelationshipError = models.RelationshipError
EmbeddingError = models.EmbeddingError
ClaimState = models.ClaimState
ClaimType = models.ClaimType
Claim = models.Claim
Relationship = models.Relationship
ClaimFilter = models.ClaimFilter
DataConfig = models.DataConfig
QueryResult = models.QueryResult
ProcessingStats = models.ProcessingStats
ProcessingResult = models.ProcessingResult
validate_claim_id = models.validate_claim_id
validate_confidence = models.validate_confidence
generate_claim_id = models.generate_claim_id

# Core models
CoreClaim = core_models.Claim
CoreClaimState = core_models.ClaimState
CoreClaimType = core_models.ClaimType
ClaimScope = core_models.ClaimScope

# Repository classes
ClaimRepository = repositories.ClaimRepository
DataManagerClaimRepository = repositories.DataManagerClaimRepository
RepositoryFactory = repositories.RepositoryFactory
get_data_manager = repositories.get_data_manager

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

# Mock the get_data_manager function
def mock_get_data_manager(use_mock_embeddings=False):
    return MockDataManager()

# Patch the get_data_manager function in repositories module
repositories._get_data_manager = mock_get_data_manager


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
    
    def test_get_data_manager_default(self):
        """Test getting data manager with default parameters"""
        mock_dm = MockDataManager()
        
        # Patch the _get_data_manager function temporarily
        original_get_dm = repositories._get_data_manager
        repositories._get_data_manager = lambda use_mock_embeddings=False: mock_dm
        
        try:
            result = get_data_manager()
            assert result == mock_dm
        finally:
            # Restore original function
            repositories._get_data_manager = original_get_dm
    
    def test_get_data_manager_with_mock_embeddings(self):
        """Test getting data manager with mock embeddings"""
        mock_dm = MockDataManager()
        
        # Patch the _get_data_manager function temporarily
        original_get_dm = repositories._get_data_manager
        repositories._get_data_manager = lambda use_mock_embeddings=True: mock_dm
        
        try:
            result = get_data_manager(use_mock_embeddings=True)
            assert result == mock_dm
        finally:
            # Restore original function
            repositories._get_data_manager = original_get_dm


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