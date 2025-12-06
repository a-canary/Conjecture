#!/usr/bin/env python3
"""
Comprehensive Tests for Data Layer Repositories
Tests claim repository operations, database interactions with mocking, and error handling
"""

import pytest
import sys
import os
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.models import Claim, ClaimState, ClaimType, ClaimScope
from src.data.repositories import (
    ClaimRepository, DataManagerClaimRepository, RepositoryFactory,
    get_data_manager
)
from src.data.data_manager import DataManager


class TestClaimRepository:
    """Test abstract ClaimRepository class"""
    
    def test_claim_repository_is_abstract(self):
        """Test that ClaimRepository cannot be instantiated directly"""
        with pytest.raises(TypeError):
            ClaimRepository()


class TestDataManagerClaimRepository:
    """Test DataManagerClaimRepository implementation"""
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create a mock DataManager"""
        mock_dm = AsyncMock(spec=DataManager)
        return mock_dm
    
    @pytest.fixture
    def repository(self, mock_data_manager):
        """Create repository with mock data manager"""
        return DataManagerClaimRepository(mock_data_manager)
    
    @pytest.fixture
    def sample_claim_data(self):
        """Sample claim data for testing"""
        return {
            'content': 'Test claim content',
            'confidence': 0.8,
            'tags': ['test', 'example'],
            'state': ClaimState.EXPLORE.value,
            'claim_type': 'concept'
        }
    
    @pytest.fixture
    def sample_claim(self):
        """Sample claim object for testing"""
        return Claim(
            id="c1234567",
            content="Test claim content",
            confidence=0.8,
            tags=["test", "example"],
            state=ClaimState.EXPLORE
        )
    
    @pytest.mark.asyncio
    async def test_create_claim_success(self, repository, mock_data_manager, sample_claim_data, sample_claim):
        """Test successful claim creation"""
        # Mock the data manager response
        mock_data_manager.create_claim.return_value = sample_claim
        
        # Call the repository method
        result = await repository.create(sample_claim_data)
        
        # Verify the result
        assert result == sample_claim
        assert isinstance(result, Claim)
        
        # Verify the data manager was called correctly
        mock_data_manager.create_claim.assert_called_once_with(
            content=sample_claim_data['content'],
            confidence=sample_claim_data.get('confidence', 0.0),
            tags=['test', 'example', 'concept'],  # Should include both tags and claim_type
            state=ClaimState.EXPLORE
        )
    
    @pytest.mark.asyncio
    async def test_create_claim_with_claim_type_mapping(self, repository, mock_data_manager, sample_claim):
        """Test claim creation with claim_type to tag mapping"""
        claim_data = {
            'content': 'Test claim content',
            'confidence': 0.8,
            'tags': ['existing'],
            'claim_type': 'concept'
        }
        
        mock_data_manager.create_claim.return_value = sample_claim
        
        result = await repository.create(claim_data)
        
        # Verify claim_type was added to tags
        expected_tags = ['existing', 'concept', 'concept']  # existing tags + claim_type + concept (for backwards compatibility)
        mock_data_manager.create_claim.assert_called_once_with(
            content=claim_data['content'],
            confidence=claim_data.get('confidence', 0.0),
            tags=expected_tags,
            state=ClaimState.EXPLORE
        )
    
    @pytest.mark.asyncio
    async def test_create_claim_without_tags(self, repository, mock_data_manager, sample_claim):
        """Test claim creation without existing tags"""
        claim_data = {
            'content': 'Test claim content',
            'confidence': 0.8
        }
        
        mock_data_manager.create_claim.return_value = sample_claim
        
        result = await repository.create(claim_data)
        
        # Verify empty tags list is used
        mock_data_manager.create_claim.assert_called_once_with(
            content=claim_data['content'],
            confidence=claim_data.get('confidence', 0.0),
            tags=[],
            state=ClaimState.EXPLORE
        )
    
    @pytest.mark.asyncio
    async def test_create_claim_exception(self, repository, mock_data_manager):
        """Test claim creation with exception"""
        claim_data = {
            'content': 'Test claim content',
            'confidence': 0.8
        }
        
        # Mock exception from data manager
        mock_data_manager.create_claim.side_effect = Exception("Database error")
        
        # Should raise the exception
        with pytest.raises(Exception, match="Database error"):
            await repository.create(claim_data)
    
    @pytest.mark.asyncio
    async def test_get_by_id_success(self, repository, mock_data_manager, sample_claim):
        """Test successful claim retrieval by ID"""
        claim_id = "c1234567"
        mock_data_manager.get_claim.return_value = sample_claim
        
        result = await repository.get_by_id(claim_id)
        
        assert result == sample_claim
        mock_data_manager.get_claim.assert_called_once_with(claim_id)
    
    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repository, mock_data_manager):
        """Test claim retrieval when claim not found"""
        claim_id = "c1234567"
        mock_data_manager.get_claim.side_effect = Exception("Claim not found")
        
        result = await repository.get_by_id(claim_id)
        
        assert result is None
        mock_data_manager.get_claim.assert_called_once_with(claim_id)
    
    @pytest.mark.asyncio
    async def test_get_by_id_other_exception(self, repository, mock_data_manager):
        """Test claim retrieval with other exceptions"""
        claim_id = "c1234567"
        mock_data_manager.get_claim.side_effect = Exception("Connection error")
        
        result = await repository.get_by_id(claim_id)
        
        assert result is None
        mock_data_manager.get_claim.assert_called_once_with(claim_id)
    
    @pytest.mark.asyncio
    async def test_update_claim_success(self, repository, mock_data_manager, sample_claim):
        """Test successful claim update"""
        claim_id = "c1234567"
        updates = {
            'confidence': 0.9,
            'state': ClaimState.VALIDATED.value
        }
        
        mock_data_manager.update_claim.return_value = sample_claim
        
        result = await repository.update(claim_id, updates)
        
        assert result == sample_claim
        mock_data_manager.update_claim.assert_called_once_with(claim_id, updates)
    
    @pytest.mark.asyncio
    async def test_update_claim_exception(self, repository, mock_data_manager):
        """Test claim update with exception"""
        claim_id = "c1234567"
        updates = {'confidence': 0.9}
        
        mock_data_manager.update_claim.side_effect = Exception("Update failed")
        
        # Should raise the exception
        with pytest.raises(Exception, match="Update failed"):
            await repository.update(claim_id, updates)
    
    @pytest.mark.asyncio
    async def test_delete_claim_success(self, repository, mock_data_manager):
        """Test successful claim deletion"""
        claim_id = "c1234567"
        
        # Mock successful deletion (no exception)
        mock_data_manager.delete_claim.return_value = None
        
        result = await repository.delete(claim_id)
        
        assert result is True
        mock_data_manager.delete_claim.assert_called_once_with(claim_id)
    
    @pytest.mark.asyncio
    async def test_delete_claim_not_found(self, repository, mock_data_manager):
        """Test claim deletion when claim not found"""
        claim_id = "c1234567"
        
        mock_data_manager.delete_claim.side_effect = Exception("Claim not found")
        
        result = await repository.delete(claim_id)
        
        assert result is False
        mock_data_manager.delete_claim.assert_called_once_with(claim_id)
    
    @pytest.mark.asyncio
    async def test_delete_claim_other_exception(self, repository, mock_data_manager):
        """Test claim deletion with other exceptions"""
        claim_id = "c1234567"
        
        mock_data_manager.delete_claim.side_effect = Exception("Connection error")
        
        result = await repository.delete(claim_id)
        
        assert result is False
        mock_data_manager.delete_claim.assert_called_once_with(claim_id)
    
    @pytest.mark.asyncio
    async def test_search_claims_success(self, repository, mock_data_manager):
        """Test successful claim search"""
        query = "test query"
        limit = 10
        
        # Mock search result
        mock_search_result = Mock()
        mock_search_result.claims = [
            Claim(id="c1111111", content="Test claim 1", confidence=0.8),
            Claim(id="c2222222", content="Test claim 2", confidence=0.9)
        ]
        
        mock_data_manager.search_claims.return_value = mock_search_result
        
        result = await repository.search(query, limit)
        
        assert len(result) == 2
        assert result[0].id == "c1111111"
        assert result[1].id == "c2222222"
        mock_data_manager.search_claims.assert_called_once_with(query, limit=limit)
    
    @pytest.mark.asyncio
    async def test_search_claims_default_limit(self, repository, mock_data_manager):
        """Test claim search with default limit"""
        query = "test query"
        
        mock_search_result = Mock()
        mock_search_result.claims = []
        
        mock_data_manager.search_claims.return_value = mock_search_result
        
        await repository.search(query)
        
        # Should use default limit of 10
        mock_data_manager.search_claims.assert_called_once_with(query, limit=10)
    
    @pytest.mark.asyncio
    async def test_search_claims_exception(self, repository, mock_data_manager):
        """Test claim search with exception"""
        query = "test query"
        
        mock_data_manager.search_claims.side_effect = Exception("Search failed")
        
        # Should raise the exception
        with pytest.raises(Exception, match="Search failed"):
            await repository.search(query)
    
    @pytest.mark.asyncio
    async def test_list_by_state_success(self, repository, mock_data_manager):
        """Test successful claim listing by state"""
        state = ClaimState.VALIDATED
        
        # Mock list result
        mock_list_result = Mock()
        mock_list_result.claims = [
            Claim(id="c1111111", content="Validated claim 1", confidence=0.8, state=ClaimState.VALIDATED),
            Claim(id="c2222222", content="Validated claim 2", confidence=0.9, state=ClaimState.VALIDATED)
        ]
        
        mock_data_manager.list_claims.return_value = mock_list_result
        
        result = await repository.list_by_state(state)
        
        assert len(result) == 2
        assert all(claim.state == ClaimState.VALIDATED for claim in result)
        mock_data_manager.list_claims.assert_called_once_with({"state": state})
    
    @pytest.mark.asyncio
    async def test_list_by_state_exception(self, repository, mock_data_manager):
        """Test claim listing by state with exception"""
        state = ClaimState.EXPLORE
        
        mock_data_manager.list_claims.side_effect = Exception("List failed")
        
        # Should raise the exception
        with pytest.raises(Exception, match="List failed"):
            await repository.list_by_state(state)


class TestRepositoryFactory:
    """Test RepositoryFactory class"""
    
    def test_create_claim_repository(self):
        """Test creating a claim repository"""
        mock_data_manager = Mock(spec=DataManager)
        
        repository = RepositoryFactory.create_claim_repository(mock_data_manager)
        
        assert isinstance(repository, DataManagerClaimRepository)
        assert repository.data_manager == mock_data_manager


class TestGetDataManager:
    """Test get_data_manager function"""
    
    @patch('src.data.repositories.get_data_manager')
    def test_get_data_manager_default(self, mock_get_data_manager):
        """Test get_data_manager with default parameters"""
        mock_get_data_manager.return_value = Mock(spec=DataManager)
        
        result = get_data_manager()
        
        assert isinstance(result, DataManager)
        mock_get_data_manager.assert_called_once_with(False)
    
    @patch('src.data.repositories.get_data_manager')
    def test_get_data_manager_with_mock_embeddings(self, mock_get_data_manager):
        """Test get_data_manager with mock embeddings"""
        mock_get_data_manager.return_value = Mock(spec=DataManager)
        
        result = get_data_manager(use_mock_embeddings=True)
        
        assert isinstance(result, DataManager)
        mock_get_data_manager.assert_called_once_with(True)


class TestRepositoryIntegration:
    """Integration tests for repository operations"""
    
    @pytest.fixture
    def mock_data_manager(self):
        """Create a comprehensive mock DataManager"""
        mock_dm = AsyncMock(spec=DataManager)
        
        # Setup default return values
        mock_dm.create_claim.return_value = Claim(
            id="c1234567",
            content="Test claim",
            confidence=0.8,
            state=ClaimState.EXPLORE
        )
        
        mock_dm.get_claim.return_value = Claim(
            id="c1234567",
            content="Test claim",
            confidence=0.8,
            state=ClaimState.EXPLORE
        )
        
        mock_dm.update_claim.return_value = Claim(
            id="c1234567",
            content="Updated claim",
            confidence=0.9,
            state=ClaimState.VALIDATED
        )
        
        mock_search_result = Mock()
        mock_search_result.claims = []
        mock_dm.search_claims.return_value = mock_search_result
        
        mock_list_result = Mock()
        mock_list_result.claims = []
        mock_dm.list_claims.return_value = mock_list_result
        
        return mock_dm
    
    @pytest.fixture
    def repository(self, mock_data_manager):
        """Create repository with comprehensive mock data manager"""
        return DataManagerClaimRepository(mock_data_manager)
    
    @pytest.mark.asyncio
    async def test_full_claim_lifecycle(self, repository, mock_data_manager):
        """Test complete claim lifecycle: create -> read -> update -> delete"""
        # Create claim
        claim_data = {
            'content': 'Lifecycle test claim',
            'confidence': 0.8,
            'tags': ['test', 'lifecycle']
        }
        
        created_claim = await repository.create(claim_data)
        assert created_claim.content == 'Lifecycle test claim'
        
        # Read claim
        retrieved_claim = await repository.get_by_id(created_claim.id)
        assert retrieved_claim == created_claim
        
        # Update claim
        updates = {
            'confidence': 0.9,
            'state': ClaimState.VALIDATED.value
        }
        updated_claim = await repository.update(created_claim.id, updates)
        assert updated_claim.confidence == 0.9
        assert updated_claim.state == ClaimState.VALIDATED
        
        # Delete claim
        delete_result = await repository.delete(created_claim.id)
        assert delete_result is True
        
        # Verify all data manager calls
        mock_data_manager.create_claim.assert_called_once()
        mock_data_manager.get_claim.assert_called_once()
        mock_data_manager.update_claim.assert_called_once()
        mock_data_manager.delete_claim.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_bulk_operations(self, repository, mock_data_manager):
        """Test bulk operations and search functionality"""
        # Setup mock search results
        search_results = [
            Claim(id="c1111111", content="Search result 1", confidence=0.8),
            Claim(id="c2222222", content="Search result 2", confidence=0.9),
            Claim(id="c3333333", content="Search result 3", confidence=0.7)
        ]
        
        mock_search_result = Mock()
        mock_search_result.claims = search_results
        mock_data_manager.search_claims.return_value = mock_search_result
        
        # Test search
        results = await repository.search("test query", limit=5)
        assert len(results) == 3
        assert results[0].content == "Search result 1"
        
        # Setup mock list results
        list_results = [
            Claim(id="c4444444", content="Validated claim 1", confidence=0.8, state=ClaimState.VALIDATED),
            Claim(id="c5555555", content="Validated claim 2", confidence=0.9, state=ClaimState.VALIDATED)
        ]
        
        mock_list_result = Mock()
        mock_list_result.claims = list_results
        mock_data_manager.list_claims.return_value = mock_list_result
        
        # Test list by state
        validated_claims = await repository.list_by_state(ClaimState.VALIDATED)
        assert len(validated_claims) == 2
        assert all(claim.state == ClaimState.VALIDATED for claim in validated_claims)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, repository, mock_data_manager):
        """Test error handling and recovery scenarios"""
        # Test creation failure
        mock_data_manager.create_claim.side_effect = Exception("Creation failed")
        
        with pytest.raises(Exception, match="Creation failed"):
            await repository.create({'content': 'Test', 'confidence': 0.8})
        
        # Reset side effect and try again
        mock_data_manager.create_claim.side_effect = None
        mock_data_manager.create_claim.return_value = Claim(
            id="c1234567",
            content="Test claim",
            confidence=0.8
        )
        
        # Should succeed now
        result = await repository.create({'content': 'Test', 'confidence': 0.8})
        assert result.id == "c1234567"
        
        # Test get failure
        mock_data_manager.get_claim.side_effect = Exception("Get failed")
        result = await repository.get_by_id("c1234567")
        assert result is None
        
        # Reset and try again
        mock_data_manager.get_claim.side_effect = None
        mock_data_manager.get_claim.return_value = Claim(
            id="c1234567",
            content="Test claim",
            confidence=0.8
        )
        
        result = await repository.get_by_id("c1234567")
        assert result is not None


class TestRepositoryEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @pytest.mark.asyncio
    async def test_empty_claim_data(self, repository, mock_data_manager):
        """Test creating claim with minimal data"""
        claim_data = {
            'content': 'Minimal claim'
        }
        
        mock_claim = Claim(
            id="c1234567",
            content="Minimal claim",
            confidence=0.0,  # Default confidence
            state=ClaimState.EXPLORE  # Default state
        )
        mock_data_manager.create_claim.return_value = mock_claim
        
        result = await repository.create(claim_data)
        
        assert result.content == "Minimal claim"
        assert result.confidence == 0.0
        assert result.state == ClaimState.EXPLORE
        
        # Verify call with defaults
        mock_data_manager.create_claim.assert_called_once_with(
            content="Minimal claim",
            confidence=0.0,
            tags=[],
            state=ClaimState.EXPLORE
        )
    
    @pytest.mark.asyncio
    async def test_large_search_limit(self, repository, mock_data_manager):
        """Test search with large limit"""
        mock_search_result = Mock()
        mock_search_result.claims = []
        mock_data_manager.search_claims.return_value = mock_search_result
        
        await repository.search("test", limit=1000)
        
        mock_data_manager.search_claims.assert_called_once_with("test", limit=1000)
    
    @pytest.mark.asyncio
    async def test_zero_search_limit(self, repository, mock_data_manager):
        """Test search with zero limit"""
        mock_search_result = Mock()
        mock_search_result.claims = []
        mock_data_manager.search_claims.return_value = mock_search_result
        
        await repository.search("test", limit=0)
        
        mock_data_manager.search_claims.assert_called_once_with("test", limit=0)
    
    @pytest.mark.asyncio
    async def test_negative_search_limit(self, repository, mock_data_manager):
        """Test search with negative limit (should still work)"""
        mock_search_result = Mock()
        mock_search_result.claims = []
        mock_data_manager.search_claims.return_value = mock_search_result
        
        await repository.search("test", limit=-5)
        
        mock_data_manager.search_claims.assert_called_once_with("test", limit=-5)
    
    @pytest.mark.asyncio
    async def test_empty_search_query(self, repository, mock_data_manager):
        """Test search with empty query"""
        mock_search_result = Mock()
        mock_search_result.claims = []
        mock_data_manager.search_claims.return_value = mock_search_result
        
        await repository.search("")
        
        mock_data_manager.search_claims.assert_called_once_with("", limit=10)
    
    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, repository, mock_data_manager):
        """Test search with special characters"""
        mock_search_result = Mock()
        mock_search_result.claims = []
        mock_data_manager.search_claims.return_value = mock_search_result
        
        special_query = "test with 'quotes' and \"double quotes\" and \n newlines"
        await repository.search(special_query)
        
        mock_data_manager.search_claims.assert_called_once_with(special_query, limit=10)
    
    @pytest.mark.asyncio
    async def test_unicode_content(self, repository, mock_data_manager):
        """Test claim with unicode content"""
        unicode_content = "Test claim with unicode: 🚀 ñáéíóú 中文"
        
        mock_claim = Claim(
            id="c1234567",
            content=unicode_content,
            confidence=0.8
        )
        mock_data_manager.create_claim.return_value = mock_claim
        
        claim_data = {'content': unicode_content, 'confidence': 0.8}
        result = await repository.create(claim_data)
        
        assert result.content == unicode_content
        mock_data_manager.create_claim.assert_called_once_with(
            content=unicode_content,
            confidence=0.8,
            tags=[],
            state=ClaimState.EXPLORE
        )
    
    @pytest.mark.asyncio
    async def test_very_long_content(self, repository, mock_data_manager):
        """Test claim with very long content"""
        long_content = "x" * 10000  # 10k characters
        
        mock_claim = Claim(
            id="c1234567",
            content=long_content,
            confidence=0.8
        )
        mock_data_manager.create_claim.return_value = mock_claim
        
        claim_data = {'content': long_content, 'confidence': 0.8}
        result = await repository.create(claim_data)
        
        assert result.content == long_content
        mock_data_manager.create_claim.assert_called_once_with(
            content=long_content,
            confidence=0.8,
            tags=[],
            state=ClaimState.EXPLORE
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])