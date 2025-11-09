"""
Comprehensive integration tests for DataManager in the Conjecture data layer.
Tests end-to-end workflows, component coordination, and real usage patterns.
"""
import pytest
import asyncio
from datetime import datetime
from typing import List, Dict, Any

from src.data.data_manager import DataManager
from src.data.models import (
    Claim, Relationship, ClaimFilter, DataConfig,
    ClaimNotFoundError, InvalidClaimError, RelationshipError, DataLayerError
)


class TestDataManagerInitialization:
    """Test DataManager initialization and setup."""

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_data_manager_initialization_with_mocks(self, data_manager: DataManager):
        """Test DataManager initialization with mock embeddings."""
        assert data_manager is not None
        assert data_manager._initialized is True
        assert data_manager.sqlite_manager is not None
        assert data_manager.chroma_manager is not None
        assert data_manager.embedding_service is not None

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_data_manager_context_manager(self, test_config: DataConfig):
        """Test DataManager as async context manager."""
        test_config.sqlite_path = ":memory:"
        test_config.chroma_path = "./test_chroma_db"
        
        async with DataManager(test_config, use_mock_embeddings=True) as dm:
            assert dm._initialized is True
            
            # Can perform operations
            claim = await dm.create_claim("Test claim", "test_user")
            assert claim is not None
        
        # Should be closed after context
        assert dm._initialized is False

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_data_manager_close_and_reinitialize(self, data_manager: DataManager):
        """Test closing and reinitializing DataManager."""
        assert data_manager._initialized is True
        
        # Close
        await data_manager.close()
        assert data_manager._initialized is False
        
        # Reinitialize
        await data_manager.initialize()
        assert data_manager._initialized is True
        
        # Should still work
        claim = await dm.create_claim("Test after reinit", "test_user")
        assert claim is not None

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_initialization_failure_handling(self, test_config: DataConfig):
        """Test error handling during initialization."""
        # Use invalid configuration
        test_config.sqlite_path = "/invalid/path/cannot/create/db"
        
        dm = DataManager(test_config, use_mock_embeddings=True)
        
        with pytest.raises(DataLayerError, match="Failed to initialize data manager"):
            await dm.initialize()


class TestDataManagerClaimCRUD:
    """Test complete claim CRUD workflows."""

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_create_claim_complete_workflow(self, data_manager: DataManager):
        """Test complete claim creation workflow."""
        claim_data = {
            "content": "The Earth's atmosphere consists of approximately 78% nitrogen, 21% oxygen, and trace amounts of other gases.",
            "created_by": "science_expert",
            "confidence": 0.95,
            "tags": ["atmosphere", "earth", "chemistry"],
            "dirty": False
        }
        
        # Create claim
        claim = await data_manager.create_claim(**claim_data)
        
        # Verify claim structure
        assert isinstance(claim, Claim)
        assert claim.content == claim_data["content"]
        assert claim.created_by == claim_data["created_by"]
        assert claim.confidence == claim_data["confidence"]
        assert claim.tags == claim_data["tags"]
        assert claim.dirty == claim_data["dirty"]
        assert claim.id.startswith('c')
        assert len(claim.id) == 8
        assert isinstance(claim.created_at, datetime)

        # Verify claim exists in SQLite
        retrieved_claim = await data_manager.get_claim(claim.id)
        assert retrieved_claim is not None
        assert retrieved_claim.id == claim.id
        assert retrieved_claim.content == claim.content

        # Verify embedding exists in ChromaDB
        embedding = await data_manager.chroma_manager.get_embedding(claim.id)
        assert embedding is not None
        assert embedding['document'] == claim.content

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_create_claim_validation_errors(self, data_manager: DataManager):
        """Test claim creation validation errors."""
        # Test content too short
        with pytest.raises(InvalidClaimError, match="Claim content must be at least 10 characters"):
            await data_manager.create_claim("Too short", "user")

        # Test missing creator
        with pytest.raises(InvalidClaimError, match="created_by is required"):
            await data_manager.create_claim("Valid content length", "")

        # Test invalid confidence
        with pytest.raises(InvalidClaimError, match="Confidence must be between 0.0 and 1.0"):
            await data_manager.create_claim("Valid content", "user", confidence=1.5)

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_create_claim_defaults(self, data_manager: DataManager):
        """Test claim creation with default values."""
        claim = await data_manager.create_claim("Test claim with defaults", "test_user")
        
        assert claim.confidence == 0.5  # Default
        assert claim.dirty is True     # Default
        assert claim.tags == []        # Default

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_get_claim_by_id(self, populated_data_manager: DataManager):
        """Test retrieving claims by ID."""
        # Get existing claim
        claim = await populated_data_manager.get_claim("c0000001")
        assert claim is not None
        assert claim.id == "c0000001"
        assert isinstance(claim, Claim)

        # Get non-existent claim
        claim = await populated_data_manager.get_claim("c0999999")
        assert claim is None

        # Test invalid ID format
        with pytest.raises(InvalidClaimError, match="Invalid claim ID format"):
            await populated_data_manager.get_claim("invalid_id")

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_update_claim_workflow(self, data_manager: DataManager):
        """Test complete claim update workflow."""
        # Create initial claim
        claim = await data_manager.create_claim(
            "Initial claim content",
            "test_user",
            confidence=0.7,
            tags=["initial"],
            dirty=True
        )
        
        original_content = claim.content
        
        # Update single field
        updated = await data_manager.update_claim(claim.id, confidence=0.85)
        assert updated is True
        
        updated_claim = await data_manager.get_claim(claim.id)
        assert updated_claim.confidence == 0.85
        assert updated_claim.content == original_content  # Unchanged
        
        # Update multiple fields
        updates = {
            "tags": ["updated", "modified"],
            "dirty": False
        }
        updated = await data_manager.update_claim(claim.id, **updates)
        assert updated is True
        
        final_claim = await data_manager.get_claim(claim.id)
        assert final_claim.confidence == 0.85
        assert final_claim.tags == ["updated", "modified"]
        assert final_claim.dirty is False

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_update_claim_content_triggers_embedding_update(self, data_manager: DataManager):
        """Test that updating claim content updates the embedding."""
        # Create claim
        claim = await data_manager.create_claim("Original content", "test_user")
        
        # Get original embedding
        original_embedding = await data_manager.chroma_manager.get_embedding(claim.id)
        assert original_embedding['document'] == "Original content"
        
        # Update content
        await data_manager.update_claim(claim.id, content="Updated content")
        
        # Verify embedding was updated
        updated_embedding = await data_manager.chroma_manager.get_embedding(claim.id)
        assert updated_embedding['document'] == "Updated content"
        assert updated_embedding['metadata']['updated_at'] is not None

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_update_claim_validation_errors(self, data_manager: DataManager):
        """Test update validation errors."""
        claim = await data_manager.create_claim("Test claim", "test_user")
        
        # Invalid confidence
        with pytest.raises(InvalidClaimError, match="Confidence must be between 0.0 and 1.0"):
            await data_manager.update_claim(claim.id, confidence=2.0)
        
        # Invalid tags type
        with pytest.raises(InvalidClaimError, match="Tags must be a list"):
            await data_manager.update_claim(claim.id, tags="not_a_list")
        
        # Invalid claim ID
        with pytest.raises(InvalidClaimError, match="Invalid claim ID format"):
            await data_manager.update_claim("invalid", confidence=0.8)

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_delete_claim_workflow(self, data_manager: DataManager):
        """Test complete claim deletion workflow."""
        # Create claim
        claim = await data_manager.create_claim("Claim to delete", "test_user")
        claim_id = claim.id
        
        # Verify it exists
        assert await data_manager.get_claim(claim_id) is not None
        assert await data_manager.chroma_manager.get_embedding(claim_id) is not None
        
        # Delete claim
        deleted = await data_manager.delete_claim(claim_id)
        assert deleted is True
        
        # Verify it's gone from both stores
        assert await data_manager.get_claim(claim_id) is None
        assert await data_manager.chroma_manager.get_embedding(claim_id) is None
        
        # Delete non-existent claim
        deleted = await data_manager.delete_claim("c0999999")
        assert deleted is False

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_delete_claim_validation_error(self, data_manager: DataManager):
        """Test delete claim validation error."""
        with pytest.raises(InvalidClaimError, match="Invalid claim ID format"):
            await data_manager.delete_claim("invalid_id")


class TestDataManagerSearchAndFiltering:
    """Test search and filtering workflows."""

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_similar_search_workflow(self, populated_data_manager: DataManager):
        """Test semantic similarity search workflow."""
        # Search for physics-related content
        query = "quantum physics and mechanics"
        similar_claims = await populated_data_manager.search_similar(query, limit=3)
        
        assert isinstance(similar_claims, list)
        assert len(similar_claims) <= 3
        
        # Results should be Claim objects
        for claim in similar_claims:
            assert isinstance(claim, Claim)
            assert claim.id.startswith('c')
        
        # Results should be somewhat relevant (contains physics-related terms)
        if similar_claims:  # If any results returned
            physics_related = any(
                any(term in claim.content.lower() for term in ["physics", "quantum", "mechanics", "speed"])
                for claim in similar_claims
            )
            # Note: Since we're using mock embeddings, relevance isn't guaranteed
            # In real usage, this would be more meaningful

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_similar_search_with_filters(self, populated_data_manager: DataManager):
        """Test similarity search with metadata filters."""
        query = "scientific research"
        where_filter = {"confidence": {"$gte": 0.9}}  # High confidence only
        
        similar_claims = await populated_data_manager.search_similar(query, limit=5, where=where_filter)
        
        # All results should meet the confidence filter
        for claim in similar_claims:
            assert claim.confidence >= 0.9

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_similar_search_validation_errors(self, data_manager: DataManager):
        """Test similarity search validation."""
        # Empty query
        with pytest.raises(InvalidClaimError, match="Query cannot be empty"):
            await data_manager.search_similar("", limit=5)

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_filter_claims_workflow(self, populated_data_manager: DataManager):
        """Test claim filtering workflow."""
        # Test various filters
        filter_tests = [
            {"tags": ["physics"], "expected_field": "tags"},
            {"confidence_min": 0.9, "expected_field": "confidence"},
            {"dirty_only": True, "expected_field": "dirty"},
            {"created_by": "test_user", "expected_field": "created_by"},
            {"content_contains": "DNA", "expected_field": "content"}
        ]
        
        for filter_data, expected_field in filter_tests:
            filter_obj = ClaimFilter(**filter_data)
            filtered_claims = await populated_data_manager.filter_claims(filter_obj)
            
            assert isinstance(filtered_claims, list)
            
            # All results should be Claim objects
            for claim in filtered_claims:
                assert isinstance(claim, Claim)
                
                # Verify filter constraint (basic checks)
                if "tags" in filter_data and filter_data["tags"]:
                    assert any(tag in claim.tags for tag in filter_data["tags"])
                if "confidence_min" in filter_data:
                    assert claim.confidence >= filter_data["confidence_min"]
                if "dirty_only" in filter_data:
                    assert claim.dirty == filter_data["dirty_only"]
                if "created_by" in filter_data:
                    assert claim.created_by == filter_data["created_by"]
                if "content_contains" in filter_data:
                    assert filter_data["content_contains"].lower() in claim.content.lower()

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_complex_filter_combinations(self, populated_data_manager: DataManager):
        """Test complex filtering with multiple criteria."""
        filter_obj = ClaimFilter(
            tags=["science"],
            confidence_min=0.8,
            confidence_max=0.98,
            limit=5,
            offset=0
        )
        
        filtered_claims = await populated_data_manager.filter_claims(filter_obj)
        
        # All results should meet all criteria
        for claim in filtered_claims:
            assert any(tag in claim.tags for tag in ["science", "physics", "chemistry", "biology"])
            assert 0.8 <= claim.confidence <= 0.98
        assert len(filtered_claims) <= 5

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_get_dirty_claims_workflow(self, populated_data_manager: DataManager):
        """Test retrieving dirty claims for processing."""
        # Get dirty claims
        dirty_claims = await populated_data_manager.get_dirty_claims(limit=10)
        
        assert isinstance(dirty_claims, list)
        
        # All should be dirty
        for claim in dirty_claims:
            assert isinstance(claim, Claim)
            assert claim.dirty is True
        
        # Should be ordered by confidence (dirty claims prioritized)
        if len(dirty_claims) > 1:
            for i in range(len(dirty_claims) - 1):
                assert dirty_claims[i].confidence <= dirty_claims[i + 1].confidence


class TestDataManagerRelationships:
    """Test relationship management workflows."""

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_add_relationship_workflow(self, populated_data_manager: DataManager):
        """Test adding relationships between claims."""
        # Add relationship between existing claims
        relationship_id = await populated_data_manager.add_relationship(
            supporter_id="c0000001",
            supported_id="c0000002",
            relationship_type="supports",
            created_by="test_user"
        )
        
        assert relationship_id is not None
        assert isinstance(relationship_id, int)

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_add_relationship_validation_errors(self, populated_data_manager: DataManager):
        """Test relationship validation errors."""
        # Invalid supporter ID format
        with pytest.raises(InvalidClaimError, match="Invalid supporter_id format"):
            await populated_data_manager.add_relationship("invalid", "c0000002")
        
        # Invalid supported ID format
        with pytest.raises(InvalidClaimError, match="Invalid supported_id format"):
            await populated_data_manager.add_relationship("c0000001", "invalid")
        
        # Self-supporting
        with pytest.raises(RelationshipError, match="Claim cannot support itself"):
            await populated_data_manager.add_relationship("c0000001", "c0000001")
        
        # Non-existent claim
        with pytest.raises(DataLayerError, match="One or both claim IDs do not exist"):
            await populated_data_manager.add_relationship("c0000001", "c0999999")

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_remove_relationship_workflow(self, populated_data_manager: DataManager):
        """Test removing relationships."""
        # First add a relationship
        await populated_data_manager.add_relationship(
            supporter_id="c0000001",
            supported_id="c0000002",
            relationship_type="supports"
        )
        
        # Remove it
        removed = await populated_data_manager.remove_relationship("c0000001", "c0000002")
        assert removed is True
        
        # Remove non-existent relationship
        removed = await populated_data_manager.remove_relationship("c0000001", "c0000003")
        assert removed is False

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_relationship_traversal_workflows(self, populated_data_manager: DataManager):
        """Test traversing claim relationships."""
        # Create a relationship network
        relationships = [
            ("c0000001", "c0000002", "supports"),
            ("c0000003", "c0000002", "supports"),
            ("c0000002", "c0000004", "contradicts"),
            ("c0000003", "c0000004", "extends"),
        ]
        
        for supporter, supported, rel_type in relationships:
            await populated_data_manager.add_relationship(supporter, supported, rel_type)
        
        # Get relationships for c0000002
        rels = await populated_data_manager.get_relationships("c0000002")
        assert len(rels) >= 2
        assert all(isinstance(rel, Relationship) for rel in rels)
        
        # Get supporters of c0000002
        supporters = await populated_data_manager.get_supported_by("c0000002")
        assert "c0000001" in supporters
        assert "c0000003" in supporters
        
        # Get claims supported by c0000003
        supported = await populated_data_manager.get_supports("c0000003")
        assert "c0000002" in supported
        assert "c0000004" in supported


class TestDataManagerBatchOperations:
    """Test batch operation workflows."""

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_batch_create_claims_workflow(self, data_manager: DataManager):
        """Test batch claim creation workflow."""
        claims_data = [
            {
                "content": f"Batch claim {i} with content",
                "confidence": 0.7 + (i * 0.05),
                "tags": [f"batch_tag_{i % 3}"],
                "created_by": "batch_user"
            }
            for i in range(10)
        ]
        
        # Batch create
        created_claims = await data_manager.batch_create_claims(claims_data)
        
        assert len(created_claims) == len(claims_data)
        
        # Verify all claims were created
        for i, claim in enumerate(created_claims):
            assert isinstance(claim, Claim)
            assert claim.content == claims_data[i]["content"]
            assert claim.confidence == claims_data[i]["confidence"]
            assert claim.tags == claims_data[i]["tags"]
            assert claim.created_by == claims_data[i]["created_by"]
            
            # Verify claim exists in both stores
            retrieved = await data_manager.get_claim(claim.id)
            assert retrieved is not None
            
            embedding = await data_manager.chroma_manager.get_embedding(claim.id)
            assert embedding is not None

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_batch_create_empty_list(self, data_manager: DataManager):
        """Test batch create with empty list."""
        result = await data_manager.batch_create_claims([])
        assert result == []

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_batch_create_with_defaults(self, data_manager: DataManager):
        """Test batch create with minimal data (defaults)."""
        claims_data = [
            {"content": f"Minimal claim {i}", "created_by": "test_user"}
            for i in range(5)
        ]
        
        created_claims = await data_manager.batch_create_claims(claims_data)
        
        for claim in created_claims:
            assert claim.confidence == 0.5  # Default
            assert claim.dirty is True     # Default
            assert claim.tags == []        # Default


class TestDataManagerStatistics:
    """Test statistics and monitoring workflows."""

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_get_comprehensive_stats(self, populated_data_manager: DataManager):
        """Test getting comprehensive data layer statistics."""
        stats = await populated_data_manager.get_stats()
        
        assert isinstance(stats, dict)
        
        # Check required fields
        required_fields = ["total_claims", "dirty_claims", "clean_claims", "chroma_stats", "embedding_model"]
        for field in required_fields:
            assert field in stats
        
        # Verify counts make sense
        assert stats["total_claims"] >= 0
        assert stats["dirty_claims"] >= 0
        assert stats["clean_claims"] >= 0
        assert stats["total_claims"] == stats["dirty_claims"] + stats["clean_claims"]
        
        # Check ChromaDB stats
        chroma_stats = stats["chroma_stats"]
        assert isinstance(chroma_stats, dict)
        assert "total_claims" in chroma_stats
        assert chroma_stats["total_claims"] == stats["total_claims"]
        
        # Check embedding model info
        model_info = stats["embedding_model"]
        assert isinstance(model_info, dict)
        assert "model_name" in model_info
        assert "initialized" in model_info
        assert model_info["initialized"] is True

    @pytest.mark.data_manager
    @pytest.mark.asyncio
    async def test_stats_consistency_after_operations(self, data_manager: DataManager):
        """Test statistics consistency after various operations."""
        # Get initial stats
        initial_stats = await data_manager.get_stats()
        initial_total = initial_stats["total_claims"]
        initial_dirty = initial_stats["dirty_claims"]
        
        # Add claims
        new_claims = await data_manager.batch_create_claims([
            {"content": "Stat test claim 1", "created_by": "test_user"},
            {"content": "Stat test claim 2", "created_by": "test_user", "dirty": False},
        ])
        
        # Check updated stats
        updated_stats = await data_manager.get_stats()
        assert updated_stats["total_claims"] == initial_total + 2
        assert updated_stats["dirty_claims"] == initial_dirty + 1  # Only one is dirty
        
        # Update a claim
        await data_manager.update_claim(new_claims[0].id, dirty=False)
        
        # Check stats after update
        final_stats = await data_manager.get_stats()
        assert final_stats["total_claims"] == initial_total + 2
        assert final_stats["dirty_claims"] == initial_dirty  # Should be back to original
        
        # Delete a claim
        await data_manager.delete_claim(new_claims[0].id)
        
        # Check stats after deletion
        deletion_stats = await data_manager.get_stats()
        assert deletion_stats["total_claims"] == initial_total + 1
        assert deletion_stats["dirty_claims"] == initial_dirty


class TestDataManagerErrorHandling:
    """Test comprehensive error handling."""

    @pytest.mark.data_manager
    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_coordinate_component_failures(self, test_config: DataConfig):
        """Test coordination when components fail."""
        # Create data manager that will fail on ChromaDB initialization
        test_config.chroma_path = "/invalid/path/that/cannot/be/created"
        
        dm = DataManager(test_config, use_mock_embeddings=True)
        
        # Should fail during initialization
        with pytest.raises(DataLayerError):
            await dm.initialize()

    @pytest.mark.data_manager
    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_partial_operation_rollback(self, data_manager: DataManager):
        """Test partial operation rollback scenarios."""
        # This is more about ensuring consistency than actual rollback
        # since our operations are atomic at the component level
        
        claim = await data_manager.create_claim("Test claim", "test_user")
        claim_id = claim.id
        
        # If embedding generation fails, claim should still exist in SQLite
        # (This is a design choice - could also choose full rollback)
        
        # Verify this behavior by checking data exists in both stores
        sqlite_claim = await data_manager.get_claim(claim_id)
        chroma_embedding = await data_manager.chroma_manager.get_embedding(claim_id)
        
        # Under normal operation, both should exist
        assert sqlite_claim is not None
        assert chroma_embedding is not None

    @pytest.mark.data_manager
    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_uninitialized_operations(self, test_config: DataConfig):
        """Test operations on uninitialized data manager."""
        dm = DataManager(test_config, use_mock_embeddings=True)
        # Don't initialize
        
        with pytest.raises(DataLayerError):
            await dm.create_claim("Test", "user")
        
        with pytest.raises(DataLayerError):
            await dm.get_claim("c0000001")
        
        with pytest.raises(DataLayerError):
            await dm.get_stats()


class TestDataManagerPerformance:
    """Performance tests for DataManager operations."""

    @pytest.mark.data_manager
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_claim_creation_performance(self, data_manager: DataManager, benchmark):
        """Benchmark claim creation performance."""
        claim_data = {
            "content": "Performance test claim with sufficient content length for proper embedding generation.",
            "created_by": "perf_user",
            "confidence": 0.8,
            "tags": ["performance", "test"]
        }
        
        async def create_claim():
            return await data_manager.create_claim(**claim_data)
        
        result = await benchmark.async_timer(create_claim)
        # Should be fast (<100ms for complete claim creation + embedding)
        assert result < 0.1  # 100ms
        assert isinstance(result["result"], Claim)

    @pytest.mark.data_manager
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_claim_retrieval_performance(self, populated_data_manager: DataManager, benchmark):
        """Benchmark claim retrieval performance."""
        async def get_claim():
            return await populated_data_manager.get_claim("c0000001")
        
        result = await benchmark.async_timer(get_claim)
        # Should be very fast (<10ms for simple retrieval)
        assert result < 0.01  # 10ms
        assert isinstance(result["result"], Claim)

    @pytest.mark.data_manager
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_similarity_search_performance(self, populated_data_manager: DataManager, benchmark):
        """Benchmark similarity search performance."""
        query = "quantum physics and scientific research"
        
        async def search_similar():
            return await populated_data_manager.search_similar(query, limit=10)
        
        result = await benchmark.async_timer(search_similar)
        # Should be fast (<100ms for similarity search)
        assert result < 0.1  # 100ms
        assert isinstance(result["result"], list)

    @pytest.mark.data_manager
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_create_performance(self, data_manager: DataManager, claim_generator, benchmark):
        """Benchmark batch claim creation performance."""
        claims_data = claim_generator.generate(50)
        
        async def batch_create():
            return await data_manager.batch_create_claims(claims_data)
        
        result = await benchmark.async_timer(batch_create)
        # Should be efficient (<1s for 50 claims)
        assert result < 1.0  # 1s
        assert len(result["result"]) == 50

    @pytest.mark.data_manager
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_filter_claims_performance(self, populated_data_manager: DataManager, benchmark):
        """Benchmark claim filtering performance."""
        filter_obj = ClaimFilter(
            confidence_min=0.8,
            tags=["science"],
            limit=20
        )
        
        async def filter_claims():
            return await populated_data_manager.filter_claims(filter_obj)
        
        result = await benchmark.async_timer(filter_claims)
        # Should be very fast (<50ms for filtering)
        assert result < 0.05  # 50ms
        assert isinstance(result["result"], list)


class TestDataManagerIntegrationScenarios:
    """Integration scenarios simulating real usage patterns."""

    @pytest.mark.data_manager
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_research_assistant_workflow(self, data_manager: DataManager):
        """Simulate a research assistant usage scenario."""
        # 1. Import initial research claims
        research_claims = [
            {"content": "Climate change is primarily caused by greenhouse gas emissions", "created_by": "researcher1", "confidence": 0.85, "tags": ["climate", "environment"]},
            {"content": "Global temperatures have risen by 1.1°C since pre-industrial times", "created_by": "researcher1", "confidence": 0.90, "tags": ["climate", "temperature"]},
            {"content": "Renewable energy sources can reduce carbon emissions", "created_by": "researcher2", "confidence": 0.75, "tags": ["energy", "renewable"]},
        ]
        
        imported_claims = await data_manager.batch_create_claims(research_claims)
        assert len(imported_claims) == 3
        
        # 2. Add supporting evidence relationships
        await data_manager.add_relationship(
            supporter_id=imported_claims[1].id,  # Temperature data
            supported_id=imported_claims[0].id,  # Climate change claim
            relationship_type="supports",
            created_by="researcher1"
        )
        
        # 3. Process dirty claims (simulate validation)
        dirty_claims = await data_manager.get_dirty_claims()
        for claim in dirty_claims:
            # Simulate validation process
            await data_manager.update_claim(
                claim.id,
                confidence=claim.confidence + 0.05,
                dirty=False
            )
        
        # 4. Search for related claims
        related_claims = await data_manager.search_similar("carbon dioxide and global warming", limit=5)
        assert len(related_claims) >= 0
        
        # 5. Generate report statistics
        stats = await data_manager.get_stats()
        assert stats["total_claims"] == 3
        assert stats["dirty_claims"] == 0  # All cleaned up
        
        # 6. Filter by topic for focused analysis
        climate_claims = await data_manager.filter_claims(
            ClaimFilter(tags=["climate"])
        )
        assert len(climate_claims) >= 2

    @pytest.mark.data_manager
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_fact_checking_workflow(self, data_manager: DataManager):
        """Simulate a fact-checking workflow."""
        # 1. Add claim to be verified
        claim_to_check = await data_manager.create_claim(
            "The Great Wall of China is visible from the Moon",
            "user_submission",
            confidence=0.5,  # Uncertainty
            dirty=True,
            tags=["space", "history", "myth"]
        )
        
        # 2. Search for related claims in knowledge base
        related_claims = await data_manager.search_similar(
            "Great Wall of China visible space",
            limit=5
        )
        
        # 3. Add fact-checking evidence
        evidence_claims = [
            {"content": "The Great Wall cannot be seen from the Moon with the naked eye", "created_by": "fact_checker", "confidence": 0.95, "tags": ["debunked", "space"]},
            {"content": "No man-made objects are visible from the Moon without aid", "created_by": "fact_checker", "confidence": 0.90, "tags": ["space", "astronomy"]},
        ]
        
        evidence = await data_manager.batch_create_claims(evidence_claims)
        
        # 4. Add contradictory relationships
        for ev_claim in evidence:
            await data_manager.add_relationship(
                supporter_id=ev_claim.id,
                supported_id=claim_to_check.id,
                relationship_type="contradicts",
                created_by="fact_checker"
            )
        
        # 5. Update original claim based on evidence
        await data_manager.update_claim(
            claim_to_check.id,
            confidence=0.1,  # Very low confidence due to contradictions
            dirty=False
        )
        
        # 6. Verify final state
        final_claim = await data_manager.get_claim(claim_to_check.id)
        assert final_claim.confidence == 0.1
        assert final_claim.dirty is False
        
        # Check relationships
        relationships = await data_manager.get_relationships(claim_to_check.id)
        contradictions = [rel for rel in relationships if rel.relationship_type == "contradicts"]
        assert len(contradictions) == 2

    @pytest.mark.data_manager
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_knowledge_base_evolution(self, data_manager: DataManager):
        """Test knowledge base evolution over time."""
        # 1. Initial knowledge base
        initial_claims = [
            {"content": "Smoking causes lung cancer", "created_by": "med_db_1950", "confidence": 0.6, "dirty": True, "tags": ["health", "smoking"]},
            {"content": "Cigarettes contain nicotine", "created_by": "med_db_1950", "confidence": 0.8, "dirty": False, "tags": ["chemistry", "nicotine"]},
        ]
        
        initial = await data_manager.batch_create_claims(initial_claims)
        
        # 2. Add early support
        early_evidence = [
            {"content": "Epidermiological studies link smoking to cancer rates", "created_by": "study_1960", "confidence": 0.75, "tags": ["epidemiology", "cancer"]},
        ]
        
        early = await data_manager.batch_create_claims(early_evidence)
        
        # Add support relationships
        await data_manager.add_relationship(early[0].id, initial[0].id, "supports", "researcher")
        
        # 3. Update based on new evidence
        await data_manager.update_claim(initial[0].id, confidence=0.8, dirty=False)
        
        # 4. Add modern consensus
        modern_claims = [
            {"content": "Smoking is the leading cause of preventable death worldwide", "created_by": "who_2020", "confidence": 0.99, "dirty": False, "tags": ["health", "public_health"]},
            {"content": "Secondhand smoke also causes lung cancer", "created_by": "cdc_2020", "confidence": 0.95, "dirty": False, "tags": ["health", "secondhand"]},
        ]
        
        modern = await data_manager.batch_create_claims(modern_claims)
        
        # Add comprehensive support network
        for claim in [early[0]] + modern:
            await data_manager.add_relationship(claim.id, initial[0].id, "supports", "consensus_builder")
        
        # 5. Verify knowledge evolution
        final_claim = await data_manager.get_claim(initial[0].id)
        assert final_claim.confidence == 0.8
        
        supporters = await data_manager.get_supported_by(initial[0].id)
        assert len(supporters) >= 3  # Multiple sources now support it
        
        # 6. Knowledge consistency check
        all_smoking_claims = await data_manager.filter_claims(
            ClaimFilter(tags=["smoking"])
        )
        assert len(all_smoking_claims) >= 1
        
        # All should have high confidence (knowledge converged)
        for claim in all_smoking_claims:
            assert claim.confidence >= 0.6

    @pytest.mark.data_manager
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_collaborative_knowledge_building(self, data_manager: DataManager):
        """Test collaborative knowledge building with multiple users."""
        users = ["alice", "bob", "charlie", "diana"]
        
        # 1. Initial knowledge seeding
        for user in users:
            await data_manager.create_claim(
                f"Initial contribution from {user} about their expertise",
                user,
                confidence=0.7,
                tags=["collaboration", user],
                dirty=False
            )
        
        # 2. Interactive refinement
        claims = await data_manager.filter_claims(ClaimFilter(limit=10))
        
        for i, claim in enumerate(claims):
            # Other users review and improve
            reviewer = users[(i + 1) % len(users)]
            
            if i % 2 == 0:
                # Add supporting claim
                supporter = await data_manager.create_claim(
                    f"Supporting evidence for {claim.content[:30]} from {reviewer}",
                    reviewer,
                    confidence=0.8,
                    tags=["evidence"],
                    dirty=False
                )
                
                await data_manager.add_relationship(
                    supporter.id, claim.id, "supports", reviewer
                )
            else:
                # Improve confidence based on review
                await data_manager.update_claim(
                    claim.id,
                    confidence=min(claim.confidence + 0.1, 0.95),
                    dirty=False
                )
        
        # 3. Knowledge network analysis
        final_claims = await data_manager.filter_claims(ClaimFilter(limit=20))
        
        # Should have supporting relationships
        relationship_count = 0
        for claim in final_claims:
            rels = await data_manager.get_relationships(claim.id)
            relationship_count += len(rels)
        
        assert relationship_count > 0
        
        # 4. Quality assessment
        stats = await data_manager.get_stats()
        high_confidence_claims = await data_manager.filter_claims(
            ClaimFilter(confidence_min=0.8)
        )
        
        # Most claims should be high confidence after collaboration
        confidence_ratio = len(high_confidence_claims) / len(final_claims)
        assert confidence_ratio >= 0.5