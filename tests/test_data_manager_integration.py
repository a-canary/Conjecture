"""
Integration tests for DataManager.
"""
import pytest
import asyncio
import tempfile
import shutil
import os

from src.data.data_manager import DataManager
from src.data.models import Claim, ClaimFilter, DataConfig


@pytest.fixture
async def data_manager():
    """Create a temporary data manager for testing."""
    temp_dir = tempfile.mkdtemp()
    
    config = DataConfig(
        sqlite_path=os.path.join(temp_dir, "test.db"),
        chroma_path=os.path.join(temp_dir, "chroma"),
        use_mock_embeddings=True
    )
    
    manager = DataManager(config, use_mock_embeddings=True)
    await manager.initialize()
    
    yield manager
    
    await manager.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestDataManagerIntegration:
    """Test DataManager integration functionality."""
    
    @pytest.mark.asyncio
    async def test_full_claim_lifecycle(self, data_manager):
        """Test complete claim lifecycle: create, read, update, delete."""
        # Create claim
        claim = await data_manager.create_claim(
            content="Machine learning algorithms require large datasets",
            created_by="test_user",
            confidence=0.8,
            tags=["ml", "algorithms"]
        )
        
        assert claim.id.startswith("c")
        assert claim.content == "Machine learning algorithms require large datasets"
        assert claim.confidence == 0.8
        assert claim.tags == ["ml", "algorithms"]
        
        # Read claim
        retrieved = await data_manager.get_claim(claim.id)
        assert retrieved is not None
        assert retrieved.id == claim.id
        assert retrieved.content == claim.content
        
        # Update claim
        success = await data_manager.update_claim(
            claim.id, 
            confidence=0.9,
            dirty=False
        )
        assert success is True
        
        updated = await data_manager.get_claim(claim.id)
        assert updated.confidence == 0.9
        assert updated.dirty is False
        
        # Delete claim
        success = await data_manager.delete_claim(claim.id)
        assert success is True
        
        deleted = await data_manager.get_claim(claim.id)
        assert deleted is None
    
    @pytest.mark.asyncio
    async def test_similarity_search_workflow(self, data_manager):
        """Test similarity search workflow."""
        # Create claims
        claims = [
            ("Machine learning uses statistical methods", "ml", "user1"),
            ("Deep learning is a subset of machine learning", "dl", "user1"),
            ("Python is a popular programming language", "python", "user2"),
            ("Neural networks mimic brain structure", "neural", "user1")
        ]
        
        created_claims = []
        for content, tag, user in claims:
            claim = await data_manager.create_claim(
                content=content,
                created_by=user,
                tags=[tag]
            )
            created_claims.append(claim)
        
        # Search for similar claims
        similar_claims = await data_manager.search_similar(
            "artificial intelligence and machine learning",
            limit=3
        )
        
        assert len(similar_claims) <= 3
        # Should find ML-related claims
        found_contents = [claim.content for claim in similar_claims]
        ml_related = any("machine learning" in content.lower() for content in found_contents)
        assert ml_related
    
    @pytest.mark.asyncio
    async def test_relationship_management_workflow(self, data_manager):
        """Test relationship management workflow."""
        # Create claims
        claim1 = await data_manager.create_claim(
            content="Basic machine learning concept",
            created_by="user1",
            tags=["ml", "basic"]
        )
        claim2 = await data_manager.create_claim(
            content="Advanced machine learning techniques",
            created_by="user1", 
            tags=["ml", "advanced"]
        )
        claim3 = await data_manager.create_claim(
            content="Deep learning fundamentals",
            created_by="user2",
            tags=["dl", "basic"]
        )
        
        # Add relationships
        rel_id1 = await data_manager.add_relationship(
            supporter_id=claim1.id,
            supported_id=claim2.id,
            relationship_type="supports",
            created_by="user1"
        )
        assert rel_id1 is not None
        
        rel_id2 = await data_manager.add_relationship(
            supporter_id=claim3.id,
            supported_id=claim2.id,
            relationship_type="supports",
            created_by="user1"
        )
        assert rel_id2 is not None
        
        # Get relationships for claim2
        relationships = await data_manager.get_relationships(claim2.id)
        assert len(relationships) == 2
        
        # Check supporters and supported
        supporters = await data_manager.get_supported_by(claim2.id)
        supports = await data_manager.get_supports(claim2.id)
        
        assert len(supporters) == 2
        assert claim1.id in supporters
        assert claim3.id in supporters
        assert len(supports) == 0
        
        # Remove relationship
        success = await data_manager.remove_relationship(
            supporter_id=claim1.id,
            supported_id=claim2.id
        )
        assert success is True
        
        # Verify removal
        supporters = await data_manager.get_supported_by(claim2.id)
        assert len(supporters) == 1
        assert claim3.id in supporters
    
    @pytest.mark.asyncio
    async def test_filter_claims_workflow(self, data_manager):
        """Test claim filtering workflow."""
        # Create diverse claims
        claims_data = [
            ("ML basics", 0.9, ["ml", "basic"], True, "user1"),
            ("DL basics", 0.7, ["dl", "basic"], False, "user1"),
            ("Python basics", 0.8, ["python", "basic"], True, "user2"),
            ("Advanced ML", 0.6, ["ml", "advanced"], True, "user1"),
            ("Advanced DL", 0.5, ["dl", "advanced"], False, "user2")
        ]
        
        created_claims = []
        for content, confidence, tags, dirty, user in claims_data:
            claim = await data_manager.create_claim(
                content=content,
                created_by=user,
                confidence=confidence,
                tags=tags,
                dirty=dirty
            )
            created_claims.append(claim)
        
        # Filter by tags
        filter_obj = ClaimFilter(tags=["ml"])
        ml_claims = await data_manager.filter_claims(filter_obj)
        assert len(ml_claims) == 2
        assert all("ml" in claim.tags for claim in ml_claims)
        
        # Filter by confidence range
        filter_obj = ClaimFilter(confidence_min=0.7, confidence_max=0.9)
        high_confidence_claims = await data_manager.filter_claims(filter_obj)
        assert len(high_confidence_claims) == 2
        assert all(0.7 <= claim.confidence <= 0.9 for claim in high_confidence_claims)
        
        # Filter by dirty flag
        filter_obj = ClaimFilter(dirty_only=True)
        dirty_claims = await data_manager.filter_claims(filter_obj)
        assert len(dirty_claims) == 3
        assert all(claim.dirty for claim in dirty_claims)
        
        # Filter by creator
        filter_obj = ClaimFilter(created_by="user1")
        user1_claims = await data_manager.filter_claims(filter_obj)
        assert len(user1_claims) == 3
        assert all(claim.created_by == "user1" for claim in user1_claims)
        
        # Combined filters
        filter_obj = ClaimFilter(
            tags=["basic"],
            confidence_min=0.7,
            dirty_only=True
        )
        combined_claims = await data_manager.filter_claims(filter_obj)
        assert len(combined_claims) == 1
        assert combined_claims[0].tags == ["ml", "basic"]
    
    @pytest.mark.asyncio
    async def test_batch_operations_workflow(self, data_manager):
        """Test batch operations workflow."""
        # Prepare batch data
        batch_data = [
            {
                "content": "Batch claim 1",
                "created_by": "batch_user",
                "confidence": 0.7,
                "tags": ["batch", "test"]
            },
            {
                "content": "Batch claim 2", 
                "created_by": "batch_user",
                "confidence": 0.8,
                "tags": ["batch", "test"]
            },
            {
                "content": "Batch claim 3",
                "created_by": "batch_user", 
                "confidence": 0.9,
                "tags": ["batch", "test"]
            }
        ]
        
        # Create claims in batch
        created_claims = await data_manager.batch_create_claims(batch_data)
        assert len(created_claims) == 3
        
        # Verify all claims were created
        for i, claim in enumerate(created_claims):
            assert claim.content == batch_data[i]["content"]
            assert claim.confidence == batch_data[i]["confidence"]
            assert claim.tags == batch_data[i]["tags"]
            
            # Verify can retrieve each claim
            retrieved = await data_manager.get_claim(claim.id)
            assert retrieved is not None
            assert retrieved.id == claim.id
        
        # Test similarity search with batch-created claims
        similar_claims = await data_manager.search_similar("batch test claims", limit=5)
        found_batch_claims = [claim for claim in similar_claims if "batch" in claim.content.lower()]
        assert len(found_batch_claims) == 3
    
    @pytest.mark.asyncio
    async def test_dirty_claims_workflow(self, data_manager):
        """Test dirty claims management workflow."""
        # Create claims with different dirty states
        clean_claim = await data_manager.create_claim(
            content="Clean claim",
            created_by="user1",
            confidence=0.9,
            dirty=False
        )
        
        dirty_claims = []
        for i in range(3):
            claim = await data_manager.create_claim(
                content=f"Dirty claim {i+1}",
                created_by="user1",
                confidence=0.5 + i * 0.1,
                dirty=True
            )
            dirty_claims.append(claim)
        
        # Get dirty claims
        retrieved_dirty = await data_manager.get_dirty_claims(limit=10)
        assert len(retrieved_dirty) == 3
        
        # Should be ordered by confidence (ascending)
        confidences = [claim.confidence for claim in retrieved_dirty]
        assert confidences == sorted(confidences)
        
        # Mark one claim as clean
        await data_manager.update_claim(dirty_claims[0].id, dirty=False)
        
        # Check dirty claims again
        retrieved_dirty = await data_manager.get_dirty_claims(limit=10)
        assert len(retrieved_dirty) == 2
        assert dirty_claims[0].id not in [claim.id for claim in retrieved_dirty]
    
    @pytest.mark.asyncio
    async def test_statistics_workflow(self, data_manager):
        """Test statistics and monitoring workflow."""
        # Get initial stats
        initial_stats = await data_manager.get_stats()
        assert initial_stats["total_claims"] == 0
        assert initial_stats["dirty_claims"] == 0
        assert initial_stats["clean_claims"] == 0
        
        # Create various claims
        claims = [
            ("Claim 1", 0.8, True, ["tag1"]),
            ("Claim 2", 0.6, False, ["tag2"]),
            ("Claim 3", 0.9, True, ["tag1", "tag3"])
        ]
        
        for content, confidence, dirty, tags in claims:
            await data_manager.create_claim(
                content=content,
                created_by="stats_user",
                confidence=confidence,
                dirty=dirty,
                tags=tags
            )
        
        # Get updated stats
        updated_stats = await data_manager.get_stats()
        assert updated_stats["total_claims"] == 3
        assert updated_stats["dirty_claims"] == 2
        assert updated_stats["clean_claims"] == 1
        
        # Verify chroma stats
        assert updated_stats["chroma_stats"]["total_claims"] == 3
        
        # Verify embedding model info
        model_info = updated_stats["embedding_model"]
        assert model_info["model_name"] == "all-MiniLM-L6-v2"
        assert model_info["initialized"] is True
    
    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, data_manager):
        """Test error handling in various scenarios."""
        # Test invalid claim creation
        with pytest.raises(Exception):  # Should raise InvalidClaimError
            await data_manager.create_claim(
                content="Short",  # Too short
                created_by="user1"
            )
        
        with pytest.raises(Exception):  # Should raise InvalidClaimError
            await data_manager.create_claim(
                content="Valid content length",
                created_by=""  # Empty creator
            )
        
        # Test invalid claim ID operations
        with pytest.raises(Exception):  # Should raise InvalidClaimError
            await data_manager.get_claim("invalid_id")
        
        with pytest.raises(Exception):  # Should raise InvalidClaimError
            await data_manager.update_claim("invalid_id", confidence=0.8)
        
        # Test relationship errors
        claim1 = await data_manager.create_claim(
            content="Claim 1",
            created_by="user1"
        )
        
        # Non-existent claim in relationship
        with pytest.raises(Exception):  # Should raise DataLayerError
            await data_manager.add_relationship(
                supporter_id=claim1.id,
                supported_id="c9999999"  # Non-existent
            )
        
        # Self-relationship
        with pytest.raises(Exception):  # Should raise RelationshipError
            await data_manager.add_relationship(
                supporter_id=claim1.id,
                supported_id=claim1.id  # Same claim
            )
    
    @pytest.mark.asyncio
    async def test_context_manager_usage(self):
        """Test using DataManager as context manager."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            config = DataConfig(
                sqlite_path=os.path.join(temp_dir, "context_test.db"),
                chroma_path=os.path.join(temp_dir, "context_chroma")
            )
            
            async with DataManager(config, use_mock_embeddings=True) as manager:
                # Use manager within context
                claim = await manager.create_claim(
                    content="Context manager test",
                    created_by="test_user"
                )
                
                retrieved = await manager.get_claim(claim.id)
                assert retrieved is not None
                assert retrieved.content == "Context manager test"
            
            # Manager should be automatically closed after context
            # Verify by trying to use it (should fail)
            with pytest.raises(Exception):
                await manager.get_claim(claim.id)
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)