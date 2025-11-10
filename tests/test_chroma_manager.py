"""
Unit tests for ChromaManager.
"""
import pytest
import asyncio
import tempfile
import shutil
import os

from src.data.chroma_manager import ChromaManager
from src.data.models import Claim, DataLayerError


@pytest.fixture
async def chroma_manager():
    """Create a temporary ChromaDB manager for testing."""
    temp_dir = tempfile.mkdtemp()
    manager = ChromaManager(temp_dir)
    await manager.initialize()
    
    yield manager
    
    await manager.close()
    # Clean up
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_claim():
    """Create a sample claim for testing."""
    return Claim(
        id="c0000001",
        content="This is a sample claim about machine learning",
        confidence=0.8,
        tags=["ml", "test"],
        created_by="test_user"
    )


@pytest.fixture
def sample_embedding():
    """Create a sample embedding for testing."""
    return [0.1] * 384  # 384-dimensional embedding


class TestChromaManager:
    """Test ChromaManager functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, chroma_manager):
        """Test ChromaDB initialization."""
        assert chroma_manager.client is not None
        assert chroma_manager.collection is not None
        assert chroma_manager.collection.name == "claims"
    
    @pytest.mark.asyncio
    async def test_add_embedding(self, chroma_manager, sample_claim, sample_embedding):
        """Test adding an embedding."""
        await chroma_manager.add_embedding(sample_claim, sample_embedding)
        
        # Verify embedding was added
        retrieved = await chroma_manager.get_embedding(sample_claim.id)
        assert retrieved is not None
        assert retrieved['id'] == sample_claim.id
        assert retrieved['document'] == sample_claim.content
        assert retrieved['embedding'] == sample_embedding
        assert retrieved['metadata']['confidence'] == sample_claim.confidence
        assert retrieved['metadata']['tags'] == "ml,test"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_embedding(self, chroma_manager):
        """Test retrieving a non-existent embedding."""
        result = await chroma_manager.get_embedding("c9999999")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_embedding(self, chroma_manager, sample_claim, sample_embedding):
        """Test updating an embedding."""
        # Add initial embedding
        await chroma_manager.add_embedding(sample_claim, sample_embedding)
        
        # Update claim and embedding
        updated_claim = Claim(
            id=sample_claim.id,
            content="Updated content about deep learning",
            confidence=0.9,
            tags=["dl", "updated"],
            created_by="test_user"
        )
        updated_embedding = [0.2] * 384
        
        await chroma_manager.update_embedding(updated_claim, updated_embedding)
        
        # Verify update
        retrieved = await chroma_manager.get_embedding(sample_claim.id)
        assert retrieved is not None
        assert retrieved['document'] == "Updated content about deep learning"
        assert retrieved['embedding'] == updated_embedding
        assert retrieved['metadata']['confidence'] == 0.9
        assert retrieved['metadata']['tags'] == "dl,updated"
    
    @pytest.mark.asyncio
    async def test_delete_embedding(self, chroma_manager, sample_claim, sample_embedding):
        """Test deleting an embedding."""
        # Add embedding
        await chroma_manager.add_embedding(sample_claim, sample_embedding)
        
        # Verify it exists
        retrieved = await chroma_manager.get_embedding(sample_claim.id)
        assert retrieved is not None
        
        # Delete embedding
        await chroma_manager.delete_embedding(sample_claim.id)
        
        # Verify deletion
        retrieved = await chroma_manager.get_embedding(sample_claim.id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_search_similar_by_embedding(self, chroma_manager, sample_claim, sample_embedding):
        """Test similarity search by embedding."""
        # Add multiple claims
        claims = [
            Claim(id="c0000001", content="Machine learning algorithms", confidence=0.8, tags=["ml"], created_by="user1"),
            Claim(id="c0000002", content="Deep neural networks", confidence=0.7, tags=["dl"], created_by="user1"),
            Claim(id="c0000003", content="Python programming", confidence=0.9, tags=["python"], created_by="user2"),
        ]
        
        embeddings = [
            [0.1] * 384,  # Similar to sample
            [0.2] * 384,  # Less similar
            [0.9] * 384,  # Very different
        ]
        
        for claim, embedding in zip(claims, embeddings):
            await chroma_manager.add_embedding(claim, embedding)
        
        # Search with sample embedding
        results = await chroma_manager.search_similar(sample_embedding, limit=3)
        
        assert len(results) == 3
        # Results should be ordered by similarity (distance)
        assert results[0]['id'] == "c0000001"  # Most similar
        assert results[1]['id'] == "c0000002"
        assert results[2]['id'] == "c0000003"  # Least similar
    
    @pytest.mark.asyncio
    async def test_search_similar_with_filters(self, chroma_manager):
        """Test similarity search with metadata filters."""
        # Add claims with different metadata
        claims = [
            Claim(id="c0000001", content="ML algorithms", confidence=0.8, tags=["ml"], created_by="user1"),
            Claim(id="c0000002", content="DL networks", confidence=0.7, tags=["dl"], created_by="user1"),
            Claim(id="c0000003", content="Python code", confidence=0.9, tags=["python"], created_by="user2"),
        ]
        
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        
        for claim, embedding in zip(claims, embeddings):
            await chroma_manager.add_embedding(claim, embedding)
        
        # Search with confidence filter
        query_embedding = [0.15] * 384
        where_filter = {"confidence": {"$gte": 0.8}}
        
        results = await chroma_manager.search_similar(query_embedding, where=where_filter)
        
        assert len(results) == 1
        assert results[0]['id'] == "c0000001"
        assert results[0]['metadata']['confidence'] == 0.8
    
    @pytest.mark.asyncio
    async def test_search_by_text(self, chroma_manager):
        """Test text-based similarity search."""
        # Add claims
        claims = [
            Claim(id="c0000001", content="Machine learning is a subset of AI", confidence=0.8, tags=["ml"], created_by="user1"),
            Claim(id="c0000002", content="Deep learning uses neural networks", confidence=0.7, tags=["dl"], created_by="user1"),
            Claim(id="c0000003", content="Python is a programming language", confidence=0.9, tags=["python"], created_by="user2"),
        ]
        
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        
        for claim, embedding in zip(claims, embeddings):
            await chroma_manager.add_embedding(claim, embedding)
        
        # Search by text
        results = await chroma_manager.search_by_text("neural networks", limit=2)
        
        assert len(results) <= 2
        # Should find claims related to neural networks
        found_ids = [result['id'] for result in results]
        assert "c0000002" in found_ids  # Most relevant
    
    @pytest.mark.asyncio
    async def test_batch_add_embeddings(self, chroma_manager):
        """Test batch addition of embeddings."""
        claims = [
            Claim(id="c0000001", content="Batch claim 1", confidence=0.5, tags=["batch"], created_by="user1"),
            Claim(id="c0000002", content="Batch claim 2", confidence=0.6, tags=["batch"], created_by="user1"),
            Claim(id="c0000003", content="Batch claim 3", confidence=0.7, tags=["batch"], created_by="user2"),
        ]
        
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        
        await chroma_manager.batch_add_embeddings(claims, embeddings)
        
        # Verify all embeddings were added
        for claim in claims:
            retrieved = await chroma_manager.get_embedding(claim.id)
            assert retrieved is not None
            assert retrieved['document'] == claim.content
    
    @pytest.mark.asyncio
    async def test_batch_add_embeddings_mismatch(self, chroma_manager):
        """Test batch add with mismatched claims and embeddings."""
        claims = [
            Claim(id="c0000001", content="Claim 1", confidence=0.5, created_by="user1"),
            Claim(id="c0000002", content="Claim 2", confidence=0.6, created_by="user1"),
        ]
        
        embeddings = [[0.1] * 384]  # Only one embedding for two claims
        
        with pytest.raises(DataLayerError):
            await chroma_manager.batch_add_embeddings(claims, embeddings)
    
    @pytest.mark.asyncio
    async def test_batch_update_embeddings(self, chroma_manager):
        """Test batch update of embeddings."""
        # Add initial embeddings
        claims = [
            Claim(id="c0000001", content="Original 1", confidence=0.5, tags=["original"], created_by="user1"),
            Claim(id="c0000002", content="Original 2", confidence=0.6, tags=["original"], created_by="user1"),
        ]
        
        embeddings = [[0.1] * 384, [0.2] * 384]
        await chroma_manager.batch_add_embeddings(claims, embeddings)
        
        # Update claims and embeddings
        updated_claims = [
            Claim(id="c0000001", content="Updated 1", confidence=0.8, tags=["updated"], created_by="user1"),
            Claim(id="c0000002", content="Updated 2", confidence=0.9, tags=["updated"], created_by="user1"),
        ]
        
        updated_embeddings = [[0.8] * 384, [0.9] * 384]
        await chroma_manager.batch_update_embeddings(updated_claims, updated_embeddings)
        
        # Verify updates
        for claim in updated_claims:
            retrieved = await chroma_manager.get_embedding(claim.id)
            assert retrieved is not None
            assert retrieved['document'] == claim.content
            assert retrieved['metadata']['confidence'] == claim.confidence
    
    @pytest.mark.asyncio
    async def test_batch_delete_embeddings(self, chroma_manager):
        """Test batch deletion of embeddings."""
        # Add embeddings
        claims = [
            Claim(id="c0000001", content="Claim 1", confidence=0.5, created_by="user1"),
            Claim(id="c0000002", content="Claim 2", confidence=0.6, created_by="user1"),
            Claim(id="c0000003", content="Claim 3", confidence=0.7, created_by="user2"),
        ]
        
        embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        await chroma_manager.batch_add_embeddings(claims, embeddings)
        
        # Delete two embeddings
        await chroma_manager.batch_delete_embeddings(["c0000001", "c0000003"])
        
        # Verify deletions
        assert await chroma_manager.get_embedding("c0000001") is None
        assert await chroma_manager.get_embedding("c0000003") is None
        # Verify remaining
        assert await chroma_manager.get_embedding("c0000002") is not None
    
    @pytest.mark.asyncio
    async def test_get_collection_stats(self, chroma_manager):
        """Test getting collection statistics."""
        # Initially should be empty
        stats = await chroma_manager.get_collection_stats()
        assert stats['total_claims'] == 0
        assert stats['collection_name'] == "claims"
        
        # Add some embeddings
        claims = [
            Claim(id="c0000001", content="Claim 1", confidence=0.5, created_by="user1"),
            Claim(id="c0000002", content="Claim 2", confidence=0.6, created_by="user1"),
        ]
        
        embeddings = [[0.1] * 384, [0.2] * 384]
        await chroma_manager.batch_add_embeddings(claims, embeddings)
        
        # Check stats again
        stats = await chroma_manager.get_collection_stats()
        assert stats['total_claims'] == 2
    
    @pytest.mark.asyncio
    async def test_reset_collection(self, chroma_manager, sample_claim, sample_embedding):
        """Test resetting the collection."""
        # Add some data
        await chroma_manager.add_embedding(sample_claim, sample_embedding)
        
        # Verify data exists
        assert await chroma_manager.get_embedding(sample_claim.id) is not None
        
        # Reset collection
        await chroma_manager.reset_collection()
        
        # Verify data is gone
        assert await chroma_manager.get_embedding(sample_claim.id) is None
        
        # Verify collection still exists but is empty
        stats = await chroma_manager.get_collection_stats()
        assert stats['total_claims'] == 0
    
    @pytest.mark.asyncio
    async def test_error_handling_not_initialized(self):
        """Test error handling when manager is not initialized."""
        manager = ChromaManager("/fake/path")
        # Don't initialize
        
        with pytest.raises(DataLayerError):
            await manager.add_embedding(None, None)
        
        with pytest.raises(DataLayerError):
            await manager.get_embedding("c0000001")
        
        with pytest.raises(DataLayerError):
            await manager.search_similar([0.1] * 384)