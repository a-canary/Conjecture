"""
Comprehensive unit tests for ChromaManager in the Conjecture data layer.
Tests vector operations, similarity search, batch operations, and error handling.
"""
import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from src.data.chroma_manager import ChromaManager
from src.data.models import Claim, DataLayerError


class TestChromaManagerInitialization:
    """Test ChromaManager initialization and setup."""

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_chroma_manager_initialization(self, chroma_manager: ChromaManager):
        """Test that ChromaManager initializes correctly."""
        assert chroma_manager is not None
        assert chroma_manager.client is not None
        assert chroma_manager.collection is not None
        assert chroma_manager.chroma_path != ""

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_collection_creation(self, chroma_manager: ChromaManager):
        """Test that collection is created with proper metadata."""
        collection = chroma_manager.collection
        
        assert collection is not None
        assert collection.name == "claims"
        
        # Check collection metadata
        metadata = collection.metadata
        assert metadata is not None
        assert "description" in metadata
        assert "created_at" in metadata

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_close_connection(self, chroma_manager: ChromaManager):
        """Test that ChromaManager can be closed properly."""
        await chroma_manager.close()
        assert chroma_manager.client is None
        assert chroma_manager.collection is None

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_initialization_error_handling(self):
        """Test error handling during initialization."""
        # Test with invalid path
        invalid_manager = ChromaManager("")
        
        with pytest.raises(DataLayerError, match="Failed to initialize ChromaDB"):
            await invalid_manager.initialize()


class TestChromaManagerEmbeddingOperations:
    """Test embedding CRUD operations."""

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_add_embedding(self, chroma_manager: ChromaManager, valid_claim: Claim, embedding_generator):
        """Test adding a single embedding."""
        embedding = embedding_generator.generate(1)[0]
        
        # Add embedding
        await chroma_manager.add_embedding(valid_claim, embedding)
        
        # Verify embedding was added
        retrieved = await chroma_manager.get_embedding(valid_claim.id)
        assert retrieved is not None
        assert retrieved['id'] == valid_claim.id
        assert retrieved['document'] == valid_claim.content
        assert len(retrieved['embedding']) > 0

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_add_embedding_not_initialized(self, valid_claim: Claim, embedding_generator):
        """Test adding embedding when ChromaDB not initialized."""
        manager = ChromaManager("/invalid/path")
        embedding = embedding_generator.generate(1)[0]
        
        with pytest.raises(DataLayerError, match="ChromaDB not initialized"):
            await manager.add_embedding(valid_claim, embedding)

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_get_embedding_existing(self, chroma_manager: ChromaManager, valid_claim: Claim, embedding_generator):
        """Test retrieving an existing embedding."""
        embedding = embedding_generator.generate(1)[0]
        await chroma_manager.add_embedding(valid_claim, embedding)
        
        retrieved = await chroma_manager.get_embedding(valid_claim.id)
        
        assert retrieved is not None
        assert retrieved['id'] == valid_claim.id
        assert retrieved['document'] == valid_claim.content
        assert 'metadata' in retrieved
        assert 'embedding' in retrieved
        
        # Check metadata contains expected fields
        metadata = retrieved['metadata']
        assert 'confidence' in metadata
        assert 'created_by' in metadata
        assert 'created_at' in metadata
        assert 'dirty' in metadata
        assert metadata['confidence'] == valid_claim.confidence

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_get_embedding_nonexistent(self, chroma_manager: ChromaManager):
        """Test retrieving a non-existent embedding."""
        result = await chroma_manager.get_embedding("c0999999")
        assert result is None

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_update_embedding(self, chroma_manager: ChromaManager, valid_claim: Claim, embedding_generator):
        """Test updating an existing embedding."""
        # Add initial embedding
        embedding1 = embedding_generator.generate(1)[0]
        await chroma_manager.add_embedding(valid_claim, embedding1)
        
        # Update claim content
        updated_claim = Claim(
            id=valid_claim.id,
            content="Updated content for testing",
            confidence=0.95,
            dirty=False,
            tags=["updated"],
            created_by=valid_claim.created_by
        )
        
        # Update with new embedding
        embedding2 = embedding_generator.generate(1)[0]
        await chroma_manager.update_embedding(updated_claim, embedding2)
        
        # Verify update
        retrieved = await chroma_manager.get_embedding(valid_claim.id)
        assert retrieved is not None
        assert retrieved['document'] == "Updated content for testing"
        assert retrieved['metadata']['confidence'] == 0.95
        assert retrieved['metadata']['dirty'] is False

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_delete_embedding(self, chroma_manager: ChromaManager, valid_claim: Claim, embedding_generator):
        """Test deleting an embedding."""
        embedding = embedding_generator.generate(1)[0]
        await chroma_manager.add_embedding(valid_claim, embedding)
        
        # Verify embedding exists
        retrieved = await chroma_manager.get_embedding(valid_claim.id)
        assert retrieved is not None
        
        # Delete embedding
        await chroma_manager.delete_embedding(valid_claim.id)
        
        # Verify it's gone
        retrieved = await chroma_manager.get_embedding(valid_claim.id)
        assert retrieved is None

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_delete_embedding_not_initialized(self):
        """Test deleting embedding when ChromaDB not initialized."""
        manager = ChromaManager("/invalid/path")
        
        with pytest.raises(DataLayerError, match="ChromaDB not initialized"):
            await manager.delete_embedding("c0000001")


class TestChromaManagerSimilaritySearch:
    """Test similarity search functionality."""

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_search_similar_by_embedding(self, chroma_manager: ChromaManager, sample_claims_data, embedding_generator):
        """Test similarity search using embedding query."""
        # Add sample claims with embeddings
        claims = [Claim(**data) for data in sample_claims_data]
        embeddings = embedding_generator.generate(len(claims))
        
        for claim, embedding in zip(claims, embeddings):
            await chroma_manager.add_embedding(claim, embedding)
        
        # Search for similar claims
        query_embedding = embeddings[0]  # Use first claim's embedding as query
        results = await chroma_manager.search_similar(query_embedding, limit=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        
        # Check result structure
        if results:
            result = results[0]
            assert 'id' in result
            assert 'content' in result
            assert 'metadata' in result
            assert 'distance' in result
            assert result['id'] in [claim.id for claim in claims]

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_search_similar_with_filters(self, chroma_manager: ChromaManager, sample_claims_data, embedding_generator):
        """Test similarity search with metadata filters."""
        # Add sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        embeddings = embedding_generator.generate(len(claims))
        
        for claim, embedding in zip(claims, embeddings):
            await chroma_manager.add_embedding(claim, embedding)
        
        # Search with filter
        query_embedding = embeddings[0]
        where_filter = {"confidence": {"$gte": 0.9}}  # High confidence claims
        results = await chroma_manager.search_similar(query_embedding, limit=5, where=where_filter)
        
        # All results should meet filter criteria
        for result in results:
            assert result['metadata']['confidence'] >= 0.9

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_search_by_text(self, chroma_manager: ChromaManager, sample_claims_data, embedding_generator):
        """Test similarity search using text query."""
        # Add sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        embeddings = embedding_generator.generate(len(claims))
        
        for claim, embedding in zip(claims, embeddings):
            await chroma_manager.add_embedding(claim, embedding)
        
        # Search by text
        query_text = "speed of light"
        results = await chroma_manager.search_by_text(query_text, limit=5)
        
        assert isinstance(results, list)
        
        # Results should contain claims related to speed/light
        if results:
            for result in results:
                assert 'id' in result
                assert 'content' in result
                assert 'metadata' in result
                assert 'distance' in result

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_search_by_text_with_filters(self, chroma_manager: ChromaManager, sample_claims_data, embedding_generator):
        """Test text search with metadata filters."""
        # Add sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        embeddings = embedding_generator.generate(len(claims))
        
        for claim, embedding in zip(claims, embeddings):
            await chroma_manager.add_embedding(claim, embedding)
        
        # Search with tag filter
        query_text = "科学的"  # Scientific in Chinese
        where_filter = {"tags": {"$contains": "physics"}}
        results = await chroma_manager.search_by_text(query_text, limit=10, where=where_filter)
        
        # All results should have physics tag
        for result in results:
            assert "physics" in result['metadata']['tags']

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_search_empty_collection(self, chroma_manager: ChromaManager, embedding_generator):
        """Test searching empty collection."""
        query_embedding = embedding_generator.generate(1)[0]
        results = await chroma_manager.search_similar(query_embedding, limit=5)
        
        assert results == []

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_search_not_initialized(self, embedding_generator):
        """Test searching when ChromaDB not initialized."""
        manager = ChromaManager("/invalid/path")
        query_embedding = embedding_generator.generate(1)[0]
        
        with pytest.raises(DataLayerError, match="ChromaDB not initialized"):
            await manager.search_similar(query_embedding)


class TestChromaManagerBatchOperations:
    """Test batch operations for efficiency."""

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_batch_add_embeddings(self, chroma_manager: ChromaManager, sample_claims_data, embedding_generator):
        """Test adding multiple embeddings efficiently."""
        claims = [Claim(**data) for data in sample_claims_data]
        embeddings = embedding_generator.generate(len(claims))
        
        # Batch add
        await chroma_manager.batch_add_embeddings(claims, embeddings)
        
        # Verify all embeddings were added
        for claim in claims:
            retrieved = await chroma_manager.get_embedding(claim.id)
            assert retrieved is not None
            assert retrieved['id'] == claim.id

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_batch_add_mismatched_lengths(self, chroma_manager: ChromaManager, sample_claims_data, embedding_generator):
        """Test batch add with mismatched lengths raises error."""
        claims = [Claim(**data) for data in sample_claims_data]
        embeddings = embedding_generator.generate(len(claims) - 1)  # One less embedding
        
        with pytest.raises(DataLayerError, match="Claims and embeddings must have same length"):
            await chroma_manager.batch_add_embeddings(claims, embeddings)

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_batch_update_embeddings(self, chroma_manager: ChromaManager, sample_claims_data, embedding_generator):
        """Test updating multiple embeddings efficiently."""
        claims = [Claim(**data) for data in sample_claims_data]
        embeddings = embedding_generator.generate(len(claims))
        
        # Add initial embeddings
        await chroma_manager.batch_add_embeddings(claims, embeddings)
        
        # Update claims
        for claim in claims:
            claim.confidence = 0.99
            claim.dirty = False
        
        # Generate new embeddings
        new_embeddings = embedding_generator.generate(len(claims))
        
        # Batch update
        await chroma_manager.batch_update_embeddings(claims, new_embeddings)
        
        # Verify updates
        for claim in claims:
            retrieved = await chroma_manager.get_embedding(claim.id)
            assert retrieved is not None
            assert retrieved['metadata']['confidence'] == 0.99
            assert retrieved['metadata']['dirty'] is False

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_batch_delete_embeddings(self, chroma_manager: ChromaManager, sample_claims_data, embedding_generator):
        """Test deleting multiple embeddings efficiently."""
        claims = [Claim(**data) for data in sample_claims_data]
        embeddings = embedding_generator.generate(len(claims))
        
        # Add embeddings
        await chroma_manager.batch_add_embeddings(claims, embeddings)
        
        # Verify they exist
        for claim in claims:
            retrieved = await chroma_manager.get_embedding(claim.id)
            assert retrieved is not None
        
        # Batch delete
        claim_ids = [claim.id for claim in claims]
        await chroma_manager.batch_delete_embeddings(claim_ids)
        
        # Verify they're deleted
        for claim in claims:
            retrieved = await chroma_manager.get_embedding(claim.id)
            assert retrieved is None

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_batch_operations_empty_lists(self, chroma_manager: ChromaManager):
        """Test batch operations with empty lists."""
        empty_claims = []
        empty_embeddings = []
        empty_ids = []
        
        # These should not raise errors
        await chroma_manager.batch_add_embeddings(empty_claims, empty_embeddings)
        await chroma_manager.batch_update_embeddings(empty_claims, empty_embeddings)
        await chroma_manager.batch_delete_embeddings(empty_ids)


class TestChromaManagerStatistics:
    """Test statistics and utility functions."""

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_get_collection_stats(self, chroma_manager: ChromaManager):
        """Test getting collection statistics."""
        stats = await chroma_manager.get_collection_stats()
        
        assert isinstance(stats, dict)
        assert 'total_claims' in stats
        assert 'collection_name' in stats
        assert 'path' in stats
        
        assert stats['collection_name'] == "claims"
        assert stats['total_claims'] == 0  # Initially empty

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_get_collection_stats_with_data(self, chroma_manager: ChromaManager, sample_claims_data, embedding_generator):
        """Test collection stats after adding data."""
        claims = [Claim(**data) for data in sample_claims_data]
        embeddings = embedding_generator.generate(len(claims))
        
        # Add data
        await chroma_manager.batch_add_embeddings(claims, embeddings)
        
        # Check stats
        stats = await chroma_manager.get_collection_stats()
        assert stats['total_claims'] == len(claims)

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_reset_collection(self, chroma_manager: ChromaManager, sample_claims_data, embedding_generator):
        """Test resetting the entire collection."""
        # Add some data first
        claims = [Claim(**data) for data in sample_claims_data]
        embeddings = embedding_generator.generate(len(claims))
        await chroma_manager.batch_add_embeddings(claims, embeddings)
        
        # Verify data exists
        stats_before = await chroma_manager.get_collection_stats()
        assert stats_before['total_claims'] > 0
        
        # Reset collection
        await chroma_manager.reset_collection()
        
        # Verify data is gone
        stats_after = await chroma_manager.get_collection_stats()
        assert stats_after['total_claims'] == 0


class TestChromaManagerErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_add_embedding_error_propagation(self, chroma_manager: ChromaManager, valid_claim: Claim):
        """Test that embedding errors are properly propagated."""
        # Use invalid embedding (wrong type)
        invalid_embedding = "not_a_list"
        
        with pytest.raises(DataLayerError, match="Failed to add embedding"):
            await chroma_manager.add_embedding(valid_claim, invalid_embedding)

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_search_with_invalid_embedding(self, chroma_manager: ChromaManager):
        """Test search with invalid embedding type."""
        invalid_embedding = "invalid"
        
        with pytest.raises(DataLayerError, match="Similarity search failed"):
            await chroma_manager.search_similar(invalid_embedding)

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_get_embedding_not_initialized(self):
        """Test getting embedding when ChromaDB not initialized."""
        manager = ChromaManager("/invalid/path")
        
        with pytest.raises(DataLayerError, match="ChromaDB not initialized"):
            await manager.get_embedding("c0000001")

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_get_stats_not_initialized(self):
        """Test getting stats when ChromaDB not initialized."""
        manager = ChromaManager("/invalid/path")
        
        with pytest.raises(DataLayerError, match="ChromaDB not initialized"):
            await manager.get_collection_stats()

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_reset_collection_not_initialized(self):
        """Test resetting collection when ChromaDB not initialized."""
        manager = ChromaManager("/invalid/path")
        
        with pytest.raises(DataLayerError, match="ChromaDB not initialized"):
            await manager.reset_collection()


class TestChromaManagerPerformance:
    """Performance tests for ChromaDB operations."""

    @pytest.mark.chroma
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_embedding_performance(self, chroma_manager: ChromaManager, valid_claim: Claim, embedding_generator, benchmark):
        """Benchmark single embedding operations."""
        embedding = embedding_generator.generate(1)[0]
        
        async def add_and_get():
            await chroma_manager.add_embedding(valid_claim, embedding)
            return await chroma_manager.get_embedding(valid_claim.id)
        
        result = await benchmark.async_timer(add_and_get)
        # Should be fast (<100ms for single embedding)
        assert result < 0.1  # 100ms
        assert result["result"] is not None

    @pytest.mark.chroma
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_embedding_performance(self, chroma_manager: ChromaManager, performance_test_claims, embedding_generator, benchmark):
        """Benchmark batch embedding operations."""
        embeddings = embedding_generator.generate(len(performance_test_claims))
        
        async def batch_operations():
            await chroma_manager.batch_add_embeddings(performance_test_claims, embeddings)
            return await chroma_manager.get_collection_stats()
        
        result = await benchmark.async_timer(batch_operations)
        # Should be reasonable (<2s for 100 claims)
        assert result < 2.0  # 2s
        assert result["result"]["total_claims"] == len(performance_test_claims)

    @pytest.mark.chroma
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_similarity_search_performance(self, chroma_manager: ChromaManager, performance_test_claims, embedding_generator, benchmark):
        """Benchmark similarity search performance."""
        # Add data first
        embeddings = embedding_generator.generate(len(performance_test_claims))
        await chroma_manager.batch_add_embeddings(performance_test_claims, embeddings)
        
        query_embedding = embeddings[0]
        
        async def search_similar():
            return await chroma_manager.search_similar(query_embedding, limit=10)
        
        result = await benchmark.async_timer(search_similar)
        # Should be fast (<100ms for similarity search)
        assert result < 0.1  # 100ms
        assert isinstance(result["result"], list)

    @pytest.mark.chroma
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_text_search_performance(self, chroma_manager: ChromaManager, performance_test_claims, embedding_generator, benchmark):
        """Benchmark text-based search performance."""
        # Add data first
        embeddings = embedding_generator.generate(len(performance_test_claims))
        await chroma_manager.batch_add_embeddings(performance_test_claims, embeddings)
        
        query_text = "performance test claim"
        
        async def search_by_text():
            return await chroma_manager.search_by_text(query_text, limit=10)
        
        result = await benchmark.async_timer(search_by_text)
        # Should be fast (<100ms for text search)
        assert result < 0.1  # 100ms
        assert isinstance(result["result"], list)

    @pytest.mark.chroma
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, chroma_manager: ChromaManager, large_claim_dataset, embedding_generator):
        """Test performance with larger datasets."""
        import time
        
        # Create a subset for performance testing (100 claims)
        test_claims_data = large_claim_dataset[:100]
        claims = []
        
        for i, data in enumerate(test_claims_data):
            data["id"] = f"c{i+1:07d}"
            claims.append(Claim(**data))
        
        embeddings = embedding_generator.generate(len(claims))
        
        # Benchmark batch insert
        start_time = time.time()
        await chroma_manager.batch_add_embeddings(claims, embeddings)
        insert_time = time.time() - start_time
        
        # Benchmark search
        query_embedding = embeddings[0]
        start_time = time.time()
        results = await chroma_manager.search_similar(query_embedding, limit=20)
        search_time = time.time() - start_time
        
        # Performance assertions
        assert insert_time < 5.0   # Should insert 100 claims in <5s
        assert search_time < 0.2   # Should search in <200ms
        assert len(results) <= 20


class TestChromaManagerScalability:
    """Scalability tests with larger datasets."""

    @pytest.mark.chroma
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_scaling(self, chroma_manager: ChromaManager, claim_generator, embedding_generator):
        """Test memory usage scaling with dataset size."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Add data in batches and monitor memory
        batch_sizes = [50, 100, 200, 400]  # Increasing batch sizes
        
        for batch_size in batch_sizes:
            # Generate claims
            claims_data = claim_generator.generate(batch_size)
            claims = []
            
            for i, data in enumerate(claims_data):
                data["id"] = f"c{len(claims) + 1:07d}"
                claims.append(Claim(**data))
            
            # Generate embeddings
            embeddings = embedding_generator.generate(len(claims))
            
            # Add to ChromaDB
            await chroma_manager.batch_add_embeddings(claims, embeddings)
            
            # Check memory growth
            current_memory = process.memory_info().rss
            memory_growth_mb = (current_memory - initial_memory) / 1024 / 1024
            
            # Memory growth should be reasonable (<200MB for 750 claims)
            assert memory_growth_mb < 200

    @pytest.mark.chroma
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_search_performance_scaling(self, chroma_manager: ChromaManager, embedding_generator):
        """Test how search performance scales with collection size."""
        import time
        
        collection_sizes = [10, 50, 100, 200]
        search_times = []
        
        base_embedding = embedding_generator.generate(1)[0]
        
        for size in collection_sizes:
            # Add claims of this size
            claims = []
            for i in range(size):
                claim = Claim(
                    id=f"c{len(claims) + 1:07d}",
                    content=f"Performance scaling test claim {i}",
                    confidence=0.7,
                    created_by="scaling_test"
                )
                claims.append(claim)
            
            embeddings = embedding_generator.generate(len(claims))
            await chroma_manager.batch_add_embeddings(claims, embeddings)
            
            # Time a search
            start_time = time.time()
            results = await chroma_manager.search_similar(base_embedding, limit=10)
            search_time = time.time() - start_time
            
            search_times.append(search_time)
            assert len(results) <= 10
        
        # Search time shouldn't grow linearly (should be sub-linear growth)
        if len(search_times) >= 2:
            # Last search should not be >10x first search for 20x data
            assert search_times[-1] < search_times[0] * 10

    @pytest.mark.chroma
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, chroma_manager: ChromaManager, embedding_generator):
        """Test concurrent ChromaDB operations."""
        import asyncio
        
        async def add_claims_batch(start_id: int, count: int):
            """Add a batch of claims concurrently."""
            claims = []
            for i in range(count):
                claim = Claim(
                    id=f"c{start_id + i:07d}",
                    content=f"Concurrent claim {start_id + i}",
                    confidence=0.7,
                    created_by="concurrent_test"
                )
                claims.append(claim)
            
            embeddings = embedding_generator.generate(len(claims))
            await chroma_manager.batch_add_embeddings(claims, embeddings)
            return len(claims)
        
        # Run concurrent batches
        tasks = [
            add_claims_batch(1000, 50),
            add_claims_batch(1050, 50),
            add_claims_batch(1100, 50),
            add_claims_batch(1150, 50)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify all were added
        total_added = sum(results)
        assert total_added == 200
        
        stats = await chroma_manager.get_collection_stats()
        assert stats['total_claims'] == total_added


class TestChromaManagerIntegration:
    """Integration tests simulating real ChromaDB usage patterns."""

    @pytest.mark.chroma
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_embedding_lifecycle(self, chroma_manager: ChromaManager, embedding_generator):
        """Test complete lifecycle of embeddings."""
        # Create claim
        claim = Claim(
            id="c0000001",
            content="Integration test claim for embedding lifecycle",
            confidence=0.7,
            dirty=True,
            tags=["integration", "lifecycle"],
            created_by="integration_user"
        )
        
        # Add embedding
        embedding1 = embedding_generator.generate(1)[0]
        await chroma_manager.add_embedding(claim, embedding1)
        
        # Retrieve and verify
        retrieved = await chroma_manager.get_embedding(claim.id)
        assert retrieved is not None
        assert retrieved['metadata']['dirty'] is True
        
        # Update claim and embedding
        claim.confidence = 0.85
        claim.dirty = False
        embedding2 = embedding_generator.generate(1)[0]
        await chroma_manager.update_embedding(claim, embedding2)
        
        # Verify update
        updated = await chroma_manager.get_embedding(claim.id)
        assert updated['metadata']['confidence'] == 0.85
        assert updated['metadata']['dirty'] is False
        
        # Test search functionality
        search_results = await chroma_manager.search_similar(embedding2, limit=5)
        assert len(search_results) >= 1
        assert search_results[0]['id'] == claim.id
        
        # Delete embedding
        await chroma_manager.delete_embedding(claim.id)
        deleted = await chroma_manager.get_embedding(claim.id)
        assert deleted is None

    @pytest.mark.chroma
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_semantic_search_workflow(self, chroma_manager: ChromaManager, embedding_generator):
        """Test complete semantic search workflow."""
        # Create a knowledge base
        knowledge_base = [
            {"content": "Python is a high-level programming language", "tags": ["programming", "python"]},
            {"content": "Machine learning uses statistical techniques", "tags": ["AI", "ML"]},
            {"content": "Databases store structured data efficiently", "tags": ["database", "storage"]},
            {"content": "Neural networks mimic brain function", "tags": ["AI", "neural"]},
            {"content": "SQL is used for database queries", "tags": ["database", "SQL"]},
        ]
        
        # Add to ChromaDB
        claims = []
        for i, data in enumerate(knowledge_base):
            claim = Claim(
                id=f"c{i+1:07d}",
                content=data["content"],
                confidence=0.8,
                tags=data["tags"],
                created_by="knowledge_base"
            )
            claims.append(claim)
        
        embeddings = embedding_generator.generate(len(claims))
        await chroma_manager.batch_add_embeddings(claims, embeddings)
        
        # Test semantic search scenarios
        
        # Scenario 1: Find similar programming content
        query_embedding = embeddings[0]  # Python programming
        programming_results = await chroma_manager.search_similar(query_embedding, limit=3)
        assert len(programming_results) >= 1
        
        # Scenario 2: Filtered search for AI content
        ai_filter = {"tags": {"$contains": "AI"}}
        ai_results = await chroma_manager.search_similar(embeddings[1], limit=5, where=ai_filter)
        assert len(ai_results) >= 1
        for result in ai_results:
            assert "AI" in result['metadata']['tags']
        
        # Scenario 3: Text-based search
        text_results = await chroma_manager.search_by_text("database query", limit=3)
        assert len(text_results) >= 1
        for result in text_results:
            assert any(term in result['content'].lower() for term in ["database", "query"])

    @pytest.mark.chroma
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_batch_update_workflow(self, chroma_manager: ChromaManager, embedding_generator):
        """Test batch update workflow for knowledge refinement."""
        # Initial knowledge base
        initial_data = [
            {"content": "Water boils at 100C", "confidence": 0.5, "dirty": True},
            {"content": "Earth has 1 moon", "confidence": 0.6, "dirty": True},
            {"content": "Python is compiled", "confidence": 0.3, "dirty": True},
        ]
        
        # Add initial claims
        claims = []
        for i, data in enumerate(initial_data):
            claim = Claim(
                id=f"c{i+1:07d}",
                content=data["content"],
                confidence=data["confidence"],
                dirty=data["dirty"],
                created_by="initial_import"
            )
            claims.append(claim)
        
        embeddings = embedding_generator.generate(len(claims))
        await chroma_manager.batch_add_embeddings(claims, embeddings)
        
        # Knowledge refinement process
        refined_claims = []
        for claim in claims:
            # Simulate validation and refinement
            if "moon" in claim.content.lower():
                claim.confidence = 0.9  # Strong evidence
                claim.dirty = False
            elif "Python" in claim.content:
                claim.content = "Python is interpreted"  # Correction
                claim.confidence = 0.8
                claim.dirty = False
            else:
                claim.dirty = False  # Accept as is
            
            refined_claims.append(claim)
        
        # Batch update
        new_embeddings = embedding_generator.generate(len(refined_claims))
        await chroma_manager.batch_update_embeddings(refined_claims, new_embeddings)
        
        # Verify updates
        for claim in refined_claims:
            retrieved = await chroma_manager.get_embedding(claim.id)
            assert retrieved is not None
            assert retrieved['metadata']['dirty'] is False
        
        # Verify specific corrections
        python_claim = await chroma_manager.get_embedding("c0000003")
        assert "interpreted" in python_claim['document']

    @pytest.mark.chroma
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_comprehensive_error_recovery(self, chroma_manager: ChromaManager, embedding_generator):
        """Test error recovery scenarios."""
        # Add some initial data
        claims = []
        for i in range(5):
            claim = Claim(
                id=f"c{i+1:07d}",
                content=f"Error recovery test claim {i}",
                confidence=0.7,
                created_by="error_test"
            )
            claims.append(claim)
        
        embeddings = embedding_generator.generate(len(claims))
        await chroma_manager.batch_add_embeddings(claims, embeddings)
        
        initial_count = (await chroma_manager.get_collection_stats())['total_claims']
        assert initial_count == 5
        
        # Simulate partial failure during batch operation
        try:
            # Try to update with mismatched data (should fail)
            invalid_claims = claims[:3]
            invalid_embeddings = embeddings[:2]  # Mismatched length
            await chroma_manager.batch_update_embeddings(invalid_claims, invalid_embeddings)
            assert False, "Should have raised DataLayerError"
        except DataLayerError:
            pass  # Expected
        
        # Verify original data is still intact
        final_count = (await chroma_manager.get_collection_stats())['total_claims']
        assert final_count == 5  # Should be unchanged
        
        # Verify all data is still accessible
        for claim in claims:
            retrieved = await chroma_manager.get_embedding(claim.id)
            assert retrieved is not None


# Mock tests for environments without ChromaDB
class TestChromaManagerWithMocks:
    """Tests using mocks for ChromaDB operations."""

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test mocked_initialization(self):
        """Test initialization using mocked ChromaDB client."""
        with patch('src.data.chroma_manager.chromadb') as mock_chromadb:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client
            
            manager = ChromaManager("/mock/path")
            await manager.initialize()
            
            assert manager.client == mock_client
            assert manager.collection == mock_collection
            mock_chromadb.PersistentClient.assert_called_once()

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_mocked_search_results(self):
        """Test search with mocked ChromaDB results."""
        with patch('src.data.chroma_manager.chromadb') as mock_chromadb:
            # Setup mocks
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.query.return_value = {
                "ids": [["c0000001", "c0000002"]],
                "documents": [["Document 1", "Document 2"]],
                "metadatas": [[{"confidence": 0.9}, {"confidence": 0.8}]],
                "distances": [[0.1, 0.2]]
            }
            
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client
            
            # Test search
            manager = ChromaManager("/mock/path")
            await manager.initialize()
            
            query_embedding = [0.1] * 384  # Mock embedding
            results = await manager.search_similar(query_embedding, limit=10)
            
            # Verify result formatting
            assert len(results) == 2
            assert results[0]['id'] == "c0000001"
            assert results[0]['content'] == "Document 1"
            assert results[0]['distance'] == 0.1

    @pytest.mark.chroma
    @pytest.mark.asyncio
    async def test_mocked_error_scenarios(self):
        """Test error handling with mocked ChromaDB failures."""
        with patch('src.data.chroma_manager.chromadb') as mock_chromadb:
            mock_client = Mock()
            mock_collection = Mock()
            
            # Simulate ChromaDB error
            mock_collection.add.side_effect = Exception("ChromaDB internal error")
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chromadb.PersistentClient.return_value = mock_client
            
            manager = ChromaManager("/mock/path")
            await manager.initialize()
            
            claim = Claim(
                id="c0000001",
                content="Test claim",
                confidence=0.7,
                created_by="test_user"
            )
            embedding = [0.1] * 384
            
            # Should wrap ChromaDB errors in DataLayerError
            with pytest.raises(DataLayerError, match="Failed to add embedding"):
                await manager.add_embedding(claim, embedding)