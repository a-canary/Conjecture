"""
Comprehensive tests for EmbeddingService in the Conjecture data layer.
Tests both real and mock implementations, similarity computation, and performance.
"""
import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock

from src.data.embedding_service import EmbeddingService, MockEmbeddingService
from src.data.models import DataLayerError


class TestEmbeddingServiceInitialization:
    """Test EmbeddingService initialization and setup."""

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_real_embedding_service_initialization(self, real_embedding_service: EmbeddingService):
        """Test real embedding service initialization."""
        assert real_embedding_service is not None
        assert real_embedding_service.model_name == "all-MiniLM-L6-v2"
        assert real_embedding_service.model is not None
        assert real_embedding_service._embedding_dim is not None
        assert real_embedding_service.embedding_dimension > 0

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_mock_embedding_service_initialization(self, mock_embedding_service: MockEmbeddingService):
        """Test mock embedding service initialization."""
        assert mock_embedding_service is not None
        assert mock_embedding_service.model_name == "mock-model"
        assert mock_embedding_service.embedding_dim == 384
        assert mock_embedding_service.call_count == 0

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_embedding_service_close(self, real_embedding_service: EmbeddingService):
        """Test that embedding service can be closed properly."""
        await real_embedding_service.close()
        # After closing, executor should be shut down
        assert real_embedding_service.executor is None

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_real_model_initialization_error(self):
        """Test error handling during real model initialization."""
        with patch('src.data.embedding_service.SentenceTransformer') as mock_transformer:
            mock_transformer.side_effect = Exception("Model loading failed")
            
            service = EmbeddingService("invalid-model")
            
            with pytest.raises(DataLayerError, match="Failed to initialize embedding model"):
                await service.initialize()


class TestEmbeddingServiceBasicOperations:
    """Test basic embedding generation operations."""

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_real_generate_single_embedding(self, real_embedding_service: EmbeddingService):
        """Test generating a single embedding with real service."""
        text = "This is a test sentence for embedding generation."
        
        embedding = await real_embedding_service.generate_embedding(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)
        
        # Check it's a reasonable embedding size (typical for sentence-transformers)
        assert 300 <= len(embedding) <= 800

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_mock_generate_single_embedding(self, mock_embedding_service: MockEmbeddingService):
        """Test generating a single embedding with mock service."""
        text1 = "Test sentence one"
        text2 = "Test sentence two"
        
        embedding1 = await mock_embedding_service.generate_embedding(text1)
        embedding2 = await mock_embedding_service.generate_embedding(text2)
        
        assert isinstance(embedding1, list)
        assert len(embedding1) == 384
        assert all(isinstance(x, (int, float)) for x in embedding1)
        
        # Mock should produce consistent embeddings for same text
        embedding1_again = await mock_embedding_service.generate_embedding(text1)
        assert embedding1 == embedding1_again
        
        # Different texts should produce different embeddings
        assert embedding1 != embedding2
        
        # Call count should increment
        assert mock_embedding_service.call_count == 3

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_real_generate_batch_embeddings(self, real_embedding_service: EmbeddingService):
        """Test generating batch embeddings with real service."""
        texts = [
            "First test sentence",
            "Second test sentence",
            "Third test sentence"
        ]
        
        embeddings = await real_embedding_service.generate_batch_embeddings(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(x, (int, float)) for x in embedding)
        
        # All embeddings should have same dimension
        dimensions = [len(emb) for emb in embeddings]
        assert len(set(dimensions)) == 1

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_mock_generate_batch_embeddings(self, mock_embedding_service: MockEmbeddingService):
        """Test generating batch embeddings with mock service."""
        texts = ["Sentence 1", "Sentence 2", "Sentence 3", "Sentence 4", "Sentence 5"]
        
        embeddings = await mock_embedding_service.generate_batch_embeddings(texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        
        for i, embedding in enumerate(embeddings):
            assert len(embedding) == 384
            # Mock should be deterministic
            single_emb = await mock_embedding_service.generate_embedding(texts[i])
            assert embedding == single_emb

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_generate_empty_batch(self, mock_embedding_service: MockEmbeddingService):
        """Test generating embeddings for empty list."""
        embeddings = await mock_embedding_service.generate_batch_embeddings([])
        assert embeddings == []

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_generate_not_initialized(self):
        """Test generating embedding when service not initialized."""
        service = EmbeddingService("test-model")
        
        with pytest.raises(DataLayerError, match="Embedding service not initialized"):
            await service.generate_embedding("test")


class TestEmbeddingServiceSimilarity:
    """Test similarity computation functionality."""

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_compute_similarity_real_embeddings(self, real_embedding_service: EmbeddingService):
        """Test computing similarity with real embeddings."""
        text1 = "The cat sat on the mat"
        text2 = "A cat was sitting on a mat"
        text3 = "The airplane flew across the sky"
        
        emb1 = await real_embedding_service.generate_embedding(text1)
        emb2 = await real_embedding_service.generate_embedding(text2)
        emb3 = await real_embedding_service.generate_embedding(text3)
        
        # Similar sentences should have high similarity
        sim_12 = await real_embedding_service.compute_similarity(emb1, emb2)
        
        # Different sentences should have lower similarity
        sim_13 = await real_embedding_service.compute_similarity(emb1, emb3)
        
        assert isinstance(sim_12, float)
        assert isinstance(sim_13, float)
        assert -1.0 <= sim_12 <= 1.0
        assert -1.0 <= sim_13 <= 1.0
        
        # Similar sentences should be more similar
        assert sim_12 > sim_13

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_compute_similarity_mock_embeddings(self, mock_embedding_service: MockEmbeddingService):
        """Test computing similarity with mock embeddings."""
        text1 = "Similar text about cats"
        text2 = "Similar text about felines"
        text3 = "Completely different text about physics"
        
        emb1 = await mock_embedding_service.generate_embedding(text1)
        emb2 = await mock_embedding_service.generate_embedding(text2)
        emb3 = await mock_embedding_service.generate_embedding(text3)
        
        sim_12 = await mock_embedding_service.compute_similarity(emb1, emb2)
        sim_13 = await mock_embedding_service.compute_similarity(emb1, emb3)
        
        assert isinstance(sim_12, float)
        assert isinstance(sim_13, float)
        assert -1.0 <= sim_12 <= 1.0
        assert -1.0 <= sim_13 <= 1.0

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_find_most_similar(self, mock_embedding_service: MockEmbeddingService):
        """Test finding most similar embedding from candidates."""
        query_text = "Find the most similar to this text about animals"
        candidate_texts = [
            "This text is about dogs and pets",
            "This text is about programming languages",
            "This text is about cooking recipes",
            "This text is about cats and felines"
        ]
        
        query_embedding = await mock_embedding_service.generate_embedding(query_text)
        candidate_embeddings = await mock_embedding_service.generate_batch_embeddings(candidate_texts)
        
        most_similar_index = await mock_embedding_service.find_most_similar(query_embedding, candidate_embeddings)
        
        assert isinstance(most_similar_index, int)
        assert 0 <= most_similar_index < len(candidate_texts)
        
        # The most similar should be one of the animal-related texts
        most_similar_text = candidate_texts[most_similar_index]
        assert "dogs" in most_similar_text or "cats" in most_similar_text

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_find_most_similar_empty_candidates(self, mock_embedding_service: MockEmbeddingService):
        """Test finding most similar with empty candidate list."""
        query_embedding = await mock_embedding_service.generate_embedding("query")
        
        result = await mock_embedding_service.find_most_similar(query_embedding, [])
        
        assert result == -1

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_similarity_edge_cases(self, mock_embedding_service: MockEmbeddingService):
        """Test similarity computation edge cases."""
        # Zero vectors
        zero_vector1 = [0.0] * 384
        zero_vector2 = [0.0] * 384
        
        similarity = await mock_embedding_service.compute_similarity(zero_vector1, zero_vector2)
        assert similarity == 0.0
        
        # Unit vectors
        unit_vector1 = [1.0] + [0.0] * 383
        unit_vector2 = [0.0, 1.0] + [0.0] * 382
        
        similarity = await mock_embedding_service.compute_similarity(unit_vector1, unit_vector2)
        assert similarity == 0.0
        
        # Identical vectors
        similarity = await mock_embedding_service.compute_similarity(unit_vector1, unit_vector1)
        assert abs(similarity - 1.0) < 1e-10


class TestEmbeddingServiceModelInfo:
    """Test model information and properties."""

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_real_embedding_dimension(self, real_embedding_service: EmbeddingService):
        """Test getting embedding dimension from real service."""
        dimension = real_embedding_service.embedding_dimension
        assert isinstance(dimension, int)
        assert dimension > 0
        
        # Should match actual model dimension
        text = "Test"
        embedding = await real_embedding_service.generate_embedding(text)
        assert len(embedding) == dimension

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_mock_embedding_dimension(self, mock_embedding_service: MockEmbeddingService):
        """Test getting embedding dimension from mock service."""
        dimension = mock_embedding_service.embedding_dimension
        assert dimension == 384

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_get_real_model_info(self, real_embedding_service: EmbeddingService):
        """Test getting model info from real service."""
        info = real_embedding_service.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "embedding_dimension" in info
        assert "initialized" in info
        
        assert info["model_name"] == "all-MiniLM-L6-v2"
        assert info["embedding_dimension"] > 0
        assert info["initialized"] is True

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_get_mock_model_info(self, mock_embedding_service: MockEmbeddingService):
        """Test getting model info from mock service."""
        info = mock_embedding_service.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "embedding_dimension" in info
        assert "initialized" in info
        assert "call_count" in info
        
        assert info["model_name"] == "mock-model"
        assert info["embedding_dimension"] == 384
        assert info["initialized"] is True
        assert info["call_count"] >= 0


class TestEmbeddingServicePerformance:
    """Performance tests for embedding operations."""

    @pytest.mark.embeddings
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_embedding_performance_real(self, real_embedding_service: EmbeddingService, benchmark):
        """Benchmark single embedding generation with real service."""
        text = "Performance test sentence for embedding generation."
        
        async def generate_embedding():
            return await real_embedding_service.generate_embedding(text)
        
        result = await benchmark.async_timer(generate_embedding)
        # Real models might be slower, but should be reasonable (<500ms)
        assert result < 0.5  # 500ms
        assert len(result["result"]) > 0

    @pytest.mark.embeddings
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_embedding_performance_mock(self, mock_embedding_service: MockEmbeddingService, benchmark):
        """Benchmark single embedding generation with mock service."""
        text = "Mock performance test sentence."
        
        async def generate_embedding():
            return await mock_embedding_service.generate_embedding(text)
        
        result = await benchmark.async_timer(generate_embedding)
        # Mock should be very fast (<10ms)
        assert result < 0.01  # 10ms
        assert len(result["result"]) == 384

    @pytest.mark.embeddings
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_embedding_performance_real(self, real_embedding_service: EmbeddingService, benchmark):
        """Benchmark batch embedding generation with real service."""
        texts = [f"Batch test sentence {i}" for i in range(10)]
        
        async def generate_batch_embeddings():
            return await real_embedding_service.generate_batch_embeddings(texts)
        
        result = await benchmark.async_timer(generate_batch_embeddings)
        # Batch should be more efficient than individual calls (<2s for 10 texts)
        assert result < 2.0  # 2s
        assert len(result["result"]) == 10

    @pytest.mark.embeddings
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_embedding_performance_mock(self, mock_embedding_service: MockEmbeddingService, benchmark):
        """Benchmark batch embedding generation with mock service."""
        texts = [f"Mock batch test sentence {i}" for i in range(100)]
        
        async def generate_batch_embeddings():
            return await mock_embedding_service.generate_batch_embeddings(texts)
        
        result = await benchmark.async_timer(generate_batch_embeddings)
        # Mock batch should be very fast (<50ms for 100 texts)
        assert result < 0.05  # 50ms
        assert len(result["result"]) == 100

    @pytest.mark.embeddings
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_similarity_computation_performance(self, mock_embedding_service: MockEmbeddingService, benchmark):
        """Benchmark similarity computation."""
        embedding1 = await mock_embedding_service.generate_embedding("Text 1")
        embedding2 = await mock_embedding_service.generate_embedding("Text 2")
        
        async def compute_similarity():
            return await mock_embedding_service.compute_similarity(embedding1, embedding2)
        
        result = await benchmark.async_timer(compute_similarity)
        # Similarity computation should be very fast (<1ms)
        assert result < 0.001  # 1ms
        assert isinstance(result["result"], float)

    @pytest.mark.embeddings
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_most_similar_search_performance(self, mock_embedding_service: MockEmbeddingService, benchmark):
        """Benchmark most similar search."""
        query_embedding = await mock_embedding_service.generate_embedding("Query")
        candidate_texts = [f"Candidate text {i}" for i in range(1000)]
        candidate_embeddings = await mock_embedding_service.generate_batch_embeddings(candidate_texts)
        
        async def find_most_similar():
            return await mock_embedding_service.find_most_similar(query_embedding, candidate_embeddings)
        
        result = await benchmark.async_timer(find_most_similar)
        # Should scale reasonably (<100ms for 1000 candidates)
        assert result < 0.1  # 100ms
        assert 0 <= result["result"] < len(candidate_embeddings)


class TestEmbeddingServiceScalability:
    """Scalability tests with larger datasets."""

    @pytest.mark.embeddings
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_batch_processing(self, mock_embedding_service: MockEmbeddingService):
        """Test processing large batches efficiently."""
        batch_sizes = [100, 500, 1000, 2000]
        
        for batch_size in batch_sizes:
            texts = [f"Large batch test text {i}" for i in range(batch_size)]
            
            # Process batch
            embeddings = await mock_embedding_service.generate_batch_embeddings(texts)
            
            assert len(embeddings) == batch_size
            
            # All embeddings should have correct dimension
            for embedding in embeddings:
                assert len(embedding) == 384
            
            # All embeddings should be different
            unique_embeddings = set(tuple(emb) for emb in embeddings[:10])  # Check first 10
            assert len(unique_embeddings) == 10

    @pytest.mark.embeddings
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_usage_scaling(self, mock_embedding_service: MockEmbeddingService):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Generate many embeddings
        for i in range(10):
            texts = [f"Memory test text {i}_{j}" for j in range(100)]
            embeddings = await mock_embedding_service.generate_batch_embeddings(texts)
            del embeddings  # Force cleanup
        
        # Check memory growth
        current_memory = process.memory_info().rss
        memory_growth_mb = (current_memory - initial_memory) / 1024 / 1024
        
        # Should not grow more than 50MB for 1000 embeddings
        assert memory_growth_mb < 50

    @pytest.mark.embeddings
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self, mock_embedding_service: MockEmbeddingService):
        """Test concurrent embedding generation."""
        import asyncio
        
        async def generate_batch(start_id: int, count: int):
            texts = [f"Concurrent test {start_id}_{i}" for i in range(count)]
            return await mock_embedding_service.generate_batch_embeddings(texts)
        
        # Run concurrent batches
        tasks = [
            generate_batch(1000, 50),
            generate_batch(2000, 50),
            generate_batch(3000, 50),
            generate_batch(4000, 50)
        ]
        
        results = await asyncio.gather(*tasks)
        
        total_embeddings = sum(len(batch) for batch in results)
        assert total_embeddings == 200
        
        # All embeddings should be valid
        for batch in results:
            for embedding in batch:
                assert len(embedding) == 384

    @pytest.mark.embeddings
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_real_model_scalability(self, real_embedding_service: EmbeddingService):
        """Test real model performance with scaling."""
        import time
        
        batch_sizes = [1, 10, 50]
        times = []
        
        for batch_size in batch_sizes:
            texts = [f"Scalability test text {i}" for i in range(batch_size)]
            
            start_time = time.time()
            embeddings = await real_embedding_service.generate_batch_embeddings(texts)
            end_time = time.time()
            
            processing_time = end_time - start_time
            times.append(processing_time)
            
            assert len(embeddings) == batch_size
            
            # Check efficiency (should be better than linear scaling)
            if len(times) >= 2:
                # Time per embedding should decrease with batch size
                time_per_embedding_current = times[-1] / batch_sizes[-1]
                time_per_embedding_previous = times[-2] / batch_sizes[-2]
                assert time_per_embedding_current <= time_per_embedding_previous * 1.2  # Allow some overhead


class TestEmbeddingServiceIntegration:
    """Integration tests for embedding service usage patterns."""

    @pytest.mark.embeddings
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_embedding_workflow(self, mock_embedding_service: MockEmbeddingService):
        """Test complete embedding workflow."""
        # 1. Generate embeddings for a document corpus
        documents = [
            "Document 1: Introduction to machine learning",
            "Document 2: Deep learning and neural networks",
            "Document 3: Natural language processing techniques",
            "Document 4: Computer vision and image recognition",
            "Document 5: Reinforcement learning fundamentals"
        ]
        
        document_embeddings = await mock_embedding_service.generate_batch_embeddings(documents)
        assert len(document_embeddings) == len(documents)
        
        # 2. Process a user query
        user_query = "What are the basics of deep learning?"
        query_embedding = await mock_embedding_service.generate_embedding(user_query)
        
        # 3. Find most similar documents
        most_similar_index = await mock_embedding_service.find_most_similar(
            query_embedding, document_embeddings
        )
        
        assert 0 <= most_similar_index < len(documents)
        most_similar_doc = documents[most_similar_index]
        
        # Should find document 2 (deep learning)
        assert "deep learning" in most_similar_doc.lower()
        
        # 4. Compute similarity scores for all documents
        similarities = []
        for doc_embedding in document_embeddings:
            similarity = await mock_embedding_service.compute_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        assert len(similarities) == len(documents)
        assert all(isinstance(sim, float) for sim in similarities)
        
        # Most similar document should have highest similarity
        highest_sim_index = similarities.index(max(similarities))
        assert highest_sim_index == most_similar_index

    @pytest.mark.embeddings
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_semantic_search_simulation(self, mock_embedding_service: MockEmbeddingService):
        """Test semantic search use case."""
        # Create a knowledge base
        knowledge_base = [
            "Python is a high-level programming language",
            "JavaScript is used for web development",
            "Machine learning uses statistical methods",
            "Databases store and manage data",
            "Neural networks are inspired by the brain",
            "React is a JavaScript library",
            "Scikit-learn is a Python ML library",
            "SQL is used for database queries"
        ]
        
        # Generate embeddings for knowledge base
        kb_embeddings = await mock_embedding_service.generate_batch_embeddings(knowledge_base)
        
        # Test different query scenarios
        query_scenarios = [
            ("programming languages", ["Python", "JavaScript"]),
            ("machine learning", ["Machine learning", "Neural networks", "Scikit-learn"]),
            ("databases", ["Databases", "SQL"]),
            ("web development", ["JavaScript", "React"])
        ]
        
        for query, expected_keywords in query_scenarios:
            query_embedding = await mock_embedding_service.generate_embedding(query)
            
            # Find top 3 most similar
            most_similar_index = await mock_embedding_service.find_most_similar(
                query_embedding, kb_embeddings
            )
            
            # Verify result makes semantic sense
            assert 0 <= most_similar_index < len(knowledge_base)
            result_text = knowledge_base[most_similar_index]
            
            # Should contain at least one expected keyword
            assert any(keyword.lower() in result_text.lower() for keyword in [kw.lower() for kw in expected_keywords])

    @pytest.mark.embeddings
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_consistency(self, mock_embedding_service: MockEmbeddingService):
        """Test embedding consistency across multiple calls."""
        text = "Consistency test text"
        
        # Generate embedding multiple times
        embeddings = []
        for _ in range(10):
            embedding = await mock_embedding_service.generate_embedding(text)
            embeddings.append(embedding)
        
        # All should be identical (deterministic)
        for i in range(1, len(embeddings)):
            assert embeddings[i] == embeddings[0]
        
        # Batch generation should also be consistent
        batch_embeddings = []
        for _ in range(5):
            batch = await mock_embedding_service.generate_batch_embeddings([text])
            batch_embeddings.append(batch[0])
        
        for batch_emb in batch_embeddings:
            assert batch_emb == embeddings[0]

    @pytest.mark.embeddings
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_quality_validation(self, real_embedding_service: EmbeddingService):
        """Test basic quality validation for real embeddings."""
        if not real_embedding_service.model:
            pytest.skip("Real embedding model not available")
        
        # Test that similar sentences have higher similarity
        similar_pairs = [
            ("The cat sat on the mat", "A cat was sitting on the mat"),
            ("Machine learning is powerful", "ML is a powerful technique"),
            ("Python is easy to learn", "Learning Python is straightforward")
        ]
        
        dissimilar_pairs = [
            ("The cat sat on the mat", "Stock prices fluctuate daily"),
            ("Machine learning is powerful", "Cooking pasta requires boiling water"),
            ("Python is easy to learn", "Mountains are formed by tectonic activity")
        ]
        
        similar_similarities = []
        dissimilar_similarities = []
        
        # Compute similarities for similar pairs
        for text1, text2 in similar_pairs:
            emb1 = await real_embedding_service.generate_embedding(text1)
            emb2 = await real_embedding_service.generate_embedding(text2)
            sim = await real_embedding_service.compute_similarity(emb1, emb2)
            similar_similarities.append(sim)
        
        # Compute similarities for dissimilar pairs
        for text1, text2 in dissimilar_pairs:
            emb1 = await real_embedding_service.generate_embedding(text1)
            emb2 = await real_embedding_service.generate_embedding(text2)
            sim = await real_embedding_service.compute_similarity(emb1, emb2)
            dissimilar_similarities.append(sim)
        
        # Similar pairs should have higher average similarity
        avg_similar = np.mean(similar_similarities)
        avg_dissimilar = np.mean(dissimilar_similarities)
        
        assert avg_similar > avg_dissimilar
        assert avg_similar > 0.5  # Should be at least somewhat similar
        assert avg_dissimilar < 0.5  # Should be less similar


class TestEmbeddingServiceErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.embeddings
    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_generate_embedding_error(self, real_embedding_service: EmbeddingService):
        """Test error handling during embedding generation."""
        with patch.object(real_embedding_service.model, 'encode') as mock_encode:
            mock_encode.side_effect = Exception("Model inference failed")
            
            with pytest.raises(DataLayerError, match="Failed to generate embedding"):
                await real_embedding_service.generate_embedding("test")

    @pytest.mark.embeddings
    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_batch_embedding_generation_error(self, real_embedding_service: EmbeddingService):
        """Test error handling during batch embedding generation."""
        with patch.object(real_embedding_service.model, 'encode') as mock_encode:
            mock_encode.side_effect = Exception("Batch model inference failed")
            
            with pytest.raises(DataLayerError, match="Failed to generate batch embeddings"):
                await real_embedding_service.generate_batch_embeddings(["test1", "test2"])

    @pytest.mark.embeddings
    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_similarity_computation_error(self):
        """Test similarity computation with malformed embeddings."""
        service = MockEmbeddingService()
        await service.initialize()
        
        # Different length embeddings
        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [1.0, 2.0]  # Different length
        
        similarity = await service.compute_similarity(embedding1, embedding2)
        assert isinstance(similarity, float)

    @pytest.mark.embeddings
    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_empty_and_invalid_inputs(self, mock_embedding_service: MockEmbeddingService):
        """Test handling of empty and invalid inputs."""
        # Empty string
        embedding = await mock_embedding_service.generate_embedding("")
        assert len(embedding) == 384
        
        # Very long text
        long_text = "A" * 10000
        embedding = await mock_embedding_service.generate_embedding(long_text)
        assert len(embedding) == 384
        
        # Special characters
        special_text = "特殊字符 🚀 émojis\n\t\v\f\r"
        embedding = await mock_embedding_service.generate_embedding(special_text)
        assert len(embedding) == 384

    @pytest.mark.embeddings
    @pytest.mark.error_handling
    @pytest.mark.asyncio
    async def test_executor_shutdown_handling(self, real_embedding_service: EmbeddingService):
        """Test behavior when executor is shut down."""
        # Close the service
        await real_embedding_service.close()
        
        # Try to use it after closing
        with pytest.raises(DataLayerError):
            await real_embedding_service.generate_embedding("test after close")


# Test with different model configurations
class TestEmbeddingServiceModelVariants:
    """Test embedding service with different model configurations."""

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_custom_mock_dimensions(self):
        """Test mock embedding service with custom dimensions."""
        custom_service = MockEmbeddingService(embedding_dim=768)
        await custom_service.initialize()
        
        embedding = await custom_service.generate_embedding("test")
        assert len(embedding) == 768
        
        info = custom_service.get_model_info()
        assert info["embedding_dimension"] == 768

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_model_dimension_defaults(self):
        """Test default model dimensions for different model names."""
        test_cases = [
            ("all-MiniLM-L6-v2", 384),
            ("all-mpnet-base-v2", 768),
            ("multi-qa-mpnet-base-dot-v1", 768),
            ("unknown-model", 384),  # Default fallback
        ]
        
        for model_name, expected_dim in test_cases:
            service = EmbeddingService(model_name)
            assert service.embedding_dimension == expected_dim

    @pytest.mark.embeddings
    @pytest.mark.asyncio
    async def test_service_with_different_configs(self):
        """Test creating services with different configurations."""
        configs = [
            {"model_name": "model1"},
            {"model_name": "model2", "embedding_dim": 512},
        ]
        
        services = []
        for config in configs:
            if "embedding_dim" in config:
                service = MockEmbeddingService(
                    model_name=config["model_name"],
                    embedding_dim=config["embedding_dim"]
                )
            else:
                service = EmbeddingService(config["model_name"])
            
            await service.initialize()
            services.append(service)
            
            info = service.get_model_info()
            assert info["model_name"] == config["model_name"]
            assert info["initialized"] is True
        
        # Clean up
        for service in services:
            await service.close()