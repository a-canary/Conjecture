"""
Unit tests for EmbeddingService.
"""

import pytest
import pytest_asyncio
import asyncio
import numpy as np

from src.data.embedding_service import EmbeddingService, MockEmbeddingService
from src.data.models import DataLayerError


class TestEmbeddingService:
    """Test EmbeddingService functionality."""

    @pytest_asyncio.fixture
    async def embedding_service(self):
        """Create an embedding service for testing."""
        service = EmbeddingService("all-MiniLM-L6-v2")
        await service.initialize()
        yield service
        await service.close()

    @pytest.mark.asyncio
    async def test_initialization(self, embedding_service):
        """Test service initialization."""
        assert embedding_service.model is not None
        assert embedding_service.embedding_dimension == 384

    @pytest.mark.asyncio
    async def test_generate_embedding(self, embedding_service):
        """Test generating a single embedding."""
        text = "This is a test sentence for embedding generation."
        embedding = await embedding_service.generate_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

        # Check that embedding is normalized (approximately)
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        assert abs(norm - 1.0) < 0.1  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_generate_batch_embeddings(self, embedding_service):
        """Test generating multiple embeddings."""
        texts = ["First test sentence", "Second test sentence", "Third test sentence"]

        embeddings = await embedding_service.generate_batch_embeddings(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == 3

        for i, embedding in enumerate(embeddings):
            assert isinstance(embedding, list)
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)

            # Check normalization
            embedding_array = np.array(embedding)
            norm = np.linalg.norm(embedding_array)
            assert abs(norm - 1.0) < 0.1

    @pytest.mark.asyncio
    async def test_generate_batch_embeddings_empty(self, embedding_service):
        """Test generating embeddings for empty list."""
        embeddings = await embedding_service.generate_batch_embeddings([])
        assert embeddings == []

    @pytest.mark.asyncio
    async def test_compute_similarity(self, embedding_service):
        """Test computing similarity between embeddings."""
        # Create two similar embeddings
        text1 = "Machine learning is about algorithms"
        text2 = "Machine learning involves algorithms"
        text3 = "The weather is nice today"

        emb1 = await embedding_service.generate_embedding(text1)
        emb2 = await embedding_service.generate_embedding(text2)
        emb3 = await embedding_service.generate_embedding(text3)

        # Similar texts should have high similarity
        sim_12 = await embedding_service.compute_similarity(emb1, emb2)
        sim_13 = await embedding_service.compute_similarity(emb1, emb3)

        assert isinstance(sim_12, float)
        assert isinstance(sim_13, float)
        assert 0.0 <= sim_12 <= 1.0
        assert 0.0 <= sim_13 <= 1.0

        # Similar texts should be more similar than dissimilar ones
        assert sim_12 > sim_13

    @pytest.mark.asyncio
    async def test_compute_similarity_identical(self, embedding_service):
        """Test similarity of identical embeddings."""
        text = "Test sentence"
        embedding = await embedding_service.generate_embedding(text)

        similarity = await embedding_service.compute_similarity(embedding, embedding)
        assert abs(similarity - 1.0) < 1e-6  # Should be exactly 1.0

    @pytest.mark.asyncio
    async def test_compute_similarity_zero_vectors(self, embedding_service):
        """Test similarity with zero vectors."""
        zero_vector = [0.0] * 384
        normal_vector = [0.1] * 384

        # Normalize normal vector
        norm = np.linalg.norm(normal_vector)
        normal_vector = [x / norm for x in normal_vector]

        similarity = await embedding_service.compute_similarity(
            zero_vector, normal_vector
        )
        assert similarity == 0.0

    @pytest.mark.asyncio
    async def test_find_most_similar(self, embedding_service):
        """Test finding most similar embedding."""
        query_text = "Machine learning algorithms"
        candidates = [
            "Deep learning networks",
            "Python programming",
            "Algorithm design",
        ]

        query_embedding = await embedding_service.generate_embedding(query_text)
        candidate_embeddings = [
            await embedding_service.generate_embedding(text) for text in candidates
        ]

        most_similar_idx = await embedding_service.find_most_similar(
            query_embedding, candidate_embeddings
        )

        assert isinstance(most_similar_idx, int)
        assert 0 <= most_similar_idx < len(candidates)

        # The most similar should be "Deep learning networks" (most related to ML)
        assert most_similar_idx == 0

    @pytest.mark.asyncio
    async def test_find_most_similar_empty(self, embedding_service):
        """Test finding most similar with empty candidates."""
        query_embedding = [0.1] * 384
        candidate_embeddings = []

        most_similar_idx = await embedding_service.find_most_similar(
            query_embedding, candidate_embeddings
        )

        assert most_similar_idx == -1

    @pytest.mark.asyncio
    async def test_get_model_info(self, embedding_service):
        """Test getting model information."""
        info = embedding_service.get_model_info()

        assert isinstance(info, dict)
        assert info["model_name"] == "all-MiniLM-L6-v2"
        assert info["embedding_dimension"] == 384
        assert info["initialized"] is True

    @pytest.mark.asyncio
    async def test_error_handling_not_initialized(self):
        """Test error handling when service is not initialized."""
        service = EmbeddingService("all-MiniLM-L6-v2")
        # Don't initialize

        with pytest.raises(DataLayerError):
            await service.generate_embedding("test")

        with pytest.raises(DataLayerError):
            await service.generate_batch_embeddings(["test1", "test2"])


class TestMockEmbeddingService:
    """Test MockEmbeddingService functionality."""

    @pytest_asyncio.fixture
    async def mock_service(self):
        """Create a mock embedding service for testing."""
        service = MockEmbeddingService("mock-model", 384)
        await service.initialize()
        yield service
        await service.close()

    @pytest.mark.asyncio
    async def test_mock_initialization(self, mock_service):
        """Test mock service initialization."""
        assert mock_service.model_name == "mock-model"
        assert mock_service.embedding_dim == 384
        assert mock_service.call_count == 0

    @pytest.mark.asyncio
    async def test_mock_generate_embedding(self, mock_service):
        """Test generating mock embeddings."""
        text = "Test text"
        embedding = await mock_service.generate_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
        assert mock_service.call_count == 1

        # Check that embedding is normalized
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        assert abs(norm - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_mock_deterministic_embeddings(self, mock_service):
        """Test that mock embeddings are deterministic."""
        text = "Test text"

        # Generate embedding twice
        emb1 = await mock_service.generate_embedding(text)
        emb2 = await mock_service.generate_embedding(text)

        # Should be identical
        assert emb1 == emb2
        assert mock_service.call_count == 2

    @pytest.mark.asyncio
    async def test_mock_different_texts_different_embeddings(self, mock_service):
        """Test that different texts produce different embeddings."""
        text1 = "First text"
        text2 = "Second text"

        emb1 = await mock_service.generate_embedding(text1)
        emb2 = await mock_service.generate_embedding(text2)

        # Should be different
        assert emb1 != emb2
        assert mock_service.call_count == 2

    @pytest.mark.asyncio
    async def test_mock_batch_embeddings(self, mock_service):
        """Test batch generation of mock embeddings."""
        texts = ["Text 1", "Text 2", "Text 3"]

        embeddings = await mock_service.generate_batch_embeddings(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
        assert mock_service.call_count == 3

    @pytest.mark.asyncio
    async def test_mock_similarity_computation(self, mock_service):
        """Test mock similarity computation."""
        # Create embeddings
        emb1 = await mock_service.generate_embedding("Text 1")
        emb2 = await mock_service.generate_embedding("Text 2")

        similarity = await mock_service.compute_similarity(emb1, emb2)

        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    @pytest.mark.asyncio
    async def test_mock_find_most_similar(self, mock_service):
        """Test finding most similar with mock service."""
        query_embedding = [0.1] * 384
        candidate_embeddings = [[0.2] * 384, [0.8] * 384, [0.3] * 384]

        most_similar_idx = await mock_service.find_most_similar(
            query_embedding, candidate_embeddings
        )

        assert isinstance(most_similar_idx, int)
        assert 0 <= most_similar_idx < len(candidate_embeddings)

    @pytest.mark.asyncio
    async def test_mock_get_model_info(self, mock_service):
        """Test getting mock model info."""
        info = mock_service.get_model_info()

        assert isinstance(info, dict)
        assert info["model_name"] == "mock-model"
        assert info["embedding_dimension"] == 384
        assert info["initialized"] is True
        assert info["call_count"] == mock_service.call_count

    @pytest.mark.asyncio
    async def test_mock_custom_dimensions(self):
        """Test mock service with custom dimensions."""
        service = MockEmbeddingService("custom-model", 768)
        await service.initialize()

        embedding = await service.generate_embedding("test")

        assert len(embedding) == 768
        assert service.embedding_dimension == 768

        await service.close()
