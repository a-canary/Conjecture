"""
Local embedding manager using sentence-transformers.
Provides fast, lightweight local text embeddings.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import json
from pathlib import Path

# Import sentence-transformers conditionally
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class LocalEmbeddingManager:
    """
    Lightweight local embedding manager using sentence-transformers.
    Optimized for fast startup and minimal memory usage.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or self._get_default_cache_dir()
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=2)  # Reduced for lighter resource usage
        self._embedding_dim = None
        self._initialized = False
        self._model_info = {}

        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

    def _get_default_cache_dir(self) -> str:
        """Get default cache directory for models."""
        home_dir = Path.home()
        cache_dir = home_dir / ".conjecture" / "models"
        return str(cache_dir)

    async def initialize(self) -> None:
        """Initialize the embedding model with optimized loading."""
        if self._initialized:
            return

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )

        try:
            logger.info(f"Initializing local embedding model: {self.model_name}")

            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            # Configure model for minimal memory usage
            model_kwargs = {
                'cache_folder': self.cache_dir,
                'device': 'cpu'  # Force CPU for consistency
            }

            self.model = await loop.run_in_executor(
                self.executor,
                lambda: SentenceTransformer(self.model_name, **model_kwargs)
            )

            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            self._model_info = {
                "model_name": self.model_name,
                "embedding_dimension": self._embedding_dim,
                "max_seq_length": getattr(self.model, 'max_seq_length', 512),
                "cache_dir": self.cache_dir,
                "initialized": True
            }

            self._initialized = True
            logger.info(f"Local embedding model initialized. Dimension: {self._embedding_dim}")

            # Cache model info
            await self._save_model_info()

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise RuntimeError(f"Embedding initialization failed: {e}")

    async def close(self) -> None:
        """Close the embedding service and clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        self.model = None
        self._initialized = False
        logger.info("Local embedding manager closed")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not self._initialized:
            await self.initialize()

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Preprocess text (basic normalization)
            clean_text = text.strip()[:2000]  # Limit length to avoid memory issues

            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self.executor,
                self.model.encode,
                clean_text
            )

            # Convert to list and ensure type consistency
            embedding_list = embedding.astype(np.float32).tolist()
            return embedding_list

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")

    async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        if not self._initialized:
            await self.initialize()

        if not texts:
            return []

        try:
            # Clean and validate texts
            clean_texts = [text.strip()[:2000] for text in texts if text and text.strip()]
            if not clean_texts:
                return []

            all_embeddings = []

            # Process in batches to manage memory
            loop = asyncio.get_event_loop()
            for i in range(0, len(clean_texts), batch_size):
                batch_texts = clean_texts[i:i + batch_size]

                batch_embeddings = await loop.run_in_executor(
                    self.executor,
                    self.model.encode,
                    batch_texts
                )

                # Convert to list and ensure type consistency
                for embedding in batch_embeddings:
                    embedding_list = embedding.astype(np.float32).tolist()
                    all_embeddings.append(embedding_list)

            return all_embeddings

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}")

    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            # Convert to numpy arrays
            emb1 = np.array(embedding1, dtype=np.float32)
            emb2 = np.array(embedding2, dtype=np.float32)

            # Compute cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = float(dot_product / (norm1 * norm2))
            # Clamp to [0, 1] range
            similarity = max(0.0, min(1.0, similarity))
            return similarity

        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0

    async def find_most_similar(self, query_embedding: List[float],
                              candidate_embeddings: List[List[float]]) -> int:
        """Find index of most similar embedding to query."""
        if not candidate_embeddings:
            return -1

        try:
            similarities = []
            for candidate in candidate_embeddings:
                similarity = await self.compute_similarity(query_embedding, candidate)
                similarities.append(similarity)

            return int(np.argmax(similarities))
        except Exception as e:
            logger.error(f"Failed to find most similar: {e}")
            return -1

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self._embedding_dim is None:
            # Default dimension for common models
            model_dimensions = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "multi-qa-mpnet-base-dot-v1": 768
            }
            self._embedding_dim = model_dimensions.get(self.model_name, 384)
        return self._embedding_dim

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        info = self._model_info.copy()
        info['available'] = SENTENCE_TRANSFORMERS_AVAILABLE
        info['initialized'] = self._initialized
        return info

    async def _save_model_info(self) -> None:
        """Save model info to cache for faster startup."""
        try:
            info_file = Path(self.cache_dir) / "model_info.json"
            with open(info_file, 'w') as f:
                json.dump(self._model_info, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save model info: {e}")

    async def load_cached_info(self) -> bool:
        """Load cached model info to speed up startup."""
        try:
            info_file = Path(self.cache_dir) / "model_info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    cached_info = json.load(f)
                    if cached_info.get('model_name') == self.model_name:
                        self._model_info = cached_info
                        self._embedding_dim = cached_info.get('embedding_dimension', 384)
                        logger.info("Loaded cached model info")
                        return True
        except Exception as e:
            logger.warning(f"Failed to load cached model info: {e}")
        return False

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the embedding service."""
        health = {
            'service': 'local_embeddings',
            'status': 'healthy' if self._initialized else 'unhealthy',
            'model_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'initialized': self._initialized,
            'cache_dir': self.cache_dir
        }

        if self._initialized:
            try:
                # Test with a simple embedding
                test_embedding = await self.generate_embedding("test")
                health['test_embedding'] = len(test_embedding) == self.embedding_dimension
            except Exception as e:
                health['status'] = 'degraded'
                health['error'] = str(e)

        return health


class MockEmbeddingManager:
    """Mock embedding manager for testing and development."""

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self._initialized = True
        self.call_count = 0

    async def initialize(self) -> None:
        """Initialize mock service."""
        pass

    async def close(self) -> None:
        """Close mock service."""
        pass

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate deterministic mock embedding."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        self.call_count += 1
        # Generate deterministic but pseudo-random embedding based on text
        hash_val = hash(text) % (2 ** 31)
        np.random.seed(hash_val)
        embedding = np.random.normal(0, 1, self.embedding_dim)
        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.astype(np.float32).tolist()

    async def generate_embeddings_batch(self, texts: List[str], batch_size: int = 8) -> List[List[float]]:
        """Generate mock embeddings for multiple texts."""
        return [await self.generate_embedding(text) for text in texts if text and text.strip()]

    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity."""
        try:
            emb1 = np.array(embedding1, dtype=np.float32)
            emb2 = np.array(embedding2, dtype=np.float32)

            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = float(dot_product / (norm1 * norm2))
            return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.0

    async def find_most_similar(self, query_embedding: List[float],
                              candidate_embeddings: List[List[float]]) -> int:
        """Find index of most similar embedding."""
        if not candidate_embeddings:
            return -1

        similarities = []
        for candidate in candidate_embeddings:
            similarity = await self.compute_similarity(query_embedding, candidate)
            similarities.append(similarity)

        return int(np.argmax(similarities))

    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model info."""
        return {
            "service": "mock_embeddings",
            "model_name": f"mock-dim{self.embedding_dim}",
            "embedding_dimension": self.embedding_dim,
            "initialized": True,
            "call_count": self.call_count
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for mock service."""
        return {
            'service': 'mock_embeddings',
            'status': 'healthy',
            'initialized': True,
            'embedding_dimension': self.embedding_dim,
            'call_count': self.call_count
        }