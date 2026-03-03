"""
Vector Store for Semantic Search

Per T-0004: FAISS+SQLite for vector search.
Provides embedding generation and similarity search for claims.
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_faiss = None
_model = None


def _get_faiss():
    """Lazy load faiss."""
    global _faiss
    if _faiss is None:
        import faiss
        _faiss = faiss
    return _faiss


def _get_embedding_model():
    """Lazy load sentence transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Use a small, fast model for embeddings
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence-transformers model: all-MiniLM-L6-v2")
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback")
            _model = "fallback"
    return _model


def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding vector for text.

    Args:
        text: Text to embed

    Returns:
        Embedding vector as numpy array (384 dimensions for MiniLM)
    """
    model = _get_embedding_model()

    if model == "fallback":
        # Simple fallback: hash-based embedding (not semantic, but works)
        # This is a placeholder - real semantic search requires sentence-transformers
        import hashlib
        hash_bytes = hashlib.sha384(text.encode()).digest()
        return np.frombuffer(hash_bytes, dtype=np.float32)[:96]

    return model.encode(text, convert_to_numpy=True)


class VectorStore:
    """FAISS-based vector store for claim embeddings.

    Stores embeddings in a FAISS index for fast similarity search.
    Maps vector indices to claim IDs via a JSON mapping file.
    """

    def __init__(self, index_path: str = "data/vectors.faiss", dimension: int = 384):
        """Initialize the vector store.

        Args:
            index_path: Path to store the FAISS index
            dimension: Embedding dimension (384 for MiniLM-L6-v2)
        """
        self.index_path = Path(index_path)
        self.mapping_path = self.index_path.with_suffix('.json')
        self.dimension = dimension
        self._index = None
        self._id_to_idx: Dict[str, int] = {}  # claim_id -> faiss index
        self._idx_to_id: Dict[int, str] = {}  # faiss index -> claim_id
        self._initialized = False

    def initialize(self) -> None:
        """Initialize or load the FAISS index."""
        if self._initialized:
            return

        faiss = _get_faiss()

        # Try to load existing index
        if self.index_path.exists() and self.mapping_path.exists():
            try:
                self._index = faiss.read_index(str(self.index_path))
                with open(self.mapping_path, 'r') as f:
                    mappings = json.load(f)
                    self._id_to_idx = mappings.get('id_to_idx', {})
                    # JSON keys are strings, convert back to int for idx_to_id
                    self._idx_to_id = {int(k): v for k, v in mappings.get('idx_to_id', {}).items()}
                logger.info(f"Loaded FAISS index with {self._index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load index, creating new: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

        self._initialized = True

    def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        faiss = _get_faiss()
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self._index = faiss.IndexFlatIP(self.dimension)
        self._id_to_idx = {}
        self._idx_to_id = {}
        logger.info(f"Created new FAISS index (dimension={self.dimension})")

    def save(self) -> None:
        """Save the index to disk."""
        if not self._initialized or self._index is None:
            return

        faiss = _get_faiss()

        # Ensure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(self.index_path))

        # Save ID mappings as JSON
        with open(self.mapping_path, 'w') as f:
            json.dump({
                'id_to_idx': self._id_to_idx,
                'idx_to_id': {str(k): v for k, v in self._idx_to_id.items()}
            }, f)

        logger.debug(f"Saved FAISS index with {self._index.ntotal} vectors")

    def add(self, claim_id: str, text: str) -> None:
        """Add a claim embedding to the index.

        Args:
            claim_id: Unique claim identifier
            text: Claim content to embed
        """
        if not self._initialized:
            self.initialize()

        # Skip if already indexed
        if claim_id in self._id_to_idx:
            return

        # Generate embedding
        embedding = generate_embedding(text)

        # Normalize for cosine similarity
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        embedding = embedding.reshape(1, -1).astype(np.float32)

        # Pad or truncate to match index dimension
        if embedding.shape[1] < self.dimension:
            embedding = np.pad(embedding, ((0, 0), (0, self.dimension - embedding.shape[1])))
        elif embedding.shape[1] > self.dimension:
            embedding = embedding[:, :self.dimension]

        # Add to index
        idx = self._index.ntotal
        self._index.add(embedding)
        self._id_to_idx[claim_id] = idx
        self._idx_to_id[idx] = claim_id

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar claims.

        Args:
            query: Query text to search for
            k: Number of results to return

        Returns:
            List of (claim_id, similarity_score) tuples, sorted by similarity
        """
        if not self._initialized:
            self.initialize()

        if self._index.ntotal == 0:
            return []

        # Generate query embedding
        query_embedding = generate_embedding(query)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        # Pad or truncate
        if query_embedding.shape[1] < self.dimension:
            query_embedding = np.pad(query_embedding, ((0, 0), (0, self.dimension - query_embedding.shape[1])))
        elif query_embedding.shape[1] > self.dimension:
            query_embedding = query_embedding[:, :self.dimension]

        # Search
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_embedding, k)

        # Map indices to claim IDs
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx in self._idx_to_id:
                claim_id = self._idx_to_id[idx]
                score = float(scores[0][i])
                results.append((claim_id, score))

        return results

    def remove(self, claim_id: str) -> bool:
        """Remove a claim from the index.

        Note: FAISS doesn't support true deletion. We mark it as removed
        and rebuild periodically. For now, just remove from mapping.

        Args:
            claim_id: Claim ID to remove

        Returns:
            True if claim was found and removed from mapping
        """
        if claim_id in self._id_to_idx:
            idx = self._id_to_idx.pop(claim_id)
            self._idx_to_id.pop(idx, None)
            return True
        return False

    def count(self) -> int:
        """Return number of vectors in index."""
        if not self._initialized:
            return 0
        return self._index.ntotal if self._index else 0
