"""
Lightweight vector store using FAISS + SQLite.
Provides fast, local vector search without ChromaDB complexity.
"""

import asyncio
import logging
import json
import sqlite3
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from pathlib import Path
import aiosqlite
import hashlib

# Import FAISS conditionally
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = logging.getLogger(__name__)

class LocalVectorStore:
    """
    Lightweight vector store combining FAISS for fast similarity search
    and SQLite for metadata storage. Replaces ChromaDB with a simpler,
    faster local solution.
    """

    def __init__(self, db_path: str = "data/local_vector_store.db", 
                 index_type: str = "ivf_flat",
                 use_faiss: bool = True,
                 nlist: int = 100):
        self.db_path = db_path
        self.index_type = index_type
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.nlist = nlist
        
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.conn: Optional[sqlite3.Connection] = None
        self.faiss_index = None
        self.dimension = 384
        self._id_to_idx = {}
        self._idx_to_id = []
        
        self._initialized = False

    async def initialize(self, dimension: int = 384) -> None:
        """Initialize the vector store with FAISS index and SQLite database."""
        if self._initialized:
            return

        try:
            self.dimension = dimension
            await self._init_sqlite_db()
            if self.use_faiss:
                await self._init_faiss_index()
            else:
                logger.info("FAISS not available, using SQLite-only vector search")
            self._initialized = True
            logger.info(f"Local vector store initialized with dimension {dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise RuntimeError(f"Vector store initialization failed: {e}")

    async def _init_sqlite_db(self) -> None:
        """Initialize SQLite database for metadata storage."""
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS vector_metadata (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    metadata TEXT,
                    embedding_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS search_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    query_hash TEXT,
                    num_results INTEGER,
                    execution_time_ms REAL
                )
            ''')
            
            await conn.commit()

    async def _init_faiss_index(self) -> None:
        """Initialize FAISS index based on configured type."""
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")
        
        try:
            existing_count = await self._get_vector_count()
            
            if existing_count == 0:
                if self.index_type == "ivf_flat":
                    quantizer = faiss.IndexFlatL2(self.dimension)
                    self.faiss_index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
                else:
                    self.faiss_index = faiss.IndexFlatL2(self.dimension)
            else:
                await self._rebuild_faiss_index()
            
            logger.info(f"FAISS index initialized with type: {self.index_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            self.use_faiss = False

    async def _get_vector_count(self) -> int:
        """Get count of existing vectors in database."""
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.execute('SELECT COUNT(*) FROM vector_metadata') as cursor:
                result = await cursor.fetchone()
                return result[0] if result else 0

    async def _create_and_populate_faiss_index(self, embeddings: List[np.ndarray]) -> None:
        """Create and populate FAISS index from a list of embeddings."""
        if not self.use_faiss:
            return

        embeddings_array = np.array(embeddings, dtype=np.float32)
        n_vectors = len(embeddings_array)

        if self.index_type == "ivf_flat" and n_vectors > 1000:
            nlist = min(100, max(1, n_vectors // 10))
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.faiss_index.train(embeddings_array)
        else:
            self.faiss_index = faiss.IndexFlatL2(self.dimension)

        self.faiss_index.add(embeddings_array)
        logger.info(f"Created and populated FAISS index with {n_vectors} vectors")

    async def _rebuild_faiss_index(self) -> None:
        """Rebuild FAISS index from existing embeddings in database."""
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.execute('''
                SELECT id, embedding FROM vector_metadata 
                WHERE embedding IS NOT NULL
            ''') as cursor:
                rows = await cursor.fetchall()
        
        if not rows:
            if self.index_type == "ivf_flat":
                nlist = 1
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            else:
                self.faiss_index = faiss.IndexFlatL2(self.dimension)
            return
        
        embeddings = []
        self._id_to_idx = {}
        self._idx_to_id = []
        
        for idx, (claim_id, embedding_blob) in enumerate(rows):
            embedding = pickle.loads(embedding_blob)
            embeddings.append(embedding)
            self._id_to_idx[claim_id] = idx
            self._idx_to_id.append(claim_id)
        
        await self._create_and_populate_faiss_index(embeddings)
        logger.info(f"Rebuilt FAISS index with {len(embeddings)} vectors")

    async def add_vector(self, 
                        claim_id: str, 
                        content: str, 
                        embedding: List[float],
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a vector to the store."""
        if not self._initialized:
            await self.initialize(len(embedding))
        
        try:
            # Calculate embedding hash for deduplication
            embedding_hash = self._calculate_embedding_hash(embedding)
            
            # Convert embedding to numpy array
            embedding_array = np.array(embedding, dtype=np.float32)
            
            # Store in SQLite
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('''
                    INSERT OR REPLACE INTO vector_metadata 
                    (id, content, embedding, metadata, embedding_hash, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    claim_id,
                    content,
                    pickle.dumps(embedding_array),
                    json.dumps(metadata or {}),
                    embedding_hash,
                    datetime.utcnow().isoformat()
                ))
                await conn.commit()
            
            # Add to FAISS index if available and not already present
            if self.use_faiss and self.faiss_index and claim_id not in self._id_to_idx:
                idx = len(self._idx_to_id)
                self._id_to_idx[claim_id] = idx
                self._idx_to_id.append(claim_id)
                
                if self.index_type == "ivf_flat" and hasattr(self.faiss_index, 'is_trained'):
                    if not self.faiss_index.is_trained:
                        # Train index with this vector
                        self.faiss_index.train(embedding_array.reshape(1, -1))
                
                self.faiss_index.add(embedding_array.reshape(1, -1))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vector for {claim_id}: {e}")
            return False

    async def add_vectors_batch(self, 
                               vectors_data: List[Tuple[str, str, List[float], Optional[Dict[str, Any]]]]) -> int:
        """Add multiple vectors efficiently."""
        if not self._initialized:
            await self.initialize(len(vectors_data[0][2]) if vectors_data else 384)
        
        if not vectors_data:
            return 0
        
        success_count = 0
        
        try:
            batch_data = []
            embeddings_for_faiss = []
            new_ids = []
            
            for claim_id, content, embedding, metadata in vectors_data:
                embedding_hash = self._calculate_embedding_hash(embedding)
                embedding_array = np.array(embedding, dtype=np.float32)
                
                batch_data.append((
                    claim_id,
                    content,
                    pickle.dumps(embedding_array),
                    json.dumps(metadata or {}),
                    embedding_hash,
                    datetime.utcnow().isoformat()
                ))
                
                if self.use_faiss and claim_id not in self._id_to_idx:
                    embeddings_for_faiss.append(embedding_array)
                    new_ids.append(claim_id)
            
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.executemany('''
                    INSERT OR REPLACE INTO vector_metadata 
                    (id, content, embedding, metadata, embedding_hash, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', batch_data)
                await conn.commit()
            
            if self.use_faiss and self.faiss_index and embeddings_for_faiss:
                await self._create_and_populate_faiss_index(embeddings_for_faiss)

            success_count = len(vectors_data)
            logger.info(f"Added {success_count} vectors in batch")
            
        except Exception as e:
            logger.error(f"Failed to add batch vectors: {e}")
        
        return success_count

    async def search_similar(self, 
                           query_embedding: List[float],
                           limit: int = 10,
                           threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")
        
        start_time = datetime.utcnow()
        
        try:
            if self.use_faiss and self.faiss_index:
                results = await self._search_with_faiss(query_embedding, limit, threshold)
            else:
                results = await self._search_with_sqlite(query_embedding, limit, threshold)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            await self._log_search_stats(query_embedding, len(results), execution_time)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def _search_with_faiss(self, 
                               query_embedding: List[float],
                               limit: int,
                               threshold: float) -> List[Dict[str, Any]]:
        """Search using FAISS index."""
        query_array = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        search_k = min(limit * 3, self.faiss_index.ntotal)
        
        distances, indices = self.faiss_index.search(query_array, search_k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            
            similarity = 1 / (1 + distance)
            
            if similarity >= threshold:
                claim_id = self._idx_to_id[idx]
                metadata = await self._get_metadata(claim_id)
                
                if metadata:
                    results.append({
                        'id': claim_id,
                        'content': metadata.get('content', ''),
                        'metadata': json.loads(metadata.get('metadata', '{}')),
                        'similarity': similarity,
                        'distance': float(distance)
                    })
            
            if len(results) >= limit:
                break
        
        return results[:limit]

    async def _search_with_sqlite(self, 
                                 query_embedding: List[float],
                                 limit: int,
                                 threshold: float) -> List[Dict[str, Any]]:
        """Search using SQLite (fallback when FAISS not available)."""
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.execute('''
                SELECT id, content, metadata, embedding 
                FROM vector_metadata 
                WHERE embedding IS NOT NULL
            ''') as cursor:
                rows = await cursor.fetchall()
        
        results = []
        query_array = np.array(query_embedding, dtype=np.float32)
        
        for claim_id, content, metadata_json, embedding_blob in rows:
            stored_embedding = pickle.loads(embedding_blob)
            stored_array = np.array(stored_embedding, dtype=np.float32)
            
            similarity = self._cosine_similarity(query_array, stored_array)
            
            if similarity >= threshold:
                metadata = json.loads(metadata_json or '{}')
                results.append({
                    'id': claim_id,
                    'content': content,
                    'metadata': metadata,
                    'similarity': similarity,
                    'distance': 1 - similarity
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    async def _get_metadata(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific claim ID."""
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.execute('''
                SELECT content, metadata FROM vector_metadata WHERE id = ?
            ''', (claim_id,)) as cursor:
                row = await cursor.fetchone()
                if row:
                    return {'content': row[0], 'metadata': row[1]}
                return None

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))

    def _calculate_embedding_hash(self, embedding: List[float]) -> str:
        """Calculate hash for embedding deduplication."""
        embedding_str = json.dumps(embedding, sort_keys=True)
        return hashlib.md5(embedding_str.encode()).hexdigest()

    async def _log_search_stats(self, query_embedding: List[float], 
                               num_results: int, 
                               execution_time: float) -> None:
        """Log search performance statistics."""
        query_hash = self._calculate_embedding_hash(query_embedding[:10])  # Hash first 10 dims
        
        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute('''
                INSERT INTO search_stats (query_hash, num_results, execution_time_ms)
                VALUES (?, ?, ?)
            ''', (query_hash, num_results, execution_time))
            await conn.commit()

    async def get_vector(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific vector by ID."""
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")
        
        async with aiosqlite.connect(self.db_path) as conn:
            async with conn.execute('''
                SELECT id, content, embedding, metadata, embedding_hash, created_at, updated_at
                FROM vector_metadata WHERE id = ?
            ''', (claim_id,)) as cursor:
                row = await cursor.fetchone()
                
                if row:
                    embedding = pickle.loads(row[2]) if row[2] else None
                    return {
                        'id': row[0],
                        'content': row[1],
                        'embedding': embedding.tolist() if embedding is not None else None,
                        'metadata': json.loads(row[3] or '{}'),
                        'embedding_hash': row[4],
                        'created_at': row[5],
                        'updated_at': row[6]
                    }
                return None

    async def _update_sqlite_vector(self, claim_id: str, updates: Dict[str, Any]) -> None:
        """Update a vector in the SQLite database."""
        async with aiosqlite.connect(self.db_path) as conn:
            # Validate column names to prevent SQL injection
            allowed_columns = {
                'content', 'embedding', 'metadata', 'embedding_hash', 'updated_at'
            }
            
            # Filter updates to only include allowed columns
            filtered_updates = {
                k: v for k, v in updates.items()
                if k in allowed_columns
            }
            
            if not filtered_updates:
                logger.warning(f"No valid columns to update for claim {claim_id}")
                return
            
            # Build parameterized query safely
            set_clause = ', '.join(f"{k} = ?" for k in filtered_updates.keys())
            values = list(filtered_updates.values()) + [claim_id]
            
            # Use parameterized query to prevent SQL injection
            query = f'''
                UPDATE vector_metadata
                SET {set_clause}
                WHERE id = ?
            '''
            
            await conn.execute(query, values)
            await conn.commit()

    async def _update_faiss_vector(self, claim_id: str, embedding: List[float]) -> None:
        """Update a vector in the FAISS index."""
        if self.use_faiss and self.faiss_index and claim_id in self._id_to_idx:
            idx = self._id_to_idx[claim_id]
            embedding_array = np.array(embedding, dtype=np.float32)
            
            self.faiss_index.remove_ids(np.array([idx]))
            self.faiss_index.add(embedding_array.reshape(1, -1))

    async def update_vector(self, 
                           claim_id: str, 
                           content: Optional[str] = None,
                           embedding: Optional[List[float]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update a vector in the store."""
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")
        
        try:
            current = await self.get_vector(claim_id)
            if not current:
                return False
            
            updates = {
                'content': content if content is not None else current['content'],
                'embedding_hash': current['embedding_hash'],
                'updated_at': datetime.utcnow().isoformat()
            }
            
            if embedding is not None:
                embedding_array = np.array(embedding, dtype=np.float32)
                updates['embedding'] = pickle.dumps(embedding_array)
                updates['embedding_hash'] = self._calculate_embedding_hash(embedding)
            
            if metadata is not None:
                updates['metadata'] = json.dumps(metadata)
            
            await self._update_sqlite_vector(claim_id, updates)
            
            if embedding is not None:
                await self._update_faiss_vector(claim_id, embedding)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update vector {claim_id}: {e}")
            return False

    async def delete_vector(self, claim_id: str) -> bool:
        """Delete a vector from the store."""
        if not self._initialized:
            raise RuntimeError("Vector store not initialized")
        
        try:
            # Delete from SQLite
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('DELETE FROM vector_metadata WHERE id = ?', (claim_id,))
                await conn.commit()
            
            # Remove from FAISS if available
            if self.use_faiss and self.faiss_index and claim_id in self._id_to_idx:
                idx = self._id_to_idx[claim_id]
                self.faiss_index.remove_ids(np.array([idx]))
                
                # Update mappings
                del self._id_to_idx[claim_id]
                self._idx_to_id.pop(idx)
                
                # Rebuild mappings to maintain consistency
                self._id_to_idx = {}
                self._idx_to_id = []
                
                async with aiosqlite.connect(self.db_path) as conn:
                    async with conn.execute('SELECT id FROM vector_metadata ORDER BY created_at') as cursor:
                        rows = await cursor.fetchall()
                        for i, (claim_id,) in enumerate(rows):
                            self._id_to_idx[claim_id] = i
                            self._idx_to_id.append(claim_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vector {claim_id}: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if not self._initialized:
            return {'initialized': False}
        
        try:
            # Get basic stats from SQLite
            async with aiosqlite.connect(self.db_path) as conn:
                async with conn.execute('SELECT COUNT(*) FROM vector_metadata') as cursor:
                    total_vectors = (await cursor.fetchone())[0]
                
                async with conn.execute('''
                    SELECT AVG(execution_time_ms), COUNT(*) 
                    FROM search_stats 
                    WHERE query_time > datetime('now', '-1 hour')
                ''') as cursor:
                    search_stats = await cursor.fetchone()
                    avg_search_time = search_stats[0] if search_stats[0] else 0
                    recent_searches = search_stats[1] if search_stats[1] else 0
            
            stats = {
                'initialized': True,
                'total_vectors': total_vectors,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'use_faiss': self.use_faiss,
                'avg_search_time_ms': avg_search_time,
                'recent_searches': recent_searches,
                'faiss_index_size': getattr(self.faiss_index, 'ntotal', 0) if self.faiss_index else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'initialized': True, 'error': str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Check health of vector store."""
        health = {
            'service': 'local_vector_store',
            'initialized': self._initialized,
            'faiss_available': FAISS_AVAILABLE,
            'use_faiss': self.use_faiss,
            'database_path': self.db_path,
            'status': 'healthy'
        }
        
        if not self._initialized:
            health['status'] = 'uninitialized'
            return health
        
        try:
            # Test database connection
            async with aiosqlite.connect(self.db_path) as conn:
                await conn.execute('SELECT 1')
            
            # Test FAISS if used
            if self.use_faiss and self.faiss_index:
                health['faiss_index_size'] = self.faiss_index.ntotal
            
            # Test with a simple search
            if await self._get_vector_count() > 0:
                test_embedding = np.random.normal(0, 1, self.dimension).tolist()
                results = await self.search_similar(test_embedding, limit=1)
                health['search_test'] = 'passed'
            else:
                health['search_test'] = 'no_vectors'
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
        
        return health

    async def close(self) -> None:
        """Close the vector store and clean up resources."""
        try:
            # FAISS doesn't need explicit cleanup
            self.faiss_index = None
            self._id_to_idx = {}
            self._idx_to_id = []
            self._initialized = False
            logger.info("Local vector store closed")
            
        except Exception as e:
            logger.error(f"Error closing vector store: {e}")
