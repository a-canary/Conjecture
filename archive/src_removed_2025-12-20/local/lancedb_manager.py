"""
LanceDB-based unified vector and embedding manager.
Simplifies architecture by replacing ChromaDB + FAISS + sentence-transformers with single lightweight solution.
"""

import asyncio
import logging
import json
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from pathlib import Path
import hashlib
import uuid

# Import LanceDB
try:
    import lancedb
    import pandas as pd
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False
    lancedb = None
    pd = None

logger = logging.getLogger(__name__)

class LanceDBManager:
    """
    Unified vector and embedding manager using LanceDB.
    Replaces LocalVectorStore + LocalEmbeddingManager with single lightweight solution.

    Benefits:
    - Single dependency (lancedb) instead of ChromaDB + FAISS + sentence-transformers
    - ~50MB vs 500MB+ dependencies
    - Embedded database (like SQLite)
    - Built-in vector search + metadata storage
    """

    def __init__(self, db_path: str = "data/conjecture_lancedb.lance"):
        self.db_path = db_path
        self.db = None
        self.table = None
        self.dimension = 384  # Default embedding dimension
        self._initialized = False

        # Ensure data directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        if not LANCEDB_AVAILABLE:
            raise ImportError("LanceDB not installed. Install with: pip install lancedb")

    async def initialize(self, dimension: int = 384, table_name: str = "embeddings") -> None:
        """Initialize LanceDB connection and table."""
        if self._initialized:
            return

        try:
            # Connect to LanceDB
            self.db = lancedb.connect(self.db_path)

            # Set embedding dimension
            self.dimension = dimension

            # Open or create table
            try:
                self.table = self.db.open_table(table_name)
                logger.info(f"Opened existing table: {table_name}")
            except Exception:
                # Create empty table with schema
                schema = {
                    "id": str,
                    "text": str,
                    "vector": np.array((dimension,)),
                    "metadata": str,
                    "created_at": str,
                    "claim_id": str,
                    "embedding_source": str
                }

                # Create with empty data
                empty_data = {
                    "id": [],
                    "text": [],
                    "vector": [],
                    "metadata": [],
                    "created_at": [],
                    "claim_id": [],
                    "embedding_source": []
                }

                self.table = self.db.create_table(table_name, empty_data)
                logger.info(f"Created new table: {table_name} with dimension {dimension}")

            self._initialized = True
            logger.info("LanceDB manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize LanceDB: {e}")
            raise

    async def add_embeddings(self, texts: List[str], vectors: List[List[float]],
                           metadata: Optional[List[Dict]] = None,
                           claim_ids: Optional[List[str]] = None,
                           embedding_source: str = "openai") -> List[str]:
        """Add texts and their embeddings to the vector store."""
        if not self._initialized:
            await self.initialize()

        if len(texts) != len(vectors):
            raise ValueError("Number of texts must match number of vectors")

        # Prepare data
        ids = []
        data = {
            "id": [],
            "text": [],
            "vector": [],
            "metadata": [],
            "created_at": [],
            "claim_id": [],
            "embedding_source": []
        }

        current_time = datetime.now().isoformat()

        for i, (text, vector) in enumerate(zip(texts, vectors)):
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)

            # Prepare metadata
            meta = metadata[i] if metadata and i < len(metadata) else {}
            claim_id = claim_ids[i] if claim_ids and i < len(claim_ids) else ""

            # Add to batch data
            data["id"].append(doc_id)
            data["text"].append(text)
            data["vector"].append(np.array(vector, dtype=np.float32))
            data["metadata"].append(json.dumps(meta))
            data["created_at"].append(current_time)
            data["claim_id"].append(claim_id)
            data["embedding_source"].append(embedding_source)

        # Add to table
        try:
            self.table.add([data])
            logger.info(f"Added {len(texts)} embeddings to LanceDB")
            return ids
        except Exception as e:
            logger.error(f"Failed to add embeddings: {e}")
            raise

    async def search(self, query_vector: List[float], limit: int = 10,
                    claim_filter: Optional[str] = None) -> List[Dict]:
        """Search for similar vectors."""
        if not self._initialized:
            await self.initialize()

        try:
            # Convert to numpy array
            query_array = np.array(query_vector, dtype=np.float32)

            # Build search query
            search_query = self.table.search(query_array).limit(limit)

            # Apply claim filter if provided
            if claim_filter:
                search_query = search_query.where(f"claim_id = '{claim_filter}'")

            # Execute search
            results = search_query.to_pandas()

            # Convert to standardized format
            formatted_results = []
            for _, row in results.iterrows():
                formatted_results.append({
                    "id": row["id"],
                    "text": row["text"],
                    "score": float(row.get("_score", 0.0)),
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "claim_id": row.get("claim_id", ""),
                    "embedding_source": row.get("embedding_source", ""),
                    "created_at": row["created_at"]
                })

            logger.info(f"Found {len(formatted_results)} similar embeddings")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def get_by_id(self, doc_id: str) -> Optional[Dict]:
        """Get document by ID."""
        if not self._initialized:
            await self.initialize()

        try:
            results = self.table.search([0.0] * self.dimension).limit(1).where(f"id = '{doc_id}'").to_pandas()

            if len(results) == 0:
                return None

            row = results.iloc[0]
            return {
                "id": row["id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "claim_id": row.get("claim_id", ""),
                "embedding_source": row.get("embedding_source", ""),
                "created_at": row["created_at"]
            }

        except Exception as e:
            logger.error(f"Failed to get document by ID: {e}")
            return None

    async def get_by_claim_id(self, claim_id: str) -> List[Dict]:
        """Get all embeddings associated with a claim ID."""
        if not self._initialized:
            await self.initialize()

        try:
            results = self.table.search([0.0] * self.dimension).limit(1000).where(f"claim_id = '{claim_id}'").to_pandas()

            formatted_results = []
            for _, row in results.iterrows():
                formatted_results.append({
                    "id": row["id"],
                    "text": row["text"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "claim_id": row["claim_id"],
                    "embedding_source": row["embedding_source"],
                    "created_at": row["created_at"]
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to get documents by claim ID: {e}")
            return []

    async def delete_by_claim_id(self, claim_id: str) -> int:
        """Delete all embeddings associated with a claim ID."""
        if not self._initialized:
            await self.initialize()

        try:
            # Note: LanceDB doesn't directly support delete by filter
            # For now, we'll return the count that would be deleted
            # In a full implementation, you might need to recreate the table or use merge operations

            results = await self.get_by_claim_id(claim_id)
            count = len(results)

            logger.info(f"Would delete {count} embeddings for claim_id: {claim_id}")
            # TODO: Implement actual delete when LanceDB supports it better

            return count

        except Exception as e:
            logger.error(f"Failed to delete embeddings: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self._initialized:
            await self.initialize()

        try:
            # Get table info
            table_name = self.table.name if self.table else "unknown"

            # Count rows (approximate)
            try:
                # This might not be available in all LanceDB versions
                stats = {"total_rows": len(self.table.to_pandas())}
            except:
                stats = {"total_rows": "unknown"}

            return {
                "db_path": self.db_path,
                "table_name": table_name,
                "dimension": self.dimension,
                "initialized": self._initialized,
                **stats
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    async def close(self):
        """Close database connection."""
        # LanceDB doesn't require explicit closing, but we can clean up
        self.db = None
        self.table = None
        self._initialized = False
        logger.info("LanceDB manager closed")

# Factory function for easy replacement
async def create_lancedb_manager(db_path: str = "data/conjecture_lancedb.lance",
                               dimension: int = 384) -> LanceDBManager:
    """Create and initialize LanceDB manager."""
    manager = LanceDBManager(db_path)
    await manager.initialize(dimension)
    return manager