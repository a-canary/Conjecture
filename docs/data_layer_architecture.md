# Data Layer Architecture Design

## Overview
This document outlines the architecture for Conjecture's data layer, implementing a SQLite + ChromaDB hybrid approach that prioritizes simplicity, self-hosting, and free solutions while meeting all project requirements.

## Architecture Components

### 1. Storage Layer

#### Primary Database: SQLite
- **Purpose**: Structured data storage (claims, relationships, metadata)
- **Location**: `./data/conjecture.db`
- **Advantages**: Serverless, zero-config, ACID compliant, excellent Python support

#### Vector Database: ChromaDB
- **Purpose**: Vector embeddings and similarity search
- **Location**: `./data/vector_db`
- **Advantages**: Native Python, simple API, built-in embedding management

### 2. Data Manager Interface

```python
class DataManager:
    """
    Unified data management interface that coordinates SQLite and ChromaDB
    """
    
    def __init__(self, config: DataConfig):
        self.sqlite_manager = SQLiteManager(config.sqlite_path)
        self.chroma_manager = ChromaManager(config.chroma_path)
        self.embedding_service = EmbeddingService(config.embedding_model)
    
    # Claim Management
    async def create_claim(self, content: str, **kwargs) -> Claim
    async def get_claim(self, claim_id: str) -> Optional[Claim]
    async def update_claim(self, claim_id: str, **updates) -> bool
    async def delete_claim(self, claim_id: str) -> bool
    
    # Search and Discovery
    async def search_similar(self, content: str, limit: int = 10) -> List[Claim]
    async def filter_claims(self, filters: dict, limit: int = 100) -> List[Claim]
    async def get_dirty_claims(self, limit: int = 50) -> List[Claim]
    
    # Relationship Management
    async def add_relationship(self, supporter_id: str, supported_id: str, 
                             relationship_type: str = "supports") -> bool
    async def remove_relationship(self, supporter_id: str, supported_id: str) -> bool
    async def get_relationships(self, claim_id: str) -> List[Relationship]
    
    # Batch Operations
    async def batch_create_claims(self, claims: List[dict]) -> List[Claim]
    async def batch_update_claims(self, updates: List[dict]) -> List[bool]
```

### 3. Database Schema

#### SQLite Schema

```sql
-- Claims table
CREATE TABLE claims (
    id VARCHAR(20) PRIMARY KEY,  -- c####### format
    content TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    dirty BOOLEAN NOT NULL DEFAULT true,
    tags TEXT NOT NULL DEFAULT '[]',  -- JSON array
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(50),
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Claim relationships junction table
CREATE TABLE claim_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    supporter_id VARCHAR(20) NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    supported_id VARCHAR(20) NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    relationship_type VARCHAR(20) NOT NULL DEFAULT 'supports',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(50),
    UNIQUE(supporter_id, supported_id, relationship_type)
);

-- Indexes for performance
CREATE INDEX idx_claims_confidence ON claims(confidence);
CREATE INDEX idx_claims_dirty ON claims(dirty);
CREATE INDEX idx_claims_created_at ON claims(created_at);
CREATE INDEX idx_claims_created_by ON claims(created_by);
CREATE INDEX idx_relationships_supporter ON claim_relationships(supporter_id);
CREATE INDEX idx_relationships_supported ON claim_relationships(supported_id);
CREATE INDEX idx_relationships_type ON claim_relationships(relationship_type);

-- Full-text search trigger
CREATE VIRTUAL TABLE claims_fts USING fts5(content, content='claims', content_rowid='rowid');
CREATE TRIGGER claims_fts_insert AFTER INSERT ON claims BEGIN
    INSERT INTO claims_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER claims_fts_delete AFTER DELETE ON claims BEGIN
    INSERT INTO claims_fts(claims_fts, rowid, content) VALUES('delete', old.rowid, old.content);
END;
CREATE TRIGGER claims_fts_update AFTER UPDATE ON claims BEGIN
    INSERT INTO claims_fts(claims_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    INSERT INTO claims_fts(rowid, content) VALUES (new.rowid, new.content);
END;
```

#### ChromaDB Collection Structure

```python
# ChromaDB collection for claim embeddings
collection = client.create_collection(
    name="claims",
    metadata={
        "description": "Claim embeddings for semantic similarity search",
        "embedding_model": "all-MiniLM-L6-v2"
    }
)

# Document structure in ChromaDB
{
    "id": "c0000001",           # Claim ID
    "document": "claim content", # Original text
    "metadata": {
        "confidence": 0.85,
        "tags": ["concept", "ai"],
        "created_by": "user123",
        "created_at": "2025-01-07T10:00:00Z",
        "dirty": true
    },
    "embeddings": [0.1, 0.2, ...] # Vector embedding
}
```

### 4. Component Implementation

#### SQLite Manager

```python
class SQLiteManager:
    """Handles all SQLite database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection_pool = aiosqlite.create_pool(db_path)
    
    async def initialize(self):
        """Create database schema"""
        async with self.connection_pool.acquire() as conn:
            await conn.executescript(SCHEMA_SQL)
            await conn.commit()
    
    async def create_claim(self, claim: Claim) -> str:
        """Insert new claim into SQLite"""
        async with self.connection_pool.acquire() as conn:
            cursor = await conn.execute("""
                INSERT INTO claims (id, content, confidence, dirty, tags, created_by)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (claim.id, claim.content, claim.confidence, 
                  claim.dirty, json.dumps(claim.tags), claim.created_by))
            await conn.commit()
            return claim.id
    
    async def get_claim(self, claim_id: str) -> Optional[dict]:
        """Retrieve claim by ID"""
        async with self.connection_pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT * FROM claims WHERE id = ?
            """, (claim_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None
    
    async def update_claim(self, claim_id: str, updates: dict) -> bool:
        """Update claim fields"""
        if not updates:
            return False
            
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [claim_id]
        
        async with self.connection_pool.acquire() as conn:
            cursor = await conn.execute(f"""
                UPDATE claims SET {set_clause}, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, values)
            await conn.commit()
            return cursor.rowcount > 0
```

#### ChromaDB Manager

```python
class ChromaManager:
    """Handles all ChromaDB vector operations"""
    
    def __init__(self, chroma_path: str):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection("claims")
    
    async def add_embedding(self, claim_id: str, content: str, 
                          embedding: List[float], metadata: dict):
        """Add claim embedding to ChromaDB"""
        self.collection.add(
            ids=[claim_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata]
        )
    
    async def search_similar(self, query_embedding: List[float], 
                           limit: int = 10) -> List[dict]:
        """Search for similar claims by embedding"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit
        )
        
        return [
            {
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
            for i in range(len(results["ids"][0]))
        ]
    
    async def update_embedding(self, claim_id: str, content: str,
                             embedding: List[float], metadata: dict):
        """Update existing embedding"""
        self.collection.update(
            ids=[claim_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata]
        )
    
    async def delete_embedding(self, claim_id: str):
        """Remove embedding from ChromaDB"""
        self.collection.delete(ids=[claim_id])
```

### 5. Data Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Claim(BaseModel):
    id: str = Field(..., regex=r'^c\d{7}$')
    content: str = Field(..., min_length=10)
    confidence: float = Field(..., ge=0.0, le=1.0)
    dirty: bool = True
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str
    updated_at: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Relationship(BaseModel):
    id: int
    supporter_id: str
    supported_id: str
    relationship_type: str = "supports"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None

class DataConfig(BaseModel):
    sqlite_path: str = "./data/conjecture.db"
    chroma_path: str = "./data/vector_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    
class ClaimFilter(BaseModel):
    tags: Optional[List[str]] = None
    confidence_min: Optional[float] = None
    confidence_max: Optional[float] = None
    dirty_only: Optional[bool] = None
    created_by: Optional[str] = None
    content_contains: Optional[str] = None
```

### 6. Configuration Management

```python
class ConfigManager:
    """Manages data layer configuration from environment variables"""
    
    @staticmethod
    def get_data_config() -> DataConfig:
        return DataConfig(
            sqlite_path=os.getenv("CONJECTURE_DB_PATH", "./data/conjecture.db"),
            chroma_path=os.getenv("CONJECTURE_CHROMA_PATH", "./data/vector_db"),
            embedding_model=os.getenv("CONJECTURE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
```

### 7. Error Handling and Validation

```python
class DataLayerError(Exception):
    """Base exception for data layer operations"""
    pass

class ClaimNotFoundError(DataLayerError):
    """Raised when a claim is not found"""
    pass

class InvalidClaimError(DataLayerError):
    """Raised when claim validation fails"""
    pass

class RelationshipError(DataLayerError):
    """Raised when relationship operations fail"""
    pass

def validate_claim_id(claim_id: str) -> bool:
    """Validate claim ID format (c#######)"""
    import re
    return bool(re.match(r'^c\d{7}$', claim_id))

def validate_confidence(confidence: float) -> bool:
    """Validate confidence score range"""
    return 0.0 <= confidence <= 1.0
```

### 8. Performance Optimizations

#### Caching Layer
```python
class CacheManager:
    """Simple in-memory cache for frequently accessed claims"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl
    
    async def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Get claim from cache if available and not expired"""
        if claim_id in self.cache:
            claim, timestamp = self.cache[claim_id]
            if time.time() - timestamp < self.ttl:
                return claim
            else:
                del self.cache[claim_id]
        return None
    
    async def set_claim(self, claim: Claim):
        """Cache claim with timestamp"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[claim.id] = (claim, time.time())
```

#### Batch Operations
```python
class BatchProcessor:
    """Handles batch operations for better performance"""
    
    def __init__(self, data_manager: DataManager, batch_size: int = 100):
        self.data_manager = data_manager
        self.batch_size = batch_size
    
    async def batch_create_claims(self, claims_data: List[dict]) -> List[Claim]:
        """Create multiple claims efficiently"""
        results = []
        for i in range(0, len(claims_data), self.batch_size):
            batch = claims_data[i:i + self.batch_size]
            batch_results = await self._process_create_batch(batch)
            results.extend(batch_results)
        return results
    
    async def _process_create_batch(self, batch: List[dict]) -> List[Claim]:
        """Process a single batch of claim creations"""
        # Generate embeddings for the batch
        contents = [claim["content"] for claim in batch]
        embeddings = await self.data_manager.embedding_service.generate_batch_embeddings(contents)
        
        # Create claims with embeddings
        claims = []
        for i, claim_data in enumerate(batch):
            claim = Claim(
                id=generate_claim_id(),
                content=claim_data["content"],
                confidence=claim_data.get("confidence", 0.5),
                tags=claim_data.get("tags", []),
                created_by=claim_data.get("created_by", "system")
            )
            claims.append(claim)
        
        # Store in both databases
        await self._store_claims_batch(claims, embeddings)
        return claims
```

## Implementation Phases

### Phase 1: Core Infrastructure
1. Set up SQLite database with schema
2. Implement basic CRUD operations
3. Create ChromaDB integration
4. Build unified DataManager interface

### Phase 2: Advanced Features
1. Add relationship management
2. Implement similarity search
3. Add filtering and querying
4. Create batch operations

### Phase 3: Performance and Reliability
1. Add caching layer
2. Implement error handling
3. Add performance monitoring
4. Create backup/restore functionality

### Phase 4: Testing and Documentation
1. Comprehensive unit tests
2. Integration tests
3. Performance benchmarks
4. API documentation

## Success Metrics

### Functional Requirements
- [ ] All CRUD operations working
- [ ] Vector similarity search functional
- [ ] Relationship management complete
- [ ] Filtering and search working

### Performance Requirements
- [ ] <10ms for simple claim retrieval
- [ ] <100ms for similarity search
- [ ] Support for 10,000+ claims
- [ ] <100MB memory usage for medium datasets

### Quality Requirements
- [ ] >90% test coverage
- [ ] Zero data corruption issues
- [ ] Comprehensive error handling
- [ ] Clear documentation

---

This architecture provides a solid foundation for Conjecture's data layer, balancing simplicity with functionality while ensuring scalability and reliability.