"""
Shared fixtures and configuration for Conjecture data layer tests.
"""
import pytest
import asyncio
import tempfile
import os
import shutil
from typing import List, Dict, Any
from datetime import datetime
import numpy as np

from src.data.models import (
    Claim, Relationship, ClaimFilter, DataConfig,
    ClaimNotFoundError, InvalidClaimError, RelationshipError, DataLayerError
)
from src.data.sqlite_manager import SQLiteManager
from src.data.chroma_manager import ChromaManager
from src.data.embedding_service import EmbeddingService, MockEmbeddingService
from src.data.data_manager import DataManager


# Test data fixtures

@pytest.fixture
def sample_claim_data() -> Dict[str, Any]:
    """Sample claim data for testing."""
    return {
        "id": "c0000001",
        "content": "The Earth revolves around the Sun in an elliptical orbit.",
        "confidence": 0.95,
        "dirty": True,
        "tags": ["astronomy", "science", "physics"],
        "created_at": datetime.utcnow()
    }


@pytest.fixture
def sample_claims_data() -> List[Dict[str, Any]]:
    """Multiple sample claims for batch tests."""
    return [
        {
            "id": "c0000001",
            "content": "Water boils at 100 degrees Celsius at sea level.",
            "confidence": 0.98,
            "dirty": False,
            "tags": ["chemistry", "physics"]
        },
        {
            "id": "c0000002", 
            "content": "The speed of light in vacuum is approximately 299,792,458 meters per second.",
            "confidence": 0.99,
            "dirty": False,
            "tags": ["physics", "constants"],
            "created_by": "test_user"
        },
        {
            "id": "c0000003",
            "content": "DNA is the genetic material that carries hereditary information.",
            "confidence": 0.95,
            "dirty": True,
            "tags": ["biology", "genetics"],
            "created_by": "test_user"
        },
        {
            "id": "c0000004",
            "content": "Mount Everest is the highest mountain above sea level.",
            "confidence": 0.92,
            "dirty": False,
            "tags": ["geography", "earth"],
            "created_by": "test_user"
        },
        {
            "id": "c0000005",
            "content": "The human heart has four chambers: two atria and two ventricles.",
            "confidence": 0.97,
            "dirty": True,
            "tags": ["biology", "anatomy"],
            "created_by": "test_user"
        }
    ]


@pytest.fixture
def sample_relationship_data() -> Dict[str, Any]:
    """Sample relationship data for testing."""
    return {
        "id": 1,
        "supporter_id": "c0000001",
        "supported_id": "c0000002",
        "relationship_type": "supports",
        "created_by": "test_user",
        "created_at": datetime.utcnow()
    }


@pytest.fixture
def test_config() -> DataConfig:
    """Test configuration for data components."""
    return DataConfig(
        sqlite_path=":memory:",  # In-memory SQLite for testing
        chroma_path="./test_chroma_db",
        embedding_model="all-MiniLM-L6-v2",
        cache_size=100,
        cache_ttl=60,
        batch_size=50
    )


@pytest.fixture
def temp_dir() -> str:
    """Temporary directory for test files."""
    temp_path = tempfile.mkdtemp(prefix="conjecture_test_")
    yield temp_path
    # Cleanup
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_sqlite_db(temp_dir: str) -> str:
    """Temporary SQLite database file."""
    return os.path.join(temp_dir, "test.db")


@pytest.fixture
def temp_chroma_db(temp_dir: str) -> str:
    """Temporary ChromaDB directory."""
    chroma_path = os.path.join(temp_dir, "chroma")
    os.makedirs(chroma_path, exist_ok=True)
    return chroma_path


# Component fixtures

@pytest.fixture
async def sqlite_manager(temp_sqlite_db: str) -> SQLiteManager:
    """SQLite manager fixture for testing."""
    manager = SQLiteManager(temp_sqlite_db)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
async def chroma_manager(temp_chroma_db: str) -> ChromaManager:
    """Chroma manager fixture for testing."""
    manager = ChromaManager(temp_chroma_db)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
async def mock_embedding_service() -> MockEmbeddingService:
    """Mock embedding service fixture for testing."""
    service = MockEmbeddingService(embedding_dim=384)
    await service.initialize()
    yield service
    await service.close()


@pytest.fixture
async def real_embedding_service() -> EmbeddingService:
    """Real embedding service fixture for testing (slower)."""
    pytest.importorskip("sentence_transformers")
    service = EmbeddingService("all-MiniLM-L6-v2")
    await service.initialize()
    yield service
    await service.close()


@pytest.fixture
async def data_manager(test_config: DataConfig) -> DataManager:
    """Data manager fixture with mock embeddings for testing."""
    # Use temp directories for testing
    test_config.sqlite_path = ":memory:"
    test_config.chroma_path = "./test_chroma_db"
    
    manager = DataManager(test_config, use_mock_embeddings=True)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
async def populated_data_manager(test_config: DataConfig, sample_claims_data: List[Dict[str, Any]]) -> DataManager:
    """Data manager fixture populated with sample claims."""
    test_config.sqlite_path = ":memory:"
    test_config.chroma_path = "./test_chroma_db"
    
    manager = DataManager(test_config, use_mock_embeddings=True)
    await manager.initialize()
    
    # Add sample claims
    for claim_data in sample_claims_data:
        await manager.create_claim(**claim_data)
    
    yield manager
    await manager.close()


# Test claim fixtures

@pytest.fixture
def valid_claim() -> Claim:
    """Valid claim for testing."""
    return Claim(
        id="c0000001",
        content="The Earth's atmosphere contains approximately 21% oxygen.",
        confidence=0.95,
        dirty=False,
        tags=["chemistry", "atmosphere"],
        created_by="test_user"
    )


@pytest.fixture
def invalid_claim() -> Dict[str, Any]:
    """Invalid claim data for error testing."""
    return {
        "id": "invalid_id",
        "content": "Too short",
        "confidence": 1.5,  # Invalid confidence
        "tags": ["", "valid_tag"],  # Empty tag
        "created_by": ""
    }


@pytest.fixture
def valid_relationship() -> Relationship:
    """Valid relationship for testing."""
    return Relationship(
        supporter_id="c0000001",
        supported_id="c0000002",
        relationship_type="supports",
        created_by="test_user"
    )


@pytest.fixture
def claim_filters() -> List[ClaimFilter]:
    """Various claim filters for testing."""
    return [
        ClaimFilter(tags=["physics"]),
        ClaimFilter(confidence_min=0.9),
        ClaimFilter(confidence_max=0.95),
        ClaimFilter(dirty_only=True),
        ClaimFilter(created_by="test_user"),
        ClaimFilter(content_contains="DNA"),
        ClaimFilter(limit=5, offset=0)
    ]


# Test fixtures for performance testing

@pytest.fixture
def large_claim_dataset() -> List[Dict[str, Any]]:
    """Large dataset for performance testing."""
    claims = []
    content_templates = [
        "Scientific statement about {}: This is claim number {}.",
        "Research finding related to {}: The data shows result {}.",
        "Theory about {}: According to hypothesis {}, this is true.",
        "Observation of {}: Evidence {} supports this conclusion.",
        "Fact about {}: Historical document {} confirms this."
    ]
    domains = ["physics", "chemistry", "biology", "mathematics", "astronomy", "geology", "psychology"]
    
    for i in range(1000):
        template = content_templates[i % len(content_templates)]
        domain = domains[i % len(domains)]
        
        claims.append({
            "content": template.format(domain, i + 1),
            "confidence": 0.5 + (i % 50) / 100.0,  # Vary confidence
            "tags": [domain, f"category_{i % 10}"],
            "created_by": f"user_{i % 20}",
            "dirty": i % 3 == 0  # Every third claim is dirty
        })
    
    return claims


@pytest.fixture
def performance_test_claims() -> List[Claim]:
    """Claims specifically designed for performance testing."""
    claims = []
    for i in range(100):
        claims.append(Claim(
            id=f"c{i:07d}",
            content=f"Performance test claim {i} with sufficient content length for embedding testing.",
            confidence=0.5 + (i % 50) / 100.0,
            dirty=i % 3 == 0,
            tags=[f"perf_tag_{i % 5}", f"category_{i % 10}"],
            created_by=f"perf_user_{i % 10}"
        ))
    return claims


# Async test event loop fixture
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Benchmark fixtures
@pytest.fixture
def benchmark_embedding() -> List[float]:
    """Standard embedding for similarity benchmarks."""
    np.random.seed(42)  # Deterministic for consistent benchmarks
    embedding = np.random.normal(0, 1, 384)
    return (embedding / np.linalg.norm(embedding)).tolist()


@pytest.fixture
def benchmark_embeddings() -> List[List[float]]:
    """Multiple embeddings for batch benchmarking."""
    np.random.seed(42)
    embeddings = []
    for i in range(100):
        embedding = np.random.normal(0, 1, 384)
        embeddings.append((embedding / np.linalg.norm(embedding)).tolist())
    return embeddings


# Mock fixtures for external dependencies
@pytest.fixture
def mock_chroma_response():
    """Mock ChromaDB query response."""
    return {
        "ids": [["c0000001", "c0000002"]],
        "documents": [["Content 1", "Content 2"]],
        "metadatas": [[{"confidence": 0.9}, {"confidence": 0.8}]],
        "distances": [[0.1, 0.2]]
    }


# Error handling fixtures
@pytest.fixture
def database_error_scenarios() -> Dict[str, Any]:
    """Scenarios for testing database error handling."""
    return {
        "connection_lost": "Connection to database lost",
        "constraint_violation": "UNIQUE constraint failed",
        "foreign_key_violation": "FOREIGN KEY constraint failed",
        "disk_full": "database or disk is full",
        "permission_denied": "attempt to write a readonly database"
    }


# Test utilities
@pytest.fixture
def claim_validator():
    """Utility function to validate claim integrity."""
    def validate(claim: Claim) -> bool:
        try:
            # Basic validation
            if not claim.id or not claim.id.startswith('c') or len(claim.id) != 8:
                return False
            if len(claim.content) < 10:
                return False
            if not (0.0 <= claim.confidence <= 1.0):
                return False
            if not claim.created_by:
                return False
            return True
        except:
            return False
    
    return validate


# Helper fixtures for test data generation
@pytest.fixture
def claim_generator():
    """Function to generate test claims with specified parameters."""
    def generate(count: int = 10, **kwargs) -> List[Dict[str, Any]]:
        claims = []
        defaults = {
            "confidence": 0.7,
            "dirty": False,
            "tags": ["test"],
            "created_by": "generator"
        }
        
        for i in range(count):
            claim_data = {
                "content": f"Generated claim {i} for testing purposes.",
                **defaults,
                **kwargs
            }
            claims.append(claim_data)
        
        return claims
    
    return generate


@pytest.fixture
def embedding_generator():
    """Function to generate test embeddings."""
    def generate(count: int = 1, dimension: int = 384) -> List[List[float]]:
        np.random.seed(42)  # Deterministic
        embeddings = []
        for _ in range(count):
            embedding = np.random.normal(0, 1, dimension)
            normalized = embedding / np.linalg.norm(embedding)
            embeddings.append(normalized.tolist())
        return embeddings
    
    return generate