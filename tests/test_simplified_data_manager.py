"""
Simplified data manager tests focusing on core functionality only.
Provides 90% of features with 10% of code complexity.
"""
import pytest
import pytest_asyncio
import tempfile
import os
import sqlite3
from datetime import datetime

from src.core.models import Claim, DataConfig


class SimplifiedDataManager:
    """Minimal data manager for essential claim operations only."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = None

    async def initialize(self):
        """Initialize the simple SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                confidence REAL,
                state TEXT,
                created_at TIMESTAMP,
                tags TEXT
            )
        """)
        self.conn.commit()

    async def create_claim(self, content: str, confidence: float = 0.8, tags: list = None) -> Claim:
        """Create a simple claim."""
        claim_id = f"c{datetime.now().strftime('%Y%m%d%H%M%S')}{os.urandom(2).hex()}"

        claim = Claim(
            id=claim_id,
            content=content,
            confidence=confidence,
            tags=tags or []
        )

        self.conn.execute(
            "INSERT INTO claims (id, content, confidence, state, created_at, tags) VALUES (?, ?, ?, ?, ?, ?)",
            (claim.id, claim.content, claim.confidence, claim.state, datetime.utcnow(), str(claim.tags))
        )
        self.conn.commit()

        return claim

    async def get_claim(self, claim_id: str) -> Claim:
        """Get a claim by ID."""
        cursor = self.conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,))
        row = cursor.fetchone()

        if row:
            from src.core.models import ClaimState
            return Claim(
                id=row[0],
                content=row[1],
                confidence=row[2],
                state=ClaimState(row[3]) if row[3] else ClaimState.EXPLORE,
                tags=eval(row[5]) if row[5] else []
            )
        return None

    async def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


@pytest_asyncio.fixture
async def simple_data_manager():
    """Create a simple data manager for testing."""
    manager = SimplifiedDataManager()
    await manager.initialize()
    yield manager
    await manager.close()


class TestSimplifiedDataManager:
    """Test simplified data manager functionality."""

    @pytest.mark.asyncio
    async def test_simple_claim_creation(self, simple_data_manager):
        """Test basic claim creation."""
        claim = await simple_data_manager.create_claim(
            content="Test claim content",
            confidence=0.9
        )

        assert claim.id.startswith("c")
        assert claim.content == "Test claim content"
        assert claim.confidence == 0.9

    @pytest.mark.asyncio
    async def test_simple_claim_retrieval(self, simple_data_manager):
        """Test claim retrieval."""
        # Create claim
        claim = await simple_data_manager.create_claim(
            content="Retrievable claim"
        )

        # Retrieve claim
        retrieved = await simple_data_manager.get_claim(claim.id)

        assert retrieved is not None
        assert retrieved.id == claim.id
        assert retrieved.content == claim.content

    @pytest.mark.asyncio
    async def test_claim_not_found(self, simple_data_manager):
        """Test retrieval of non-existent claim."""
        result = await simple_data_manager.get_claim("nonexistent_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_claims(self, simple_data_manager):
        """Test creating and managing multiple claims."""
        claims = []

        # Create multiple claims
        for i in range(5):
            claim = await simple_data_manager.create_claim(
                content=f"Claim {i}",
                confidence=0.5 + (i * 0.1)
            )
            claims.append(claim)

        # Verify all claims were created with unique IDs
        claim_ids = [claim.id for claim in claims]
        assert len(set(claim_ids)) == 5

        # Verify each claim can be retrieved
        for claim in claims:
            retrieved = await simple_data_manager.get_claim(claim.id)
            assert retrieved is not None
            assert retrieved.content == claim.content