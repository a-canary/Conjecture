"""
Unit tests for SQLiteManager.
"""
import pytest
import asyncio
import tempfile
import os
from datetime import datetime

from src.data.sqlite_manager import SQLiteManager
from src.data.models import Claim, Relationship, ClaimFilter, ClaimNotFoundError


@pytest.fixture
async def sqlite_manager():
    """Create a temporary SQLite manager for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    manager = SQLiteManager(db_path)
    await manager.initialize()
    
    yield manager
    
    await manager.close()
    # Clean up
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_claim():
    """Create a sample claim for testing."""
    return Claim(
        id="c0000001",
        content="This is a sample claim for testing purposes",
        confidence=0.8,
        tags=["test", "sample"],
        created_by="test_user"
    )


class TestSQLiteManager:
    """Test SQLiteManager functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, sqlite_manager):
        """Test database initialization."""
        # Should not raise any exceptions
        assert sqlite_manager.connection_pool is not None
    
    @pytest.mark.asyncio
    async def test_create_claim(self, sqlite_manager, sample_claim):
        """Test creating a claim."""
        result = await sqlite_manager.create_claim(sample_claim)
        assert result == sample_claim.id
        
        # Verify claim was created
        retrieved = await sqlite_manager.get_claim(sample_claim.id)
        assert retrieved is not None
        assert retrieved['id'] == sample_claim.id
        assert retrieved['content'] == sample_claim.content
        assert retrieved['confidence'] == sample_claim.confidence
        assert retrieved['tags'] == sample_claim.tags
    
    @pytest.mark.asyncio
    async def test_create_duplicate_claim(self, sqlite_manager, sample_claim):
        """Test creating a duplicate claim."""
        # Create first claim
        await sqlite_manager.create_claim(sample_claim)
        
        # Try to create duplicate
        with pytest.raises(Exception):  # Should raise DataLayerError
            await sqlite_manager.create_claim(sample_claim)
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_claim(self, sqlite_manager):
        """Test retrieving a non-existent claim."""
        result = await sqlite_manager.get_claim("c9999999")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_claim(self, sqlite_manager, sample_claim):
        """Test updating a claim."""
        # Create claim
        await sqlite_manager.create_claim(sample_claim)
        
        # Update confidence
        updates = {"confidence": 0.9, "dirty": False}
        result = await sqlite_manager.update_claim(sample_claim.id, updates)
        assert result is True
        
        # Verify update
        retrieved = await sqlite_manager.get_claim(sample_claim.id)
        assert retrieved['confidence'] == 0.9
        assert retrieved['dirty'] is False
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_claim(self, sqlite_manager):
        """Test updating a non-existent claim."""
        updates = {"confidence": 0.9}
        result = await sqlite_manager.update_claim("c9999999", updates)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_claim(self, sqlite_manager, sample_claim):
        """Test deleting a claim."""
        # Create claim
        await sqlite_manager.create_claim(sample_claim)
        
        # Delete claim
        result = await sqlite_manager.delete_claim(sample_claim.id)
        assert result is True
        
        # Verify deletion
        retrieved = await sqlite_manager.get_claim(sample_claim.id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_claim(self, sqlite_manager):
        """Test deleting a non-existent claim."""
        result = await sqlite_manager.delete_claim("c9999999")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_filter_claims_by_tags(self, sqlite_manager):
        """Test filtering claims by tags."""
        # Create multiple claims
        claims = [
            Claim(id="c0000001", content="Claim 1", confidence=0.5, tags=["tag1"], created_by="user1"),
            Claim(id="c0000002", content="Claim 2", confidence=0.6, tags=["tag2"], created_by="user1"),
            Claim(id="c0000003", content="Claim 3", confidence=0.7, tags=["tag1", "tag2"], created_by="user2"),
        ]
        
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Filter by tag1
        filter_obj = ClaimFilter(tags=["tag1"])
        results = await sqlite_manager.filter_claims(filter_obj)
        assert len(results) == 2
        assert all("tag1" in result['tags'] for result in results)
        
        # Filter by tag2
        filter_obj = ClaimFilter(tags=["tag2"])
        results = await sqlite_manager.filter_claims(filter_obj)
        assert len(results) == 2
        assert all("tag2" in result['tags'] for result in results)
    
    @pytest.mark.asyncio
    async def test_filter_claims_by_confidence(self, sqlite_manager):
        """Test filtering claims by confidence range."""
        # Create claims with different confidence levels
        claims = [
            Claim(id="c0000001", content="Claim 1", confidence=0.3, created_by="user1"),
            Claim(id="c0000002", content="Claim 2", confidence=0.6, created_by="user1"),
            Claim(id="c0000003", content="Claim 3", confidence=0.9, created_by="user2"),
        ]
        
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Filter by confidence range
        filter_obj = ClaimFilter(confidence_min=0.5, confidence_max=0.8)
        results = await sqlite_manager.filter_claims(filter_obj)
        assert len(results) == 1
        assert results[0]['confidence'] == 0.6
    
    @pytest.mark.asyncio
    async def test_filter_claims_by_dirty_flag(self, sqlite_manager):
        """Test filtering claims by dirty flag."""
        # Create claims
        claims = [
            Claim(id="c0000001", content="Claim 1", confidence=0.5, dirty=True, created_by="user1"),
            Claim(id="c0000002", content="Claim 2", confidence=0.6, dirty=False, created_by="user1"),
        ]
        
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Filter dirty claims only
        filter_obj = ClaimFilter(dirty_only=True)
        results = await sqlite_manager.filter_claims(filter_obj)
        assert len(results) == 1
        assert results[0]['dirty'] is True
        
        # Filter clean claims only
        filter_obj = ClaimFilter(dirty_only=False)
        results = await sqlite_manager.filter_claims(filter_obj)
        assert len(results) == 1
        assert results[0]['dirty'] is False
    
    @pytest.mark.asyncio
    async def test_get_dirty_claims(self, sqlite_manager):
        """Test retrieving dirty claims."""
        # Create claims
        claims = [
            Claim(id="c0000001", content="Claim 1", confidence=0.3, dirty=True, created_by="user1"),
            Claim(id="c0000002", content="Claim 2", confidence=0.6, dirty=False, created_by="user1"),
            Claim(id="c0000003", content="Claim 3", confidence=0.9, dirty=True, created_by="user2"),
        ]
        
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Get dirty claims
        dirty_claims = await sqlite_manager.get_dirty_claims()
        assert len(dirty_claims) == 2
        assert all(claim['dirty'] for claim in dirty_claims)
        
        # Should be ordered by confidence (ascending)
        assert dirty_claims[0]['confidence'] == 0.3
        assert dirty_claims[1]['confidence'] == 0.9
    
    @pytest.mark.asyncio
    async def test_search_claims_by_content(self, sqlite_manager):
        """Test full-text search of claim content."""
        # Create claims
        claims = [
            Claim(id="c0000001", content="Machine learning is a subset of AI", confidence=0.8, created_by="user1"),
            Claim(id="c0000002", content="Deep learning uses neural networks", confidence=0.7, created_by="user1"),
            Claim(id="c0000003", content="Python is a programming language", confidence=0.9, created_by="user2"),
        ]
        
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Search for "learning"
        results = await sqlite_manager.search_claims_by_content("learning")
        assert len(results) == 2
        assert all("learning" in result['content'].lower() for result in results)
    
    @pytest.mark.asyncio
    async def test_add_relationship(self, sqlite_manager):
        """Test adding relationships between claims."""
        # Create claims
        claim1 = Claim(id="c0000001", content="Claim 1", confidence=0.8, created_by="user1")
        claim2 = Claim(id="c0000002", content="Claim 2", confidence=0.7, created_by="user1")
        
        await sqlite_manager.create_claim(claim1)
        await sqlite_manager.create_claim(claim2)
        
        # Add relationship
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002",
            relationship_type="supports",
            created_by="user1"
        )
        
        relationship_id = await sqlite_manager.add_relationship(relationship)
        assert relationship_id is not None
        assert isinstance(relationship_id, int)
    
    @pytest.mark.asyncio
    async def test_add_relationship_nonexistent_claims(self, sqlite_manager):
        """Test adding relationship with non-existent claims."""
        relationship = Relationship(
            supporter_id="c9999999",  # Non-existent
            supported_id="c9999998",  # Non-existent
            relationship_type="supports"
        )
        
        with pytest.raises(Exception):  # Should raise DataLayerError
            await sqlite_manager.add_relationship(relationship)
    
    @pytest.mark.asyncio
    async def test_add_duplicate_relationship(self, sqlite_manager):
        """Test adding duplicate relationship."""
        # Create claims
        claim1 = Claim(id="c0000001", content="Claim 1", confidence=0.8, created_by="user1")
        claim2 = Claim(id="c0000002", content="Claim 2", confidence=0.7, created_by="user1")
        
        await sqlite_manager.create_claim(claim1)
        await sqlite_manager.create_claim(claim2)
        
        # Add first relationship
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002",
            relationship_type="supports"
        )
        await sqlite_manager.add_relationship(relationship)
        
        # Try to add duplicate
        with pytest.raises(Exception):  # Should raise DataLayerError
            await sqlite_manager.add_relationship(relationship)
    
    @pytest.mark.asyncio
    async def test_remove_relationship(self, sqlite_manager):
        """Test removing relationships."""
        # Create claims and relationship
        claim1 = Claim(id="c0000001", content="Claim 1", confidence=0.8, created_by="user1")
        claim2 = Claim(id="c0000002", content="Claim 2", confidence=0.7, created_by="user1")
        
        await sqlite_manager.create_claim(claim1)
        await sqlite_manager.create_claim(claim2)
        
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002",
            relationship_type="supports"
        )
        await sqlite_manager.add_relationship(relationship)
        
        # Remove relationship
        result = await sqlite_manager.remove_relationship("c0000001", "c0000002", "supports")
        assert result is True
        
        # Try to remove again
        result = await sqlite_manager.remove_relationship("c0000001", "c0000002", "supports")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_relationships(self, sqlite_manager):
        """Test getting relationships for a claim."""
        # Create claims
        claim1 = Claim(id="c0000001", content="Claim 1", confidence=0.8, created_by="user1")
        claim2 = Claim(id="c0000002", content="Claim 2", confidence=0.7, created_by="user1")
        claim3 = Claim(id="c0000003", content="Claim 3", confidence=0.6, created_by="user2")
        
        for claim in [claim1, claim2, claim3]:
            await sqlite_manager.create_claim(claim)
        
        # Add relationships
        rel1 = Relationship(supporter_id="c0000001", supported_id="c0000002", relationship_type="supports")
        rel2 = Relationship(supporter_id="c0000003", supported_id="c0000001", relationship_type="supports")
        
        await sqlite_manager.add_relationship(rel1)
        await sqlite_manager.add_relationship(rel2)
        
        # Get relationships for claim1
        relationships = await sqlite_manager.get_relationships("c0000001")
        assert len(relationships) == 2
        
        # Check relationship directions
        supporter_rels = [r for r in relationships if r['supporter_id'] == "c0000001"]
        supported_rels = [r for r in relationships if r['supported_id'] == "c0000001"]
        
        assert len(supporter_rels) == 1
        assert len(supported_rels) == 1
    
    @pytest.mark.asyncio
    async def test_get_supported_by_and_supports(self, sqlite_manager):
        """Test getting supported_by and supports relationships."""
        # Create claims
        claim1 = Claim(id="c0000001", content="Claim 1", confidence=0.8, created_by="user1")
        claim2 = Claim(id="c0000002", content="Claim 2", confidence=0.7, created_by="user1")
        claim3 = Claim(id="c0000003", content="Claim 3", confidence=0.6, created_by="user2")
        
        for claim in [claim1, claim2, claim3]:
            await sqlite_manager.create_claim(claim)
        
        # Add relationships: claim1 supports claim2, claim3 supports claim1
        rel1 = Relationship(supporter_id="c0000001", supported_id="c0000002", relationship_type="supports")
        rel2 = Relationship(supporter_id="c0000003", supported_id="c0000001", relationship_type="supports")
        
        await sqlite_manager.add_relationship(rel1)
        await sqlite_manager.add_relationship(rel2)
        
        # Test get_supported_by (who supports claim1)
        supported_by = await sqlite_manager.get_supported_by("c0000001")
        assert len(supported_by) == 1
        assert "c0000003" in supported_by
        
        # Test get_supports (what claim1 supports)
        supports = await sqlite_manager.get_supports("c0000001")
        assert len(supports) == 1
        assert "c0000002" in supports
    
    @pytest.mark.asyncio
    async def test_batch_create_claims(self, sqlite_manager):
        """Test batch creation of claims."""
        claims = [
            Claim(id="c0000001", content="Batch claim 1", confidence=0.5, created_by="user1"),
            Claim(id="c0000002", content="Batch claim 2", confidence=0.6, created_by="user1"),
            Claim(id="c0000003", content="Batch claim 3", confidence=0.7, created_by="user2"),
        ]
        
        result_ids = await sqlite_manager.batch_create_claims(claims)
        assert len(result_ids) == 3
        assert result_ids == ["c0000001", "c0000002", "c0000003"]
        
        # Verify all claims were created
        for claim_id in result_ids:
            retrieved = await sqlite_manager.get_claim(claim_id)
            assert retrieved is not None
    
    @pytest.mark.asyncio
    async def test_get_claim_count(self, sqlite_manager):
        """Test getting claim count."""
        # Initially should be 0
        count = await sqlite_manager.get_claim_count()
        assert count == 0
        
        # Create some claims
        claims = [
            Claim(id="c0000001", content="Claim 1", confidence=0.5, created_by="user1"),
            Claim(id="c0000002", content="Claim 2", confidence=0.6, created_by="user1"),
        ]
        
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Check count
        count = await sqlite_manager.get_claim_count()
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_get_dirty_claim_count(self, sqlite_manager):
        """Test getting dirty claim count."""
        # Create claims with different dirty flags
        claims = [
            Claim(id="c0000001", content="Claim 1", confidence=0.5, dirty=True, created_by="user1"),
            Claim(id="c0000002", content="Claim 2", confidence=0.6, dirty=False, created_by="user1"),
            Claim(id="c0000003", content="Claim 3", confidence=0.7, dirty=True, created_by="user2"),
        ]
        
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Check dirty count
        dirty_count = await sqlite_manager.get_dirty_claim_count()
        assert dirty_count == 2