"""
Comprehensive unit tests for SQLiteManager.
Tests CRUD operations, relationships, filtering, and error handling.
"""

import pytest
import asyncio
import aiosqlite
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, Any

from src.data.optimized_sqlite_manager import OptimizedSQLiteManager as SQLiteManager
from src.data.models import (
    Claim, Relationship, ClaimFilter, DataConfig,
    ClaimNotFoundError, InvalidClaimError, RelationshipError, DataLayerError
)

class TestSQLiteManagerInitialization:
    """Test SQLiteManager initialization and setup"""

    @pytest.mark.asyncio
    async def test_initialization_success(self, temp_sqlite_db: str):
        """Test successful initialization"""
        manager = SQLiteManager(temp_sqlite_db)
        await manager.initialize()
        
        assert manager._initialized is True
        assert manager.db_path == temp_sqlite_db
        assert manager.connection_pool is not None
        
        await manager.close()

    @pytest.mark.asyncio
    async def test_schema_creation(self, temp_sqlite_db: str):
        """Test database schema creation"""
        manager = SQLiteManager(temp_sqlite_db)
        await manager.initialize()
        
        # Check tables exist
        async with aiosqlite.connect(temp_sqlite_db) as conn:
            cursor = await conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('claims', 'claim_relationships', 'claims_fts')
            """)
            tables = [row[0] for row in await cursor.fetchall()]
            
            assert 'claims' in tables
            assert 'claim_relationships' in tables
            assert 'claims_fts' in tables
        
        await manager.close()

    @pytest.mark.asyncio
    async def test_pragma_settings(self, temp_sqlite_db: str):
        """Test SQLite PRAGMA settings"""
        manager = SQLiteManager(temp_sqlite_db)
        await manager.initialize()
        
        async with manager.connection_pool.get_connection() as conn:
            cursor = await conn.execute("PRAGMA journal_mode")
            result = await cursor.fetchone()
            assert result[0] == "wal"
        
        await manager.close()

    @pytest.mark.asyncio
    async def test_close_connection(self, temp_sqlite_db: str):
        """Test connection cleanup"""
        manager = SQLiteManager(temp_sqlite_db)
        await manager.initialize()
        await manager.close()
        
        assert manager.connection_pool is None
        assert manager._initialized is False

class TestSQLiteManagerClaimsCRUD:
    """Test claim CRUD operations"""

    @pytest.mark.asyncio
    async def test_create_claim_success(self, sqlite_manager: SQLiteManager, valid_claim: Claim):
        """Test successful claim creation"""
        claim_id = await sqlite_manager.create_claim(valid_claim)
        
        assert claim_id == valid_claim.id
        
        # Verify claim was created
        retrieved = await sqlite_manager.get_claim(claim_id)
        assert retrieved is not None
        assert retrieved['id'] == claim_id
        assert retrieved['content'] == valid_claim.content

    @pytest.mark.asyncio
    async def test_create_duplicate_claim(self, sqlite_manager: SQLiteManager, valid_claim: Claim):
        """Test creating duplicate claim raises error"""
        await sqlite_manager.create_claim(valid_claim)
        
        with pytest.raises(DataLayerError, match="Claim with ID .* already exists"):
            await sqlite_manager.create_claim(valid_claim)

    @pytest.mark.asyncio
    async def test_get_claim_existing(self, sqlite_manager: SQLiteManager, valid_claim: Claim):
        """Test retrieving existing claim"""
        await sqlite_manager.create_claim(valid_claim)
        
        retrieved = await sqlite_manager.get_claim(valid_claim.id)
        
        assert retrieved is not None
        assert retrieved['id'] == valid_claim.id
        assert retrieved['content'] == valid_claim.content
        assert retrieved['confidence'] == valid_claim.confidence
        assert retrieved['created_by'] == valid_claim.created_by

    @pytest.mark.asyncio
    async def test_get_claim_nonexistent(self, sqlite_manager: SQLiteManager):
        """Test retrieving non-existent claim"""
        result = await sqlite_manager.get_claim("c09999999")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_claim_single_field(self, sqlite_manager: SQLiteManager, valid_claim: Claim):
        """Test updating single claim field"""
        await sqlite_manager.create_claim(valid_claim)
        
        updates = {"confidence": 0.99}
        success = await sqlite_manager.update_claim(valid_claim.id, updates)
        
        assert success is True
        
        retrieved = await sqlite_manager.get_claim(valid_claim.id)
        assert retrieved['confidence'] == 0.99

    @pytest.mark.asyncio
    async def test_update_claim_multiple_fields(self, sqlite_manager: SQLiteManager, valid_claim: Claim):
        """Test updating multiple claim fields"""
        await sqlite_manager.create_claim(valid_claim)
        
        updates = {
            "confidence": 0.88,
            "dirty": True,
            "tags": ["updated", "test"]
        }
        success = await sqlite_manager.update_claim(valid_claim.id, updates)
        
        assert success is True
        
        retrieved = await sqlite_manager.get_claim(valid_claim.id)
        assert retrieved['confidence'] == 0.88
        assert retrieved['dirty'] is True
        assert "updated" in retrieved['tags']

    @pytest.mark.asyncio
    async def test_update_nonexistent_claim(self, sqlite_manager: SQLiteManager):
        """Test updating non-existent claim"""
        updates = {"confidence": 0.99}
        success = await sqlite_manager.update_claim("c09999999", updates)
        assert success is False

    @pytest.mark.asyncio
    async def test_update_empty_updates(self, sqlite_manager: SQLiteManager, valid_claim: Claim):
        """Test updating claim with empty updates"""
        await sqlite_manager.create_claim(valid_claim)
        
        success = await sqlite_manager.update_claim(valid_claim.id, {})
        assert success is False

    @pytest.mark.asyncio
    async def test_delete_claim_success(self, sqlite_manager: SQLiteManager, valid_claim: Claim):
        """Test successful claim deletion"""
        await sqlite_manager.create_claim(valid_claim)
        
        # Verify claim exists
        retrieved = await sqlite_manager.get_claim(valid_claim.id)
        assert retrieved is not None
        
        # Delete claim
        success = await sqlite_manager.delete_claim(valid_claim.id)
        assert success is True
        
        # Verify claim is deleted
        retrieved = await sqlite_manager.get_claim(valid_claim.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_claim(self, sqlite_manager: SQLiteManager):
        """Test deleting non-existent claim"""
        success = await sqlite_manager.delete_claim("c09999999")
        assert success is False

    @pytest.mark.asyncio
    async def test_batch_create_claims(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test batch claim creation"""
        claims = [Claim(**data) for data in sample_claims_data]
        
        created_ids = await sqlite_manager.batch_create_claims(claims)
        
        assert len(created_ids) == len(claims)
        for i, claim_id in enumerate(created_ids):
            assert claim_id == claims[i].id
            
            # Verify claim was created
            retrieved = await sqlite_manager.get_claim(claim_id)
            assert retrieved is not None

    @pytest.mark.asyncio
    async def test_batch_create_empty_list(self, sqlite_manager: SQLiteManager):
        """Test batch creating empty claim list"""
        created_ids = await sqlite_manager.batch_create_claims([])
        assert created_ids == []

    @pytest.mark.asyncio
    async def test_batch_create_with_duplicates(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test batch creating with duplicate IDs"""
        # Create duplicate IDs
        sample_claims_data[1]["id"] = sample_claims_data[0]["id"]
        claims = [Claim(**data) for data in sample_claims_data]
        
        with pytest.raises(DataLayerError, match="One or more claim IDs already exist"):
            await sqlite_manager.batch_create_claims(claims)

class TestSQLiteManagerRelationships:
    """Test relationship management"""

    @pytest.mark.asyncio
    async def test_add_relationship_success(self, sqlite_manager: SQLiteManager, valid_relationship: Relationship):
        """Test adding relationship successfully"""
        # Create participating claims
        claim1 = Claim(id="c0000001", content="First claim", confidence=0.7, created_by="user")
        claim2 = Claim(id="c0000002", content="Second claim", confidence=0.8, created_by="user")
        
        await sqlite_manager.create_claim(claim1)
        await sqlite_manager.create_claim(claim2)
        
        relationship_id = await sqlite_manager.add_relationship(valid_relationship)
        assert relationship_id is not None
        assert isinstance(relationship_id, int)

    @pytest.mark.asyncio
    async def test_add_relationship_nonexistent_claims(self, sqlite_manager: SQLiteManager):
        """Test adding relationship with non-existent claims"""
        relationship = Relationship(
            supporter_id="c09999999",  # Non-existent
            supported_id="c09999998",  # Non-existent
            created_by="user"
        )
        
        with pytest.raises(RelationshipError, match="One or both claim IDs do not exist"):
            await sqlite_manager.add_relationship(relationship)

    @pytest.mark.asyncio
    async def test_add_duplicate_relationship(self, sqlite_manager: SQLiteManager, valid_relationship: Relationship):
        """Test adding duplicate relationship"""
        # Create participating claims
        claim1 = Claim(id="c0000001", content="First claim", confidence=0.7, created_by="user")
        claim2 = Claim(id="c0000002", content="Second claim", confidence=0.8, created_by="user")
        
        await sqlite_manager.create_claim(claim1)
        await sqlite_manager.create_claim(claim2)
        
        # Add relationship twice
        await sqlite_manager.add_relationship(valid_relationship)
        
        with pytest.raises(RelationshipError, match="Relationship already exists"):
            await sqlite_manager.add_relationship(valid_relationship)

    @pytest.mark.asyncio
    async def test_remove_relationship_success(self, sqlite_manager: SQLiteManager, valid_relationship: Relationship):
        """Test removing relationship successfully"""
        # Create participating claims and relationship
        claim1 = Claim(id="c0000001", content="First claim", confidence=0.7, created_by="user")
        claim2 = Claim(id="c0000002", content="Second claim", confidence=0.8, created_by="user")
        
        await sqlite_manager.create_claim(claim1)
        await sqlite_manager.create_claim(claim2)
        await sqlite_manager.add_relationship(valid_relationship)
        
        # Remove relationship
        success = await sqlite_manager.remove_relationship(
            valid_relationship.supporter_id,
            valid_relationship.supported_id,
            valid_relationship.relationship_type
        )
        assert success is True

    @pytest.mark.asyncio
    async def test_remove_nonexistent_relationship(self, sqlite_manager: SQLiteManager):
        """Test removing non-existent relationship"""
        success = await sqlite_manager.remove_relationship("c0000001", "c0000002")
        assert success is False

    @pytest.mark.asyncio
    async def test_get_relationships(self, sqlite_manager: SQLiteManager):
        """Test getting relationships for claim"""
        # Create claims
        claims = [
            Claim(id="c0000001", content="Claim 1", confidence=0.7, created_by="user"),
            Claim(id="c0000002", content="Claim 2", confidence=0.8, created_by="user"),
            Claim(id="c0000003", content="Claim 3", confidence=0.6, created_by="user")
        ]
        
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Add relationships
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002",
            created_by="user"
        )
        await sqlite_manager.add_relationship(relationship)
        
        # Get relationships
        relationships = await sqlite_manager.get_relationships("c0000001")
        assert len(relationships) == 1
        assert relationships[0]["supporter_id"] == "c0000001"
        assert relationships[0]["supported_id"] == "c0000002"

    @pytest.mark.asyncio
    async def test_get_supported_by(self, sqlite_manager: SQLiteManager):
        """Test getting claims that support a claim"""
        # Create claims
        main_claim = Claim(id="c0000001", content="Main claim", confidence=0.7, created_by="user")
        supporter1 = Claim(id="c0000002", content="Supporter 1", confidence=0.8, created_by="user")
        supporter2 = Claim(id="c0000003", content="Supporter 2", confidence=0.6, created_by="user")
        
        for claim in [main_claim, supporter1, supporter2]:
            await sqlite_manager.create_claim(claim)
        
        # Add support relationships
        relationship1 = Relationship(
            supporter_id="c0000002",
            supported_id="c0000001",
            created_by="user"
        )
        relationship2 = Relationship(
            supporter_id="c0000003",
            supported_id="c0000001",
            created_by="user"
        )
        
        await sqlite_manager.add_relationship(relationship1)
        await sqlite_manager.add_relationship(relationship2)
        
        # Get supporters
        supporters = await sqlite_manager.get_supported_by("c0000001")
        assert len(supporters) == 2
        assert "c0000002" in supporters
        assert "c0000003" in supporters

    @pytest.mark.asyncio
    async def test_get_supports(self, sqlite_manager: SQLiteManager):
        """Test getting claims supported by a claim"""
        # Create claims
        supporter = Claim(id="c0000001", content="Supporting claim", confidence=0.8, created_by="user")
        supported1 = Claim(id="c0000002", content="Supported 1", confidence=0.7, created_by="user")
        supported2 = Claim(id="c0000003", content="Supported 2", confidence=0.6, created_by="user")
        
        for claim in [supporter, supported1, supported2]:
            await sqlite_manager.create_claim(claim)
        
        # Add support relationships
        relationship1 = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002",
            created_by="user"
        )
        relationship2 = Relationship(
            supporter_id="c0000001",
            supported_id="c0000003",
            created_by="user"
        )
        
        await sqlite_manager.add_relationship(relationship1)
        await sqlite_manager.add_relationship(relationship2)
        
        # Get supported claims
        supported = await sqlite_manager.get_supports("c0000001")
        assert len(supported) == 2
        assert "c0000002" in supported
        assert "c0000003" in supported

class TestSQLiteManagerFiltering:
    """Test claim filtering and search"""

    @pytest.mark.asyncio
    async def test_filter_claims_no_criteria(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test filtering with no criteria"""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        filter_obj = ClaimFilter()
        results = await sqlite_manager.filter_claims(filter_obj)
        
        assert len(results) == len(sample_claims_data)

    @pytest.mark.asyncio
    async def test_filter_by_tags(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test filtering by tags"""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Filter by physics tag
        filter_obj = ClaimFilter(tags=["physics"])
        results = await sqlite_manager.filter_claims(filter_obj)
        
        # Should return claims with physics tag
        for result in results:
            tag_list = result.get('tags', [])
            tag_str = ','.join(tag_list) if isinstance(tag_list, list) else tag_list
            assert "physics" in tag_str

    @pytest.mark.asyncio
    async def test_filter_by_confidence_range(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test filtering by confidence range"""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Filter by confidence range
        filter_obj = ClaimFilter(confidence_min=0.9, confidence_max=0.95)
        results = await sqlite_manager.filter_claims(filter_obj)
        
        # Should return claims within range
        for result in results:
            assert 0.9 <= result['confidence'] <= 0.95

    @pytest.mark.asyncio
    async def test_filter_dirty_claims(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test filtering dirty claims"""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Filter dirty claims
        filter_obj = ClaimFilter(dirty_only=True)
        results = await sqlite_manager.filter_claims(filter_obj)
        
        # Should return only dirty claims
        for result in results:
            assert result['dirty'] is True

    @pytest.mark.asyncio
    async def test_filter_by_creator(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test filtering by creator"""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Filter by creator
        filter_obj = ClaimFilter(created_by="test_user")
        results = await sqlite_manager.filter_claims(filter_obj)
        
        # Should return claims by specified creator
        for result in results:
            assert result['created_by'] == "test_user"

    @pytest.mark.asyncio
    async def test_filter_with_pagination(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test filtering with pagination"""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Test pagination
        filter_obj = ClaimFilter(limit=2, offset=1)
        results = await sqlite_manager.filter_claims(filter_obj)
        
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_complex_filter_combinations(self, sqlite_manager: SQLiteManager):
        """Test complex filtering with multiple criteria"""
        # Create claims with varied properties
        claims_data = [
            {
                "id": "c0000001",
                "content": "Physics claim about quantum mechanics",
                "confidence": 0.9,
                "dirty": True,
                "tags": ["physics"],
                "created_by": "scientist"
            },
            {
                "id": "c0000002",
                "content": "Chemistry claim about reactions",
                "confidence": 0.7,
                "dirty": False,
                "tags": ["chemistry"],
                "created_by": "scientist"
            },
            {
                "id": "c0000003",
                "content": "Biology claim about DNA",
                "confidence": 0.85,
                "dirty": True,
                "tags": ["biology"],
                "created_by": "student"
            }
        ]
        
        claims = [Claim(**data) for data in claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Complex filter: physics claims by scientist with high confidence
        filter_obj = ClaimFilter(
            tags=["physics"],
            confidence_min=0.8,
            created_by="scientist"
        )
        results = await sqlite_manager.filter_claims(filter_obj)
        
        # Should match specific criteria
        assert len(results) == 1
        assert results[0]['id'] == "c0000001"

    @pytest.mark.asyncio
    async def test_get_dirty_claims(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test getting dirty claims specifically"""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Get dirty claims
        dirty_claims = await sqlite_manager.get_dirty_claims()
        
        # Should return only dirty claims
        for claim in dirty_claims:
            assert claim['dirty'] is True

    @pytest.mark.asyncio
    async def test_full_text_search(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test full-text search functionality"""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Search for specific terms
        search_results = await sqlite_manager.search_claims_by_content("speed", limit=5)
        
        # Should return claims matching search term
        for result in search_results:
            assert "speed" in result['content'].lower()

class TestSQLiteManagerStatistics:
    """Test statistics and utility functions"""

    @pytest.mark.asyncio
    async def test_get_statistics_empty_db(self, sqlite_manager: SQLiteManager):
        """Test getting statistics from empty database"""
        stats = await sqlite_manager.get_statistics()
        
        assert stats['total_claims'] == 0
        assert stats['total_relationships'] == 0
        assert stats['dirty_claims'] == 0
        assert stats['average_confidence'] == 0.0

    @pytest.mark.asyncio
    async def test_get_statistics_with_data(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test getting statistics with data"""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Add some relationships
        relationship = Relationship(
            supporter_id="c0000001",
            supported_id="c0000002",
            created_by="user"
        )
        await sqlite_manager.add_relationship(relationship)
        
        stats = await sqlite_manager.get_statistics()
        
        assert stats['total_claims'] == len(sample_claims_data)
        assert stats['total_relationships'] == 1
        assert stats['dirty_claims'] >= 0
        assert stats['average_confidence'] > 0.0

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, sqlite_manager: SQLiteManager):
        """Test cleanup of old data"""
        # Create old claim
        old_claim = Claim(
            id="cold00001",
            content="Old claim",
            confidence=0.5,
            created_by="test",
            created_at=datetime.utcnow() - timedelta(days=35)  # 35 days old
        )
        await sqlite_manager.create_claim(old_claim)
        
        # Create recent claim
        recent_claim = Claim(
            id="crec00001",
            content="Recent claim",
            confidence=0.5,
            created_by="test",
            created_at=datetime.utcnow() - timedelta(days=5)  # 5 days old
        )
        await sqlite_manager.create_claim(recent_claim)
        
        # Cleanup old data (30 days)
        deleted_count = await sqlite_manager.cleanup_old_data(days_old=30)
        
        # Should have deleted old claim
        assert deleted_count > 0
        
        # Recent claim should still exist
        retrieved = await sqlite_manager.get_claim("crec00001")
        assert retrieved is not None
        
        # Old claim should be deleted
        retrieved = await sqlite_manager.get_claim("cold00001")
        assert retrieved is None

class TestSQLiteManagerErrorHandling:
    """Test error handling and edge cases"""

    @pytest.mark.asyncio
    async def test_uninitialized_error(self):
        """Test operations on uninitialized manager"""
        manager = SQLiteManager(":memory:")
        
        with pytest.raises(DataLayerError, match="SQLite database not initialized"):
            await manager.get_claim("c0000001")

    @pytest.mark.asyncio
    async def test_invalid_claim_data(self, sqlite_manager: SQLiteManager):
        """Test handling invalid claim data"""
        invalid_claim = Claim(
            id="invalid",
            content="Too short",  # Violates validation
            confidence=1.5,  # Invalid range
            created_by="user"
        )
        
        with pytest.raises(Exception):  # Should raise validation error
            await sqlite_manager.create_claim(invalid_claim)

    @pytest.mark.asyncio
    async def test_database_connection_error(self):
        """Test handling database connection errors"""
        # Use invalid path
        manager = SQLiteManager("/invalid/path/that/does/not/exist/test.db")
        
        with pytest.raises(DataLayerError):
            await manager.initialize()