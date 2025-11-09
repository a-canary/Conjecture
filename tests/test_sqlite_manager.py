"""
Comprehensive unit tests for SQLiteManager in the Conjecture data layer.
Tests CRUD operations, relationships, filtering, and performance.
"""
import pytest
import asyncio
import aiosqlite
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.data.sqlite_manager import SQLiteManager
from src.data.models import (
    Claim, Relationship, ClaimFilter, DataConfig,
    ClaimNotFoundError, InvalidClaimError, RelationshipError, DataLayerError
)


class TestSQLiteManagerInitialization:
    """Test SQLiteManager initialization and setup."""

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_sqlite_manager_initialization(self, sqlite_manager: SQLiteManager):
        """Test that SQLiteManager initializes correctly."""
        assert sqlite_manager is not None
        assert sqlite_manager.connection_pool is not None
        assert sqlite_manager.db_path != ""

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_schema_creation(self, sqlite_manager: SQLiteManager):
        """Test that database schema is created properly."""
        async with sqlite_manager.connection_pool.acquire() as conn:
            # Check that tables exist
            cursor = await conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('claims', 'claim_relationships', 'claims_fts')
            """)
            tables = [row[0] for row in await cursor.fetchall()]
            
            assert 'claims' in tables
            assert 'claim_relationships' in tables
            assert 'claims_fts' in tables

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_indexes_creation(self, sqlite_manager: SQLiteManager):
        """Test that performance indexes are created."""
        async with sqlite_manager.connection_pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name LIKE 'idx_%'
            """)
            indexes = [row[0] for row in await cursor.fetchall()]
            
            # Check that expected indexes exist
            expected_indexes = [
                'idx_claims_confidence',
                'idx_claims_dirty',
                'idx_claims_created_at',
                'idx_claims_created_by',
                'idx_relationships_supporter',
                'idx_relationships_supported',
                'idx_relationships_type'
            ]
            
            for expected_index in expected_indexes:
                assert expected_index in indexes

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_fts_triggers_creation(self, sqlite_manager: SQLiteManager):
        """Test that full-text search triggers are created."""
        async with sqlite_manager.connection_pool.acquire() as conn:
            cursor = await conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='trigger' AND name LIKE 'claims_fts_%'
            """)
            triggers = [row[0] for row in await cursor.fetchall()]
            
            expected_triggers = ['claims_fts_insert', 'claims_fts_delete', 'claims_fts_update']
            for expected_trigger in expected_triggers:
                assert expected_trigger in triggers

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_close_connection_pool(self, sqlite_manager: SQLiteManager):
        """Test that connection pool can be closed properly."""
        await sqlite_manager.close()
        assert sqlite_manager.connection_pool is None


class TestSQLiteManagerClaimsCRUD:
    """Test CRUD operations for claims."""

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_create_claim(self, sqlite_manager: SQLiteManager, valid_claim: Claim):
        """Test creating a new claim."""
        result_id = await sqlite_manager.create_claim(valid_claim)
        
        assert result_id == valid_claim.id

        # Verify claim was actually created
        retrieved_claim = await sqlite_manager.get_claim(valid_claim.id)
        assert retrieved_claim is not None
        assert retrieved_claim['id'] == valid_claim.id
        assert retrieved_claim['content'] == valid_claim.content

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_create_duplicate_claim(self, sqlite_manager: SQLiteManager, valid_claim: Claim):
        """Test creating a claim with duplicate ID raises error."""
        await sqlite_manager.create_claim(valid_claim)
        
        with pytest.raises(DataLayerError, match="Claim with ID .* already exists"):
            await sqlite_manager.create_claim(valid_claim)

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_get_claim_existing(self, sqlite_manager: SQLiteManager, valid_claim: Claim):
        """Test retrieving an existing claim."""
        await sqlite_manager.create_claim(valid_claim)
        
        retrieved = await sqlite_manager.get_claim(valid_claim.id)
        
        assert retrieved is not None
        assert retrieved['id'] == valid_claim.id
        assert retrieved['content'] == valid_claim.content
        assert retrieved['confidence'] == valid_claim.confidence
        assert retrieved['dirty'] == valid_claim.dirty
        assert retrieved['tags'] == valid_claim.tags
        assert retrieved['created_by'] == valid_claim.created_by
        assert isinstance(retrieved['created_at'], datetime)

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_get_claim_nonexistent(self, sqlite_manager: SQLiteManager):
        """Test retrieving a non-existent claim returns None."""
        result = await sqlite_manager.get_claim("c0999999")
        assert result is None

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_update_claim_single_field(self, sqlite_manager: SQLiteManager, valid_claim: Claim):
        """Test updating a single field of a claim."""
        await sqlite_manager.create_claim(valid_claim)
        
        # Update confidence
        updates = {"confidence": 0.99}
        updated = await sqlite_manager.update_claim(valid_claim.id, updates)
        
        assert updated is True
        
        # Verify update
        retrieved = await sqlite_manager.get_claim(valid_claim.id)
        assert retrieved['confidence'] == 0.99
        assert retrieved['content'] == valid_claim.content  # Unchanged

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_update_claim_multiple_fields(self, sqlite_manager: SQLiteManager, valid_claim: Claim):
        """Test updating multiple fields of a claim."""
        await sqlite_manager.create_claim(valid_claim)
        
        updates = {
            "confidence": 0.88,
            "dirty": False,
            "tags": ["new_tag", "another_tag"]
        }
        updated = await sqlite_manager.update_claim(valid_claim.id, updates)
        
        assert updated is True
        
        retrieved = await sqlite_manager.get_claim(valid_claim.id)
        assert retrieved['confidence'] == 0.88
        assert retrieved['dirty'] is False
        assert retrieved['tags'] == ["new_tag", "another_tag"]

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_update_nonexistent_claim(self, sqlite_manager: SQLiteManager):
        """Test updating a non-existent claim."""
        updates = {"confidence": 0.99}
        updated = await sqlite_manager.update_claim("c0999999", updates)
        assert updated is False

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_update_claim_empty_updates(self, sqlite_manager: SQLiteManager, valid_claim: Claim):
        """Test updating claim with empty updates does nothing."""
        await sqlite_manager.create_claim(valid_claim)
        
        updated = await sqlite_manager.update_claim(valid_claim.id, {})
        assert updated is False

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_delete_claim(self, sqlite_manager: SQLiteManager, valid_claim: Claim):
        """Test deleting a claim."""
        await sqlite_manager.create_claim(valid_claim)
        
        # Verify claim exists
        retrieved = await sqlite_manager.get_claim(valid_claim.id)
        assert retrieved is not None
        
        # Delete claim
        deleted = await sqlite_manager.delete_claim(valid_claim.id)
        assert deleted is True
        
        # Verify claim is gone
        retrieved = await sqlite_manager.get_claim(valid_claim.id)
        assert retrieved is None

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_delete_nonexistent_claim(self, sqlite_manager: SQLiteManager):
        """Test deleting a non-existent claim."""
        deleted = await sqlite_manager.delete_claim("c0999999")
        assert deleted is False


class TestSQLiteManagerRelationships:
    """Test relationship management operations."""

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_add_relationship(self, sqlite_manager: SQLiteManager, valid_relationship: Relationship):
        """Test adding a relationship between claims."""
        # Create the referenced claims first
        claim1 = Claim(id="c0000001", content="First claim", confidence=0.7, created_by="user")
        claim2 = Claim(id="c0000002", content="Second claim", confidence=0.8, created_by="user")
        
        await sqlite_manager.create_claim(claim1)
        await sqlite_manager.create_claim(claim2)
        
        relationship_id = await sqlite_manager.add_relationship(valid_relationship)
        
        assert relationship_id is not None
        assert isinstance(relationship_id, int)

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_add_relationship_nonexistent_claim(self, sqlite_manager: SQLiteManager):
        """Test adding relationship with non-existent claim fails."""
        relationship = Relationship(
            supporter_id="c0999999",  # Non-existent
            supported_id="c0000001",
            relationship_type="supports"
        )
        
        with pytest.raises(DataLayerError, match="One or both claim IDs do not exist"):
            await sqlite_manager.add_relationship(relationship)

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_add_duplicate_relationship(self, sqlite_manager: SQLiteManager, valid_relationship: Relationship):
        """Test adding duplicate relationship fails."""
        # Create claims
        claim1 = Claim(id="c0000001", content="First claim", confidence=0.7, created_by="user")
        claim2 = Claim(id="c0000002", content="Second claim", confidence=0.8, created_by="user")
        
        await sqlite_manager.create_claim(claim1)
        await sqlite_manager.create_claim(claim2)
        
        # Add relationship twice
        await sqlite_manager.add_relationship(valid_relationship)
        
        with pytest.raises(DataLayerError, match="Relationship already exists"):
            await sqlite_manager.add_relationship(valid_relationship)

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_remove_relationship(self, sqlite_manager: SQLiteManager, valid_relationship: Relationship):
        """Test removing a relationship."""
        # Create claims and relationship
        claim1 = Claim(id="c0000001", content="First claim", confidence=0.7, created_by="user")
        claim2 = Claim(id="c0000002", content="Second claim", confidence=0.8, created_by="user")
        
        await sqlite_manager.create_claim(claim1)
        await sqlite_manager.create_claim(claim2)
        await sqlite_manager.add_relationship(valid_relationship)
        
        # Remove relationship
        removed = await sqlite_manager.remove_relationship(
            valid_relationship.supporter_id,
            valid_relationship.supported_id,
            valid_relationship.relationship_type
        )
        assert removed is True

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_remove_nonexistent_relationship(self, sqlite_manager: SQLiteManager):
        """Test removing non-existent relationship."""
        removed = await sqlite_manager.remove_relationship("c0000001", "c0000002")
        assert removed is False

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_get_relationships(self, sqlite_manager: SQLiteManager):
        """Test retrieving relationships for a claim."""
        # Create multiple claims and relationships
        claims = [
            Claim(id="c0000001", content="Claim 1", confidence=0.7, created_by="user"),
            Claim(id="c0000002", content="Claim 2", confidence=0.8, created_by="user"),
            Claim(id="c0000003", content="Claim 3", confidence=0.6, created_by="user")
        ]
        
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Add relationships: c0000001 supports c0000002, c0000003 supports c0000001
        rel1 = Relationship(supporter_id="c0000001", supported_id="c0000002", relationship_type="supports")
        rel2 = Relationship(supporter_id="c0000003", supported_id="c0000001", relationship_type="supports")
        
        await sqlite_manager.add_relationship(rel1)
        await sqlite_manager.add_relationship(rel2)
        
        # Get relationships for c0000001
        relationships = await sqlite_manager.get_relationships("c0000001")
        
        assert len(relationships) == 2
        relationship_ids = [rel['id'] for rel in relationships]
        assert len(relationship_ids) == 2  # Should have both relationships

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_get_supported_by(self, sqlite_manager: SQLiteManager):
        """Test getting claims that support a given claim."""
        # Create claims
        claims = [
            Claim(id="c0000001", content="Main claim", confidence=0.7, created_by="user"),
            Claim(id="c0000002", content="Support 1", confidence=0.8, created_by="user"),
            Claim(id="c0000003", content="Support 2", confidence=0.6, created_by="user")
        ]
        
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Add support relationships
        rel1 = Relationship(supporter_id="c0000002", supported_id="c0000001", relationship_type="supports")
        rel2 = Relationship(supporter_id="c0000003", supported_id="c0000001", relationship_type="supports")
        
        await sqlite_manager.add_relationship(rel1)
        await sqlite_manager.add_relationship(rel2)
        
        # Get supporters for main claim
        supporters = await sqlite_manager.get_supported_by("c0000001")
        
        assert len(supporters) == 2
        assert "c0000002" in supporters
        assert "c0000003" in supporters

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_get_supports(self, sqlite_manager: SQLiteManager):
        """Test getting claims that a given claim supports."""
        # Create claims
        claims = [
            Claim(id="c0000001", content="Supporting claim", confidence=0.8, created_by="user"),
            Claim(id="c0000002", content="Supported 1", confidence=0.7, created_by="user"),
            Claim(id="c0000003", content="Supported 2", confidence=0.6, created_by="user")
        ]
        
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Add support relationships
        rel1 = Relationship(supporter_id="c0000001", supported_id="c0000002", relationship_type="supports")
        rel2 = Relationship(supporter_id="c0000001", supported_id="c0000003", relationship_type="supports")
        
        await sqlite_manager.add_relationship(rel1)
        await sqlite_manager.add_relationship(rel2)
        
        # Get supported claims
        supported = await sqlite_manager.get_supports("c0000001")
        
        assert len(supported) == 2
        assert "c0000002" in supported
        assert "c0000003" in supported


class TestSQLiteManagerFiltering:
    """Test claim filtering and search capabilities."""

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_filter_claims_no_filter(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test filtering with no criteria (returns all claims)."""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        filter_obj = ClaimFilter()
        results = await sqlite_manager.filter_claims(filter_obj)
        
        assert len(results) == len(sample_claims_data)

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_filter_by_tags(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test filtering by tags."""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Filter by physics tag
        filter_obj = ClaimFilter(tags=["physics"])
        results = await sqlite_manager.filter_claims(filter_obj)
        
        # Should return claims with physics tag
        assert len(results) >= 1
        for result in results:
            assert "physics" in result['tags']

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_filter_by_confidence_min(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test filtering by minimum confidence."""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Filter by minimum confidence
        filter_obj = ClaimFilter(confidence_min=0.95)
        results = await sqlite_manager.filter_claims(filter_obj)
        
        # Should return claims with confidence >= 0.95
        for result in results:
            assert result['confidence'] >= 0.95

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_filter_by_confidence_range(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test filtering by confidence range."""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Filter by confidence range
        filter_obj = ClaimFilter(confidence_min=0.9, confidence_max=0.95)
        results = await sqlite_manager.filter_claims(filter_obj)
        
        # Should return claims within confidence range
        for result in results:
            assert 0.9 <= result['confidence'] <= 0.95

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_filter_dirty_claims(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test filtering dirty claims."""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Filter dirty claims only
        filter_obj = ClaimFilter(dirty_only=True)
        results = await sqlite_manager.filter_claims(filter_obj)
        
        # Should return only dirty claims
        for result in results:
            assert result['dirty'] is True

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_filter_by_creator(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test filtering by creator."""
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

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_filter_by_content_contains(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test filtering by content search."""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Filter by content
        filter_obj = ClaimFilter(content_contains="speed")
        results = await sqlite_manager.filter_claims(filter_obj)
        
        # Should return claims containing the search term
        for result in results:
            assert "speed" in result['content'].lower()

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_filter_with_pagination(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test filtering with pagination."""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Test pagination
        filter_obj = ClaimFilter(limit=2, offset=1)
        results = await sqlite_manager.filter_claims(filter_obj)
        
        assert len(results) <= 2

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_get_dirty_claims(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test getting dirty claims specifically."""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Get dirty claims
        dirty_claims = await sqlite_manager.get_dirty_claims()
        
        # Should return only dirty claims
        for claim in dirty_claims:
            assert claim['dirty'] is True

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_full_text_search(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test full-text search functionality."""
        # Create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Search for specific terms
        search_results = await sqlite_manager.search_claims_by_content("speed", limit=5)
        
        # Should return claims matching the search term
        for result in search_results:
            assert "speed" in result['content'].lower()

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_complex_filter_combinations(self, sqlite_manager: SQLiteManager):
        """Test complex filtering with multiple criteria."""
        # Create claims with various properties
        claims_data = [
            {"id": "c0000001", "content": "Physics claim about quantum mechanics", "confidence": 0.9, "dirty": True, "tags": ["physics"], "created_by": "scientist"},
            {"id": "c0000002", "content": "Chemistry claim about reactions", "confidence": 0.7, "dirty": False, "tags": ["chemistry"], "created_by": "scientist"},
            {"id": "c0000003", "content": "Biology claim about DNA", "confidence": 0.85, "dirty": True, "tags": ["biology"], "created_by": "student"},
            {"id": "c0000004", "content": "Physics claim about relativity", "confidence": 0.95, "dirty": False, "tags": ["physics"], "created_by": "scientist"},
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
        for result in results:
            assert "physics" in result['tags']
            assert result['confidence'] >= 0.8
            assert result['created_by'] == "scientist"


class TestSQLiteManagerBatchOperations:
    """Test batch operations for efficiency."""

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_batch_create_claims(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test creating multiple claims efficiently."""
        claims = [Claim(**data) for data in sample_claims_data]
        
        # Batch create
        created_ids = await sqlite_manager.batch_create_claims(claims)
        
        assert len(created_ids) == len(claims)
        for i, claim_id in enumerate(created_ids):
            assert claim_id == claims[i].id
        
        # Verify all claims were created
        for claim in claims:
            retrieved = await sqlite_manager.get_claim(claim.id)
            assert retrieved is not None

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_batch_create_empty_list(self, sqlite_manager: SQLiteManager):
        """Test batch creating empty list of claims."""
        created_ids = await sqlite_manager.batch_create_claims([])
        assert created_ids == []

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_batch_create_large_number(self, sqlite_manager: SQLiteManager, claim_generator):
        """Test batch creating large number of claims."""
        claims_data = claim_generator.generate(100)
        claims = [Claim(**data) for data in claims_data]
        
        created_ids = await sqlite_manager.batch_create_claims(claims)
        
        assert len(created_ids) == 100
        
        # Verify a few random claims
        import random
        for _ in range(5):
            claim = random.choice(claims)
            retrieved = await sqlite_manager.get_claim(claim.id)
            assert retrieved is not None


class TestSQLiteManagerStatistics:
    """Test statistics and count operations."""

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_get_claim_count(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test getting total claim count."""
        # Initially should be 0
        count = await sqlite_manager.get_claim_count()
        assert count == 0
        
        # Add claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Count should match
        count = await sqlite_manager.get_claim_count()
        assert count == len(claims)

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_get_dirty_claim_count(self, sqlite_manager: SQLiteManager, sample_claims_data):
        """Test getting dirty claim count."""
        # Add claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        # Count dirty claims
        dirty_count = await sqlite_manager.get_dirty_claim_count()
        
        # Verify count matches actual dirty claims
        dirty_claims = [claim for claim in claims if claim.dirty]
        assert dirty_count == len(dirty_claims)


class TestSQLiteManagerErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_foreign_key_constraint_violation(self, sqlite_manager: SQLiteManager):
        """Test foreign key constraint violation handling."""
        relationship = Relationship(
            supporter_id="c0999999",  # Non-existent claim
            supported_id="c0999998",  # Non-existent claim
            relationship_type="supports"
        )
        
        with pytest.raises(DataLayerError, match="One or both claim IDs do not exist"):
            await sqlite_manager.add_relationship(relationship)

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_invalid_sql_operations(self, sqlite_manager: SQLiteManager):
        """Test handling of invalid SQL operations."""
        # This would normally cause SQL errors internally
        # Test that error handling prevents crashes
        with pytest.raises(DataLayerError):
            await sqlite_manager.update_claim("invalid_id", {"confidence": 999.99})

    @pytest.mark.sqlite
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, sqlite_manager: SQLiteManager):
        """Test concurrent database operations."""
        async def create_claims(start_id: int, count: int):
            for i in range(count):
                claim = Claim(
                    id=f"c{start_id + i:07d}",
                    content=f"Concurrent claim {i}",
                    confidence=0.7,
                    created_by="concurrent_user"
                )
                await sqlite_manager.create_claim(claim)
        
        # Run concurrent operations
        await asyncio.gather(
            create_claims(1000001, 5),
            create_claims(1000006, 5),
            create_claims(1000011, 5)
        )
        
        # Verify all claims were created
        count = await sqlite_manager.get_claim_count()
        assert count == 15


class TestSQLiteManagerPerformance:
    """Performance tests for SQLite operations."""

    @pytest.mark.sqlite
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_simple_query_performance(self, sqlite_manager: SQLiteManager, benchmark):
        """Benchmark simple claim retrieval performance."""
        # Setup: create a claim
        claim = Claim(
            id="c0000001",
            content="Performance test claim",
            confidence=0.7,
            created_by="perf_user"
        )
        await sqlite_manager.create_claim(claim)
        
        async def get_claim():
            return await sqlite_manager.get_claim("c0000001")
        
        result = await benchmark.async_timer(get_claim)
        # Should be very fast (<10ms for simple query)
        assert result < 0.01  # 10ms
        assert result["result"] is not None

    @pytest.mark.sqlite
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_filter_query_performance(self, sqlite_manager: SQLiteManager, sample_claims_data, benchmark):
        """Benchmark filter query performance."""
        # Setup: create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        filter_obj = ClaimFilter(tags=["physics"])
        
        async def filter_claims():
            return await sqlite_manager.filter_claims(filter_obj)
        
        result = await benchmark.async_timer(filter_claims)
        # Should be fast (<50ms for filter query)
        assert result < 0.05  # 50ms
        assert len(result["result"]) >= 0

    @pytest.mark.sqlite
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_create_performance(self, sqlite_manager: SQLiteManager, claim_generator, benchmark):
        """Benchmark batch create performance."""
        claims_data = claim_generator.generate(100)
        claims = [Claim(**data) for data in claims_data]
        
        async def batch_create():
            return await sqlite_manager.batch_create_claims(claims)
        
        result = await benchmark.async_timer(batch_create)
        # Should be reasonable (<1s for 100 claims)
        assert result < 1.0  # 1s
        assert len(result["result"]) == 100

    @pytest.mark.sqlite
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_full_text_search_performance(self, sqlite_manager: SQLiteManager, sample_claims_data, benchmark):
        """Benchmark full-text search performance."""
        # Setup: create sample claims
        claims = [Claim(**data) for data in sample_claims_data]
        for claim in claims:
            await sqlite_manager.create_claim(claim)
        
        async def search_claims():
            return await sqlite_manager.search_claims_by_content("speed", limit=10)
        
        result = await benchmark.async_timer(search_claims)
        # Should be fast (<100ms for FTS)
        assert result < 0.1  # 100ms
        assert isinstance(result["result"], list)

    @pytest.mark.sqlite
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_relationship_query_performance(self, sqlite_manager: SQLiteManager, benchmark):
        """Benchmark relationship query performance."""
        # Setup: create claims and relationships
        base_claim = Claim(id="c0000001", content="Base claim", confidence=0.7, created_by="user")
        await sqlite_manager.create_claim(base_claim)
        
        # Create many supporting claims
        for i in range(50):
            claim = Claim(
                id=f"c0000{i+2:04d}",
                content=f"Supporting claim {i}",
                confidence=0.8,
                created_by="user"
            )
            await sqlite_manager.create_claim(claim)
            
            rel = Relationship(
                supporter_id=claim.id,
                supported_id=base_claim.id,
                relationship_type="supports"
            )
            await sqlite_manager.add_relationship(rel)
        
        async def get_supporters():
            return await sqlite_manager.get_supported_by("c0000001")
        
        result = await benchmark.async_timer(get_supporters)
        # Should be fast (<100ms for relationships)
        assert result < 0.1  # 100ms
        assert len(result["result"]) == 50


class TestSQLiteManagerScalability:
    """Scalability tests with larger datasets."""

    @pytest.mark.sqlite
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, sqlite_manager: SQLiteManager, large_claim_dataset):
        """Test performance with large dataset (1000 claims)."""
        import time
        
        # Create claims in batches
        batch_size = 100
        claims_created = 0
        
        start_time = time.time()
        
        for i in range(0, len(large_claim_dataset), batch_size):
            batch_data = large_claim_dataset[i:i + batch_size]
            # Assign sequential IDs
            for j, data in enumerate(batch_data):
                data["id"] = f"c{claims_created + j + 1:07d}"
            
            claims = [Claim(**data) for data in batch_data]
            await sqlite_manager.batch_create_claims(claims)
            claims_created += len(batch_data)
        
        creation_time = time.time() - start_time
        
        # Test query performance with large dataset
        start_time = time.time()
        all_claims = await sqlite_manager.filter_claims(ClaimFilter(limit=100))
        query_time = time.time() - start_time
        
        # Performance benchmarks
        assert creation_time < 10.0  # Should create 1000 claims in <10s
        assert query_time < 0.1     # Should query in <100ms
        assert len(all_claims) == 100
        
        # Test filtering performance
        start_time = time.time()
        physics_claims = await sqlite_manager.filter_claims(ClaimFilter(tags=["physics"]))
        filter_time = time.time() - start_time
        
        assert filter_time < 0.05   # Filter should be very fast

    @pytest.mark.sqlite
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_usage_scaling(self, sqlite_manager: SQLiteManager, claim_generator):
        """Test memory usage doesn't grow excessively."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create many claims in batches
        for batch in range(5):
            claims_data = claim_generator.generate(200)
            claims = []
            
            for i, data in enumerate(claims_data):
                data["id"] = f"c{batch * 200 + i + 1:07d}"
                claims.append(Claim(**data))
            
            await sqlite_manager.batch_create_claims(claims)
            
            # Check memory doesn't grow too much
            current_memory = process.memory_info().rss
            memory_growth = (current_memory - initial_memory) / 1024 / 1024  # MB
            
            # Should not grow more than 100MB for 1000 claims
            assert memory_growth < 100


# Integration tests that simulate real-world usage patterns
class TestSQLiteManagerIntegration:
    """Integration tests simulating real usage patterns."""

    @pytest.mark.sqlite
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_claim_lifecycle(self, sqlite_manager: SQLiteManager):
        """Test complete lifecycle of a claim."""
        # Create claim
        claim = Claim(
            id="c0000001",
            content="Test claim for lifecycle testing",
            confidence=0.5,
            dirty=True,
            tags=["test", "lifecycle"],
            created_by="integration_user"
        )
        
        claim_id = await sqlite_manager.create_claim(claim)
        assert claim_id == claim.id
        
        # Read claim
        retrieved = await sqlite_manager.get_claim(claim_id)
        assert retrieved is not None
        assert retrieved['content'] == claim.content
        
        # Update claim
        updates = {
            "confidence": 0.8,
            "dirty": False,
            "tags": ["test", "lifecycle", "updated"]
        }
        updated = await sqlite_manager.update_claim(claim_id, updates)
        assert updated is True
        
        # Verify update
        updated_claim = await sqlite_manager.get_claim(claim_id)
        assert updated_claim['confidence'] == 0.8
        assert updated_claim['dirty'] is False
        assert "updated" in updated_claim['tags']
        
        # Add relationships
        supporting_claim = Claim(
            id="c0000002",
            content="Supporting evidence",
            confidence=0.9,
            created_by="integration_user"
        )
        await sqlite_manager.create_claim(supporting_claim)
        
        relationship = Relationship(
            supporter_id="c0000002",
            supported_id="c0000001",
            relationship_type="supports"
        )
        relationship_id = await sqlite_manager.add_relationship(relationship)
        assert relationship_id is not None
        
        # Verify relationship
        supporters = await sqlite_manager.get_supported_by("c0000001")
        assert "c0000002" in supporters
        
        # Delete claim
        deleted = await sqlite_manager.delete_claim(claim_id)
        assert deleted is True
        
        # Verify deletion
        deleted_claim = await sqlite_manager.get_claim(claim_id)
        assert deleted_claim is None

    @pytest.mark.sqlite
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complex_relationship_network(self, sqlite_manager: SQLiteManager):
        """Test managing a complex network of relationships."""
        # Create a hierarchy of claims
        claim_data = [
            {"id": "c0000001", "content": "Root claim", "confidence": 0.5},
            {"id": "c0000002", "content": "Supporting claim 1", "confidence": 0.7},
            {"id": "c0000003", "content": "Supporting claim 2", "confidence": 0.6},
            {"id": "c0000004", "content": "Evidence for claim 1", "confidence": 0.8},
            {"id": "c0000005", "content": "Evidence for claim 2", "confidence": 0.9},
        ]
        
        # Create claims
        for data in claim_data:
            claim = Claim(
                id=data["id"],
                content=data["content"],
                confidence=data["confidence"],
                created_by="relationship_test"
            )
            await sqlite_manager.create_claim(claim)
        
        # Create relationship network
        relationships = [
            ("c0000002", "c0000001", "supports"),  # 2 supports 1
            ("c0000003", "c0000001", "supports"),  # 3 supports 1
            ("c0000004", "c0000002", "supports"),  # 4 supports 2
            ("c0000005", "c0000003", "supports"),  # 5 supports 3
            ("c0000002", "c0000003", "contradicts"),  # 2 contradicts 3
        ]
        
        for supporter, supported, rel_type in relationships:
            rel = Relationship(
                supporter_id=supporter,
                supported_id=supported,
                relationship_type=rel_type
            )
            await sqlite_manager.add_relationship(rel)
        
        # Verify the network structure
        # Root claim should have 2 supporters
        root_supporters = await sqlite_manager.get_supported_by("c0000001")
        assert len(root_supporters) == 2
        assert "c0000002" in root_supporters
        assert "c0000003" in root_supporters
        
        # Check individual claim relationships
        c0000002_rels = await sqlite_manager.get_relationships("c0000002")
        assert len(c0000002_rels) == 3  # supports 1, supports by 4, contradicts 3
        
        # Test cascade deletion
        await sqlite_manager.delete_claim("c0000002")
        
        # Relationship should be removed automatically
        c0000001_supporters = await sqlite_manager.get_supported_by("c0000001")
        assert "c0000002" not in c0000001_supporters
        assert len(c0000001_supporters) == 1  # Only c0000003 remains

    @pytest.mark.sqlite
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_search_and_filter_workflow(self, sqlite_manager: SQLiteManager):
        """Test realistic search and filtering workflow."""
        # Create diverse claim dataset
        claims_data = [
            {"id": "c0000001", "content": "Quantum entanglement allows particles to be correlated", "confidence": 0.9, "tags": ["physics", "quantum"], "dirty": False},
            {"id": "c0000002", "content": "Water is composed of hydrogen and oxygen atoms", "confidence": 0.95, "tags": ["chemistry", "water"], "dirty": False},
            {"id": "c0000003", "content": "DNA mutations can lead to genetic diseases", "confidence": 0.85, "tags": ["biology", "genetics"], "dirty": True},
            {"id": "c0000004", "content": "The theory of relativity describes spacetime curvature", "confidence": 0.88, "tags": ["physics", "relativity"], "dirty": True},
            {"id": "c0000005", "content": "Climate change is caused by greenhouse gas emissions", "confidence": 0.92, "tags": ["climate", "environment"], "dirty": False},
        ]
        
        # Create claims
        for data in claims_data:
            claim = Claim(
                id=data["id"],
                content=data["content"],
                confidence=data["confidence"],
                tags=data["tags"],
                dirty=data["dirty"],
                created_by="workflow_test"
            )
            await sqlite_manager.create_claim(claim)
        
        # Workflow 1: Find high-confidence physics claims
        physics_filter = ClaimFilter(
            tags=["physics"],
            confidence_min=0.85
        )
        physics_claims = await sqlite_manager.filter_claims(physics_filter)
        assert len(physics_claims) == 2  # c0000001 and c0000004
        
        # Workflow 2: Find dirty claims needing review
        dirty_claims = await sqlite_manager.get_dirty_claims()
        assert len(dirty_claims) == 2  # c0000003 and c0000004
        
        # Workflow 3: Full-text search for specific terms
        search_results = await sqlite_manager.search_claims_by_content("quantum")
        assert len(search_results) == 1  # c0000001
        
        search_results = await sqlite_manager.search_claims_by_content("genetic")
        assert len(search_results) == 1  # c0000003
        
        # Workflow 4: Complex filtering
        complex_filter = ClaimFilter(
            confidence_min=0.8,
            confidence_max=0.9,
            dirty_only=False,
            limit=10
        )
        results = await sqlite_manager.filter_claims(complex_filter)
        
        # Should include claims with confidence between 0.8 and 0.9
        for result in results:
            assert 0.8 <= result['confidence'] <= 0.9
            assert result['dirty'] is False