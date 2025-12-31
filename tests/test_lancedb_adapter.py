"""
Comprehensive unit tests for LanceDB Adapter
Tests the refactored data layer functionality
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from typing import List

# Test imports
from pydantic import ValidationError
from src.data.models import (
    Claim,
    ClaimState,
    ClaimType,
    ClaimFilter,
    Relationship,
    DataLayerError,
    ClaimNotFoundError,
    InvalidClaimError,
)
from src.data.lancedb_adapter import LanceDBAdapter, create_lancedb_adapter
from src.core.models import ClaimScope

# Skip tests if LanceDB not available
lancedb_available = True
try:
    import lancedb
    import pandas as pd
    import numpy as np
except ImportError:
    lancedb_available = False

pytestmark = pytest.mark.skipif(not lancedb_available, reason="LanceDB not available")


class TestLanceDBAdapter:
    """Test cases for LanceDB Adapter"""

    @pytest_asyncio.fixture
    async def adapter(self):
        """Create a temporary LanceDB adapter for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.lance")
            adapter = LanceDBAdapter(db_path)
            await adapter.initialize()
            yield adapter
            await adapter.close()

    @pytest.fixture
    def sample_claim(self):
        """Create a sample claim for testing"""
        return Claim(
            id="c0000001",
            content="This is a test claim about Python programming",
            confidence=0.85,
            type=[ClaimType.CONCEPT],
            tags=["python", "programming", "test"],
            state=ClaimState.EXPLORE,
            scope=ClaimScope.USER_WORKSPACE,
            embedding=[0.1] * 384,  # Mock embedding
        )

    @pytest.fixture
    def sample_claims(self):
        """Create multiple sample claims"""
        return [
            Claim(
                id=f"c000000{i:02d}",
                content=f"Test claim {i} about mathematics and logic",
                confidence=0.6 + (i * 0.03),
                type=[ClaimType.THESIS if i % 2 == 0 else ClaimType.CONCEPT],
                tags=[f"tag{i}", "test"],
                state=ClaimState.VALIDATED if i > 5 else ClaimState.EXPLORE,
                scope=ClaimScope.USER_WORKSPACE,
                embedding=[(i * 0.01)] * 384,
            )
            for i in range(1, 11)
        ]

    @pytest.mark.asyncio
    async def test_adapter_initialization(self):
        """Test adapter initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_init.lance")
            adapter = LanceDBAdapter(db_path)

            # Should not be initialized initially
            assert not adapter._initialized

            # Initialize should work
            await adapter.initialize()
            assert adapter._initialized
            assert adapter.db is not None
            assert adapter.claims_table is not None
            assert adapter.relationships_table is not None

            await adapter.close()

    @pytest.mark.asyncio
    async def test_create_claim(self, adapter, sample_claim):
        """Test claim creation"""
        # Create claim
        result = await adapter.create_claim(sample_claim)

        # Should return the same claim
        assert result.id == sample_claim.id
        assert result.content == sample_claim.content
        assert result.confidence == sample_claim.confidence

    @pytest.mark.asyncio
    async def test_create_duplicate_claim(self, adapter, sample_claim):
        """Test creating duplicate claim should fail"""
        # Create claim first time
        await adapter.create_claim(sample_claim)

        # Try to create same claim again
        with pytest.raises(InvalidClaimError, match="already exists"):
            await adapter.create_claim(sample_claim)

    @pytest.mark.asyncio
    async def test_get_claim(self, adapter, sample_claim):
        """Test retrieving a claim"""
        # Create claim
        await adapter.create_claim(sample_claim)

        # Retrieve claim
        retrieved = await adapter.get_claim(sample_claim.id)

        assert retrieved is not None
        assert retrieved.id == sample_claim.id
        assert retrieved.content == sample_claim.content
        assert retrieved.confidence == pytest.approx(sample_claim.confidence, rel=1e-5)
        assert retrieved.state == sample_claim.state

    @pytest.mark.asyncio
    async def test_get_nonexistent_claim(self, adapter):
        """Test retrieving non-existent claim"""
        retrieved = await adapter.get_claim("c9999999")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_update_claim(self, adapter, sample_claim):
        """Test updating a claim"""
        # Create claim
        await adapter.create_claim(sample_claim)

        # Update claim
        updates = {
            "content": "Updated content",
            "confidence": 0.95,
            "state": ClaimState.VALIDATED,
        }
        updated = await adapter.update_claim(sample_claim.id, updates)

        assert updated.id == sample_claim.id
        assert updated.content == "Updated content"
        assert updated.confidence == 0.95
        assert updated.state == ClaimState.VALIDATED

    @pytest.mark.asyncio
    async def test_update_nonexistent_claim(self, adapter):
        """Test updating non-existent claim"""
        with pytest.raises(ClaimNotFoundError):
            await adapter.update_claim("c9999999", {"content": "Updated"})

    @pytest.mark.asyncio
    async def test_list_claims(self, adapter, sample_claims):
        """Test listing claims"""
        # Create multiple claims
        for claim in sample_claims:
            await adapter.create_claim(claim)

        # List all claims
        all_claims = await adapter.list_claims()
        assert len(all_claims) == len(sample_claims)

        # List with limit
        limited_claims = await adapter.list_claims(ClaimFilter(limit=5))
        assert len(limited_claims) == 5

    @pytest.mark.asyncio
    async def test_list_claims_with_filters(self, adapter, sample_claims):
        """Test listing claims with filters"""
        # Create multiple claims
        for claim in sample_claims:
            await adapter.create_claim(claim)

        # Filter by state
        explore_filter = ClaimFilter(states=[ClaimState.EXPLORE])
        explore_claims = await adapter.list_claims(explore_filter)

        for claim in explore_claims:
            assert claim.state == ClaimState.EXPLORE

        # Filter by confidence range
        confidence_filter = ClaimFilter(confidence_min=0.8, confidence_max=0.9)
        confidence_claims = await adapter.list_claims(confidence_filter)

        for claim in confidence_claims:
            assert 0.8 <= claim.confidence <= 0.9

    @pytest.mark.asyncio
    async def test_search_claims(self, adapter, sample_claims):
        """Test searching claims"""
        # Create multiple claims
        for claim in sample_claims:
            await adapter.create_claim(claim)

        # Search for "mathematics"
        math_results = await adapter.search_claims("mathematics", limit=10)

        assert len(math_results) > 0
        for claim in math_results:
            assert "mathematics" in claim.content.lower()

    @pytest.mark.asyncio
    async def test_vector_search(self, adapter, sample_claims):
        """Test vector similarity search"""
        # Create multiple claims
        for claim in sample_claims:
            await adapter.create_claim(claim)

        # Create query vector
        query_vector = [0.05] * 384

        # Perform search
        results = await adapter.vector_search(query_vector, limit=5)

        assert len(results) <= 5
        for claim, score in results:
            assert isinstance(claim, Claim)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_create_relationship(self, adapter, sample_claim):
        """Test creating a relationship"""
        # Create two claims
        claim1 = sample_claim
        claim2 = Claim(
            id="c0000002",
            content="Supporting claim for testing",
            confidence=0.9,
            type=[ClaimType.REFERENCE],
            tags=["support", "test"],
            state=ClaimState.VALIDATED,
            scope=ClaimScope.USER_WORKSPACE,
        )

        await adapter.create_claim(claim1)
        await adapter.create_claim(claim2)

        # Create relationship
        relationship = Relationship(supporter_id=claim2.id, supported_id=claim1.id)

        result = await adapter.create_relationship(relationship)

        assert result.supporter_id == claim2.id
        assert result.supported_id == claim1.id

    @pytest.mark.asyncio
    async def test_get_relationships(self, adapter, sample_claim):
        """Test retrieving relationships for a claim"""
        # Create two claims
        claim1 = sample_claim
        claim2 = Claim(
            id="c0000002",
            content="Supporting claim",
            confidence=0.9,
            type=[ClaimType.REFERENCE],
            tags=["support"],
            state=ClaimState.VALIDATED,
            scope=ClaimScope.USER_WORKSPACE,
        )

        await adapter.create_claim(claim1)
        await adapter.create_claim(claim2)

        # Create relationship
        relationship = Relationship(supporter_id=claim2.id, supported_id=claim1.id)
        await adapter.create_relationship(relationship)

        # Get relationships
        relationships = await adapter.get_relationships(claim1.id)

        assert len(relationships) == 1
        assert relationships[0].supporter_id == claim2.id
        assert relationships[0].supported_id == claim1.id

    @pytest.mark.asyncio
    async def test_get_stats(self, adapter, sample_claims):
        """Test getting database statistics"""
        # Create multiple claims
        for claim in sample_claims:
            await adapter.create_claim(claim)

        # Get stats
        stats = await adapter.get_stats()

        assert stats["total_claims"] == len(sample_claims)
        assert stats["initialized"] is True
        assert "dimension" in stats
        assert "claims_by_state" in stats

    @pytest.mark.asyncio
    async def test_claim_model_conversion(self, sample_claim):
        """Test claim model conversion methods"""
        # Test to_lancedb_dict
        claim_dict = sample_claim.to_lancedb_dict()

        assert claim_dict["id"] == sample_claim.id
        assert claim_dict["content"] == sample_claim.content
        assert claim_dict["confidence"] == sample_claim.confidence
        assert claim_dict["state"] == sample_claim.state.value

        # Test from_lancedb_result
        restored_claim = Claim.from_lancedb_result(claim_dict)

        assert restored_claim.id == sample_claim.id
        assert restored_claim.content == sample_claim.content
        assert restored_claim.confidence == sample_claim.confidence
        assert restored_claim.state == sample_claim.state

    @pytest.mark.asyncio
    async def test_factory_function(self):
        """Test the factory function for creating adapters"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "factory_test.lance")

            # Create adapter using factory
            adapter = await create_lancedb_adapter(db_path, dimension=256)

            assert adapter._initialized is True
            assert (
                adapter.dimension == 256
            )  # Factory should respect the dimension parameter

            await adapter.close()


class TestLanceDBAdapterEdgeCases:
    """Test edge cases and error conditions"""

    @pytest_asyncio.fixture
    async def adapter(self):
        """Create a temporary LanceDB adapter for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_edge.lance")
            adapter = LanceDBAdapter(db_path)
            await adapter.initialize()
            yield adapter
            await adapter.close()

    @pytest.mark.asyncio
    async def test_invalid_claim_validation(self, adapter):
        """Test that Pydantic validates claim content before adapter sees it"""
        # Test short content - Pydantic should reject this
        with pytest.raises(ValidationError):
            Claim(
                id="c0000001",
                content="Short",  # Too short - Pydantic validates
                confidence=0.5,
                type=[ClaimType.CONCEPT],
                tags=["test"],
            )

    @pytest.mark.asyncio
    async def test_invalid_confidence(self, adapter):
        """Test that Pydantic validates confidence values before adapter sees it"""
        # Test high confidence - Pydantic should reject this
        with pytest.raises(ValidationError):
            Claim(
                id="c0000001",
                content="Valid content for testing purposes",
                confidence=1.5,  # Too high - Pydantic validates
                type=[ClaimType.CONCEPT],
                tags=["test"],
            )

    @pytest.mark.asyncio
    async def test_large_number_of_claims(self, adapter):
        """Test handling large numbers of claims"""
        claims = []
        for i in range(100):
            claim = Claim(
                id=f"c{i:08d}",
                content=f"Test claim number {i} with some content",
                confidence=0.5 + (i % 50) / 100,
                type=[ClaimType.CONCEPT],
                tags=[f"tag{i}", "bulk_test"],
                state=ClaimState.EXPLORE,
                scope=ClaimScope.USER_WORKSPACE,
            )
            claims.append(claim)

        # Create all claims
        for claim in claims:
            await adapter.create_claim(claim)

        # List and verify count
        all_claims = await adapter.list_claims()
        assert len(all_claims) == 100

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, adapter):
        """Test concurrent claim operations"""

        async def create_claims(start_id: int, count: int):
            for i in range(count):
                claim = Claim(
                    id=f"c{start_id + i:08d}",
                    content=f"Concurrent claim {start_id + i}",
                    confidence=0.8,
                    type=[ClaimType.CONCEPT],
                    tags=["concurrent"],
                    state=ClaimState.EXPLORE,
                    scope=ClaimScope.USER_WORKSPACE,
                )
                await adapter.create_claim(claim)

        # Run concurrent operations
        tasks = [
            create_claims(1000, 10),
            create_claims(2000, 10),
            create_claims(3000, 10),
        ]

        await asyncio.gather(*tasks)

        # Verify all claims were created
        all_claims = await adapter.list_claims()
        assert len(all_claims) == 30


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
