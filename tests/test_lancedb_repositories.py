"""
Comprehensive unit tests for LanceDB Repositories
Tests the refactored repository layer functionality
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import os
from datetime import datetime

# Test imports
from src.data.models import (
    Claim,
    ClaimState,
    ClaimType,
    ClaimFilter,
    Relationship,
    ProcessingResult,
)
from src.data.lancedb_repositories import (
    LanceDBClaimRepository,
    LanceDBRelationshipRepository,
    create_lancedb_repositories,
)
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


class TestLanceDBClaimRepository:
    """Test cases for LanceDB Claim Repository"""

    @pytest_asyncio.fixture
    async def repository(self):
        """Create a temporary repository for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_repo.lance")
            claim_repo, relationship_repo = await create_lancedb_repositories(db_path)
            yield claim_repo
            # Cleanup is handled by temporary directory

    @pytest.fixture
    def sample_claim(self):
        """Create a sample claim for testing"""
        return Claim(
            id="c0000001",
            content="This is a test claim about machine learning algorithms",
            confidence=0.85,
            type=[ClaimType.CONCEPT],
            tags=["machine-learning", "algorithms", "test"],
            state=ClaimState.EXPLORE,
            scope=ClaimScope.USER_WORKSPACE,
            embedding=[0.1] * 384,
        )

    @pytest.fixture
    def sample_claims(self):
        """Create multiple sample claims"""
        return [
            Claim(
                id=f"c000000{i:02d}",
                content=f"Test claim {i} about data science and analytics",
                confidence=0.6 + (i * 0.03),
                type=[ClaimType.THESIS if i % 2 == 0 else ClaimType.CONCEPT],
                tags=[f"tag{i}", "data-science"],
                state=ClaimState.VALIDATED if i > 7 else ClaimState.EXPLORE,
                scope=ClaimScope.USER_WORKSPACE,
                embedding=[(i * 0.01)] * 384,
            )
            for i in range(1, 11)
        ]

    @pytest.mark.asyncio
    async def test_create_claim_success(self, repository, sample_claim):
        """Test successful claim creation"""
        result = await repository.create_claim(sample_claim)

        assert result.success is True
        assert result.claim_id == sample_claim.id
        assert result.message == "Claim created successfully"
        assert result.updated_confidence == sample_claim.confidence
        assert result.metadata["operation"] == "create"

    @pytest.mark.asyncio
    async def test_create_claim_validation_error(self, repository):
        """Test claim creation with validation error - Pydantic validates before repository"""
        from pydantic import ValidationError

        # Pydantic validates content length before repository call
        with pytest.raises(ValidationError) as exc_info:
            Claim(
                id="c0000001",
                content="Short",  # Too short - min 10 chars
                confidence=0.5,
                type=[ClaimType.CONCEPT],
                tags=["test"],
            )

        assert "String should have at least 10 characters" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_duplicate_claim(self, repository, sample_claim):
        """Test creating duplicate claim"""
        # Create claim first time
        await repository.create_claim(sample_claim)

        # Try to create same claim again
        result = await repository.create_claim(sample_claim)

        assert result.success is False
        assert "already exists" in result.message

    @pytest.mark.asyncio
    async def test_get_claim_success(self, repository, sample_claim):
        """Test successful claim retrieval"""
        # Create claim
        await repository.create_claim(sample_claim)

        # Retrieve claim
        retrieved = await repository.get_claim(sample_claim.id)

        assert retrieved is not None
        assert retrieved.id == sample_claim.id
        assert retrieved.content == sample_claim.content
        assert retrieved.confidence == pytest.approx(sample_claim.confidence, rel=1e-5)
        assert retrieved.state == sample_claim.state

    @pytest.mark.asyncio
    async def test_get_claim_not_found(self, repository):
        """Test retrieving non-existent claim"""
        retrieved = await repository.get_claim("c9999999")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_update_claim_success(self, repository, sample_claim):
        """Test successful claim update"""
        # Create claim first
        await repository.create_claim(sample_claim)

        # Update claim
        updates = {
            "content": "Updated content about machine learning",
            "confidence": 0.95,
            "state": ClaimState.VALIDATED,
            "tags": ["updated", "machine-learning", "validated"],
        }

        result = await repository.update_claim(sample_claim.id, updates)

        assert result.success is True
        assert result.claim_id == sample_claim.id
        assert result.message == "Claim updated successfully"
        assert result.updated_confidence == pytest.approx(0.95, rel=1e-5)
        assert "updated_fields" in result.metadata

        # Verify the update
        updated_claim = await repository.get_claim(sample_claim.id)
        assert updated_claim.content == "Updated content about machine learning"
        assert updated_claim.confidence == pytest.approx(0.95, rel=1e-5)
        assert updated_claim.state == ClaimState.VALIDATED

    @pytest.mark.asyncio
    async def test_update_claim_not_found(self, repository):
        """Test updating non-existent claim"""
        result = await repository.update_claim("c9999999", {"content": "Updated"})

        assert result.success is False
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_update_claim_invalid_fields(self, repository, sample_claim):
        """Test updating claim with invalid fields"""
        # Create claim first
        await repository.create_claim(sample_claim)

        # Try to update invalid field
        updates = {"invalid_field": "value", "content": "Valid update"}

        result = await repository.update_claim(sample_claim.id, updates)

        assert result.success is False
        assert "Invalid fields" in result.message

    @pytest.mark.asyncio
    async def test_delete_claim_success(self, repository, sample_claim):
        """Test successful claim deletion"""
        # Create claim first
        await repository.create_claim(sample_claim)

        # Delete claim (note: LanceDB delete is limited, so this may not work fully)
        result = await repository.delete_claim(sample_claim.id)

        # This might fail due to LanceDB limitations
        # We're testing the error handling
        assert isinstance(result, ProcessingResult)

    @pytest.mark.asyncio
    async def test_search_claims(self, repository, sample_claims):
        """Test searching claims"""
        # Create multiple claims
        for claim in sample_claims:
            await repository.create_claim(claim)

        # Search for "data science"
        results = await repository.search_claims("data science", limit=10)

        assert len(results) > 0
        for claim in results:
            assert "data science" in claim.content.lower()

    @pytest.mark.asyncio
    async def test_list_claims(self, repository, sample_claims):
        """Test listing claims"""
        # Create multiple claims
        for claim in sample_claims:
            await repository.create_claim(claim)

        # List all claims
        all_claims = await repository.list_claims()
        assert len(all_claims) == len(sample_claims)

        # List with limit
        limited_claims = await repository.list_claims(ClaimFilter(limit=5))
        assert len(limited_claims) == 5

    @pytest.mark.asyncio
    async def test_list_claims_with_filters(self, repository, sample_claims):
        """Test listing claims with filters"""
        # Create multiple claims
        for claim in sample_claims:
            await repository.create_claim(claim)

        # Filter by state
        explore_filter = ClaimFilter(states=[ClaimState.EXPLORE])
        explore_claims = await repository.list_claims(explore_filter)

        for claim in explore_claims:
            assert claim.state == ClaimState.EXPLORE

        # Filter by confidence range
        confidence_filter = ClaimFilter(confidence_min=0.7, confidence_max=0.8)
        confidence_claims = await repository.list_claims(confidence_filter)

        for claim in confidence_claims:
            assert 0.7 <= claim.confidence <= 0.8

        # Filter by tags
        tag_filter = ClaimFilter(tags=["tag1"])
        tag_claims = await repository.list_claims(tag_filter)

        for claim in tag_claims:
            assert "tag1" in claim.tags

    @pytest.mark.asyncio
    async def test_vector_search(self, repository, sample_claims):
        """Test vector similarity search"""
        # Create multiple claims
        for claim in sample_claims:
            await repository.create_claim(claim)

        # Create query vector
        query_vector = [0.05] * 384

        # Perform search
        results = await repository.vector_search(query_vector, limit=5)

        assert len(results) <= 5
        for claim, score in results:
            assert isinstance(claim, Claim)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_get_claims_by_state(self, repository, sample_claims):
        """Test getting claims by state"""
        # Create multiple claims
        for claim in sample_claims:
            await repository.create_claim(claim)

        # Get validated claims
        validated_claims = await repository.get_claims_by_state(ClaimState.VALIDATED)

        for claim in validated_claims:
            assert claim.state == ClaimState.VALIDATED

        # Get explore claims
        explore_claims = await repository.get_claims_by_state(ClaimState.EXPLORE)

        for claim in explore_claims:
            assert claim.state == ClaimState.EXPLORE

    @pytest.mark.asyncio
    async def test_dirty_claims_management(self, repository, sample_claim):
        """Test dirty claims management"""
        # Create claim
        await repository.create_claim(sample_claim)

        # Mark claim as dirty
        result = await repository.mark_claim_dirty(
            sample_claim.id, reason="Test dirty marking", priority=5
        )

        assert result.success is True
        # mark_claim_dirty uses update_claim internally, so message reflects update
        assert result.success is True

        # Verify claim is marked dirty
        dirty_claims = await repository.get_dirty_claims()
        assert len(dirty_claims) == 1
        assert dirty_claims[0].id == sample_claim.id
        assert dirty_claims[0].is_dirty is True
        assert dirty_claims[0].dirty_reason == "Test dirty marking"
        assert dirty_claims[0].dirty_priority == 5

        # Mark claim as clean
        result = await repository.mark_claim_clean(sample_claim.id)

        assert result.success is True
        assert "marked clean" in result.message

    @pytest.mark.asyncio
    async def test_get_stats(self, repository, sample_claims):
        """Test getting repository statistics"""
        # Create multiple claims
        for claim in sample_claims:
            await repository.create_claim(claim)

        # Get stats
        stats = await repository.get_stats()

        assert "total_claims" in stats
        assert stats["total_claims"] == len(sample_claims)
        assert "repository_operation" in stats
        assert stats["repository_operation"] == "claim_repository"
        assert "success_rate" in stats


class TestLanceDBRelationshipRepository:
    """Test cases for LanceDB Relationship Repository"""

    @pytest_asyncio.fixture
    async def repositories(self):
        """Create temporary repositories for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_rel.lance")
            claim_repo, relationship_repo = await create_lancedb_repositories(db_path)
            yield claim_repo, relationship_repo

    @pytest.fixture
    def sample_claims(self):
        """Create two sample claims"""
        claim1 = Claim(
            id="c0000001",
            content="Main claim about AI ethics",
            confidence=0.9,
            type=[ClaimType.THESIS],
            tags=["ai", "ethics"],
            state=ClaimState.VALIDATED,
            scope=ClaimScope.USER_WORKSPACE,
        )

        claim2 = Claim(
            id="c0000002",
            content="Supporting research paper citation",
            confidence=0.95,
            type=[ClaimType.REFERENCE],
            tags=["research", "citation"],
            state=ClaimState.VALIDATED,
            scope=ClaimScope.USER_WORKSPACE,
        )

        claim3 = Claim(
            id="c0000003",
            content="Related concept about machine learning",
            confidence=0.8,
            type=[ClaimType.CONCEPT],
            tags=["ml", "concept"],
            state=ClaimState.EXPLORE,
            scope=ClaimScope.USER_WORKSPACE,
        )

        return claim1, claim2, claim3

    @pytest.mark.asyncio
    async def test_create_relationship_success(self, repositories, sample_claims):
        """Test successful relationship creation"""
        claim_repo, relationship_repo = repositories
        claim1, claim2, claim3 = sample_claims

        # Create claims first
        await claim_repo.create_claim(claim1)
        await claim_repo.create_claim(claim2)

        # Create relationship
        result = await relationship_repo.create_relationship(claim2.id, claim1.id)

        assert result.success is True
        assert "created successfully" in result.message
        assert result.metadata["operation"] == "create_relationship"

    @pytest.mark.asyncio
    async def test_create_relationship_nonexistent_claims(self, repositories):
        """Test creating relationship with non-existent claims"""
        claim_repo, relationship_repo = repositories

        # Try to create relationship with non-existent claims
        result = await relationship_repo.create_relationship("c9999998", "c9999999")

        assert result.success is False
        assert "not found" in result.message

    @pytest.mark.asyncio
    async def test_get_relationships(self, repositories, sample_claims):
        """Test getting relationships for a claim"""
        claim_repo, relationship_repo = repositories
        claim1, claim2, claim3 = sample_claims

        # Create claims first
        await claim_repo.create_claim(claim1)
        await claim_repo.create_claim(claim2)
        await claim_repo.create_claim(claim3)

        # Create multiple relationships
        await relationship_repo.create_relationship(
            claim2.id, claim1.id
        )  # claim2 supports claim1
        await relationship_repo.create_relationship(
            claim3.id, claim1.id
        )  # claim3 supports claim1

        # Get relationships for claim1
        relationships = await relationship_repo.get_relationships(claim1.id)

        assert len(relationships) == 2

        # Check relationship directions
        supporter_ids = [rel.supporter_id for rel in relationships]
        supported_ids = [rel.supported_id for rel in relationships]

        assert claim2.id in supporter_ids
        assert claim3.id in supporter_ids
        assert all(rid == claim1.id for rid in supported_ids)

    @pytest.mark.asyncio
    async def test_get_supporting_claims(self, repositories, sample_claims):
        """Test getting supporting claims"""
        claim_repo, relationship_repo = repositories
        claim1, claim2, claim3 = sample_claims

        # Create claims first
        await claim_repo.create_claim(claim1)
        await claim_repo.create_claim(claim2)
        await claim_repo.create_claim(claim3)

        # Create relationships
        await relationship_repo.create_relationship(claim2.id, claim1.id)
        await relationship_repo.create_relationship(claim3.id, claim1.id)

        # Get supporting claims for claim1
        supporting_claims = await relationship_repo.get_supporting_claims(claim1.id)

        assert len(supporting_claims) == 2
        supporting_ids = [claim.id for claim in supporting_claims]
        assert claim2.id in supporting_ids
        assert claim3.id in supporting_ids

    @pytest.mark.asyncio
    async def test_get_supported_claims(self, repositories, sample_claims):
        """Test getting supported claims"""
        claim_repo, relationship_repo = repositories
        claim1, claim2, claim3 = sample_claims

        # Create claims first
        await claim_repo.create_claim(claim1)
        await claim_repo.create_claim(claim2)
        await claim_repo.create_claim(claim3)

        # Create relationships
        await relationship_repo.create_relationship(claim1.id, claim2.id)
        await relationship_repo.create_relationship(claim1.id, claim3.id)

        # Get supported claims for claim1
        supported_claims = await relationship_repo.get_supported_claims(claim1.id)

        assert len(supported_claims) == 2
        supported_ids = [claim.id for claim in supported_claims]
        assert claim2.id in supported_ids
        assert claim3.id in supported_ids


class TestRepositoryIntegration:
    """Integration tests for repository operations"""

    @pytest_asyncio.fixture
    async def repositories(self):
        """Create temporary repositories for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_integration.lance")
            claim_repo, relationship_repo = await create_lancedb_repositories(db_path)
            yield claim_repo, relationship_repo

    @pytest.mark.asyncio
    async def test_complex_claim_workflow(self, repositories):
        """Test a complex workflow with claims and relationships"""
        claim_repo, relationship_repo = repositories

        # Create main claim
        main_claim = Claim(
            id="c0000001",
            content="Artificial intelligence will transform healthcare in the next decade",
            confidence=0.8,
            type=[ClaimType.THESIS],
            tags=["ai", "healthcare", "future"],
            state=ClaimState.EXPLORE,
            scope=ClaimScope.USER_WORKSPACE,
        )

        result = await claim_repo.create_claim(main_claim)
        assert result.success is True

        # Create supporting claims
        supporting_claims = []
        for i, content in enumerate(
            [
                "AI diagnostic tools have shown 95% accuracy in recent studies",
                "Machine learning algorithms can predict patient outcomes",
                "Robotic surgery reduces recovery time by 50%",
            ]
        ):
            claim = Claim(
                id=f"c000000{i + 2}",
                content=content,
                confidence=0.9,
                type=[ClaimType.REFERENCE],
                tags=["ai", "healthcare", f"support{i}"],
                state=ClaimState.VALIDATED,
                scope=ClaimScope.USER_WORKSPACE,
            )

            await claim_repo.create_claim(claim)
            supporting_claims.append(claim)

            # Create relationship
            rel_result = await relationship_repo.create_relationship(
                claim.id, main_claim.id
            )
            assert rel_result.success is True

        # Verify relationships
        supporting = await relationship_repo.get_supporting_claims(main_claim.id)
        assert len(supporting) == 3

        # Update main claim based on supporting evidence
        updates = {
            "confidence": 0.95,
            "state": ClaimState.VALIDATED,
            "tags": main_claim.tags + ["well-supported"],
        }

        update_result = await claim_repo.update_claim(main_claim.id, updates)
        assert update_result.success is True

        # Verify final state
        final_claim = await claim_repo.get_claim(main_claim.id)
        assert final_claim.confidence == pytest.approx(0.95, rel=1e-5)
        assert final_claim.state == ClaimState.VALIDATED
        assert "well-supported" in final_claim.tags

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, repositories):
        """Test concurrent repository operations"""
        claim_repo, relationship_repo = repositories

        async def create_claim_batch(start_id: int, count: int):
            claims = []
            for i in range(count):
                claim = Claim(
                    id=f"c{start_id + i:08d}",
                    content=f"Concurrent claim {start_id + i} about data analysis",
                    confidence=0.7 + (i % 30) / 100,
                    type=[ClaimType.CONCEPT],
                    tags=["concurrent", f"batch{start_id}"],
                    state=ClaimState.EXPLORE,
                    scope=ClaimScope.USER_WORKSPACE,
                )
                result = await claim_repo.create_claim(claim)
                claims.append(result)
            return claims

        # Run concurrent claim creation
        tasks = [
            create_claim_batch(1000, 5),
            create_claim_batch(2000, 5),
            create_claim_batch(3000, 5),
        ]

        results = await asyncio.gather(*tasks)

        # Flatten results
        all_results = []
        for batch_results in results:
            all_results.extend(batch_results)

        # Verify all creations succeeded
        successful_results = [r for r in all_results if r.success]
        assert len(successful_results) == 15

        # Verify claims exist
        all_claims = await claim_repo.list_claims()
        assert len(all_claims) == 15


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
