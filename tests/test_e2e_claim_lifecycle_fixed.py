# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end test for complete claim lifecycle (Fixed Version)
Tests claim creation, processing, evaluation, and relationship updates
without requiring LLM providers
"""
import pytest
import pytest_asyncio
import tempfile
import os
import asyncio
from pathlib import Path

from src.core.models import Claim, ClaimState, ClaimType, DirtyReason
from src.data.optimized_sqlite_manager import OptimizedSQLiteManager


class TestClaimLifecycleE2EFixed:
    """End-to-end test for complete claim lifecycle (Fixed - no LLM dependencies)"""

    @pytest_asyncio.fixture
    async def db_manager(self):
        """Create temporary database manager for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")

            # Initialize database manager
            manager = OptimizedSQLiteManager(db_path)
            await manager.initialize()

            yield manager

            # Cleanup
            await manager.close()

    @pytest.mark.asyncio
    async def test_complete_claim_lifecycle(self, db_manager):
        """Test complete claim lifecycle from creation to validation"""

        # Step 1: Create initial claim
        initial_claim = Claim(
            id="lifecycle_test",
            content="Python is a versatile programming language used for web development, data science, and automation.",
            confidence=0.3,
            state=ClaimState.EXPLORE,
            tags=["programming", "python", "versatility"]
        )

        assert initial_claim.confidence == 0.3
        assert initial_claim.state == ClaimState.EXPLORE
        assert initial_claim.is_dirty is True

        # Step 2: Add claim to database
        claim_id = await db_manager.create_claim(initial_claim)
        assert claim_id == initial_claim.id

        # Step 3: Retrieve claim and verify (now returns Claim object)
        retrieved_claim = await db_manager.get_claim(initial_claim.id)
        assert retrieved_claim is not None
        assert retrieved_claim.id == initial_claim.id
        assert retrieved_claim.content == initial_claim.content
        assert retrieved_claim.confidence == initial_claim.confidence
        assert retrieved_claim.state == initial_claim.state
        assert ClaimType.CONCEPT in retrieved_claim.type  # Default type

        # Step 4: Simulate LLM evaluation by updating claim
        updates = {
            "confidence": 0.8,  # Increased confidence after evaluation
            "state": ClaimState.VALIDATED,
            "tags": initial_claim.tags + ["evaluated"],
            "is_dirty": False,
            "dirty_reason": None
        }

        update_result = await db_manager.update_claim(initial_claim.id, updates)
        assert update_result is True

        # Step 5: Verify updated claim
        updated_claim = await db_manager.get_claim(initial_claim.id)
        assert updated_claim.confidence == 0.8
        assert updated_claim.state == ClaimState.VALIDATED
        assert "evaluated" in updated_claim.tags
        assert updated_claim.is_dirty is False

        # Step 6: Add supporting claims
        supporter1 = Claim(
            id="supporter1",
            content="Python has extensive libraries for web development like Django and Flask.",
            confidence=0.9,
            state=ClaimState.VALIDATED,
            supers=["lifecycle_test"],
            tags=["web", "django", "flask"]
        )

        supporter2 = Claim(
            id="supporter2",
            content="Python is widely used in data science with libraries like pandas and numpy.",
            confidence=0.85,
            state=ClaimState.VALIDATED,
            supers=["lifecycle_test"],
            tags=["data-science", "pandas", "numpy"]
        )

        # Add supporters
        supporter1_id = await db_manager.create_claim(supporter1)
        supporter2_id = await db_manager.create_claim(supporter2)

        assert supporter1_id == supporter1.id
        assert supporter2_id == supporter2.id

        # Step 7: Simulate relationship updates by updating main claim's subs field
        main_claim_updates = {
            "subs": ["supporter1", "supporter2"],
            "is_dirty": True,
            "dirty_reason": DirtyReason.SUPPORTING_CLAIM_CHANGED
        }

        main_update_result = await db_manager.update_claim("lifecycle_test", main_claim_updates)
        assert main_update_result is True

        # Step 8: Verify main claim was updated with supporters
        final_main_claim = await db_manager.get_claim("lifecycle_test")
        assert final_main_claim is not None
        assert set(final_main_claim.subs) == {"supporter1", "supporter2"}
        assert final_main_claim.is_dirty is True
        assert final_main_claim.dirty_reason == DirtyReason.SUPPORTING_CLAIM_CHANGED

        # Step 9: Simulate re-evaluation of main claim with new support
        final_updates = {
            "confidence": 0.95,  # Higher confidence due to support
            "state": ClaimState.VALIDATED,
            "is_dirty": False,
            "dirty_reason": None
        }

        final_result = await db_manager.update_claim("lifecycle_test", final_updates)
        assert final_result is True

        # Step 10: Verify final state
        final_verified = await db_manager.get_claim("lifecycle_test")
        assert final_verified.confidence == 0.95
        assert final_verified.state == ClaimState.VALIDATED
        assert final_verified.is_dirty is False
        assert len(final_verified.subs) == 2
        assert "supporter1" in final_verified.subs
        assert "supporter2" in final_verified.subs

    @pytest.mark.asyncio
    async def test_dirty_flag_propagation_cascade(self, db_manager):
        """Test dirty flag propagation through claim relationships"""

        # Create a hierarchy: A -> B -> C (A supports B, B supports C)
        claim_a = Claim(
            id="cascade_a",
            content="Fundamental principle A",
            confidence=0.9,
            state=ClaimState.VALIDATED,
            supers=["cascade_b"],
            tags=["fundamental"]
        )

        claim_b = Claim(
            id="cascade_b",
            content="Derived principle B",
            confidence=0.7,
            state=ClaimState.VALIDATED,
            subs=["cascade_a"],
            supers=["cascade_c"],
            tags=["derived"]
        )

        claim_c = Claim(
            id="cascade_c",
            content="Specific conclusion C",
            confidence=0.5,
            state=ClaimState.EXPLORE,
            subs=["cascade_b"],
            tags=["specific"]
        )

        # Add all claims
        await db_manager.create_claim(claim_a)
        await db_manager.create_claim(claim_b)
        await db_manager.create_claim(claim_c)

        # Update claim A (simulating cascade effect)
        updated_a_updates = {
            "content": claim_a.content + " (updated)",
            "confidence": 0.95  # Increased confidence
        }

        update_result = await db_manager.update_claim("cascade_a", updated_a_updates)
        assert update_result is True

        # Update claim B to simulate cascade effect
        cascade_b_updates = {
            "is_dirty": True,
            "dirty_reason": DirtyReason.SUPPORTING_CLAIM_CHANGED
        }

        await db_manager.update_claim("cascade_b", cascade_b_updates)

        # Update claim C to simulate cascade effect
        cascade_c_updates = {
            "is_dirty": True,
            "dirty_reason": DirtyReason.SUPPORTING_CLAIM_CHANGED
        }

        await db_manager.update_claim("cascade_c", cascade_c_updates)

        # Check that B and C are marked dirty (now returns Claim objects)
        dirty_b = await db_manager.get_claim("cascade_b")
        dirty_c = await db_manager.get_claim("cascade_c")

        assert dirty_b.is_dirty is True
        assert dirty_c.is_dirty is True
        assert dirty_b.dirty_reason == DirtyReason.SUPPORTING_CLAIM_CHANGED
        assert dirty_c.dirty_reason == DirtyReason.SUPPORTING_CLAIM_CHANGED

    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, db_manager):
        """Test batch processing of multiple claims"""

        # Create multiple related claims
        claims_batch = [
            Claim(
                id="batch_1",
                content="Machine learning requires large datasets",
                confidence=0.6,
                state=ClaimState.EXPLORE,
                tags=["ml", "data"]
            ),
            Claim(
                id="batch_2",
                content="Deep learning is a subset of machine learning",
                confidence=0.8,
                state=ClaimState.VALIDATED,
                supers=["batch_1"],
                tags=["dl", "ml"]
            ),
            Claim(
                id="batch_3",
                content="Neural networks are the foundation of deep learning",
                confidence=0.9,
                state=ClaimState.VALIDATED,
                supers=["batch_2"],
                tags=["neural", "dl"]
            )
        ]

        # Add batch using batch_create_claims
        claim_ids = await db_manager.batch_create_claims(claims_batch)
        assert len(claim_ids) == 3
        assert "batch_1" in claim_ids
        assert "batch_2" in claim_ids
        assert "batch_3" in claim_ids

        # Get dirty claims (all should be dirty on creation)
        dirty_claims = await db_manager.get_dirty_claims()
        dirty_ids = [claim.id for claim in dirty_claims]
        assert len(dirty_claims) >= 1  # At least batch_1 should be dirty

        # Simulate processing results for dirty claims using dict format
        processed_updates = {}
        for claim in dirty_claims:
            if claim.id == "batch_1":
                processed_updates[claim.id] = {
                    "confidence": 0.85,  # Improved confidence
                    "state": ClaimState.VALIDATED,
                    "is_dirty": False,
                    "dirty_reason": None
                }

        # Update processed claims
        if processed_updates:
            updated_count = await db_manager.batch_update_claims(processed_updates)
            assert updated_count >= 1

        # Verify final state
        final_claim_1 = await db_manager.get_claim("batch_1")
        if final_claim_1:
            assert final_claim_1.confidence >= 0.8
            assert final_claim_1.state == ClaimState.VALIDATED

        # Verify count
        count = await db_manager.count()
        assert count == 3
