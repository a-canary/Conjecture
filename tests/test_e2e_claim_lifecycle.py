import asyncio
"""
End-to-end test for complete claim lifecycle
Tests claim creation, processing, evaluation, and relationship updates
"""
import pytest
import tempfile
import os
from pathlib import Path

from src.conjecture import Conjecture
from src.config.unified_config import UnifiedConfig
from src.core.models import Claim, ClaimState, DirtyReason


class TestClaimLifecycleE2E:
    """End-to-end test for complete claim lifecycle"""

    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"

            # Create test configuration with LLM disabled
            config_data = {
                "processing": {
                    "confidence_threshold": 0.9,
                    "confident_threshold": 0.7,
                    "max_context_size": 4000,
                    "batch_size": 10
                },
                "database": {
                    "database_path": f"{temp_dir}/test.db",
                    "chroma_path": f"{temp_dir}/chroma"
                },
                "workspace": {
                    "data_dir": temp_dir
                },
                "llm": {
                    "enabled": False,
                    "mock_mode": True,
                    "timeout": 1.0
                },
                "debug": True
            }

            import json
            with open(config_path, 'w') as f:
                json.dump(config_data, f)

            yield UnifiedConfig(config_path)

    @pytest.fixture
    def conjecture_instance(self, temp_config):
        """Create Conjecture instance with test configuration"""
        return Conjecture(config=temp_config)

    @pytest.mark.asyncio
    async def test_complete_claim_lifecycle(self, conjecture_instance):
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
        
        # Step 2: Add claim to system
        result = conjecture_instance.add_claim(initial_claim)
        assert result.success is True
        assert result.processed_claims == 1
        
        # Step 3: Process claim (simulate LLM evaluation)
        # This would normally call LLM, but we'll simulate the result
        processed_claim = Claim(
            id=initial_claim.id,
            content=initial_claim.content,
            confidence=0.8,  # Increased confidence after evaluation
            state=ClaimState.VALIDATED,
            tags=initial_claim.tags + ["evaluated"],
            supported_by=initial_claim.supported_by,
            supports=initial_claim.supports,
            is_dirty=False,  # Now clean after evaluation
            dirty_reason=None
        )
        
        update_result = conjecture_instance.update_claim_sync(processed_claim)
        assert update_result.success is True
        
        # Step 4: Add supporting claims
        supporter1 = Claim(
            id="supporter1",
            content="Python has extensive libraries for web development like Django and Flask.",
            confidence=0.9,
            state=ClaimState.VALIDATED,
            supports=["lifecycle_test"],
            tags=["web", "django", "flask"]
        )
        
        supporter2 = Claim(
            id="supporter2", 
            content="Python is widely used in data science with libraries like pandas and numpy.",
            confidence=0.85,
            state=ClaimState.VALIDATED,
            supports=["lifecycle_test"],
            tags=["data-science", "pandas", "numpy"]
        )
        
        # Add supporters
        conj_result1 = conjecture_instance.add_claim(supporter1)
        conj_result2 = conjecture_instance.add_claim(supporter2)
        
        assert conj_result1.success is True
        assert conj_result2.success is True
        
        # Step 5: Verify relationships and dirty flag propagation
        # Get the main claim and check it was marked dirty due to new supporters
        updated_main_claim = conjecture_instance.get_claim_sync("lifecycle_test")
        assert updated_main_claim is not None
        assert updated_main_claim.supported_by in [["supporter1"], ["supporter2"], ["supporter1", "supporter2"]]
        assert updated_main_claim.is_dirty is True  # Should be dirty due to relationship changes
        assert updated_main_claim.dirty_reason == DirtyReason.SUPPORTING_CLAIM_CHANGED
        
        # Step 6: Re-evaluate main claim with new support
        final_claim = Claim(
            id=updated_main_claim.id,
            content=updated_main_claim.content,
            confidence=0.95,  # Higher confidence due to support
            state=ClaimState.VALIDATED,
            tags=updated_main_claim.tags,
            supported_by=["supporter1", "supporter2"],
            supports=updated_main_claim.supports,
            is_dirty=False,
            dirty_reason=None
        )
        
        final_result = conjecture_instance.update_claim(final_claim)
        assert final_result.success is True
        
        # Step 7: Verify final state
        final_verified = conjecture_instance.get_claim_sync("lifecycle_test")
        assert final_verified.confidence == 0.95
        assert final_verified.state == ClaimState.VALIDATED
        assert final_verified.is_dirty is False
        assert len(final_verified.supported_by) == 2
        assert "supporter1" in final_verified.supported_by
        assert "supporter2" in final_verified.supported_by

    def test_dirty_flag_propagation_cascade(self, conjecture_instance):
        """Test dirty flag propagation through claim relationships"""
        
        # Create a hierarchy: A -> B -> C (A supports B, B supports C)
        claim_a = Claim(
            id="cascade_a",
            content="Fundamental principle A",
            confidence=0.9,
            state=ClaimState.VALIDATED,
            supports=["cascade_b"],
            tags=["fundamental"]
        )
        
        claim_b = Claim(
            id="cascade_b", 
            content="Derived principle B",
            confidence=0.7,
            state=ClaimState.VALIDATED,
            supported_by=["cascade_a"],
            supports=["cascade_c"],
            tags=["derived"]
        )
        
        claim_c = Claim(
            id="cascade_c",
            content="Specific conclusion C",
            confidence=0.5,
            state=ClaimState.EXPLORE,
            supported_by=["cascade_b"],
            tags=["specific"]
        )
        
        # Add all claims
        conj_result_a = conjecture_instance.add_claim(claim_a)
        conj_result_b = conjecture_instance.add_claim(claim_b)
        conj_result_c = conjecture_instance.add_claim(claim_c)
        
        assert all(r.success for r in [conj_result_a, conj_result_b, conj_result_c])
        
        # Update claim A (should propagate dirty flag to B and C)
        updated_a = Claim(
            id=claim_a.id,
            content=claim_a.content + " (updated)",
            confidence=0.95,  # Increased confidence
            state=claim_a.state,
            supports=claim_a.supports,
            supported_by=claim_a.supported_by,
            tags=claim_a.tags,
            is_dirty=False
        )
        
        update_result = conjecture_instance.update_claim(updated_a)
        assert update_result.success is True
        
        # Check that B and C are marked dirty
        dirty_b = conjecture_instance.get_claim_sync("cascade_b")
        dirty_c = conjecture_instance.get_claim_sync("cascade_c")
        
        assert dirty_b.is_dirty is True
        assert dirty_c.is_dirty is True
        assert dirty_b.dirty_reason == DirtyReason.SUPPORTING_CLAIM_CHANGED
        assert dirty_c.dirty_reason == DirtyReason.SUPPORTING_CLAIM_CHANGED

    def test_batch_processing_workflow(self, conjecture_instance):
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
                supports=["batch_1"],
                tags=["dl", "ml"]
            ),
            Claim(
                id="batch_3",
                content="Neural networks are the foundation of deep learning",
                confidence=0.9,
                state=ClaimState.VALIDATED,
                supports=["batch_2"],
                tags=["neural", "dl"]
            )
        ]
        
        # Add batch
        batch_result = conjecture_instance.add_claims_batch(claims_batch)
        assert batch_result.success is True
        assert batch_result.processed_claims == 3
        
        # Process dirty claims (simulate evaluation)
        dirty_claims = conjecture_instance.get_dirty_claims()
        assert len(dirty_claims) >= 1  # At least batch_1 should be dirty
        
        # Simulate processing results
        processed_claims = []
        for claim in dirty_claims:
            if claim.id == "batch_1":
                processed = Claim(
                    id=claim.id,
                    content=claim.content,
                    confidence=0.85,  # Improved confidence
                    state=ClaimState.VALIDATED,
                    supported_by=claim.supported_by,
                    supports=claim.supports,
                    tags=claim.tags,
                    is_dirty=False
                )
                processed_claims.append(processed)
        
        # Update processed claims
        if processed_claims:
            update_result = conjecture_instance.update_claims_batch(processed_claims)
            assert update_result.success is True
        
        # Verify final state
        final_claim_1 = conjecture_instance.get_claim_sync("batch_1")
        if final_claim_1:
            assert final_claim_1.confidence >= 0.8
            assert final_claim_1.state == ClaimState.VALIDATED
        
        # Test query functionality
        all_claims = conjecture_instance.get_all_claims()
        assert len(all_claims) >= 3
        
        # Test filtering by tags
        ml_claims = conjecture_instance.get_claims_by_tags(["ml"])
        assert len(ml_claims) >= 2  # batch_1 and batch_2