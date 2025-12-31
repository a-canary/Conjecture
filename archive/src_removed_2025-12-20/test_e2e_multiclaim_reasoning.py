"""
End-to-end test for multi-claim reasoning
Tests claim networks, batch processing, and confidence propagation
"""
import pytest
import tempfile
import json
from pathlib import Path

from src.conjecture import Conjecture
from src.config.unified_config import UnifiedConfig
from src.core.models import Claim, ClaimState, DirtyReason


class TestMultiClaimReasoningE2E:
    """End-to-end test for multi-claim reasoning"""

    @pytest.fixture
    def temp_config(self):
        """Create temporary configuration for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.json"
            
            config_data = {
                "processing": {
                    "confidence_threshold": 0.8,
                    "confident_threshold": 0.6,
                    "max_context_size": 8000,
                    "batch_size": 20
                },
                "database": {
                    "database_path": f"{temp_dir}/reasoning_test.db",
                    "chroma_path": f"{temp_dir}/reasoning_chroma"
                },
                "workspace": {
                    "data_dir": temp_dir
                },
                "debug": True
            }
            
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
            
            yield UnifiedConfig(config_path)

    @pytest.fixture
    def conjecture_instance(self, temp_config):
        """Create Conjecture instance with test configuration"""
        return Conjecture(config=temp_config)

    def test_claim_network_reasoning(self, conjecture_instance):
        """Test reasoning across a network of related claims"""
        
        # Create a claim network representing a reasoning chain
        reasoning_network = [
            # Root claims (foundational knowledge)
            Claim(
                id="root_evidence_1",
                content="The Earth's climate has warmed by approximately 1.1°C since pre-industrial times",
                confidence=0.95,
                state=ClaimState.VALIDATED,
                tags=["climate", "evidence", "temperature"],
                supports=["intermediate_conclusion_1", "intermediate_conclusion_2"]
            ),
            Claim(
                id="root_evidence_2",
                content="Atmospheric CO2 concentrations have increased from 280ppm to over 415ppm since 1850",
                confidence=0.98,
                state=ClaimState.VALIDATED,
                tags=["climate", "co2", "evidence"],
                supports=["intermediate_conclusion_1"]
            ),
            
            # Intermediate conclusions
            Claim(
                id="intermediate_conclusion_1",
                content="Greenhouse gas emissions are the primary driver of recent climate change",
                confidence=0.85,
                state=ClaimState.EXPLORE,
                supported_by=["root_evidence_1", "root_evidence_2"],
                supports=["final_conclusion_1"],
                tags=["climate", "greenhouse", "causation"]
            ),
            Claim(
                id="intermediate_conclusion_2",
                content="Climate change is accelerating in recent decades",
                confidence=0.8,
                state=ClaimState.EXPLORE,
                supported_by=["root_evidence_1"],
                supports=["final_conclusion_1"],
                tags=["climate", "acceleration", "trend"]
            ),
            
            # Final conclusion
            Claim(
                id="final_conclusion_1",
                content="Urgent action is needed to reduce greenhouse gas emissions",
                confidence=0.4,
                state=ClaimState.EXPLORE,
                supported_by=["intermediate_conclusion_1", "intermediate_conclusion_2"],
                tags=["climate", "policy", "action"]
            )
        ]
        
        # Step 1: Add entire network
        batch_result = conjecture_instance.add_claims_batch(reasoning_network)
        assert batch_result.success is True
        assert batch_result.processed_claims == 5
        
        # Step 2: Verify network structure
        root_1 = conjecture_instance.get_claim("root_evidence_1")
        root_2 = conjecture_instance.get_claim("root_evidence_2")
        inter_1 = conjecture_instance.get_claim("intermediate_conclusion_1")
        inter_2 = conjecture_instance.get_claim("intermediate_conclusion_2")
        final = conjecture_instance.get_claim("final_conclusion_1")
        
        # Verify relationships
        assert "intermediate_conclusion_1" in root_1.supports
        assert "intermediate_conclusion_2" in root_1.supports
        assert "intermediate_conclusion_1" in root_2.supports
        
        assert "root_evidence_1" in inter_1.supported_by
        assert "root_evidence_2" in inter_1.supported_by
        assert "root_evidence_1" in inter_2.supported_by
        
        assert "final_conclusion_1" in inter_1.supports
        assert "final_conclusion_1" in inter_2.supports
        
        assert "intermediate_conclusion_1" in final.supported_by
        assert "intermediate_conclusion_2" in final.supported_by
        
        # Step 3: Get dirty claims (should be the unvalidated intermediate and final claims)
        dirty_claims = conjecture_instance.get_dirty_claims()
        dirty_ids = [claim.id for claim in dirty_claims]
        
        assert "intermediate_conclusion_1" in dirty_ids
        assert "intermediate_conclusion_2" in dirty_ids
        assert "final_conclusion_1" in dirty_ids
        assert "root_evidence_1" not in dirty_ids  # Already validated
        assert "root_evidence_2" not in dirty_ids  # Already validated
        
        # Step 4: Simulate evaluation of intermediate claims
        evaluated_intermediate = [
            Claim(
                id="intermediate_conclusion_1",
                content=inter_1.content,
                confidence=0.92,  # Increased due to strong evidence
                state=ClaimState.VALIDATED,
                supported_by=inter_1.supported_by,
                supports=inter_1.supports,
                tags=inter_1.tags,
                is_dirty=False
            ),
            Claim(
                id="intermediate_conclusion_2",
                content=inter_2.content,
                confidence=0.88,  # Increased due to evidence
                state=ClaimState.VALIDATED,
                supported_by=inter_2.supported_by,
                supports=inter_2.supports,
                tags=inter_2.tags,
                is_dirty=False
            )
        ]
        
        update_result = conjecture_instance.update_claims_batch(evaluated_intermediate)
        assert update_result.success is True
        
        # Step 5: Check that final claim is now dirty due to updated supporters
        updated_final = conjecture_instance.get_claim("final_conclusion_1")
        assert updated_final.is_dirty is True
        assert updated_final.dirty_reason == DirtyReason.SUPPORTING_CLAIM_CHANGED
        
        # Step 6: Evaluate final claim with improved support
        evaluated_final = Claim(
            id="final_conclusion_1",
            content=final.content,
            confidence=0.85,  # Significantly improved due to strong support
            state=ClaimState.VALIDATED,
            supported_by=final.supported_by,
            supports=final.supports,
            tags=final.tags,
            is_dirty=False
        )
        
        final_update = conjecture_instance.update_claim(evaluated_final)
        assert final_update.success is True
        
        # Step 7: Verify final network state
        final_verified = conjecture_instance.get_claim("final_conclusion_1")
        assert final_verified.confidence == 0.85
        assert final_verified.state == ClaimState.VALIDATED
        assert final_verified.is_dirty is False

    def test_confidence_propagation(self, conjecture_instance):
        """Test confidence propagation through claim hierarchy"""
        
        # Create hierarchical claim structure
        hierarchy = [
            # Level 0: Base evidence (high confidence)
            Claim(
                id="base_1",
                content="Water boils at 100°C at sea level",
                confidence=0.99,
                state=ClaimState.VALIDATED,
                tags=["physics", "water", "boiling"],
                supports=["level1_1"]
            ),
            Claim(
                id="base_2",
                content="Salt raises the boiling point of water",
                confidence=0.95,
                state=ClaimState.VALIDATED,
                tags=["chemistry", "salt", "boiling"],
                supports=["level1_1"]
            ),
            
            # Level 1: Intermediate claim (medium confidence)
            Claim(
                id="level1_1",
                content="Salt water boils at a higher temperature than pure water",
                confidence=0.7,
                state=ClaimState.EXPLORE,
                supported_by=["base_1", "base_2"],
                supports=["level2_1"],
                tags=["physics", "chemistry", "solution"]
            ),
            
            # Level 2: Derived claim (low confidence initially)
            Claim(
                id="level2_1",
                content="Adding salt to water allows it to reach higher temperatures before boiling",
                confidence=0.4,
                state=ClaimState.EXPLORE,
                supported_by=["level1_1"],
                tags=["cooking", "practical", "application"]
            )
        ]
        
        # Add hierarchy
        for claim in hierarchy:
            result = conjecture_instance.add_claim(claim)
            assert result.success is True
        
        # Get support strength for each level
        from src.core.claim_operations import calculate_support_strength
        
        level1_claim = conjecture_instance.get_claim("level1_1")
        level2_claim = conjecture_instance.get_claim("level2_1")
        all_claims = hierarchy
        
        # Calculate support strength
        l1_strength, l1_count = calculate_support_strength(level1_claim, all_claims)
        l2_strength, l2_count = calculate_support_strength(level2_claim, all_claims)
        
        # Level 1 should have 2 supporters with high average confidence
        assert l1_count == 2
        assert l1_strength == (0.99 + 0.95) / 2  # ~0.97
        
        # Level 2 should have 1 supporter with medium confidence
        assert l2_count == 1
        assert l2_strength == 0.7
        
        # Simulate evaluation based on support strength
        # Level 1 confidence should increase due to strong support
        updated_l1 = Claim(
            id=level1_claim.id,
            content=level1_claim.content,
            confidence=min(0.95, level1_claim.confidence + l1_strength * 0.1),  # Boost from support
            state=ClaimState.VALIDATED,
            supported_by=level1_claim.supported_by,
            supports=level1_claim.supports,
            tags=level1_claim.tags,
            is_dirty=False
        )
        
        # Level 2 confidence should increase moderately
        updated_l2 = Claim(
            id=level2_claim.id,
            content=level2_claim.content,
            confidence=min(0.8, level2_claim.confidence + l2_strength * 0.15),  # Boost from support
            state=ClaimState.VALIDATED,
            supported_by=level2_claim.supported_by,
            supports=level2_claim.supports,
            tags=level2_claim.tags,
            is_dirty=False
        )
        
        # Update claims
        conj_result1 = conjecture_instance.update_claim(updated_l1)
        conj_result2 = conjecture_instance.update_claim(updated_l2)
        
        assert conj_result1.success is True
        assert conj_result2.success is True
        
        # Verify confidence improvements
        final_l1 = conjecture_instance.get_claim("level1_1")
        final_l2 = conjecture_instance.get_claim("level2_1")
        
        assert final_l1.confidence > level1_claim.confidence  # Should improve
        assert final_l2.confidence > level2_claim.confidence  # Should improve
        assert final_l1.confidence > final_l2.confidence  # Higher level should have higher confidence

    def test_batch_reasoning_workflow(self, conjecture_instance):
        """Test batch processing of reasoning workflow"""
        
        # Create diverse claim set for batch processing
        diverse_claims = [
            # Scientific claims
            Claim(
                id="science_1",
                content="DNA replication follows a semi-conservative mechanism",
                confidence=0.9,
                state=ClaimState.VALIDATED,
                tags=["biology", "dna", "replication"]
            ),
            Claim(
                id="science_2",
                content="CRISPR-Cas9 can edit DNA with high precision",
                confidence=0.85,
                state=ClaimState.EXPLORE,
                supported_by=["science_1"],
                tags=["biology", "crispr", "editing"]
            ),
            
            # Technology claims
            Claim(
                id="tech_1",
                content="Quantum computers can solve certain problems exponentially faster",
                confidence=0.7,
                state=ClaimState.EXPLORE,
                tags=["quantum", "computing", "complexity"]
            ),
            
            # Mathematical claims
            Claim(
                id="math_1",
                content="The Riemann hypothesis remains unproven",
                confidence=0.99,
                state=ClaimState.VALIDATED,
                tags=["mathematics", "riemann", "hypothesis"]
            ),
            Claim(
                id="math_2",
                content="Prime number distribution follows logarithmic patterns",
                confidence=0.95,
                state=ClaimState.VALIDATED,
                supported_by=["math_1"],
                supports=["science_1"],  # Cross-domain support
                tags=["mathematics", "primes", "distribution"]
            )
        ]
        
        # Step 1: Batch add
        batch_result = conjecture_instance.add_claims_batch(diverse_claims)
        assert batch_result.success is True
        assert batch_result.processed_claims == 5
        
        # Step 2: Get claims by category (tags)
        bio_claims = conjecture_instance.get_claims_by_tags(["biology"])
        quantum_claims = conjecture_instance.get_claims_by_tags(["quantum"])
        math_claims = conjecture_instance.get_claims_by_tags(["mathematics"])
        
        assert len(bio_claims) == 2
        assert len(quantum_claims) == 1
        assert len(math_claims) == 2
        
        # Step 3: Get dirty claims for batch evaluation
        dirty_claims = conjecture_instance.get_dirty_claims()
        dirty_ids = [claim.id for claim in dirty_claims]
        
        # Should include unvalidated claims
        assert "science_2" in dirty_ids
        assert "tech_1" in dirty_ids
        
        # Step 4: Simulate batch evaluation
        evaluated_claims = []
        for claim in dirty_claims:
            if claim.id == "science_2":
                evaluated = Claim(
                    id=claim.id,
                    content=claim.content,
                    confidence=0.92,  # Improved after evaluation
                    state=ClaimState.VALIDATED,
                    supported_by=claim.supported_by,
                    supports=claim.supports,
                    tags=claim.tags,
                    is_dirty=False
                )
                evaluated_claims.append(evaluated)
            elif claim.id == "tech_1":
                evaluated = Claim(
                    id=claim.id,
                    content=claim.content,
                    confidence=0.78,  # Moderately improved
                    state=ClaimState.VALIDATED,
                    supported_by=claim.supported_by,
                    supports=claim.supports,
                    tags=claim.tags,
                    is_dirty=False
                )
                evaluated_claims.append(evaluated)
        
        # Step 5: Batch update
        if evaluated_claims:
            update_result = conjecture_instance.update_claims_batch(evaluated_claims)
            assert update_result.success is True
        
        # Step 6: Verify final state and statistics
        final_stats = conjecture_instance.get_system_statistics()
        
        assert final_stats.total_claims >= 5
        assert final_stats.validated_claims >= 3  # At least the originally validated ones
        assert final_stats.average_confidence > 0.7  # Should be reasonably high
        
        # Test complex query: cross-domain relationships
        cross_domain = conjecture_instance.get_claims_with_relationships()
        assert len(cross_domain) >= 1  # Should find math_1 -> science_1 relationship