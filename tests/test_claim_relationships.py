"""
Unit tests for Claim relationship integrity
Tests claim relationships and validation without mocking
"""

import pytest
from pydantic import ValidationError

from src.core.models import Claim, ClaimState
from src.core.claim_operations import (
    validate_relationship_integrity,
    find_sub_claims,
    find_super_claims,
    calculate_support_strength,
)


class TestClaimRelationships:
    """Test claim relationship integrity and operations"""

    def test_simple_relationship_creation(self):
        """Test creating claims with simple relationships"""
        supporter = Claim(
            id="supporter1",
            content="Supporting claim",
            confidence=0.8,
            supers=["supported1"],
        )

        supported = Claim(
            id="supported1",
            content="Supported claim",
            confidence=0.6,
            subs=["supporter1"],
        )

        assert "supported1" in supporter.supers
        assert "supporter1" in supported.subs

    def test_bidirectional_relationships(self):
        """Test bidirectional relationship consistency"""
        claim_a = Claim(
            id="claim_a", content="Claim A", confidence=0.7, supers=["claim_b"]
        )

        claim_b = Claim(
            id="claim_b", content="Claim B", confidence=0.5, subs=["claim_a"]
        )

        # Verify bidirectional consistency
        assert "claim_b" in claim_a.supers
        assert "claim_a" in claim_b.subs

    def test_multiple_supporters(self):
        """Test claim with multiple supporters"""
        supported_claim = Claim(
            id="main_claim",
            content="Main claim with support",
            confidence=0.6,
            subs=["supporter1", "supporter2", "supporter3"],
        )

        supporters = [
            Claim(
                id="supporter1",
                content="Supporter 1",
                confidence=0.8,
                supers=["main_claim"],
            ),
            Claim(
                id="supporter2",
                content="Supporter 2",
                confidence=0.7,
                supers=["main_claim"],
            ),
            Claim(
                id="supporter3",
                content="Supporter 3",
                confidence=0.9,
                supers=["main_claim"],
            ),
        ]

        all_claims = [supported_claim] + supporters

        # Test finding supporters
        found_supporters = find_sub_claims(supported_claim, all_claims)
        assert len(found_supporters) == 3
        assert all(
            s.id in ["supporter1", "supporter2", "supporter3"] for s in found_supporters
        )

    def test_multiple_supported_claims(self):
        """Test claim supporting multiple other claims"""
        supporter_claim = Claim(
            id="main_supporter",
            content="Main supporting claim",
            confidence=0.9,
            supers=["supported1", "supported2", "supported3"],
        )

        supported_claims = [
            Claim(
                id="supported1",
                content="Supported 1",
                confidence=0.6,
                subs=["main_supporter"],
            ),
            Claim(
                id="supported2",
                content="Supported 2",
                confidence=0.7,
                subs=["main_supporter"],
            ),
            Claim(
                id="supported3",
                content="Supported 3",
                confidence=0.5,
                subs=["main_supporter"],
            ),
        ]

        all_claims = [supporter_claim] + supported_claims

        # Test finding supported claims
        found_supported = find_super_claims(supporter_claim, all_claims)
        assert len(found_supported) == 3
        assert all(
            s.id in ["supported1", "supported2", "supported3"] for s in found_supported
        )

    def test_relationship_validation_valid(self):
        """Test relationship validation with valid relationships"""
        claims = [
            Claim(
                id="claim1",
                content="Claim 1",
                confidence=0.8,
                supers=["claim2", "claim3"],
            ),
            Claim(
                id="claim2", content="Claim 2", confidence=0.6, subs=["claim1"]
            ),
            Claim(
                id="claim3", content="Claim 3", confidence=0.7, subs=["claim1"]
            ),
        ]

        # Validate claim1 relationships
        errors = validate_relationship_integrity(claims[0], claims)
        assert len(errors) == 0

    def test_relationship_validation_missing_supporter(self):
        """Test relationship validation with missing supporter"""
        claims = [
            Claim(
                id="claim1",
                content="Claim 1",
                confidence=0.6,
                subs=["nonexistent"],
            ),
            Claim(id="claim2", content="Claim 2", confidence=0.8),
        ]

        # Validate claim1 relationships
        errors = validate_relationship_integrity(claims[0], claims)
        assert len(errors) > 0
        assert any("nonexistent" in error for error in errors)

    def test_relationship_validation_missing_supported(self):
        """Test relationship validation with missing supported claim"""
        claims = [
            Claim(
                id="claim1", content="Claim 1", confidence=0.8, supers=["nonexistent"]
            ),
            Claim(id="claim2", content="Claim 2", confidence=0.6),
        ]

        # Validate claim1 relationships
        errors = validate_relationship_integrity(claims[0], claims)
        assert len(errors) > 0
        assert any("nonexistent" in error for error in errors)

    def test_support_strength_calculation(self):
        """Test support strength calculation"""
        supporters = [
            Claim(id="supporter1", content="Supporter 1", confidence=0.8),
            Claim(id="supporter2", content="Supporter 2", confidence=0.6),
            Claim(id="supporter3", content="Supporter 3", confidence=0.9),
        ]

        supported_claim = Claim(
            id="supported",
            content="Supported claim",
            confidence=0.5,
            subs=["supporter1", "supporter2", "supporter3"],
        )

        all_claims = supporters + [supported_claim]

        strength, count = calculate_support_strength(supported_claim, all_claims)

        # Average confidence should be (0.8 + 0.6 + 0.9) / 3 = 0.766...
        expected_strength = (0.8 + 0.6 + 0.9) / 3
        assert abs(strength - expected_strength) < 0.001
        assert count == 3

    def test_support_strength_no_supporters(self):
        """Test support strength calculation with no supporters"""
        claim = Claim(
            id="isolated", content="Isolated claim", confidence=0.5, subs=[]
        )

        strength, count = calculate_support_strength(claim, [claim])

        assert strength == 0.0
        assert count == 0

    def test_circular_relationships(self):
        """Test handling of circular relationships"""
        # Create circular relationship: A supports B, B supports C, C supports A
        claim_a = Claim(
            id="claim_a", content="Claim A", confidence=0.7, supers=["claim_b"]
        )
        claim_b = Claim(
            id="claim_b", content="Claim B", confidence=0.6, supers=["claim_c"]
        )
        claim_c = Claim(
            id="claim_c", content="Claim C", confidence=0.5, supers=["claim_a"]
        )

        # Add backward references
        claim_a.subs = ["claim_c"]
        claim_b.subs = ["claim_a"]
        claim_c.subs = ["claim_b"]

        all_claims = [claim_a, claim_b, claim_c]

        # All relationships should be valid (all claims exist)
        for claim in all_claims:
            errors = validate_relationship_integrity(claim, all_claims)
            assert len(errors) == 0

    def test_self_relationship_prevention(self):
        """Test that claims don't reference themselves"""
        # Pydantic should prevent self-referencing claims at creation time
        with pytest.raises(ValidationError) as exc_info:
            claim = Claim(
                id="self_ref",
                content="Self-referencing claim",
                confidence=0.5,
                supers=["self_ref"],  # This would be invalid
                subs=["self_ref"],  # This would also be invalid
            )

        # Verify the error message mentions self-reference
        assert "cannot reference itself" in str(exc_info.value)

    def test_complex_relationship_network(self):
        """Test complex relationship network"""
        claims = [
            # Root claims (no supporters)
            Claim(
                id="root1",
                content="Root claim 1",
                confidence=0.9,
                supers=["mid1", "mid2"],
            ),
            Claim(
                id="root2", content="Root claim 2", confidence=0.8, supers=["mid2"]
            ),
            # Middle claims (both support and are supported)
            Claim(
                id="mid1",
                content="Middle claim 1",
                confidence=0.7,
                supers=["leaf1"],
                subs=["root1"],
            ),
            Claim(
                id="mid2",
                content="Middle claim 2",
                confidence=0.6,
                supers=["leaf1", "leaf2"],
                subs=["root1", "root2"],
            ),
            # Leaf claims (only supported, don't support others)
            Claim(
                id="leaf1",
                content="Leaf claim 1",
                confidence=0.5,
                subs=["mid1", "mid2"],
            ),
            Claim(
                id="leaf2",
                content="Leaf claim 2",
                confidence=0.4,
                subs=["mid2"],
            ),
        ]

        # Test finding supporters for a middle claim
        mid2_supporters = find_sub_claims(claims[3], claims)  # mid2
        assert len(mid2_supporters) == 2
        supporter_ids = [s.id for s in mid2_supporters]
        assert "root1" in supporter_ids
        assert "root2" in supporter_ids

        # Test finding supported claims for a root claim
        root1_supported = find_super_claims(claims[0], claims)  # root1
        assert len(root1_supported) == 2
        supported_ids = [s.id for s in root1_supported]
        assert "mid1" in supported_ids
        assert "mid2" in supported_ids
