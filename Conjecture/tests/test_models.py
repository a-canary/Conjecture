"""
Test core models without external dependencies
"""

import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.core.models import Claim, ClaimState, ClaimType


def test_claim_creation():
    """Test basic claim creation and validation"""
    try:
        claim = Claim(
            id="test_001",
            content="Quantum encryption uses photon polarization states",
            confidence=0.85,
            type=[ClaimType.CONCEPT],
            tags=["quantum", "encryption", "physics"],
        )

        # Basic validations
        assert claim.id == "test_001"
        assert claim.confidence == 0.85
        assert claim.state == ClaimState.EXPLORE
        assert ClaimType.CONCEPT in claim.type
        assert len(claim.tags) == 3
        print("✅ Claim creation: PASS")
        return True
    except Exception as e:
        print(f"❌ Claim creation: FAIL - {e}")
        return False


def test_claim_validation():
    """Test claim validation rules"""
    try:
        # Test invalid confidence
        try:
            Claim(
                id="invalid_001",
                content="Test",
                confidence=1.5,  # Invalid confidence > 1.0
                type=[ClaimType.CONCEPT],
            )
            print("❌ Validation rules: FAIL - Should reject invalid confidence")
            return False
        except Exception:
            print("✅ Confidence validation: PASS")

        # Test invalid content length
        try:
            Claim(
                id="invalid_002",
                content="Too short",  # Less than 5 chars
                confidence=0.5,
                type=[ClaimType.CONCEPT],
            )
            print("❌ Validation rules: FAIL - Should reject short content")
            return False
        except Exception:
            print("✅ Content length validation: PASS")

        # Test valid claim
        valid_claim = Claim(
            id="valid_001",
            content="This is a valid claim content that meets minimum requirements",
            confidence=0.7,
            type=[ClaimType.REFERENCE],
        )
        assert (
            valid_claim.content
            == "This is a valid claim content that meets minimum requirements"
        )
        print("✅ Valid claim creation: PASS")
        return True
    except Exception as e:
        print(f"❌ Validation rules: FAIL - {e}")
        return False


def test_claim_relationships():
    """Test claim relationship management"""
    try:
        claim1 = Claim(
            id="claim_001",
            content="Base claim",
            confidence=0.5,
            type=[ClaimType.CONCEPT],
        )

        claim2 = Claim(
            id="claim_002",
            content="Supporting claim",
            confidence=0.8,
            type=[ClaimType.REFERENCE],
        )

        # Test adding support
        claim1.add_support(claim2.id)
        assert claim2.id in claim1.supported_by
        assert claim1.updated > claim1.created
        print("✅ Add support: PASS")

        # Test adding what this claim supports
        claim2.add_supports(claim1.id)
        assert claim1.id in claim2.supports
        assert claim2.updated > claim2.created
        print("✅ Add supports: PASS")

        return True
    except Exception as e:
        print(f"❌ Claim relationships: FAIL - {e}")
        return False


def test_claim_state_management():
    """Test claim state and confidence updates"""
    try:
        claim = Claim(
            id="state_test_001",
            content="Test claim for state management",
            confidence=0.3,
            type=[ClaimType.CONCEPT],
        )

        original_confidence = claim.confidence
        original_updated = claim.updated

        # Test confidence update
        claim.update_confidence(0.95)
        assert claim.confidence == 0.95
        assert claim.updated > original_updated
        print("✅ Confidence update: PASS")

        # Test invalid confidence
        try:
            claim.update_confidence(1.5)
            print("❌ State management: FAIL - Should reject invalid confidence")
            return False
        except Exception:
            print("✅ Confidence validation in update: PASS")

        # Test context formatting
        context_str = claim.format_for_context()
        expected_format = f"[{claim.id},{claim.confidence},concept,{claim.state.value}]{claim.content}"
        assert context_str == expected_format
        print("✅ Context formatting: PASS")

        return True
    except Exception as e:
        print(f"❌ State management: FAIL - {e}")
        return False


def test_chroma_metadata_conversion():
    """Test conversion to/from ChromaDB metadata format"""
    try:
        claim = Claim(
            id="metadata_test_001",
            content="Test metadata conversion",
            confidence=0.75,
            type=[ClaimType.SKILL, ClaimType.EXAMPLE],
            tags=["testing", "metadata"],
            supported_by=["support_001"],
            supports=["supported_001"],
        )

        # Test conversion to Chroma metadata
        metadata = claim.to_chroma_metadata()
        assert metadata["confidence"] == 0.75
        assert metadata["state"] == "Explore"
        assert "skill" in metadata["type"]
        assert "example" in metadata["type"]
        assert metadata["tags"] == ["testing", "metadata"]
        assert metadata["supported_by"] == ["support_001"]
        assert metadata["supports"] == ["supported_001"]
        assert "created" in metadata
        assert "updated" in metadata
        print("✅ To Chroma metadata: PASS")

        # Test conversion from Chroma result
        chroma_result_metadata = {
            "confidence": 0.9,
            "state": "Validated",
            "supported_by": ["support_002"],
            "supports": ["supported_002"],
            "type": ["thesis"],
            "tags": ["restored", "testing"],
            "created": "2024-10-31T10:00:00",
            "updated": "2024-10-31T11:00:00",
        }

        restored_claim = Claim.from_chroma_result(
            id="restored_001",
            content="Restored claim content",
            metadata=chroma_result_metadata,
        )

        assert restored_claim.id == "restored_001"
        assert restored_claim.confidence == 0.9
        assert restored_claim.state == ClaimState.VALIDATED
        assert ClaimType.THESIS in restored_claim.type
        assert restored_claim.supported_by == ["support_002"]
        print("✅ From Chroma result: PASS")

        return True
    except Exception as e:
        print(f"❌ Chroma metadata conversion: FAIL - {e}")
        return False


def run_model_tests():
    """Run all model tests"""
    print("🧪 Running Core Model Tests")
    print("=" * 40)

    tests = [
        test_claim_creation,
        test_claim_validation,
        test_claim_relationships,
        test_claim_state_management,
        test_chroma_metadata_conversion,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("=" * 40)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All model tests passed!")
        return True
    else:
        print("❌ Some model tests failed")
        return False


if __name__ == "__main__":
    success = run_model_tests()
    exit(0 if success else 1)
