"""
Simple tests for data layer - start basic and expand as needed
"""

import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.core.models import Claim, ClaimState, ClaimType
from src.data.chroma_integration import ChromaIntegration


def test_basic_claim_creation():
    """Test basic claim creation"""
    claim = Claim(
        id="c0000001",
        content="Quantum encryption uses photon polarization",
        confidence=0.85,
        type=[ClaimType.CONCEPT],
        tags=["quantum", "encryption"],
    )
    assert claim.id == "c0000001"
    assert claim.confidence == 0.85
    assert claim.state == ClaimState.EXPLORE
    print("✅ Basic claim creation: PASS")


def test_chroma_connection():
    """Test ChromaDB connection"""
    try:
        chroma = ChromaIntegration()
        print("✅ ChromaDB connection: PASS")
        return chroma
    except Exception as e:
        print(f"❌ ChromaDB connection: FAIL - {e}")
        return None


def test_add_claim():
    """Test adding a claim"""
    chroma = test_chroma_connection()
    if not chroma:
        return False

    claim = Claim(
        id="c0000002",
        content="Test claim for addition",
        confidence=0.9,
        type=[ClaimType.CONCEPT],
    )

    result = chroma.add_claim(claim)
    if result:
        print("✅ Add claim: PASS")
        return True
    else:
        print("❌ Add claim: FAIL")
        return False


def test_get_claim():
    """Test retrieving a claim"""
    chroma = ChromaIntegration()
    claim = chroma.get_claim("test_002")

    if claim and claim.id == "test_002":
        print("✅ Get claim: PASS")
        return True
    else:
        print("❌ Get claim: FAIL")
        return False


def test_update_claim():
    """Test updating a claim"""
    chroma = ChromaIntegration()

    # Get existing claim and update confidence
    claim = chroma.get_claim("test_002")
    if claim:
        claim.update_confidence(0.95)
        result = chroma.update_claim(claim)

        if result:
            # Verify update
            updated = chroma.get_claim("test_002")
            if updated and updated.confidence == 0.95:
                print("✅ Update claim: PASS")
                return True

    print("❌ Update claim: FAIL")
    return False


def test_search_claims():
    """Test searching similar claims"""
    chroma = ChromaIntegration()

    # Search for claims similar to our test claim
    claims = chroma.search_similar("Test claim", limit=5)

    if len(claims) > 0:
        print("✅ Search claims: PASS")
        return True
    else:
        print("❌ Search claims: FAIL")
        return False


def run_basic_tests():
    """Run all basic tests"""
    print("🧪 Running Data Layer Basic Tests")
    print("=" * 40)

    tests = [
        test_basic_claim_creation,
        test_add_claim,
        test_get_claim,
        test_update_claim,
        test_search_claims,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: FAIL - {e}")

    print("=" * 40)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    run_basic_tests()
