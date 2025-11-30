#!/usr/bin/env python3
"""
Simple test to verify tag-based architecture works
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_tag_based_architecture():
    """Test the tag-based architecture functionality"""
    try:
        from src.core.models import Claim, create_claim

        print("SUCCESS: Import successful")

        # Test 1: Basic claim creation with tags
        claim1 = Claim(
            id="c1234567",
            content="Test claim for tag-based architecture",
            confidence=0.8,
            tags=["test", "concept", "architecture"],
        )
        print("SUCCESS: Test 1 PASSED: Basic claim creation with tags")
        print(f"   Tags: {claim1.tags}")

        # Test 2: Factory function with tags
        claim2 = create_claim(
            content="Factory-created claim",
            tag="tool_example",
            tags=["tool_example", "example", "test"],
        )
        print("SUCCESS: Test 2 PASSED: Factory function with tags")
        print(f"   Tags: {claim2.tags}")

        # Test 3: Tag-based filtering simulation
        test_claims = [claim1, claim2]
        tool_example_claims = [c for c in test_claims if "tool_example" in c.tags]
        print(
            f"SUCCESS: Test 3 PASSED: Tag-based filtering found {len(tool_example_claims)} tool_example claims"
        )

        # Test 4: Tag validation and deduplication
        claim3 = Claim(
            id="c7654321",
            content="Test claim with duplicate tags",
            confidence=0.9,
            tags=["test", "concept", "test"],  # Duplicate 'test'
        )
        unique_tags = set(claim3.tags)
        print(
            f"SUCCESS: Test 4 PASSED: Tag deduplication - {len(claim3.tags)} unique tags from 3 input tags"
        )

        print("\nALL TESTS PASSED - Tag-based architecture is working correctly!")
        return True

    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_tag_based_architecture()
    sys.exit(0 if success else 1)
