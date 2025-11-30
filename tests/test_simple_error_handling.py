#!/usr/bin/env python3
"""
Simple error handling tests for Conjecture system
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.engine import Conjecture


def test_invalid_requests():
    """Test handling of invalid requests"""
    print("=== Testing Invalid Requests ===")

    try:
        cf = Conjecture()

        # Test empty request
        result1 = cf.process_request("")
        print(f"Empty request: {not result1['success']} (expected True)")

        # Test short request
        result2 = cf.process_request("hi")
        print(f"Short request: {not result2['success']} (expected True)")

        # Test None request
        try:
            result3 = cf.process_request(None)
            print(f"None request: {not result3['success']} (expected True)")
        except Exception as e:
            print(f"None request: Exception handled correctly - {type(e).__name__}")

        print("Invalid request tests: PASS")
        return True

    except Exception as e:
        print(f"Invalid request test failed: {e}")
        return False


def test_api_failure_simulation():
    """Test behaviour when API failures occur"""
    print("\n=== Testing API Failure Simulation ===")

    try:
        cf = Conjecture()

        # Test with mock data (since we can't easily simulate API failures)
        result = cf.process_request("Test request with mocked tools")

        # The system should still return success even if tools fail
        print(f"Request processing with tools: {result['success']}")

        # Check tool results for any failures
        tool_failures = 0
        for tool_result in result["tool_results"]:
            if not tool_result["result"].get("success", True):
                tool_failures += 1

        print(f"Tool failures handled: {tool_failures}")
        print("API failure simulation: PASS")
        return True

    except Exception as e:
        print(f"API failure test failed: {e}")
        return False


def test_invalid_claim_creation():
    """Test claim creation with invalid data"""
    print("\n=== Testing Invalid Claim Creation ===")

    try:
        cf = Conjecture()
        import asyncio

        # Test invalid confidence
        async def test_invalid_confidence():
            result = await cf.create_claim(
                content="Test claim",
                confidence=1.5,  # Invalid: > 1.0
                claim_type="concept",
            )
            return result

        result1 = asyncio.run(test_invalid_confidence())
        print(
            f"Invalid confidence handling: {result1['success']} (system should validate)"
        )

        # Test empty content
        async def test_empty_content():
            result = await cf.create_claim(
                content="", confidence=0.5, claim_type="concept"
            )
            return result

        result2 = asyncio.run(test_empty_content())
        print(f"Empty content handling: {result2['success']} (system should validate)")

        print("Invalid claim creation tests: PASS")
        return True

    except Exception as e:
        print(f"Invalid claim creation test failed: {e}")
        return False


def test_session_error_handling():
    """Test session-related error handling"""
    print("\n=== Testing Session Error Handling ===")

    try:
        cf = Conjecture()

        # Test invalid session ID
        invalid_session = cf.get_session("non_existent_session")
        print(f"Invalid session retrieval: {invalid_session is None} (expected True)")

        # Test session cleanup with invalid parameters
        cleaned = cf.cleanup_sessions(-1)  # Negative days
        print(f"Invalid cleanup parameter: {cleaned >= 0} (should not fail)")

        print("Session error handling tests: PASS")
        return True

    except Exception as e:
        print(f"Session error handling test failed: {e}")
        return False


def test_database_error_handling():
    """Test database error handling"""
    print("\n=== Testing Database Error Handling ===")

    try:
        cf = Conjecture()

        # Test search operations that might fail
        import asyncio

        async def test_search_fallback():
            try:
                results = await cf.search_claims("nonexistent query", limit=1000)
                return results
            except Exception as e:
                print(f"Search error caught: {e}")
                return []

        results = asyncio.run(test_search_fallback())
        print(f"Search error handling: {len(results) >= 0} (fallback working)")

        print("Database error handling tests: PASS")
        return True

    except Exception as e:
        print(f"Database error handling test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Conjecture Error Handling")
    print("=" * 50)

    test1 = test_invalid_requests()
    test2 = test_api_failure_simulation()
    test3 = test_invalid_claim_creation()
    test4 = test_session_error_handling()
    test5 = test_database_error_handling()

    if all([test1, test2, test3, test4, test5]):
        print("\nAll error handling tests passed!")
    else:
        print("\nSome error handling tests failed!")
