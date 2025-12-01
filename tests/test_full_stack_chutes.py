#!/usr/bin/env python3
"""
Full-Stack Test with Modular Chutes.ai Integration
Tests the complete flow: CLI → Conjecture → LLM Bridge → Chutes.ai → Response
"""

import os
import sys
import time
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from conjecture import Conjecture
from config.simple_config import Config


def test_basic_functionality():
    """Test basic Conjecture functionality with Chutes.ai"""
    print("=== Test 1: Basic Functionality ===")

    try:
        # Initialize Conjecture
        cf = Conjecture()

        # Test exploration
        print("\n--- Testing Exploration ---")
        result = cf.explore("machine learning", max_claims=3)

        if result.claims:
            print(f"✅ Exploration successful: {len(result.claims)} claims found")
            for i, claim in enumerate(result.claims, 1):
                print(f"  {i}. [{claim.confidence:.2f}] {claim.content[:80]}...")
            return True
        else:
            print("FAIL: No claims returned from exploration")
            return False

    except Exception as e:
        print(f"FAIL: Basic functionality test failed: {e}")
        return False


def test_claim_validation():
    """Test claim validation with Chutes.ai"""
    print("\n=== Test 2: Claim Validation ===")

    try:
        cf = Conjecture()

        # Test claim creation with validation
        print("\n--- Testing Claim Creation with LLM Validation ---")
        claim = cf.add_claim(
            content="The earth revolves around the sun",
            confidence=0.9,
            claim_type="concept",
            validate_with_llm=True,
        )

        print(f"✅ Claim created and validated: {claim.content}")
        print(f"   Final confidence: {claim.confidence}")
        print(f"   State: {claim.state}")

        # Test with a potentially false claim
        print("\n--- Testing False Claim Detection ---")
        false_claim = cf.add_claim(
            content="All birds can fly perfectly",
            confidence=0.8,
            claim_type="concept",
            validate_with_llm=True,
        )

        print(f"✅ False claim processed: {false_claim.content}")
        print(f"   Final confidence: {false_claim.confidence}")
        print(f"   State: {false_claim.state}")

        return True

    except Exception as e:
        print(f"❌ Claim validation test failed: {e}")
        return False


def test_llm_bridge_status():
    """Test LLM bridge status and configuration"""
    print("\n=== Test 3: LLM Bridge Status ===")

    try:
        cf = Conjecture()

        # Check bridge status
        if hasattr(cf, "llm_bridge"):
            status = cf.llm_bridge.get_status()
            print(f"✅ LLM Bridge Status:")
            print(f"   Primary available: {status['primary_available']}")
            print(f"   Fallback available: {status['fallback_available']}")

            if status["primary_available"]:
                provider_stats = status["primary_stats"]
                print(f"   Provider: {provider_stats.get('provider', 'Unknown')}")
                print(f"   Model: {provider_stats.get('model', 'Unknown')}")

            return True
        else:
            print("❌ LLM bridge not initialized")
            return False

    except Exception as e:
        print(f"❌ Bridge status test failed: {e}")
        return False


def test_performance():
    """Test performance with timing"""
    print("\n=== Test 4: Performance Testing ===")

    try:
        cf = Conjecture()

        # Test multiple exploration requests
        queries = [
            "artificial intelligence",
            "climate change",
            "quantum computing",
            "renewable energy",
        ]

        total_time = 0
        successful_requests = 0

        for query in queries:
            start_time = time.time()
            result = cf.explore(query, max_claims=2)
            end_time = time.time()

            request_time = end_time - start_time
            total_time += request_time

            if result.claims:
                successful_requests += 1
                print(
                    f"✅ '{query}': {len(result.claims)} claims in {request_time:.2f}s"
                )
            else:
                print(f"❌ '{query}': No claims in {request_time:.2f}s")

        if successful_requests > 0:
            avg_time = total_time / len(queries)
            print(f"\n📊 Performance Summary:")
            print(
                f"   Success rate: {successful_requests}/{len(queries)} ({successful_requests / len(queries) * 100:.1f}%)"
            )
            print(f"   Average time: {avg_time:.2f}s")
            print(f"   Total time: {total_time:.2f}s")

            return avg_time < 5.0  # Expect average under 5 seconds
        else:
            print("❌ No successful requests")
            return False

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and graceful degradation"""
    print("\n=== Test 5: Error Handling ===")

    try:
        cf = Conjecture()

        # Test with invalid query
        print("\n--- Testing Invalid Query ---")
        try:
            result = cf.explore("", max_claims=3)  # Empty query
            print("❌ Should have raised ValueError")
            return False
        except ValueError:
            print("✅ Correctly rejected empty query")

        # Test with invalid confidence
        print("\n--- Testing Invalid Confidence ---")
        try:
            claim = cf.add_claim("Test content", 1.5, "concept")  # Invalid confidence
            print("❌ Should have raised ValueError")
            return False
        except ValueError:
            print("✅ Correctly rejected invalid confidence")

        # Test with very short content
        print("\n--- Testing Short Content ---")
        try:
            claim = cf.add_claim("Short", 0.8, "concept")  # Too short
            print("❌ Should have raised ValueError")
            return False
        except ValueError:
            print("✅ Correctly rejected short content")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_configuration():
    """Test configuration and environment setup"""
    print("\n=== Test 6: Configuration Testing ===")

    try:
        config = Config()

        # Check Chutes.ai configuration
        print(f"Configuration loaded:")
        print(f"   Database type: {config.database_type}")
        print(f"   LLM provider: {getattr(config, 'llm_provider', 'Not set')}")
        print(f"   LLM API URL: {getattr(config, 'llm_api_url', 'Not set')}")
        print(f"   LLM model: {getattr(config, 'llm_model', 'Not set')}")

        # Check environment variables
        chutes_key = os.getenv("CHUTES_API_KEY") or os.getenv("Conjecture_LLM_API_KEY")
        if chutes_key:
            print(
                f"Chutes.ai API key: {'*' * 20}{chutes_key[-4:] if len(chutes_key) > 4 else '****'}"
            )
        else:
            print("No Chutes.ai API key found - may affect LLM functionality")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def main():
    """Run all full-stack tests"""
    print("Full-Stack Test: Modular Chutes.ai Integration")
    print("=" * 60)

    # Load environment
    load_dotenv()

    # Run tests
    tests = [
        ("Configuration", test_configuration),
        ("LLM Bridge Status", test_llm_bridge_status),
        ("Basic Functionality", test_basic_functionality),
        ("Claim Validation", test_claim_validation),
        ("Performance", test_performance),
        ("Error Handling", test_error_handling),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(
        f"\n🎯 Overall Result: {passed}/{total} tests passed ({passed / total * 100:.1f}%)"
    )

    if passed == total:
        print("All tests passed! Full-stack integration is working.")
        return True
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
