"""
Comprehensive Test Suite for LM Studio Integration
Runs all end-to-end tests for the LM Studio provider with ibm/granite-4-h-tiny model
"""

import unittest
import os
import sys
import time

sys.path.insert(0, './src')

def run_all_lm_studio_tests():
    """Run all LM Studio integration tests"""
    print("="*60)
    print("COMPREHENSIVE LM STUDIO INTEGRATION TEST SUITE")
    print("="*60)
    print("Testing Conjecture with LM Studio and ibm/granite-4-h-tiny model")
    print("Ensure LM Studio is running at http://127.0.0.1:1234")
    print("="*60)
    
    # Import test modules
    from test_lm_studio_e2e import TestLMStudioEndToEnd
    from test_lm_studio_performance import TestLMStudioPerformance
    from test_granite_model_specific import TestGraniteModelSpecific
    from test_use_cases import TestPracticalUseCases
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestLMStudioEndToEnd))
    suite.addTests(loader.loadTestsFromTestCase(TestLMStudioPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestGraniteModelSpecific))
    suite.addTests(loader.loadTestsFromTestCase(TestPracticalUseCases))
    
    # Create a custom test runner for detailed reporting
    print("\nRunning comprehensive test suite...")
    start_time = time.time()
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)
    print(f"Duration: {duration:.2f} seconds")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n✓ All tests passed!")
        print("LM Studio integration with ibm/granite-4-h-tiny is working correctly.")
    else:
        print(f"\n✗ {len(result.failures) + len(result.errors)} test(s) failed or had errors")
    
    print("="*60)
    return result


def run_smoke_tests():
    """Run a quick smoke test to verify basic functionality"""
    print("Running quick smoke test...")
    
    # Set up environment
    os.environ['Conjecture_LLM_PROVIDER'] = 'lm_studio'
    os.environ['Conjecture_LLM_API_URL'] = 'http://127.0.0.1:1234'
    os.environ['Conjecture_LLM_MODEL'] = 'ibm/granite-4-h-tiny'
    
    try:
        from contextflow import Conjecture
        cf = Conjecture()
        
        print(f"SUCCESS: Conjecture initialized with provider: {cf.config.llm_provider}")
        print(f"SUCCESS: LLM Bridge available: {cf.llm_bridge.is_available()}")
        
        # Quick functionality test
        result = cf.explore("test topic", max_claims=1)
        print(f"SUCCESS: Exploration successful, got {len(result.claims)} claims")
        
        claim = cf.add_claim(
            content="Test claim for smoke test",
            confidence=0.80,
            claim_type="concept"
        )
        print(f"SUCCESS: Claim creation successful: {claim.content[:20]}...")
        
        print("SUCCESS: Smoke test passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: Smoke test failed: {e}")
        return False


if __name__ == '__main__':
    print("LM Studio Integration Test Suite")
    print("Available commands:")
    print("  python test_suite.py smoke    - Run quick smoke test")
    print("  python test_suite.py full     - Run comprehensive test suite")
    print()
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'smoke':
        success = run_smoke_tests()
        sys.exit(0 if success else 1)
    elif len(sys.argv) > 1 and sys.argv[1] == 'full':
        result = run_all_lm_studio_tests()
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        print("Please specify test type: 'smoke' or 'full'")
        print("Example: python test_suite.py full")