#!/usr/bin/env python3
"""
Simple validation script for coding capabilities implementation.
Tests the basic functionality without requiring full Conjecture setup.
"""

import sys
import os

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_imports():
    """Test that our coding capabilities module can be imported."""
    print("--- Import Test ---")
    try:
        from tests.test_coding_capabilities import CodingCapabilitiesEvaluator
        print("PASS CodingCapabilitiesEvaluator imported successfully")
        return True
    except Exception as e:
        print(f"FAIL CodingCapabilitiesEvaluator import failed: {e}")
        return False

def test_mock_evaluator():
    """Test that the mock evaluator works correctly."""
    print("\n--- Mock Evaluator Test ---")
    try:
        import asyncio
        from tests.test_coding_capabilities import CodingCapabilitiesEvaluator
        
        evaluator = CodingCapabilitiesEvaluator(use_mock=True)
        print("PASS Mock evaluator created successfully")
        
        # Test a simple evaluation
        test_case = {
            "id": "test_001",
            "task": "Write a function that adds two numbers",
            "expected_output": "def add(a, b): return a + b",
            "category": "code_generation",
            "complexity": "simple",
            "language": "python"
        }
        
        # Run async evaluation
        result = asyncio.run(evaluator.evaluate_coding_task(test_case, "mock_model"))
        print(f"PASS Mock evaluation completed: {result.evaluation_criteria.weighted_average}")
        return True
        
    except Exception as e:
        print(f"FAIL Mock evaluator test failed: {e}")
        return False

def test_test_cases():
    """Test that coding test cases can be loaded."""
    print("\n--- Test Cases Test ---")
    try:
        from tests.test_coding_capabilities import CodingCapabilitiesEvaluator
        
        evaluator = CodingCapabilitiesEvaluator(use_mock=True)
        test_cases = evaluator.load_test_cases()
        
        print(f"PASS Loaded {len(test_cases)} coding test cases")
        
        # Check categories
        categories = set()
        for case in test_cases:
            categories.add(case['category'])
        
        print(f"PASS Found categories: {', '.join(sorted(categories))}")
        return True
        
    except Exception as e:
        print(f"FAIL Test cases loading failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("==================================================")
    print("SIMPLE CODING IMPLEMENTATION VALIDATION")
    print("==================================================")
    
    tests = [
        test_imports,
        test_mock_evaluator,
        test_test_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n==================================================")
    print(f"RESULTS: {passed}/{total} tests passed")
    print("==================================================")
    
    if passed == total:
        print("SUCCESS All tests passed! Coding capabilities implementation is working.")
        return 0
    else:
        print("ERROR Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())