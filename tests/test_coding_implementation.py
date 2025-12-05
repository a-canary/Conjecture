#!/usr/bin/env python3
"""
Test script for coding capabilities implementation
"""

import sys

def test_coding_test_cases():
    """Test loading of coding test cases"""
    print("Testing coding test cases loading...")
    
    try:
        import json
        from pathlib import Path
        
        # Test loading coding test cases
        test_case_files = [
            "research/test_cases/coding_tasks_agenting_75.json",
            "research/test_cases/coding_tasks_system_design_45.json"
        ]
        
        total_cases = 0
        for file_path in test_case_files:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    test_cases = json.load(f)
                    total_cases += len(test_cases)
                    print(f"Loaded {len(test_cases)} cases from {file_path}")
            else:
                print(f"Missing test case file: {file_path}")
        
        print(f"Total coding test cases available: {total_cases}")
        return total_cases >= 100  # Expecting at least 100 cases

def test_coding_evaluator():
    """Test coding evaluator initialization"""
    print("Testing coding evaluator...")
    
    try:
        from test_coding_capabilities import CodingCapabilitiesEvaluator
        
        # Test evaluator initialization (without LLM manager for basic test)
        evaluator = CodingCapabilitiesEvaluator(None)
        print("Coding evaluator test passed" if CodingCapabilitiesEvaluator else print("Coding evaluator test failed"))
        
        # Test evaluation criteria structure
        criteria = evaluator.evaluation_criteria
        print(f"Evaluation criteria structure: {type(criteria).__name__}")
        
        return True
        
    except Exception as e:
        print(f"Coding evaluator test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING CODING CAPABILITIES IMPLEMENTATION")
    print("=" * 60)
    
    tests = [
        ("Coding Test Cases Loading", test_coding_test_cases),
        ("Coding Evaluator", test_coding_evaluator),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            print(f"{test_name} PASSED")
            passed += 1
        else:
            print(f"{test_name} FAILED")
    
    print(f"\n--- SUMMARY ---")
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)