#!/usr/bin/env python3
"""
Quick test of Direct vs Conjecture comparison with minimal test cases
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from direct_vs_conjecture_test import run_comparison_test

def create_minimal_test_case():
    """Create a minimal test case for quick testing"""
    test_case = {
        "id": "minimal_test",
        "category": "reasoning",
        "difficulty": "easy",
        "task": "Evaluate whether remote work is beneficial for employee productivity",
        "question": "What are the key factors to consider when evaluating remote work productivity?",
        "reasoning_requirements": [
            "balanced_analysis",
            "evidence_consideration"
        ],
        "success_criteria": "Acknowledges both benefits and challenges",
        "evaluation_criteria": [
            "correctness",
            "reasoning_quality",
            "completeness",
            "coherence"
        ]
    }
    
    test_case_dir = Path(__file__).parent / "test_cases"
    test_case_dir.mkdir(exist_ok=True)
    
    test_file = test_case_dir / "minimal_test.json"
    with open(test_file, 'w') as f:
        json.dump(test_case, f, indent=2)
    
    return test_file

def run_quick_test():
    """Run a quick test with minimal setup"""
    print("Running Quick Direct vs Conjecture Test")
    print("=" * 50)
    
    # Create minimal test case
    test_file = create_minimal_test_case()
    print(f"Created test case: {test_file}")
    
    # Temporarily modify the comparison test to use fewer test cases
    import direct_vs_conjecture_test
    original_load = direct_vs_conjecture_test.load_test_cases
    
    def minimal_load_test_cases():
        # Return only our minimal test case
        with open(test_file, 'r') as f:
            case_data = json.load(f)
        return [{
            "file": test_file.name,
            "category": case_data["category"],
            "data": case_data
        }]
    
    direct_vs_conjecture_test.load_test_cases = minimal_load_test_cases
    
    try:
        # Run the comparison
        results_file = run_comparison_test()
        print(f"Test completed! Results saved to: {results_file}")
        
        # Check if results were generated
        if results_file and results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"Total comparisons: {results.get('total_comparisons', 0)}")
            if results.get('results'):
                print("✅ Test generated results successfully!")
                for result in results['results'][:2]:  # Show first 2 results
                    print(f"- {result.get('test_case', 'unknown')}: {result.get('weighted_improvement', 0):+.3f}")
            else:
                print("⚠️ No comparison results generated")
        else:
            print("❌ No results file generated")
            
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore original function
        direct_vs_conjecture_test.load_test_cases = original_load
        # Clean up test file
        if test_file.exists():
            test_file.unlink()

if __name__ == "__main__":
    run_quick_test()