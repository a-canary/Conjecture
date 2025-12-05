#!/usr/bin/env python3
"""
Generate Comprehensive Test Suite for End-to-End Pipeline Experiment
Creates 75 test cases across all categories for statistical significance
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from test_cases.test_case_generator import TestCaseGenerator

def main():
    """Generate comprehensive test suite"""
    print("Generating comprehensive test suite for end-to-end pipeline experiment...")
    
    # Initialize generator
    generator = TestCaseGenerator()
    
    # Generate 8-10 test cases per category (9 categories = 72-90 total cases)
    count_per_category = 8
    
    print(f"Generating {count_per_category} test cases per category...")
    
    # Generate comprehensive test suite
    generator.generate_test_suite(count_per_type=count_per_category)
    
    # Count generated files
    test_cases_dir = Path("research/test_cases")
    json_files = list(test_cases_dir.glob("*.json"))
    
    print(f"\nTest suite generation complete!")
    print(f"Total test cases available: {len(json_files)}")
    print(f"Target range: 50-100 test cases")
    status = "Sufficient" if len(json_files) >= 50 else "Insufficient"
    print(f"Status: {status}")
    
    # Show distribution by category
    categories = {}
    for file_path in json_files:
        try:
            import json
            with open(file_path, 'r') as f:
                test_case = json.load(f)
                category = test_case.get('category', 'unknown')
                categories[category] = categories.get(category, 0) + 1
        except:
            pass
    
    print(f"\nTest case distribution:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")
    
    print(f"\nTest suite ready for end-to-end pipeline experiment!")

if __name__ == "__main__":
    main()