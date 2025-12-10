#!/usr/bin/env python3
"""
Simple validation test for enhanced research framework configurations
Tests the setup without importing complex modules
"""

import json
import sys
from pathlib import Path

def test_configuration_files():
    """Test that all configuration files are properly set up"""
    print("Testing Configuration Files...")
    
    research_dir = Path(__file__).parent
    
    # Test research config
    config_path = research_dir / "config.json"
    
    if not config_path.exists():
        print(f"  ❌ Research config not found at {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check for required fields
        required_fields = ["judge_model", "providers", "experiments"]
        for field in required_fields:
            if field not in config:
                print(f"  ❌ Missing required field: {field}")
                return False
        
        # Check for new baseline comparison field
        if "baseline_comparison" not in config.get("experiments", {}):
            print(f"  ❌ Missing baseline_comparison in experiments")
            return False
        
        # Check judge model configuration
        judge_model = config["judge_model"]
        if "GLM-4.6" not in judge_model:
            print(f"  ❌ Judge model not configured for GLM-4.6: {judge_model}")
            return False
        
        print(f"  [OK] Research config structure validated")
        print(f"      Judge model: {judge_model}")
        print(f"      Providers: {len(config['providers'])}")
        print(f"      Baseline comparison: {config['experiments']['baseline_comparison']}")

    except Exception as e:
        print(f"  [ERROR] Error loading research config: {e}")
        return False

    # Test .env example
    env_example_path = research_dir / ".env.example"

    if not env_example_path.exists():
        print(f"  [ERROR] .env.example not found")
        return False

    with open(env_example_path, 'r') as f:
        env_content = f.read()

    required_vars = ["JUDGE_MODEL", "BASELINE_COMPARISON"]
    missing_vars = []

    for var in required_vars:
        if f"{var}=" not in env_content:
            missing_vars.append(var)

    if missing_vars:
        print(f"  [ERROR] Missing environment variables: {missing_vars}")
        return False

    print(f"  [OK] .env.example contains required variables")
    
    # Check for GLM-4.6 configuration
    if "GLM-4.6-FP8" not in env_content:
        print(f"  ❌ GLM-4.6-FP8 not found in .env.example")
        return False
    
    print(f"  ✅ GLM-4.6 configuration found in .env.example")
    
    return True

def test_new_files_created():
    """Test that new enhancement files are created"""
    print("🧪 Testing New Enhancement Files...")
    
    research_dir = Path(__file__).parent
    
    required_files = [
        "experiments/baseline_comparison.py",
        "analysis/statistical_analyzer.py",
        "test_enhanced_framework.py"
    ]
    
    all_exist = True
    
    for file_path in required_files:
        full_path = research_dir / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"  ✅ {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {file_path} not found")
            all_exist = False
    
    return all_exist

def test_statistical_analyzer_structure():
    """Test statistical analyzer structure without imports"""
    print("🧪 Testing Statistical Analyzer Structure...")
    
    analyzer_path = Path(__file__).parent / "analysis" / "statistical_analyzer.py"
    
    if not analyzer_path.exists():
        print(f"  ❌ Statistical analyzer not found")
        return False
    
    with open(analyzer_path, 'r') as f:
        content = f.read()
    
    # Check for required classes and methods
    required_elements = [
        "class StatisticalAnalyzer",
        "def calculate_cohens_d",
        "def calculate_hedges_g", 
        "def paired_t_test",
        "def wilcoxon_signed_rank_test",
        "def analyze_ab_test",
        "class StatisticalTest",
        "class ABTestResult"
    ]
    
    missing_elements = []
    
    for element in required_elements:
        if element not in content:
            missing_elements.append(element)
    
    if missing_elements:
        print(f"  ❌ Missing elements in statistical analyzer: {missing_elements}")
        return False
    
    print(f"  ✅ Statistical analyzer structure validated ({len(required_elements)} components)")
    
    # Check for proper imports
    required_imports = ["from scipy import stats", "import numpy as np", "import pandas as pd"]
    
    for import_stmt in required_imports:
        if import_stmt not in content:
            print(f"  ❌ Missing import: {import_stmt}")
            return False
    
    print(f"  ✅ Statistical analyzer imports validated")
    
    return True

def test_baseline_comparison_structure():
    """Test baseline comparison structure without imports"""
    print("🧪 Testing Baseline Comparison Structure...")
    
    comparison_path = Path(__file__).parent / "experiments" / "baseline_comparison.py"
    
    if not comparison_path.exists():
        print(f"  ❌ Baseline comparison module not found")
        return False
    
    with open(comparison_path, 'r') as f:
        content = f.read()
    
    # Check for required classes
    required_classes = [
        "class BaselineType",
        "class BaselineEngine",
        "class BaselineComparisonSuite",
        "class ComparisonResult"
    ]
    
    missing_classes = []
    
    for class_name in required_classes:
        if class_name not in content:
            missing_classes.append(class_name)
    
    if missing_classes:
        print(f"  ❌ Missing classes in baseline comparison: {missing_classes}")
        return False
    
    print(f"  ✅ Baseline comparison classes validated ({len(required_classes)} classes)")
    
    # Check for baseline types
    baseline_types = [
        'DIRECT_PROMPT = "direct_prompt"',
        'FEW_SHOT = "few_shot"',
        'CHAIN_OF_THOUGHT = "chain_of_thought"',
        'ZERO_SHOT_COT = "zero_shot_cot"'
    ]
    
    for baseline_type in baseline_types:
        if baseline_type not in content:
            print(f"  ❌ Missing baseline type: {baseline_type}")
            return False
    
    print(f"  ✅ Baseline types validated ({len(baseline_types)} types)")
    
    # Check for statistical integration
    if "StatisticalAnalyzer" not in content:
        print(f"  ❌ Statistical analyzer integration not found")
        return False
    
    print(f"  ✅ Statistical analyzer integration found")
    
    return True

def test_research_runner_integration():
    """Test research runner integration"""
    print("🧪 Testing Research Runner Integration...")
    
    runner_path = Path(__file__).parent / "run_research.py"
    
    if not runner_path.exists():
        print(f"  ❌ Research runner not found")
        return False
    
    with open(runner_path, 'r') as f:
        content = f.read()
    
    # Check for_baseline comparison integration
    required_elements = [
        "from experiments.baseline_comparison import",
        "baseline_suite = BaselineComparisonSuite",
        "async def run_baseline_comparison",
        "--baseline",
        "BASELINE_COMPARISON"
    ]
    
    missing_elements = []
    
    for element in required_elements:
        if element not in content:
            missing_elements.append(element)
    
    if missing_elements:
        print(f"  ❌ Missing integration elements: {missing_elements}")
        return False
    
    print(f"  ✅ Research runner integration validated ({len(required_elements)} elements)")
    
    return True

def main():
    """Run all validation tests"""
    print("🚀 Enhanced Research Framework Configuration Validation\n")
    print("=" * 60)
    
    tests = [
        ("Configuration Files", test_configuration_files),
        ("New Files Created", test_new_files_created),
        ("Statistical Analyzer Structure", test_statistical_analyzer_structure),
        ("Baseline Comparison Structure", test_baseline_comparison_structure),
        ("Research Runner Integration", test_research_runner_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"Running {test_name} tests...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} tests failed with exception: {e}")
            results[test_name] = False
        print()
    
    # Summary
    print("=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\nOverall Result: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 ALL CONFIGURATION TESTS PASSED!")
        print("\nEnhanced research framework is properly configured:")
        print("✅ Judge model updated to GLM-4.6")
        print("✅ Baseline comparison framework created")
        print("✅ Statistical analysis tools added")
        print("✅ A/B testing support implemented")
        print("✅ Research runner integration complete")
        
        print("\n📋 NEXT STEPS:")
        print("1. Install dependencies: pip install scipy matplotlib seaborn pandas")
        print("2. Configure API keys in research/.env")
        print("3. Run baseline comparison:")
        print("   python research/run_research.py --baseline")
        print("4. Run full research suite:")
        print("   python research/run_research.py --full")
        print("5. Review reports in research/analysis/")
        
    else:
        print("⚠️  Some configuration tests failed")
        print("Please review the issues above and fix them before proceeding.")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)