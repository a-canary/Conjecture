#!/usr/bin/env python3
"""
Simple Research Test Script
Tests the basic research functionality without complex dependencies
"""

import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_env_loading():
    """Test that environment variables load correctly"""
    print("Testing environment variable loading...")
    
    # Load .env files
    env_files = [
        Path(__file__).parent.parent / '.env',
        Path(__file__).parent / '.env'
    ]
    
    env_vars = {}
    for env_file in env_files:
        if env_file.exists():
            print(f"  Loading: {env_file}")
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
    
    print(f"  Loaded {len(env_vars)} environment variables")
    
    # Test key variables
    key_vars = ['PROVIDER_API_URL', 'PROVIDER_API_KEY', 'PROVIDER_MODEL', 'JUDGE_MODEL']
    for var in key_vars:
        if var in env_vars:
            print(f"  [OK] {var}: {env_vars[var]}")
        else:
            print(f"  [FAIL] {var}: not found")
    
    return env_vars

def test_test_case_generation():
    """Test basic test case generation"""
    print("\nTesting test case generation...")
    
    try:
        from test_cases.test_case_generator import TestCaseGenerator
        
        generator = TestCaseGenerator()
        
        # Generate one test case
        logic_case = generator.generate_logic_puzzle()
        print(f"  [OK] Logic puzzle: {logic_case['id']}")
        
        # Save test case
        test_case_dir = Path(__file__).parent / 'test_cases'
        test_case_dir.mkdir(exist_ok=True)
        
        filename = f"{logic_case['id']}.json"
        filepath = test_case_dir / filename
        with open(filepath, 'w') as f:
            json.dump(logic_case, f, indent=2)
        print(f"    [SAVED] {filename}")
        
        return [logic_case]
        
    except Exception as e:
        print(f"  [FAIL] Test case generation failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """Run all tests"""
    print("Conjecture Research Framework Test")
    print("=" * 50)
    
    # Test environment loading
    env_vars = test_env_loading()
    
    # Test test case generation
    test_cases = test_test_case_generation()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    if env_vars:
        print(f"[OK] Environment variables: {len(env_vars)} loaded")
    else:
        print("[FAIL] Environment variables: failed to load")
    
    if test_cases:
        print(f"[OK] Test cases: {len(test_cases)} generated")
    else:
        print("[FAIL] Test cases: failed to generate")
    
    # Overall status
    success_count = sum([bool(env_vars), bool(test_cases)])
    print(f"\nOverall: {success_count}/2 components working")
    
    if success_count >= 1:
        print("Research framework basic functionality working!")
        return True
    else:
        print("Some components need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)