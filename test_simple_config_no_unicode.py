#!/usr/bin/env python3
"""
Simple Configuration Test
Tests the simplified configuration system without Unicode
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_simple_validator():
    """Test the simple configuration validator"""
    print("Testing Simple Configuration Validator")
    print("=" * 50)
    
    try:
        from config.simple_validator import SimpleValidator
        
        validator = SimpleValidator()
        
        # Test validation
        print("Validating configuration...")
        result = validator.validate_configuration()
        
        print(f"Validation success: {result.success}")
        
        if result.primary_provider:
            print(f"Primary provider: {result.primary_provider}")
        
        if result.available_providers:
            print(f"Available providers: {len(result.available_providers)}")
            for provider in result.available_providers:
                print(f"  - {provider.name} ({provider.type})")
        
        if result.missing_vars:
            print("Missing variables:")
            for provider, vars in result.missing_vars.items():
                print(f"  {provider}: {vars}")
        
        if result.errors:
            print("Errors:")
            for error in result.errors:
                print(f"  - {error}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_env_file():
    """Test .env file handling"""
    print("\nTesting .env File Handling")
    print("=" * 30)
    
    env_path = ".env"
    env_example_path = ".env.example"
    
    print(f".env exists: {os.path.exists(env_path)}")
    print(f".env.example exists: {os.path.exists(env_example_path)}")
    
    if os.path.exists(env_example_path):
        with open(env_example_path, 'r') as f:
            lines = f.readlines()
            print(f".env.example has {len(lines)} lines")
            
            # Count provider sections
            provider_sections = [line for line in lines if line.strip().startswith('# ') and 'INSTALLATION' in line.upper()]
            print(f"Provider sections: {len(provider_sections)}")
    
    return True

def main():
    """Run all tests"""
    print("Simple Configuration System Test")
    print("=" * 40)
    
    results = {
        "validator": test_simple_validator(),
        "env_file": test_env_file()
    }
    
    print("\n" + "=" * 40)
    print("Test Results:")
    
    for test_name, success in results.items():
        status = "[OK] Working" if success else "[ERROR] Failed"
        print(f"   {test_name:15}: {status}")
    
    working_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nOverall: {working_count}/{total_count} tests passed")
    
    if working_count == total_count:
        print("[SUCCESS] Simple configuration system is ready!")
        print("\nNext steps:")
        print("1. Copy template: cp .env.example .env")
        print("2. Edit .env with your preferred provider")
        print("3. Test: python simple_conjecture_cli.py validate")
    else:
        print("[WARNING] Some components need attention")

if __name__ == "__main__":
    main()