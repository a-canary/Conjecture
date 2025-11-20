#!/usr/bin/env python3
"""
Test script for the simplified configuration system
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_validator():
    """Test the simple configuration validator"""
    print("Testing Simple Configuration Validator")
    print("=" * 50)
    
    try:
        from config.simple_validator import (
            validate_config, 
            print_validation_result,
            get_configured_provider,
            print_configuration_status
        )
        
        # Test validation
        print("\n1. Testing configuration validation...")
        result = validate_config()
        print_validation_result(result)
        
        # Test configured provider
        print("\n2. Testing configured provider...")
        provider = get_configured_provider()
        if provider:
            print(f"✅ Configured provider: {provider['name']} ({provider['type']})")
        else:
            print("❌ No configured provider found")
        
        # Test configuration status
        print("\n3. Testing configuration status...")
        print_configuration_status()
        
        print("\n✅ Validator test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_new_cli():
    """Test the new simplified CLI"""
    print("\n\nTesting New Simplified CLI")
    print("=" * 50)
    
    try:
        # Import the CLI module
        import simple_conjecture_cli
        
        # Test if it can be imported without errors
        print("✅ CLI module imported successfully")
        
        # Check if CLI app exists
        if hasattr(simple_conjecture_cli, 'app'):
            print("✅ CLI app found")
        else:
            print("❌ CLI app not found")
            return False
        
        print("✅ CLI test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ CLI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_env_file():
    """Test .env.example file"""
    print("\n\nTesting .env.example File")
    print("=" * 50)
    
    env_example = Path(".env.example")
    
    if not env_example.exists():
        print("❌ .env.example file not found")
        return False
    
    print(f"✅ .env.example file exists: {env_example.absolute()}")
    
    # Check file content
    try:
        with open(env_example, 'r') as f:
            content = f.read()
        
        # Check for key sections
        sections = [
            "CONJECTURE CLI PROVIDER CONFIGURATION",
            "PRIORITY 1: LOCAL SERVICES",
            "PRIORITY 2: CLOUD SERVICES", 
            "OLLAMA",
            "OPENAI",
            "USAGE INSTRUCTIONS",
            "SECURITY NOTES"
        ]
        
        for section in sections:
            if section in content:
                print(f"✅ Found section: {section}")
            else:
                print(f"❌ Missing section: {section}")
                return False
        
        print("✅ .env.example file validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error reading .env.example: {e}")
        return False

def create_test_env():
    """Create a test .env file for testing"""
    test_env_content = """# Test configuration for Conjecture
# Ollama configuration (commented out for test)
# OLLAMA_ENDPOINT=http://localhost:11434
# OLLAMA_MODEL=llama2

# OpenAI configuration (commented out for test)  
# OPENAI_API_KEY=sk-test-key-here
# OPENAI_MODEL=gpt-3.5-turbo

# Database and other settings
CONJECTURE_DB_PATH=data/test_conjecture.db
CONJECTURE_CONFIDENCE=0.7
CONJECTURE_DEBUG=false
"""
    
    test_env_path = Path(".env.test")
    try:
        with open(test_env_path, 'w') as f:
            f.write(test_env_content)
        print(f"✅ Created test .env file: {test_env_path.absolute()}")
        return True
    except Exception as e:
        print(f"❌ Could not create test .env file: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Simplified Configuration System Tests")
    print("=" * 60)
    
    # Test components
    tests = [
        ("Validator", test_validator),
        (".env.example", test_env_file),
        ("New CLI", test_new_cli)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running {test_name} Test")
        print('='*60)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Create test .env file
    create_test_env()
    
    # Summary
    print(f"\n\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:15} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The simplified configuration system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())