#!/usr/bin/env python3
"""
Basic Tests for Modular Conjecture CLI
Tests core functionality without complex dependencies
"""

import unittest
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_backend_imports():
    """Test that backend modules can be imported."""
    try:
        from src.cli.backends.local_backend import LocalBackend
        print("[OK] LocalBackend imported successfully")
        
        from src.cli.backends.cloud_backend import CloudBackend
        print("[OK] CloudBackend imported successfully")
        
        # Test backend instantiation without registry
        backends = {}
        for name, backend_class in [("local", LocalBackend), ("cloud", CloudBackend)]:
            backend = backend_class()
            backends[name] = backend
            print(f"[OK] {name.title()} backend instantiated successfully")
            
            # Test required methods
            required_methods = ['is_available', 'create_claim', 'get_claim', 'search_claims', 'analyze_claim', 'get_backend_info']
            for method in required_methods:
                if hasattr(backend, method):
                    print(f"  [OK] {name}.{method} method exists")
                else:
                    print(f"  [FAIL] {name}.{method} method missing")
                    return False
        
        print("[OK] All backends have required methods")
        return True
        
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_base_cli():
    """Test base CLI functionality."""
    try:
        from src.cli.base_cli import BaseCLI
        print("[OK] BaseCLI imported successfully")
        
        # Test that it's an abstract class
        try:
            BaseCLI()
            print("[FAIL] BaseCLI should not be instantiable (must be abstract)")
            return False
        except TypeError:
            print("[OK] BaseCLI is properly abstract")
        
        return True
    except Exception as e:
        print(f"[FAIL] BaseCLI test failed: {e}")
        return False

def test_modular_cli_import():
    """Test modular CLI import."""
    try:
        # Test without importing the full app initially
        import src.cli.modular_cli
        print("[OK] Modular CLI module imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Modular CLI import failed: {e}")
        return False

def test_console_functionality():
    """Test console functionality."""
    try:
        from rich.console import Console
        console = Console()
        print("[OK] Rich console imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Console test failed: {e}")
        return False

def main():
    """Run basic functionality tests."""
    print("=" * 60)
    print("TESTING: Basic Modular CLI Functionality")
    print("=" * 60)
    
    tests = [
        ("Backend Import Tests", test_backend_imports),
        ("Base CLI Tests", test_base_cli),
        ("Modular CLI Import", test_modular_cli_import),
        ("Console Functionality", test_console_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("SUMMARY: Test Results")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nRESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! Modular CLI is ready.")
        return True
    else:
        print("WARNING: Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)