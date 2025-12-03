"""
Test script for the streamlined configuration wizard

This script tests the wizard functionality without making actual changes.
"""

import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

def test_diagnostics():
    """Test the diagnostics module"""
    print("Testing diagnostics module...")
    
    try:
        from src.config.diagnostics import SystemDiagnostics, run_diagnostics
        
        # Test diagnostics
        diagnostics = SystemDiagnostics()
        results = diagnostics.run_all_diagnostics()
        
        print(f"Diagnostics completed")
        print(f"   Overall status: {results['summary']['overall_status']}")
        print(f"   Total checks: {results['summary']['total_checks']}")
        print(f"   Ready for setup: {results['summary']['ready_for_setup']}")

        return True

    except Exception as e:
        print(f"Diagnostics test failed: {e}")
        return False

def test_wizard_imports():
    """Test wizard imports"""
    print("Testing wizard imports...")

    try:
        from src.config.streamlined_wizard import StreamlinedConfigWizard, run_wizard
        print("Wizard imports successful")
        return True

    except Exception as e:
        print(f"Wizard import failed: {e}")
        return False

def test_wizard_initialization():
    """Test wizard initialization"""
    print("Testing wizard initialization...")

    try:
        from src.config.streamlined_wizard import StreamlinedConfigWizard

        wizard = StreamlinedConfigWizard()

        # Check providers are loaded
        assert len(wizard.providers) > 0, "No providers loaded"
        print(f"Loaded {len(wizard.providers)} providers")

        # Check project files
        assert wizard.env_file.name == '.env', "Env file not found"
        assert wizard.env_example.name == '.env.example', "Env example not found"

        print("Wizard initialization successful")
        return True

    except Exception as e:
        print(f"Wizard initialization failed: {e}")
        return False

def test_provider_configs():
    """Test provider configurations"""
    print("Testing provider configurations...")

    try:
        from src.config.streamlined_wizard import StreamlinedConfigWizard

        wizard = StreamlinedConfigWizard()

        # Test local providers
        local_providers = [p for p in wizard.providers.values() if p.type == 'local']
        assert len(local_providers) > 0, "No local providers found"
        print(f"Found {len(local_providers)} local providers")

        # Test cloud providers
        cloud_providers = [p for p in wizard.providers.values() if p.type == 'cloud']
        assert len(cloud_providers) > 0, "No cloud providers found"
        print(f"Found {len(cloud_providers)} cloud providers")

        # Test provider structure
        for key, provider in wizard.providers.items():
            assert provider.name, f"Provider {key} missing name"
            assert provider.endpoint, f"Provider {key} missing endpoint"
            assert provider.type in ['local', 'cloud'], f"Provider {key} invalid type"

        print("Provider configurations valid")
        return True

    except Exception as e:
        print(f"Provider configuration test failed: {e}")
        return False

def test_env_file_handling():
    """Test environment file handling"""
    print("Testing environment file handling...")

    try:
        from src.config.streamlined_wizard import StreamlinedConfigWizard

        wizard = StreamlinedConfigWizard()

        # Test configuration check
        is_configured = wizard._is_configured()
        print(f"Configuration check: {'configured' if is_configured else 'not configured'}")

        # Test status icon function
        pass_icon = wizard._get_status_icon('pass')
        fail_icon = wizard._get_status_icon('fail')
        assert pass_icon == '✅', "Pass icon incorrect"
        assert fail_icon == '❌', "Fail icon incorrect"

        print("Environment file handling works")
        return True

    except Exception as e:
        print(f"Environment file test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("Running Configuration Wizard Tests")
    print("=" * 50)
    
    tests = [
        test_diagnostics,
        test_wizard_imports,
        test_wizard_initialization,
        test_provider_configs,
        test_env_file_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("Test Results")
    print("=" * 20)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print("\\nAll tests passed! The wizard is ready to use.")
        print("\\nTo run the wizard:")
        print("  python setup_wizard.py")
    else:
        print("\\nSome tests failed. Please check the issues above.")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)