#!/usr/bin/env python3
"""
Test script for the Provider Discovery System

Tests:
1. Service detection functionality
2. Configuration management
3. Security features
4. CLI integration
"""

import asyncio
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from discovery.service_detector import ServiceDetector, DetectedProvider
from discovery.config_updater import ConfigUpdater
from discovery.provider_discovery import ProviderDiscovery

def test_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from discovery.service_detector import ServiceDetector, DetectedProvider
        from discovery.config_updater import ConfigUpdater
        from discovery.provider_discovery import ProviderDiscovery
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_service_detector():
    """Test service detection functionality"""
    print("\n🔍 Testing ServiceDetector...")
    
    try:
        detector = ServiceDetector(timeout=2)
        
        # Test API key validation
        openai_test_key = "sk-1234567890abcdef1234567890abcdef12345678"
        assert detector.validate_api_key_format("openai", openai_test_key), "OpenAI key validation failed"
        
        # Test API key masking
        masked = detector.mask_api_key(openai_test_key)
        assert masked.startswith("sk-1") and masked.endswith("5678") and "*" in masked, "Key masking failed"
        
        print("✅ ServiceDetector basic tests passed")
        return True
    except Exception as e:
        print(f"❌ ServiceDetector test failed: {e}")
        return False

def test_config_updater():
    """Test configuration management"""
    print("\n🔍 Testing ConfigUpdater...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            updater = ConfigUpdater(str(temp_path))
            
            # Test gitignore creation
            updater.ensure_gitignore()
            assert updater.gitignore_file.exists(), "Gitignore file not created"
            assert ".env" in updater.gitignore_file.read_text(), " .env not in gitignore"
            
            # Test API key validation
            assert updater.validate_api_key("openai", "sk-1234567890abcdef1234567890abcdef12345678"), "OpenAI key validation failed"
            
            # Test empty env reading
            env_vars = updater.read_existing_env()
            assert isinstance(env_vars, dict), "Env reading should return dict"
            
            print("✅ ConfigUpdater basic tests passed")
            return True
    except Exception as e:
        print(f"❌ ConfigUpdater test failed: {e}")
        return False

async def test_provider_discovery():
    """Test provider discovery (no actual services required)"""
    print("\n🔍 Testing ProviderDiscovery...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            discovery = ProviderDiscovery(str(temp_path), timeout=1)
            
            # Test configuration status
            status = discovery.get_configuration_status()
            assert isinstance(status, dict), "Status should be dict"
            assert 'env_file_exists' in status, "Status missing env_file_exists"
            
            # Test quick check (will likely find no providers, which is fine)
            result = await discovery.quick_check()
            assert isinstance(result, dict), "Quick check should return dict"
            assert 'success' in result, "Result missing success field"
            
            print("✅ ProviderDiscovery basic tests passed")
            return True
    except Exception as e:
        print(f"❌ ProviderDiscovery test failed: {e}")
        return False

def test_security_features():
    """Test security features"""
    print("\n🔍 Testing security features...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            updater = ConfigUpdater(str(temp_path))
            
            # Test gitignore protection
            updater.ensure_gitignore()
            gitignore_content = updater.gitignore_file.read_text()
            
            # Check for sensitive patterns
            sensitive_patterns = ['.env', '*api_key*', '*secret*']
            for pattern in sensitive_patterns:
                assert pattern in gitignore_content, f"Pattern {pattern} not in gitignore"
            
            # Test API key masking
            test_key = "sk-test1234567890abcdef1234567890abcdef12345678"
            masked = updater.mask_api_key(test_key)
            assert len(masked) == len(test_key), "Masked key should be same length"
            assert masked[4:-4] == '*' * (len(test_key) - 8), "Middle should be masked"
            assert masked[:4] == test_key[:4], "First 4 chars should be preserved"
            assert masked[-4:] == test_key[-4:], "Last 4 chars should be preserved"
            
            print("✅ Security features tests passed")
            return True
    except Exception as e:
        print(f"❌ Security features test failed: {e}")
        return False

def test_cli_integration():
    """Test CLI command integration"""
    print("\n🔍 Testing CLI integration...")
    
    try:
        # Test that the CLI file can be imported without errors
        cli_file = Path(__file__).parent / "simple_local_cli.py"
        assert cli_file.exists(), "CLI file not found"
        
        # Check that discovery commands are present
        content = cli_file.read_text()
        assert "def discover(" in content, "Discover command not found"
        assert "def config_status(" in content, "Config status command not found"
        assert "ProviderDiscovery" in content, "ProviderDiscovery import not found"
        
        print("✅ CLI integration tests passed")
        return True
    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Provider Discovery System Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_service_detector,
        test_config_updater,
        lambda: asyncio.run(test_provider_discovery()),
        test_security_features,
        test_cli_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Discovery system is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

def demo_discovery_check():
    """Demo: Quick check for available providers"""
    print("\n🔍 Demo: Checking for available providers...")
    
    async def check():
        try:
            discovery = ProviderDiscovery(timeout=2)
            result = await discovery.quick_check()
            
            if result['success']:
                summary = result['summary']
                print(f"✅ Found {summary['total_providers']} provider(s)")
                
                if summary['total_providers'] > 0:
                    print(f"• Local services: {summary['local_providers']}")
                    print(f"• Cloud services: {summary['cloud_providers']}")
                    print(f"• Total models: {summary['total_models']}")
                    
                    print("\nDetected providers:")
                    for p in result['providers']:
                        status = "✅" if p['status'] == 'available' else "❌"
                        provider_type = "🏠" if p['type'] == 'local' else "☁️"
                        print(f"{status} {provider_type} {p['name']} ({p['models_count']} models)")
                else:
                    print("ℹ️ No providers detected. This is normal if no services are running.")
            else:
                print(f"❌ Discovery failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    
    asyncio.run(check())

if __name__ == "__main__":
    # Run tests
    success = run_all_tests()
    
    # Demo discovery check
    if success:
        demo_discovery_check()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)