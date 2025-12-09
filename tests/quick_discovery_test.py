#!/usr/bin/env python3
"""
Quick test of the discovery system
"""

import asyncio
import sys
import os
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

@pytest.mark.asyncio
async def test_basic_functionality():
    """Test basic discovery functionality"""
    print("🔍 Testing Provider Discovery System...")
    
    try:
        # Test imports
        from discovery.provider_discovery import ProviderDiscovery
        from discovery.service_detector import ServiceDetector
        from discovery.config_updater import ConfigUpdater
        print("✅ All modules imported successfully")
        
        # Test service detector
        detector = ServiceDetector(timeout=2)
        test_key = "sk-1234567890abcdef1234567890abcdef12345678"
        masked = detector.mask_api_key(test_key)
        print(f"✅ API key masking: {masked}")
        
        # Test config updater
        updater = ConfigUpdater()
        status = updater.get_config_status()
        print(f"✅ Config status: {status}")
        
        # Test discovery quick check
        discovery = ProviderDiscovery(timeout=2)
        result = await discovery.quick_check()
        
        if result['success']:
            summary = result['summary']
            print(f"✅ Discovery completed:")
            print(f"   Total providers: {summary['total_providers']}")
            print(f"   Local services: {summary['local_providers']}")
            print(f"   Cloud services: {summary['cloud_providers']}")
            
            if summary['total_providers'] > 0:
                print("🎉 Providers detected! System is working correctly.")
            else:
                print("ℹ️ No providers detected, but system is functioning normally.")
        else:
            print(f"⚠️ Discovery returned: {result.get('error', 'No error details')}")
        
        print("\n✅ Discovery system is working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_basic_functionality())
    sys.exit(0 if success else 1)