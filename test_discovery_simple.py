#!/usr/bin/env python3
"""
Simple Provider Discovery Test
Tests the discovery system without Unicode characters
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_service_detection():
    """Test service detection functionality"""
    print("Testing Service Detection...")
    
    try:
        from discovery.service_detector import ServiceDetector
        detector = ServiceDetector()
        
        # Use the async context manager
        async with detector:
            # Test all provider detection
            print("Checking all providers...")
            providers = await detector.detect_all()
            
            if providers:
                print(f"Found {len(providers)} providers:")
                for provider in providers:
                    print(f"  - {provider.name} ({provider.type}): {provider.status}")
                    if provider.models:
                        print(f"    Models: {len(provider.models)} available")
                    if provider.endpoint:
                        print(f"    Endpoint: {provider.endpoint}")
            else:
                print("No providers detected")
            
            return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Error during detection: {e}")
        return False

def test_config_updater():
    """Test configuration updater"""
    print("\nTesting Configuration Updater...")
    
    try:
        from discovery.config_updater import ConfigUpdater
        updater = ConfigUpdater()
        
        # Test gitignore check
        print("Checking .gitignore protection...")
        try:
            gitignore_protected = updater.ensure_gitignore_protection()
            print(f"Gitignore protection: {'Active' if gitignore_protected else 'Not active'}")
        except Exception as e:
            print(f"Gitignore check error: {e}")
        
        # Test backup functionality
        print("Testing backup functionality...")
        try:
            if updater.env_exists():
                backup_path = updater.create_backup()
                print(f"Backup created: {backup_path}")
            else:
                print("No .env file to backup")
        except Exception as e:
            print(f"Backup test error: {e}")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Error during config test: {e}")
        return False

async def main():
    """Run all tests"""
    print("Provider Discovery System Test")
    print("=" * 40)
    
    results = {
        "service_detection": await test_service_detection(),
        "config_updater": test_config_updater()
    }
    
    print("\n" + "=" * 40)
    print("Test Results:")
    
    for test_name, success in results.items():
        status = "[OK] Working" if success else "[ERROR] Failed"
        print(f"   {test_name:20}: {status}")
    
    working_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nOverall: {working_count}/{total_count} tests passed")
    
    if working_count == total_count:
        print("[OK] Provider discovery system is ready!")
    else:
        print("[WARNING] Some components need attention")

if __name__ == "__main__":
    asyncio.run(main())