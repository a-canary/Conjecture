#!/usr/bin/env python3
"""
Simple Discovery Demo
Shows provider discovery results without Unicode issues
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def demo_discovery():
    """Demonstrate provider discovery"""
    print("Provider Discovery Demo")
    print("=" * 40)
    
    try:
        from discovery.provider_discovery import ProviderDiscovery
        
        discovery = ProviderDiscovery()
        
        # Quick check of available providers
        print("Checking available providers...")
        check_result = await discovery.quick_check()
        
        print(f"\nDiscovery Results:")
        print(f"Total providers found: {check_result.get('total_providers', 0)}")
        print(f"Local providers: {check_result.get('local_providers', 0)}")
        print(f"Cloud providers: {check_result.get('cloud_providers', 0)}")
        
        # Show provider details
        providers = check_result.get('providers', [])
        if providers:
            print(f"\nProvider Details:")
            for provider in providers:
                print(f"  - {provider['name']} ({provider['type']})")
                print(f"    Status: {provider['status']}")
                if provider.get('models'):
                    print(f"    Models: {len(provider['models'])} available")
                if provider.get('endpoint'):
                    print(f"    Endpoint: {provider['endpoint']}")
                print()
        
        # Show configuration status
        print("Configuration Status:")
        status = discovery.get_configuration_status()
        print(f"  .env file exists: {status.get('env_file_exists', False)}")
        print(f"  .env.example exists: {status.get('env_example_exists', False)}")
        print(f"  Gitignore protected: {status.get('gitignore_protected', False)}")
        
        configured = status.get('configured_providers', [])
        if configured:
            print(f"  Configured providers: {', '.join(configured)}")
        
        missing = status.get('missing_providers', [])
        if missing:
            print(f"  Missing providers: {', '.join(missing)}")
        
        return True
        
    except Exception as e:
        print(f"Error during discovery: {e}")
        return False

async def main():
    """Run the demo"""
    success = await demo_discovery()
    
    if success:
        print("\n[SUCCESS] Provider discovery system is working!")
        print("\nNext steps:")
        print("1. Run: python simple_local_cli.py discover --auto")
        print("2. Or run: python simple_local_cli.py discover --check")
    else:
        print("\n[ERROR] Provider discovery needs attention")

if __name__ == "__main__":
    asyncio.run(main())