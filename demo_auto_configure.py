#!/usr/bin/env python3
"""
Auto-Configure Demo
Shows automatic provider configuration
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def demo_auto_configure():
    """Demonstrate automatic configuration"""
    print("Auto-Configuration Demo")
    print("=" * 40)
    
    try:
        from discovery.provider_discovery import ProviderDiscovery
        
        discovery = ProviderDiscovery()
        
        # Run automatic discovery and configuration
        print("Running automatic discovery and configuration...")
        result = await discovery.run_automatic_discovery(auto_configure=True)
        
        print(f"\nAuto-Configuration Results:")
        print(f"Success: {result.get('success', False)}")
        print(f"Message: {result.get('message', 'No message')}")
        
        # Show selected provider
        selected = result.get('selected_provider')
        if selected:
            print(f"\nSelected Provider:")
            print(f"  Name: {selected.get('name', 'Unknown')}")
            print(f"  Type: {selected.get('type', 'Unknown')}")
            print(f"  Priority: {selected.get('priority', 'Unknown')}")
            
            if selected.get('endpoint'):
                print(f"  Endpoint: {selected['endpoint']}")
            
            models = selected.get('models', [])
            if models:
                print(f"  Models: {len(models)} available")
                if len(models) <= 5:
                    for model in models:
                        print(f"    - {model}")
                else:
                    print(f"    - {models[0]} (and {len(models)-1} more)")
        
        # Show configuration changes
        config_changes = result.get('config_changes', {})
        if config_changes:
            print(f"\nConfiguration Changes:")
            for key, value in config_changes.items():
                if 'key' in key.lower():
                    # Mask API keys
                    masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                    print(f"  {key}: {masked_value}")
                else:
                    print(f"  {key}: {value}")
        
        # Show updated status
        print(f"\nUpdated Configuration Status:")
        status = discovery.get_configuration_status()
        configured = status.get('configured_providers', [])
        if configured:
            print(f"  Configured providers: {', '.join(configured)}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"Error during auto-configuration: {e}")
        return False

async def main():
    """Run the demo"""
    print("This demo will automatically detect and configure the best available provider.")
    print("It will prioritize local services (Ollama/LM Studio) over cloud services.\n")
    
    success = await demo_auto_configure()
    
    if success:
        print("\n[SUCCESS] Auto-configuration completed!")
        print("\nYour CLI is now ready to use with the detected provider.")
        print("You can now run commands like:")
        print("  python simple_local_cli.py create 'Your claim here' --user yourname")
    else:
        print("\n[INFO] Auto-configuration completed with info message.")
        print("Check the output above for details.")

if __name__ == "__main__":
    asyncio.run(main())