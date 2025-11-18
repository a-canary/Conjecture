#!/usr/bin/env python3
"""
Simple Auto-Configure Demo
Shows the new simplified automatic configuration using the wizard
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_auto_configure():
    """Demonstrate automatic configuration"""
    print("🚀 Simple Auto-Configuration Demo")
    print("=" * 40)

    try:
        from config.setup_wizard import SetupWizard, quick_setup, auto_setup_ollama

        wizard = SetupWizard()

        # Check current status first
        print("Checking current configuration...")
        status = wizard.quick_status()
        
        if status['configured']:
            print(f"✅ Already configured with {status['provider']} ({status['model']})")
            print("Configuration details:")
            print(f"  Provider: {status['provider']}")
            print(f"  Type: {status['provider_type']}")
            print(f"  Model: {status['model']}")
            print(f"  API URL: {status['api_url']}")
            return True

        print("Not configured - attempting auto-configuration...")

        # Try auto-detect local providers first
        print("\n1. Detecting local providers...")
        detected = wizard.auto_detect_local()
        
        if detected:
            print(f"   Found: {', '.join(detected).title()}")
            
            # Auto-configure Ollama if available
            if 'ollama' in detected:
                print("\n2. Auto-configuring Ollama...")
                if auto_setup_ollama():
                    print("   ✅ Ollama configured successfully!")
                    print("   Quick test:")
                    status_after = wizard.quick_status()
                    print(f"      Provider: {status_after.get('provider')}")
                    print(f"      Model: {status_after.get('model')}")
                    return True
                else:
                    print("   ❌ Ollama configuration failed")
            else:
                print(f"   {detected[0].title()} detected but not auto-configured")
                print(f"   Run interactive setup to configure: python -c 'from config.setup_wizard import SetupWizard; SetupWizard().interactive_setup()'")
        else:
            print("   No local providers detected")

        # Check for cloud providers
        print("\n3. Checking cloud provider API keys...")
        cloud_providers = []
        for var in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'CHUTES_API_KEY']:
            if os.getenv(var):
                provider = var.replace('_API_KEY', '').title()
                cloud_providers.append(provider)
        
        if cloud_providers:
            print(f"   Found API keys for: {', '.join(cloud_providers)}")
            print("   Run interactive setup to configure cloud provider:")
            print("   python -c 'from config.setup_wizard import SetupWizard; SetupWizard().interactive_setup()'")
        else:
            print("   No cloud API keys found")

        print("\n❌ Auto-configuration couldn't complete automatically")
        print("\nManual setup options:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Install LM Studio: https://lmstudio.ai/") 
        print("3. Set cloud API key in environment variables")
        print("4. Run interactive setup:")
        print("   python -c 'from config.setup_wizard import SetupWizard; SetupWizard().interactive_setup()'")

        return False

    except Exception as e:
        print(f"❌ Error during auto-configuration: {e}")
        return False

def main():
    """Run the demo"""
    success = demo_auto_configure()

    if success:
        print("\n✅ Simple auto-configuration successful!")
        print("\nSimplification achieved:")
        print("  - No async complexity")
        print("  - Clear 3-step wizard")
        print("  - Direct .env file manipulation")
        print("  - Focused on common use cases")
    else:
        print("\n⚠️ Auto-configuration requires manual setup")
        print("\nThe wizard provides much clearer setup guidance than the old discovery system!")

if __name__ == "__main__":
    main()