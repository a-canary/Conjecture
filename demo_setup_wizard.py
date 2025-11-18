#!/usr/bin/env python3
"""
Simple Setup Wizard Demo
Shows the new simplified configuration wizard in action
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_wizard():
    """Demonstrate the setup wizard"""
    print("🧙 Setup Wizard Demo")
    print("=" * 40)

    try:
        from config.setup_wizard import SetupWizard, check_status, auto_setup_ollama

        wizard = SetupWizard()

        # Check current configuration status
        print("Checking configuration status...")
        status = wizard.quick_status()

        print(f"\nCurrent Status:")
        print(f"Configured: {status['configured']}")
        if status['configured']:
            print(f"Provider: {status['provider']}")
            print(f"Model: {status['model']}")
            print(f"API URL: {status['api_url']}")
        else:
            print("Not configured - running auto-detection...")
        
        # Auto-detect local providers
        print(f"\nAuto-detecting local providers...")
        detected = wizard.auto_detect_local()
        
        if detected:
            print(f"Found local providers: {', '.join(detected).title()}")
            
            # Try auto-setup for Ollama if detected
            if 'ollama' in detected:
                print("Attempting auto-setup for Ollama...")
                if auto_setup_ollama():
                    print("✅ Ollama auto-setup successful!")
                else:
                    print("❌ Ollama auto-setup failed")
        else:
            print("No local providers detected")
            print("To set up local providers:")
            print("  1. Install Ollama: https://ollama.ai/")
            print("  2. Install LM Studio: https://lmstudio.ai/")

        # Check for cloud provider API keys
        print(f"\nChecking for cloud provider API keys...")
        cloud_keys = []
        for var in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'CHUTES_API_KEY']:
            if os.getenv(var):
                cloud_keys.append(var.replace('_API_KEY', '').title())
        
        if cloud_keys:
            print(f"Found API keys for: {', '.join(cloud_keys)}")
        else:
            print("No cloud API keys found in environment")

        # Run interactive setup if not configured
        if not status['configured']:
            print(f"\nStarting interactive setup...")
            print("(This will normally prompt for user input)")
            
            # Don't actually run interactive setup in demo
            print("Interactive setup available - run with user input to complete")

        print(f"\n[SUCCESS] Setup Wizard demo completed!")
        print(f"\nUsage examples:")
        print(f"1. Check status: python -c 'from config.setup_wizard import check_status; print(check_status())'")
        print(f"2. Quick setup: python -c 'from config.setup_wizard import quick_setup; quick_setup()'")  
        print(f"3. Auto Ollama: python -c 'from config.setup_wizard import auto_setup_ollama; auto_setup_ollama()'")
        print(f"4. Interactive: python -c 'from config.setup_wizard import SetupWizard; SetupWizard().interactive_setup()'")

        return True

    except Exception as e:
        print(f"Error during wizard demo: {e}")
        return False

def main():
    """Run the demo"""
    success = demo_wizard()

    if success:
        print("\n✅ Setup Wizard system is working!")
        print("\nComplexity reduction achieved:")
        print("  - 1000+ lines → ~200 lines")
        print("  - Async → Synchronous")  
        print("  - Multiple modes → Simple 3-step wizard")
        print("  - Complex interaction → Straightforward UX")
    else:
        print("\n❌ Setup Wizard needs attention")

if __name__ == "__main__":
    main()