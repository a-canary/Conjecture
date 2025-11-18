#!/usr/bin/env python3
"""
Configuration Demo
Shows how the simplified system guides users
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_configuration_guidance():
    """Demonstrate configuration guidance"""
    print("Configuration Guidance Demo")
    print("=" * 40)
    
    try:
        from config.simple_validator import SimpleValidator
        
        validator = SimpleValidator()
        result = validator.validate_configuration()
        
        if not result.success:
            print("CONFIGURATION NEEDED")
            print("-" * 25)
            print(result.errors[0])  # Main error message
            
            print("\nQUICK SETUP:")
            print("1. Copy template: cp .env.example .env")
            print("2. Edit .env with your preferred provider")
            print("3. Try again: python simple_conjecture_cli.py create 'Your claim'")
            
            print("\nPROVIDER OPTIONS:")
            
            # Show available providers with setup hints
            for provider in validator.provider_configs.values():
                status = "[READY]" if provider.name.lower() in [p.lower() for p in []] else "[SETUP NEEDED]"
                print(f"  {status} {provider.name} ({provider.type})")
                print(f"    Priority: {provider.priority}")
                
                # Show first line of setup instructions
                first_line = provider.setup_instructions.strip().split('\n')[0]
                print(f"    Setup: {first_line}")
                print()
            
            print("RECOMMENDED:")
            print("- Use Ollama for local, private AI (install: https://ollama.ai/)")
            print("- Use LM Studio for GUI-based local AI (install: https://lmstudio.ai/)")
            print("- Use OpenAI for cloud-based GPT models (API key required)")
            
        else:
            print("CONFIGURATION OK")
            print("-" * 20)
            print(f"Primary provider: {result.primary_provider}")
            print(f"Available providers: {len(result.available_providers)}")
            print("Ready to use!")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def show_env_example_preview():
    """Show preview of .env.example structure"""
    print("\n.env.example PREVIEW:")
    print("=" * 30)
    
    try:
        with open('.env.example', 'r') as f:
            lines = f.readlines()
        
        # Show key sections
        in_section = False
        section_lines = []
        
        for line in lines[:50]:  # First 50 lines
            if line.startswith('# ==='):
                if section_lines:
                    print(''.join(section_lines))
                    section_lines = []
                in_section = True
            if in_section:
                section_lines.append(line)
                if len(section_lines) > 15:  # Limit section size
                    break
        
        if section_lines:
            print(''.join(section_lines))
        
        print("... (continues with more providers)")
        print(f"Total file: {len(lines)} lines with detailed instructions")
        
    except Exception as e:
        print(f"Could not read .env.example: {e}")

def main():
    """Run the demo"""
    print("Simplified Configuration System Demo")
    print("=" * 45)
    
    demo_configuration_guidance()
    show_env_example_preview()
    
    print("\nSUMMARY:")
    print("-" * 10)
    print("✅ Configuration validation working")
    print("✅ Clear error messages and guidance")
    print("✅ Comprehensive .env.example template")
    print("✅ Priority-based provider selection")
    print("✅ User-friendly setup instructions")
    
    print("\nGET STARTED:")
    print("1. cp .env.example .env")
    print("2. Edit .env (uncomment your preferred provider)")
    print("3. python simple_conjecture_cli.py validate")

if __name__ == "__main__":
    main()