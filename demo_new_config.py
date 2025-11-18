#!/usr/bin/env python3
"""
Provider Configuration Demo
Shows how to configure different providers with the new format
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_provider_configuration():
    """Demonstrate provider configuration"""
    print("Provider Configuration Demo")
    print("=" * 40)
    
    print("NEW FORMAT: PROVIDER_[NAME]=[BASE_URL],[API_KEY],[MODEL],[PROTOCOL]")
    print()
    
    # Show examples for each provider
    examples = {
        "Local Services": {
            "Ollama": "PROVIDER_OLLAMA=http://localhost:11434,,llama2,ollama",
            "LM Studio": "PROVIDER_LM_STUDIO=http://localhost:1234/v1,,microsoft/DialoGPT-medium,openai"
        },
        "Cloud Services": {
            "Chutes.ai": "PROVIDER_CHUTES=https://api.chutes.ai/v1,your-chutes-key,chutes-gpt-3.5-turbo,openai",
            "OpenRouter": "PROVIDER_OPENROUTER=https://openrouter.ai/api/v1,your-openrouter-key,openai/gpt-3.5-turbo,openai",
            "Groq": "PROVIDER_GROQ=https://api.groq.com/openai/v1,your-groq-key,llama3-8b-8192,openai",
            "OpenAI": "PROVIDER_OPENAI=https://api.openai.com/v1,your-openai-key,gpt-3.5-turbo,openai",
            "Anthropic": "PROVIDER_ANTHROPIC=https://api.anthropic.com,your-anthropic-key,claude-3-haiku-20240307,anthropic",
            "Google": "PROVIDER_GOOGLE=https://generativelanguage.googleapis.com,your-google-key,gemini-pro,google",
            "Cohere": "PROVIDER_COHERE=https://api.cohere.ai/v1,your-cohere-key,command,cohere"
        }
    }
    
    for category, providers in examples.items():
        print(f"{category}:")
        print("-" * len(category))
        for name, config in providers.items():
            print(f"  {name}:")
            print(f"    {config}")
        print()
    
    print("CONFIGURATION STEPS:")
    print("1. Copy template: copy .env.example .env")
    print("2. Edit .env file")
    print("3. Uncomment ONE provider line")
    print("4. Replace 'your-key' with actual API key")
    print("5. Save the file")
    print()
    
    print("PRIORITY ORDER:")
    print("1. Ollama (local)")
    print("2. LM Studio (local)")
    print("3. Chutes.ai (cloud)")
    print("4. OpenRouter (cloud)")
    print("5. OpenAI (cloud)")
    print("6. Anthropic (cloud)")
    print("7. Google (cloud)")
    print("8. Groq (cloud)")
    print("9. Cohere (cloud)")
    print()
    
    print("FORMAT BREAKDOWN:")
    print("  BASE_URL: API endpoint URL")
    print("  API_KEY: Your API key (empty for local services)")
    print("  MODEL: Model name to use")
    print("  PROTOCOL: API protocol (openai, anthropic, google, cohere, ollama)")

def test_current_config():
    """Test current configuration"""
    print("\nCurrent Configuration Test:")
    print("=" * 30)
    
    try:
        # Use new unified validator (replaces SimpleProviderValidator)
        from config.unified_validator import UnifiedConfigValidator
        
        validator = UnifiedConfigValidator()
        
        if validator.providers:
            print(f"Found {len(validator.providers)} configured provider(s):")
            
            for name, provider in validator.providers.items():
                print(f"  - {provider.name} (Priority: {provider.priority})")
                print(f"    URL: {provider.base_url}")
                print(f"    Model: {provider.model}")
                print(f"    Protocol: {provider.protocol}")
                print(f"    Type: {'Local' if provider.is_local else 'Cloud'}")
                print()
            
            primary = validator.get_primary_provider()
            if primary:
                print(f"Primary provider: {primary.name}")
            
            is_valid, errors = validator.validate_configuration()
            print(f"Configuration valid: {is_valid}")
            
            if errors:
                print("Errors:")
                for error in errors:
                    print(f"  - {error}")
        else:
            print("No providers configured")
            print("Edit .env file to configure a provider")
        
    except Exception as e:
        print(f"Error testing configuration: {e}")

def main():
    """Run the demo"""
    demo_provider_configuration()
    test_current_config()
    
    print("\nSUMMARY:")
    print("-" * 10)
    print("✅ New simplified format implemented")
    print("✅ Support for custom API base URLs")
    print("✅ Chutes.ai and OpenRouter included")
    print("✅ Groq added for ultra-fast inference")
    print("✅ Clear priority-based selection")
    print("✅ Easy one-line configuration")

if __name__ == "__main__":
    main()