#!/usr/bin/env python3
"""
Configuration Setup Demo
Shows how easy it is to uncomment and configure providers
"""

def show_setup_examples():
    """Show practical setup examples"""
    print("EASY CONFIGURATION SETUP")
    print("=" * 40)
    print()
    
    print("STEP 1: Copy template")
    print("  copy .env.example .env")
    print()
    
    print("STEP 2: Edit .env file and uncomment ONE provider")
    print()
    
    print("EXAMPLE CONFIGURATIONS:")
    print("-" * 25)
    print()
    
    # Show before/after for each provider type
    examples = [
        {
            "name": "Ollama (Local - Recommended)",
            "before": "# PROVIDER_OLLAMA=http://localhost:11434,,llama2,ollama",
            "after":  "PROVIDER_OLLAMA=http://localhost:11434,,llama2,ollama",
            "note": "No API key needed - just install and start Ollama"
        },
        {
            "name": "Chutes.ai (Cloud - Fast)",
            "before": "# PROVIDER_CHUTES=https://api.chutes.ai/v1,sk-chutes-your-api-key-here,chutes-gpt-3.5-turbo,openai",
            "after":  "PROVIDER_CHUTES=https://api.chutes.ai/v1,sk-abc123def456,chutes-gpt-3.5-turbo,openai",
            "note": "Replace 'sk-chutes-your-api-key-here' with your actual key"
        },
        {
            "name": "OpenRouter (Cloud - Multi-Model)",
            "before": "# PROVIDER_OPENROUTER=https://openrouter.ai/api/v1,sk-or-your-api-key-here,openai/gpt-3.5-turbo,openai",
            "after":  "PROVIDER_OPENROUTER=https://openrouter.ai/api/v1,sk-or-xyz789uvw456,openai/gpt-3.5-turbo,openai",
            "note": "Replace 'sk-or-your-api-key-here' with your actual key"
        },
        {
            "name": "Groq (Cloud - Ultra-Fast)",
            "before": "# PROVIDER_GROQ=https://api.groq.com/openai/v1,gsk_your-api-key-here,llama3-8b-8192,openai",
            "after":  "PROVIDER_GROQ=https://api.groq.com/openai/v1,gsk_abc123def456,llama3-8b-8192,openai",
            "note": "Replace 'gsk_your-api-key-here' with your actual key"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   Before: {example['before']}")
        print(f"   After:  {example['after']}")
        print(f"   Note:   {example['note']}")
        print()
    
    print("STEP 3: Save and validate")
    print("  python simple_conjecture_cli.py validate")
    print()
    
    print("STEP 4: Start using!")
    print("  python simple_conjecture_cli.py create 'Your claim here' --user yourname")
    print()

def show_format_explanation():
    """Explain the format in detail"""
    print("FORMAT EXPLANATION")
    print("=" * 20)
    print()
    
    print("Format: PROVIDER_[NAME]=[BASE_URL],[API_KEY],[MODEL],[PROTOCOL]")
    print()
    
    print("BREAKDOWN:")
    print("-" * 10)
    print("• BASE_URL: The API endpoint URL")
    print("• API_KEY: Your authentication key (empty for local services)")
    print("• MODEL: The model name to use")
    print("• PROTOCOL: The API protocol type")
    print()
    
    print("EXAMPLE BREAKDOWN:")
    print("-" * 20)
    example = "PROVIDER_CHUTES=https://api.chutes.ai/v1,sk-abc123,chutes-gpt-3.5-turbo,openai"
    print(f"Example: {example}")
    print()
    print("Parts:")
    print(f"  • PROVIDER_CHUTES: Provider name (Chutes.ai)")
    print(f"  • https://api.chutes.ai/v1: Base URL")
    print(f"  • sk-abc123: API key")
    print(f"  • chutes-gpt-3.5-turbo: Model name")
    print(f"  • openai: Protocol (uses OpenAI-compatible API)")
    print()
    
    print("LOCAL SERVICE EXAMPLE:")
    print("-" * 25)
    local_example = "PROVIDER_OLLAMA=http://localhost:11434,,llama2,ollama"
    print(f"Example: {local_example}")
    print()
    print("Parts:")
    print(f"  • PROVIDER_OLLAMA: Provider name (Ollama)")
    print(f"  • http://localhost:11434: Base URL")
    print(f"  • (empty): No API key needed for local service")
    print(f"  • llama2: Model name")
    print(f"  • ollama: Protocol (Ollama-specific)")
    print()

def show_priority_system():
    """Explain the priority system"""
    print("PRIORITY SYSTEM")
    print("=" * 16)
    print()
    
    priorities = [
        (1, "Ollama", "Local", "Private, offline, free"),
        (2, "LM Studio", "Local", "Private, GUI-based"),
        (3, "Chutes.ai", "Cloud", "Optimized, cost-effective"),
        (4, "OpenRouter", "Cloud", "100+ models, flexible"),
        (5, "Groq", "Cloud", "Ultra-fast inference"),
        (6, "OpenAI", "Cloud", "Most popular, reliable"),
        (7, "Anthropic", "Cloud", "Advanced reasoning"),
        (8, "Google", "Cloud", "Latest Gemini models"),
        (9, "Cohere", "Cloud", "Enterprise-grade")
    ]
    
    print("The system automatically selects the first available provider:")
    print()
    
    for priority, name, type_name, description in priorities:
        print(f"{priority}. {name} ({type_name}) - {description}")
    print()
    
    print("Only ONE provider needs to be configured.")
    print("The system will automatically use the highest priority available.")

def main():
    """Run the complete demo"""
    show_setup_examples()
    show_format_explanation()
    show_priority_system()
    
    print("SUMMARY")
    print("=" * 8)
    print("✅ Examples are now working configurations")
    print("✅ Just uncomment and replace placeholder keys")
    print("✅ Priority-based automatic selection")
    print("✅ Support for custom API base URLs")
    print("✅ All major providers included")
    print()
    print("READY TO USE!")
    print("Choose your provider, uncomment the line, replace the key, and start!")

if __name__ == "__main__":
    main()