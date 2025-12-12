#!/usr/bin/env python3
"""
Test Config Integration
Tests if we can properly read API keys from Conjecture config
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_config_reading():
    """Test reading API keys from Conjecture config"""
    try:
        from config.unified_config import get_config
        config = get_config()

        print("Configuration Reading Test")
        print("=" * 40)

        # Check providers
        for provider in config.settings.providers:
            print(f"Provider: {provider.name}")
            print(f"  URL: {provider.url}")
            print(f"  Model: {provider.model}")
            print(f"  API Key: {'SET' if provider.api else 'EMPTY'}")
            print(f"  Priority: {provider.priority}")
            print(f"  Local: {provider.is_local}")
            print()

        # Find OpenRouter provider
        openrouter_provider = None
        for provider in config.settings.providers:
            if provider.name == "openrouter":
                openrouter_provider = provider
                break

        if openrouter_provider:
            if openrouter_provider.api and openrouter_provider.api != "sk-or-your-api-key-here":
                print(f"✓ OpenRouter API key is configured: {openrouter_provider.api[:20]}...")
                return True
            else:
                print("✗ OpenRouter API key is not configured or still placeholder")
                return False
        else:
            print("✗ OpenRouter provider not found in config")
            return False

    except Exception as e:
        print(f"✗ Error reading config: {e}")
        return False

if __name__ == "__main__":
    success = test_config_reading()
    if success:
        print("\n✓ Config integration working - ready for GPT-OSS testing")
    else:
        print("\n✗ Config integration needs fixing")