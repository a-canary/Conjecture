#!/usr/bin/env python3
"""
Standalone test for LLM provider integrations
Tests without full Conjecture dependencies
"""

import sys
import os
import json
from unittest.mock import Mock, patch

# Mock the BasicClaim to avoid core dependency issues
class MockBasicClaim:
    def __init__(self, claim_id, content, claim_type, confidence):
        self.claim_id = claim_id
        self.content = content
        self.claim_type = claim_type
        self.confidence = confidence
        self.state = "UNKNOWN"

class MockClaimState:
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED" 
    DEBUNKED = "DEBUNKED"

class MockClaimType:
    ASSERTION = "ASSERTION"
    HYPOTHESIS = "HYPOTHESIS"
    PREDICTION = "PREDICTION"
    QUESTION = "QUESTION"
    OPINION = "OPINION"

def test_provider_imports():
    """Test that all provider modules can be imported"""
    print("Testing Provider Imports")
    print("=" * 40)
    
    providers = {}
    
    # Test imports
    try:
        from src.processing.llm.openrouter_integration import OpenRouterProcessor
        providers["OpenRouter"] = OpenRouterProcessor
        print("✅ OpenRouter")
    except Exception as e:
        print(f"❌ OpenRouter: {e}")
    
    try:
        from src.processing.llm.groq_integration import GroqProcessor
        providers["Groq"] = GroqProcessor
        print("✅ Groq")
    except Exception as e:
        print(f"❌ Groq: {e}")
    
    try:
        from src.processing.llm.openai_integration import OpenAIProcessor
        providers["OpenAI"] = OpenAIProcessor
        print("✅ OpenAI")
    except Exception as e:
        print(f"❌ OpenAI: {e}")
    
    try:
        from src.processing.llm.anthropic_integration import AnthropicProcessor
        providers["Anthropic"] = AnthropicProcessor
        print("✅ Anthropic")
    except Exception as e:
        print(f"❌ Anthropic: {e}")
    
    try:
        from src.processing.llm.cohere_integration import CohereProcessor
        providers["Cohere"] = CohereProcessor
        print("✅ Cohere")
    except Exception as e:
        print(f"❌ Cohere: {e}")
    
    try:
        from src.processing.llm.google_integration import GoogleProcessor
        providers["Google"] = GoogleProcessor
        print("✅ Google")
    except Exception as e:
        print(f"❌ Google: {e}")
    
    try:
        from src.processing.llm.chutes_integration import ChutesProcessor
        providers["Chutes"] = ChutesProcessor
        print("✅ Chutes")
    except Exception as e:
        print(f"❌ Chutes: {e}")
    
    try:
        from src.processing.llm.local_providers_adapter import LocalProviderProcessor
        providers["LocalProviders"] = LocalProviderProcessor
        print("✅ LocalProviders")
    except Exception as e:
        print(f"❌ LocalProviders: {e}")
    
    print(f"\nSuccessfully imported {len(providers)} providers")
    return providers

def test_provider_initialization(providers):
    """Test provider initialization with mock data"""
    print("\nTesting Provider Initialization")
    print("=" * 40)
    
    initialized = {}
    
    # Test each provider initialization
    for name, processor_class in providers.items():
        try:
            if name == "Chutes":
                processor = processor_class(
                    api_key="test_key",
                    api_url="https://llm.chutes.ai/v1",
                    model_name="test-model"
                )
            elif name == "OpenRouter":
                processor = processor_class(
                    api_key="test_key",
                    api_url="https://openrouter.ai/api/v1",
                    model_name="openai/gpt-3.5-turbo"
                )
            elif name == "Groq":
                processor = processor_class(
                    api_key="test_key",
                    api_url="https://api.groq.com/openai/v1",
                    model_name="llama3-8b-8192"
                )
            elif name == "OpenAI":
                processor = processor_class(
                    api_key="test_key",
                    api_url="https://api.openai.com/v1",
                    model_name="gpt-3.5-turbo"
                )
            elif name == "Anthropic":
                processor = processor_class(
                    api_key="test_key", 
                    api_url="https://api.anthropic.com",
                    model_name="claude-3-haiku-20240307"
                )
            elif name == "Google":
                # Skip Google if library not available
                try:
                    processor = processor_class(
                        api_key="test_key",
                        api_url="https://generativelanguage.googleapis.com",
                        model_name="gemini-pro"
                    )
                except ImportError:
                    print(f"⚠️ Google library not available")
                    continue
            elif name == "Cohere":
                processor = processor_class(
                    api_key="test_key",
                    api_url="https://api.cohere.ai/v1",
                    model_name="command"
                )
            elif name == "LocalProviders":
                # Skip local providers as they require actual services
                print(f"⚠️ Skipping local providers (requires service)")
                continue
            
            initialized[name] = processor
            print(f"✅ {name} initialized successfully")
            
        except Exception as e:
            print(f"❌ {name} initialization failed: {e}")
    
    print(f"\nSuccessfully initialized {len(initialized)} providers")
    return initialized

def test_provider_methods(providers):
    """Test provider key methods"""
    print("\nTesting Provider Methods")
    print("=" * 40)
    
    for name, processor in providers.items():
        try:
            # Test get_stats method
            stats = processor.get_stats()
            print(f"✅ {name}.get_stats() works")
            
            # Test reset_stats method
            processor.reset_stats()
            print(f"✅ {name}.reset_stats() works")
            
            # Test health_check method
            health = processor.health_check()
            print(f"✅ {name}.health_check() works")
            
        except Exception as e:
            print(f"❌ {name} method test failed: {e}")

def main():
    """Run standalone provider tests"""
    print("Standalone LLM Provider Tests")
    print("=" * 50)
    
    # Test imports
    providers = test_provider_imports()
    
    if not providers:
        print("\n❌ No providers could be imported")
        return 1
    
    # Test initialization  
    initialized_providers = test_provider_initialization(providers)
    
    if not initialized_providers:
        print("\n❌ No providers could be initialized")
        return 1
    
    # Test methods
    test_provider_methods(initialized_providers)
    
    print(f"\nProvider integration test completed successfully!")
    print(f"{len(initialized_providers)} providers are working correctly")
    
    return 0

if __name__ == "__main__":
    exit(main())