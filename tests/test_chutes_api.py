#!/usr/bin/env python3
"""
Test script for Chutes.ai API connectivity and model response
"""

import requests
import json
import os
import time

def test_chutes_api():
    """Test Chutes.ai API connectivity and model response"""
    print("=== Chutes.ai API Test ===")
    
    # Get API key from environment
    api_key = os.getenv('Conjecture_LLM_API_KEY') or os.getenv('CHUTES_API_KEY')
    
    if not api_key:
        print("Error: No API key found in environment variables")
        return False
    
    print(f"API Key: {'*' * 20}{api_key[-4:] if len(api_key) > 4 else '****'}")
    
    # Test 1: List available models
    print("\n--- Test 1: Model Listing ---")
    try:
        url = "https://llm.chutes.ai/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers, timeout=10)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            models = response.json()
            print(f"Found {len(models.get('data', []))} models")
            
            # Check for our target model
            target_model = "zai-org/GLM-4.6-turbo"
            model_found = any(m.get('id') == target_model for m in models.get('data', []))
            print(f"Target model '{target_model}': {'Available' if model_found else 'Not found'}")
            
            if model_found:
                print("✅ Model listing test: PASS")
            else:
                print("❌ Model listing test: FAIL - Target model not found")
                return False
        else:
            print(f"❌ Model listing failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Model listing error: {e}")
        return False
    
    # Test 2: Chat completion
    print("\n--- Test 2: Chat Completion ---")
    try:
        url = "https://llm.chutes.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "zai-org/GLM-4.6-turbo",
            "messages": [
                {"role": "user", "content": "Hello, respond with just: API_TEST_SUCCESS"}
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data, timeout=15)
        response_time = time.time() - start_time
        
        print(f"Status: {response.status_code}")
        print(f"Response time: {response_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
            usage = result.get('usage', {})
            
            print(f"Response: {content}")
            print(f"Input tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"Output tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"Total tokens: {usage.get('total_tokens', 'N/A')}")
            
            if "API_TEST_SUCCESS" in content:
                print("✅ Chat completion test: PASS")
                return True
            else:
                print("❌ Chat completion test: FAIL - Unexpected response")
                return False
        else:
            print(f"❌ Chat completion failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Chat completion error: {e}")
        return False

def test_model_capabilities():
    """Test specific model capabilities for Conjecture"""
    print("\n=== Model Capabilities Test ===")
    
    api_key = os.getenv('Conjecture_LLM_API_KEY') or os.getenv('CHUTES_API_KEY')
    url = "https://llm.chutes.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Test different types of prompts Conjecture might use
    test_prompts = [
        {
            "name": "Claim Generation",
            "prompt": "Generate a factual claim about artificial intelligence with confidence score between 0.0 and 1.0. Format: [CONFIDENCE: X.X] Claim content here."
        },
        {
            "name": "Claim Validation", 
            "prompt": "Evaluate this claim for factual accuracy: 'All birds can fly'. Respond with True, False, or Partially True and a brief explanation."
        },
        {
            "name": "Complex Reasoning",
            "prompt": "Explain the relationship between neural networks and deep learning in 2-3 sentences."
        }
    ]
    
    for test in test_prompts:
        print(f"\n--- Testing {test['name']} ---")
        
        data = {
            "model": "zai-org/GLM-4.6-turbo",
            "messages": [
                {"role": "user", "content": test['prompt']}
            ],
            "max_tokens": 200,
            "temperature": 0.3
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=data, timeout=20)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
                usage = result.get('usage', {})
                
                print(f"✅ {test['name']}: PASS")
                print(f"   Response time: {response_time:.2f}s")
                print(f"   Tokens used: {usage.get('total_tokens', 'N/A')}")
                print(f"   Response: {content[:100]}{'...' if len(content) > 100 else ''}")
            else:
                print(f"❌ {test['name']}: FAIL ({response.status_code})")
                print(f"   Error: {response.text[:100]}")
                
        except Exception as e:
            print(f"❌ {test['name']}: ERROR - {e}")

if __name__ == "__main__":
    print("🧪 Chutes.ai API Testing for Conjecture")
    print("=" * 50)
    
    # Test basic API connectivity
    api_success = test_chutes_api()
    
    if api_success:
        # Test model capabilities
        test_model_capabilities()
        print("\n🎉 All Chutes.ai tests passed!")
    else:
        print("\n❌ Chutes.ai API tests failed!")