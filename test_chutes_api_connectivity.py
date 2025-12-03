#!/usr/bin/env python3
"""
Test Chutes API connectivity and fix model format issues
"""

import requests
import json
import os
from pathlib import Path

def test_chutes_connectivity():
    """Test basic Chutes API connectivity"""
    
    # Try to load API key from research .env
    env_file = Path(__file__).parent / 'research' / '.env'
    api_key = None
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('CHUTES_API_KEY=') and not line.startswith('CHUTES_API_KEY=your_'):
                    api_key = line.split('=', 1)[1]
                    break
    
    if not api_key:
        # Try main .env
        main_env = Path(__file__).parent / '.env'
        if main_env.exists():
            with open(main_env, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('CHUTES_API_KEY=') and not line.startswith('CHUTES_API_KEY=your_'):
                        api_key = line.split('=', 1)[1]
                        break
    
    if not api_key:
        print("ERROR: No valid CHUTES_API_KEY found!")
        return False
    
    print(f"Using API key: {api_key[:10]}...")
    
    # Test basic connectivity
    try:
        url = "https://llm.chutes.ai/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        print(f"Testing connectivity to: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            print("[OK] Basic connectivity successful!")
            
            # Parse models
            data = response.json()
            models = data.get("data", [])
            print(f"Found {len(models)} available models")
            
            # Show first few models
            for i, model in enumerate(models[:5]):
                model_id = model.get("id", "Unknown")
                context = model.get("context_length", 0)
                modalities = model.get("output_modalities", [])
                features = model.get("supported_features", [])
                
                print(f"  {i+1}. {model_id}")
                print(f"     Context: {context:,} chars")
                print(f"     Modalities: {modalities}")
                print(f"     Features: {features}")
                print()
            
            return True, models
            
        else:
            print(f"[FAIL] Connectivity failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Exception during connectivity test: {e}")
        return False

def test_chat_completion(api_key, model_name):
    """Test chat completion with specific model"""
    
    try:
        url = "https://llm.chutes.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "Answer briefly: What is 2+2?"}
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        print(f"Testing chat completion with model: {model_name}")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("[OK] Chat completion successful!")
            
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                content = message.get("content")
                reasoning = message.get("reasoning_content")
                
                print(f"  Content: {content}")
                if reasoning:
                    print(f"  Reasoning: {reasoning}")
                
                usage = result.get("usage", {})
                print(f"  Tokens used: {usage.get('total_tokens', 0)}")
                
                return True
            else:
                print("[FAIL] No choices in response")
                print(f"Response: {json.dumps(result, indent=2)}")
                return False
        else:
            print(f"[FAIL] Chat completion failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Exception during chat completion: {e}")
        return False

def main():
    """Main test function"""
    print("CHUTES API CONNECTIVITY TEST")
    print("=" * 50)

    result = test_chutes_connectivity()

    # Get API key from environment for chat completion test
    env_file = Path(__file__).parent / 'research' / '.env'
    api_key = None

    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('CHUTES_API_KEY=') and not line.startswith('CHUTES_API_KEY=your_'):
                    api_key = line.split('=', 1)[1]
                    break

    if not api_key:
        main_env = Path(__file__).parent / '.env'
        if main_env.exists():
            with open(main_env, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('CHUTES_API_KEY=') and not line.startswith('CHUTES_API_KEY=your_'):
                        api_key = line.split('=', 1)[1]
                        break

    if result is True:
        print("\n[OK] Basic connectivity works")
        return True
    elif result and len(result) == 2:
        success, models = result
        if success and models:
            print("\n[OK] Testing chat completion...")

            # Test with the first available model
            first_model = models[0].get("id")
            if first_model:
                chat_success = test_chat_completion(api_key, first_model)
                if chat_success:
                    print("\n[SUCCESS] CHUTES API IS FULLY FUNCTIONAL!")
                    return True

        print("\n[FAIL] Chat completion test failed")
        return False
    else:
        print("\n[FAIL] Basic connectivity failed")
        return False

if __name__ == "__main__":
    main()