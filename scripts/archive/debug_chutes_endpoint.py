#!/usr/bin/env python3
"""
Debug the Chutes API endpoint issue
"""

import os
import requests
import json

def test_chutes_endpoint():
    """Test the exact endpoint configuration"""
    
    # Test direct API call to the working URL
    api_url = "https://llm.chutes.ai/v1"
    api_key = os.getenv('CHUTES_API_KEY') or os.getenv('PROVIDER_API_KEY')
    
    print(f"API URL: {api_url}")
    print(f"API Key: {api_key[:20]}...{api_key[-4:] if api_key else 'None'}")
    
    # Test the models endpoint first
    try:
        print("\n1. Testing models endpoint...")
        response = requests.get(f"{api_url}/models", headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }, timeout=30)
        
        print(f"Models endpoint status: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"Found {len(models.get('data', []))} models")
            # Look for our model
            for model in models.get('data', []):
                if 'GLM-4.6' in model.get('id', ''):
                    print(f"Found GLM-4.6 model: {model.get('id')}")
                    break
        else:
            print(f"Models endpoint error: {response.text}")
    except Exception as e:
        print(f"Models endpoint failed: {e}")
    
    # Test the chat completions endpoint
    try:
        print("\n2. Testing chat completions endpoint...")
        data = {
            "model": "zai-org/GLM-4.6",
            "messages": [{"role": "user", "content": "Hello, test message"}],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        endpoint = f"{api_url}/chat/completions"
        print(f"Making POST request to: {endpoint}")
        print(f"Data: {json.dumps(data, indent=2)}")
        
        response = requests.post(endpoint, headers=headers, json=data, timeout=30)
        
        print(f"Chat completions status: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Chat completions endpoint working!")
        else:
            print("❌ Chat completions endpoint failed")
            
    except Exception as e:
        print(f"Chat completions endpoint failed: {e}")

if __name__ == "__main__":
    test_chutes_endpoint()