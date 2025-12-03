#!/usr/bin/env python3
"""
Debug Chutes API Integration
Test and fix Chutes API calls for GLM models
"""

import requests
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def load_environment():
    """Load environment variables"""
    env_vars = {}
    env_files = [
        Path(__file__).parent.parent / '.env',
        Path(__file__).parent / '.env'
    ]
    
    for env_file in env_files:
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
    
    return env_vars

def test_chutes_api():
    """Test Chutes API with debugging"""
    print("Testing Chutes API Integration")
    print("=" * 40)
    
    # Load environment
    env_vars = load_environment()
    
    # Check required variables
    api_url = env_vars.get('PROVIDER_API_URL', 'https://llm.chutes.ai/v1')
    api_key = env_vars.get('CHUTES_API_KEY')
    
    print(f"API URL: {api_url}")
    print(f"API Key: {'SET' if api_key else 'NOT SET'}")
    
    if not api_key:
        print("ERROR: CHUTES_API_KEY not found!")
        return False
    
    # Test different endpoints and formats
    endpoints_to_test = [
        f"{api_url}/chat/completions",
        f"{api_url}/v1/chat/completions",
        f"{api_url}/models"
    ]
    
    for endpoint in endpoints_to_test:
        print(f"\nTesting endpoint: {endpoint}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Test models endpoint first
        if endpoint.endswith('/models'):
            try:
                response = requests.get(endpoint, headers=headers, timeout=30)
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text[:500]}...")
                
                if response.status_code == 200:
                    print("✅ Models endpoint works!")
                    return True
                    
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # Test chat completion
        else:
            test_data = {
                "model": "GLM-4.6",
                "messages": [
                    {"role": "user", "content": "Hello, can you respond with just 'API test successful'?"}
                ],
                "max_tokens": 50,
                "temperature": 0.1
            }
            
            try:
                response = requests.post(endpoint, headers=headers, json=test_data, timeout=30)
                print(f"Status: {response.status_code}")
                print(f"Headers: {dict(response.headers)}")
                print(f"Response: {response.text}")
                
                if response.status_code == 200:
                    try:
                        result = response.json()
                        print(f"Parsed JSON: {json.dumps(result, indent=2)}")
                        
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                            print(f"✅ Chat completion works! Response: {content}")
                            return True
                        else:
                            print("❌ No choices in response")
                            
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error: {e}")
                else:
                    print(f"❌ HTTP error: {response.status_code}")
                    
            except Exception as e:
                print(f"Error: {e}")
    
    return False

def test_alternative_formats():
    """Test alternative API formats"""
    print("\nTesting Alternative API Formats")
    print("=" * 40)
    
    env_vars = load_environment()
    api_url = env_vars.get('PROVIDER_API_URL', 'https://llm.chutes.ai/v1')
    api_key = env_vars.get('CHUTES_API_KEY')
    
    if not api_key:
        return False
    
    # Test OpenAI-compatible format
    endpoint = f"{api_url}/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Try different data formats
    test_formats = [
        {
            "name": "OpenAI Format",
            "data": {
                "model": "GLM-4.6",
                "messages": [
                    {"role": "user", "content": "Test message"}
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }
        },
        {
            "name": "Minimal Format",
            "data": {
                "model": "GLM-4.6",
                "prompt": "Test message"
            }
        },
        {
            "name": "GLM Format",
            "data": {
                "model": "zai-org/GLM-4.6-FP8",
                "messages": [
                    {"role": "user", "content": "Test message"}
                ]
            }
        }
    ]
    
    for test_format in test_formats:
        print(f"\nTesting {test_format['name']}:")
        print(f"Data: {json.dumps(test_format['data'], indent=2)}")
        
        try:
            response = requests.post(endpoint, headers=headers, json=test_format['data'], timeout=30)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text[:500]}...")
            
            if response.status_code == 200:
                print("✅ This format works!")
                return test_format['data']
                
        except Exception as e:
            print(f"Error: {e}")
    
    return None

def main():
    """Main test function"""
    print("Chutes API Debug Tool")
    print("=" * 50)
    
    # Test basic API
    if test_chutes_api():
        print("\n✅ Chutes API is working!")
        return True
    
    # Test alternative formats
    working_format = test_alternative_formats()
    if working_format:
        print(f"\n✅ Found working format: {working_format}")
        return True
    
    print("\n❌ Could not get Chutes API working")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)