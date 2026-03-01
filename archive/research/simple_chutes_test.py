#!/usr/bin/env python3
"""
Simple Chutes Test
Debug the None response issue
"""

import requests
import json
import sys
from pathlib import Path

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

def test_single_model(model_name):
    """Test a single model with detailed debugging"""
    env_vars = load_environment()
    api_url = env_vars.get('PROVIDER_API_URL', 'https://llm.chutes.ai/v1')
    api_key = env_vars.get('CHUTES_API_KEY')
    
    print(f"Testing model: {model_name}")
    print(f"API URL: {api_url}")
    print(f"API Key: {api_key[:20]}..." if api_key else "NOT SET")
    
    if not api_key:
        print("ERROR: No API key!")
        return False
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Simple test prompt
    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Say hello"}
        ],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    print(f"Request data: {json.dumps(data, indent=2)}")
    
    try:
        endpoint = f"{api_url}/chat/completions"
        print(f"Endpoint: {endpoint}")
        
        response = requests.post(endpoint, headers=headers, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"Parsed JSON: {json.dumps(result, indent=2)}")
                
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    print(f"Choice: {json.dumps(choice, indent=2)}")
                    
                    if "message" in choice:
                        message = choice["message"]
                        print(f"Message: {json.dumps(message, indent=2)}")
                        
                        if "content" in message:
                            content = message["content"]
                            print(f"Content: '{content}' (type: {type(content)}, len: {len(content) if content else 'None'})")
                            
                            if content is None:
                                print("WARNING: Content is None!")
                                return False
                            elif content == "":
                                print("WARNING: Content is empty string!")
                                return False
                            else:
                                print(f"SUCCESS: Got content: {content}")
                                return True
                        else:
                            print("ERROR: No 'content' field in message")
                            return False
                    else:
                        print("ERROR: No 'message' field in choice")
                        return False
                else:
                    print("ERROR: No 'choices' field in response")
                    return False
                    
            except json.JSONDecodeError as e:
                print(f"ERROR: JSON decode error: {e}")
                return False
        else:
            print(f"ERROR: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"ERROR: Exception occurred: {e}")
        return False

def main():
    """Main function"""
    print("Simple Chutes Test")
    print("=" * 30)
    
    # Test models one by one
    models_to_test = [
        'zai-org/GLM-4.6',
        'zai-org/GLM-4.5-Air',
        'openai/gpt-oss-20b'
    ]
    
    for model in models_to_test:
        print(f"\n{'='*50}")
        success = test_single_model(model)
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        
        if success:
            print(f"\n✅ {model} works! Using this model for research.")
            return model
    
    print(f"\n❌ No models worked successfully.")
    return None

if __name__ == "__main__":
    working_model = main()
    if working_model:
        print(f"\nUse this model in your research: {working_model}")
    else:
        print("\nAll models failed - check API key and endpoint")