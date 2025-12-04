#!/usr/bin/env python3
"""
Test with the same imports as the research script
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print(f"Added to sys.path: {str(Path(__file__).parent / 'src')}")

def load_environment():
    """Load real environment variables from .env files"""
    env_vars = {}
    env_files = [
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
    
    # Add from system environment
    import os
    env_vars.update(os.environ)
    
    return env_vars

def test_with_same_imports():
    """Test with the same imports and setup as research script"""
    
    env_vars = load_environment()
    
    # Now try making the request
    import requests
    import os
    
    model_name = 'chutes:zai-org/GLM-4.6'
    prompt = "Test with same imports as research script"
    
    try:
        if "chutes" in model_name:
            # Chutes API
            api_url = env_vars.get('PROVIDER_API_URL', 'https://llm.chutes.ai/v1')
            api_key = env_vars.get('CHUTES_API_KEY') or env_vars.get('PROVIDER_API_KEY')
            endpoint = f"{api_url}/chat/completions"
            
            print(f"INFO: API URL: {api_url}")
            print(f"INFO: Endpoint: {endpoint}")

            if not api_key:
                raise ValueError("CHUTES_API_KEY or PROVIDER_API_KEY not found in environment")

            # Extract model name
            if "GLM-4.5-Air" in model_name:
                model = "zai-org/GLM-4.5-Air"
            elif "GLM-4.6" in model_name:
                model = "zai-org/GLM-4.6"
            else:
                model = model_name.split(':')[-1]

            # Prepare request
            headers = {
                "Content-Type": "application/json"
            }

            if "chutes" in model_name:
                headers["Authorization"] = f"Bearer {api_key}"

            data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.3
            }

            print(f"Making request to: {endpoint}")
            print(f"Model: {model}")
            
            # Make request
            response = requests.post(endpoint, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            
            print(f"Response status: {response.status_code}")
            
            result = response.json()
            
            # Extract response text with GLM reasoning content support
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                
                # Check for reasoning_content first (GLM models)
                reasoning_content = message.get("reasoning_content")
                if reasoning_content:
                    print(f"SUCCESS! Got {len(reasoning_content)} characters from reasoning_content")
                    return
                
                # Fallback to standard content
                content = message.get("content")
                if content:
                    print(f"SUCCESS! Got {len(content)} characters from content")
                    return
                
                print("No content found in response")
            else:
                print(f"Unexpected response format: {result}")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_same_imports()