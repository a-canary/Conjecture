#!/usr/bin/env python3
"""
Test the exact same call that the research script makes
"""

import os
requests = __import__('requests')

def load_environment():
    """Load environment variables like the research script does"""
    import json
    env_vars = {}
    
    # Load from .env file if it exists
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    
    # Add from system environment
    env_vars.update(os.environ)
    
    return env_vars

def test_research_script_call():
    """Test the exact same API call as the research script"""
    
    env_vars = load_environment()
    
    model_name = 'chutes:zai-org/GLM-4.6'
    prompt = "Test question for debugging"
    
    # This is the exact logic from the research script
    if "chutes" in model_name:
        # Chutes API
        api_url = env_vars.get('PROVIDER_API_URL', 'https://llm.chutes.ai/v1')
        api_key = env_vars.get('CHUTES_API_KEY') or env_vars.get('PROVIDER_API_KEY')
        endpoint = f"{api_url}/chat/completions"
        
        print(f"API URL: {api_url}")
        print(f"API Key: {'*' * (len(api_key) - 4)}{api_key[-4:] if api_key else 'None'}")
        print(f"Endpoint: {endpoint}")

        if not api_key:
            print("CHUTES_API_KEY or PROVIDER_API_KEY not found in environment")
            return

        # Extract model name
        if "GLM-4.5-Air" in model_name:
            model = "zai-org/GLM-4.5-Air"
        elif "GLM-4.6" in model_name:
            model = "zai-org/GLM-4.6"
        else:
            model = model_name.split(':')[-1]
            
        print(f"Model to use: {model}")
        
        # Prepare request (exact same as research script)
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

        print(f"Request data: {data}")
        print(f"Headers: {headers}")
        
        # Make request
        try:
            response = requests.post(endpoint, headers=headers, json=data, timeout=60)
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
            
            if response.status_code == 200:
                result = response.json()
                print("SUCCESS! API call worked.")
            else:
                print(f"FAILED: Status {response.status_code}")
                
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    test_research_script_call()