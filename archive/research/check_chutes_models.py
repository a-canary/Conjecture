#!/usr/bin/env python3
"""
Check Available Models on Chutes
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

def get_available_models():
    """Get list of available models from Chutes"""
    env_vars = load_environment()
    api_url = env_vars.get('PROVIDER_API_URL', 'https://llm.chutes.ai/v1')
    api_key = env_vars.get('CHUTES_API_KEY')
    
    if not api_key:
        print("ERROR: CHUTES_API_KEY not found!")
        return []
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.get(f"{api_url}/models", headers=headers, timeout=30)
        
        if response.status_code == 200:
            models_data = response.json()
            
            if 'data' in models_data:
                models = models_data['data']
                print(f"Found {len(models)} available models:")
                print("=" * 60)
                
                # Filter for GLM and GPT models
                glm_models = []
                gpt_models = []
                other_models = []
                
                for model in models:
                    model_id = model.get('id', '')
                    model_name = model_id.split('/')[-1] if '/' in model_id else model_id
                    
                    if 'glm' in model_name.lower() or 'zai-org' in model_id.lower():
                        glm_models.append(model)
                    elif 'gpt' in model_name.lower():
                        gpt_models.append(model)
                    else:
                        other_models.append(model)
                
                # Print GLM models
                if glm_models:
                    print("\nGLM MODELS:")
                    for model in glm_models:
                        model_id = model.get('id', '')
                        max_len = model.get('max_model_len', 'N/A')
                        print(f"  - {model_id} (max_len: {max_len})")
                
                # Print GPT models
                if gpt_models:
                    print("\nGPT MODELS:")
                    for model in gpt_models:
                        model_id = model.get('id', '')
                        max_len = model.get('max_model_len', 'N/A')
                        print(f"  - {model_id} (max_len: {max_len})")
                
                # Print other notable models
                if other_models:
                    print("\nOTHER MODELS:")
                    for model in other_models[:10]:  # Show first 10
                        model_id = model.get('id', '')
                        max_len = model.get('max_model_len', 'N/A')
                        print(f"  - {model_id} (max_len: {max_len})")
                    
                    if len(other_models) > 10:
                        print(f"  ... and {len(other_models) - 10} more")
                
                return models
            else:
                print("No 'data' field in models response")
                return []
                
        else:
            print(f"Error getting models: {response.status_code}")
            print(f"Response: {response.text}")
            return []
            
    except Exception as e:
        print(f"Error: {e}")
        return []

def test_model(model_id):
    """Test a specific model"""
    env_vars = load_environment()
    api_url = env_vars.get('PROVIDER_API_URL', 'https://llm.chutes.ai/v1')
    api_key = env_vars.get('CHUTES_API_KEY')
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Hello! Please respond with just 'Test successful'."}
        ],
        "max_tokens": 20,
        "temperature": 0.1
    }
    
    try:
        response = requests.post(f"{api_url}/chat/completions", headers=headers, json=data, timeout=60)
        
        print(f"\nTesting model: {model_id}")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(f"Response: {content}")
                return True
            else:
                print("No choices in response")
                return False
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error testing model: {e}")
        return False

def main():
    """Main function"""
    print("Chutes Available Models Checker")
    print("=" * 40)
    
    models = get_available_models()
    
    if not models:
        print("No models found!")
        return False
    
    # Test a few promising models
    test_models = []
    
    for model in models:
        model_id = model.get('id', '')
        if any(keyword in model_id.lower() for keyword in ['glm', 'gpt']):
            test_models.append(model_id)
    
    if test_models:
        print(f"\nTesting {len(test_models)} promising models:")
        for model_id in test_models[:3]:  # Test first 3
            test_model(model_id)
    
    return True

if __name__ == "__main__":
    main()