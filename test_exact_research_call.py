#!/usr/bin/env python3
"""
Test the exact research script call sequence
"""

import os
import json
import requests

def load_environment():
    """Exact copy from research script"""
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

def make_llm_call(model_name, prompt, env_vars):
    """Exact copy from research script"""
    try:
        
        if "lmstudio" in model_name:
            # LM Studio
            api_url = env_vars.get('PROVIDER_API_URL', 'http://localhost:1234')
            endpoint = f"{api_url}/v1/chat/completions"

            # Extract model name
            if "ibm/granite-4-h-tiny" in model_name:
                model = "ibm/granite-4-h-tiny"
            elif "GLM-Z1-9B-0414" in model_name:
                model = "GLM-Z1-9B-0414"
            else:
                model = model_name.split(':')[-1]

        elif "chutes" in model_name:
            # Chutes API
            api_url = env_vars.get('PROVIDER_API_URL', 'https://llm.chutes.ai/v1')
            api_key = env_vars.get('CHUTES_API_KEY') or env_vars.get('PROVIDER_API_KEY')
            endpoint = f"{api_url}/chat/completions"
            
            print(f"DEBUG RESEARCH: Chutes API - URL: {api_url}, Key: {'*' * (len(api_key) - 4)}{api_key[-4:] if api_key else 'None'}")

            if not api_key:
                raise ValueError("CHUTES_API_KEY or PROVIDER_API_KEY not found in environment")

            # Extract model name
            if "GLM-4.5-Air" in model_name:
                model = "zai-org/GLM-4.5-Air"
            elif "GLM-4.6" in model_name:
                model = "zai-org/GLM-4.6"
            else:
                model = model_name.split(':')[-1]

        else:
            raise ValueError(f"Unsupported model: {model_name}")

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

        print(f"DEBUG RESEARCH: Making request to {endpoint}")
        print(f"DEBUG RESEARCH: Model: {model}")
        print(f"DEBUG RESEARCH: Headers: {headers}")

        # Make request
        response = requests.post(endpoint, headers=headers, json=data, timeout=60)
        response.raise_for_status()

        result = response.json()

        # Extract response text with GLM reasoning content support
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            print(f"DEBUG RESEARCH: Message keys: {list(message.keys())}")

            # Check for reasoning_content first (GLM models)
            reasoning_content = message.get("reasoning_content")
            print(f"DEBUG RESEARCH: reasoning_content length: {len(reasoning_content) if reasoning_content else 'None'}")
            if reasoning_content:
                return reasoning_content.strip()

            # Fallback to standard content
            content = message.get("content")
            print(f"DEBUG RESEARCH: content length: {len(content) if content else 'None'}")
            if content:
                return content.strip()

            # If no content found, log the response
            print(f"No content found in response: {result}")
            raise ValueError("No content found in API response")
        else:
            print(f"Unexpected response format: {result}")
            raise ValueError("Unexpected response format")

    except Exception as e:
        print(f"Error making LLM call to {model_name}: {e}")
        raise

def main():
    env_vars = load_environment()
    model_name = 'chutes:zai-org/GLM-4.6'
    prompt = "Test prompt to debug the issue"
    
    try:
        result = make_llm_call(model_name, prompt, env_vars)
        print(f"SUCCESS! Result length: {len(result)}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    main()