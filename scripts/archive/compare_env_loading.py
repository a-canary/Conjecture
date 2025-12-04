#!/usr/bin/env python3
"""
Compare environment loading between debug script and research script
"""

import os
import json

def load_environment():
    """Load environment variables like the research script does (exact copy)"""
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

def compare_env_loading():
    """Compare different ways of getting environment variables"""
    
    print("=== Direct os.getenv ===")
    provider_url = os.getenv('PROVIDER_API_URL', 'https://llm.chutes.ai/v1')
    provider_key = os.getenv('PROVIDER_API_KEY')
    print(f"PROVIDER_API_URL: {provider_url}")
    print(f"PROVIDER_API_KEY: {'*' * (len(provider_key) - 4)}{provider_key[-4:] if provider_key else 'None'}")
    
    print("\n=== Research script loading ===")
    env_vars = load_environment()
    provider_url_script = env_vars.get('PROVIDER_API_URL', 'https://llm.chutes.ai/v1')
    provider_key_script = env_vars.get('CHUTES_API_KEY') or env_vars.get('PROVIDER_API_KEY')
    print(f"PROVIDER_API_URL: {provider_url_script}")
    print(f"API Key: {'*' * (len(provider_key_script) - 4)}{provider_key_script[-4:] if provider_key_script else 'None'}")
    
    print("\n=== Environment dump ===")
    for key, value in env_vars.items():
        if 'API' in key or 'CHUTES' in key or 'PROVIDER' in key:
            print(f"{key}: {value[:30]}{'...' if len(value) > 30 else ''}")

if __name__ == "__main__":
    compare_env_loading()