#!/usr/bin/env python3
"""
Test script for GLM model integration with Chutes AI API
Tests both GLM-4.5-Air and GLM-4.6 models with proper response format handling
"""

import os
import requests
import json
from src.processing.llm.chutes_integration import make_chutes_request

def test_glm_model(api_url: str, api_key: str, model: str, test_prompt: str):
    """Test a specific GLM model"""
    print(f"\n{'='*60}")
    print(f"Testing {model}")
    print(f"{'='*60}")
    
    try:
        # Test the fixed API integration
        response = make_chutes_request(
            api_url=api_url,
            api_key=api_key,
            model=model,
            prompt=test_prompt,
            temperature=0.7,
            max_tokens=500
        )
        
        print(f"[SUCCESS] {model} SUCCESS!")
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:200]}{'...' if len(response) > 200 else ''}")
        
        return True, response
        
    except Exception as e:
        print(f"[FAILED] {model} FAILED: {e}")
        return False, str(e)

def test_direct_api_call(api_url: str, api_key: str, model: str, test_prompt: str):
    """Test direct API call to see raw response format"""
    print(f"\n{'='*60}")
    print(f"Testing Direct API call to {model}")
    print(f"{'='*60}")
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": test_prompt}],
            "temperature": 0.7,
            "max_tokens": 200,
        }

        response = requests.post(f"{api_url}/chat/completions", headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        print("Raw Response Structure:")
        print(json.dumps(result, indent=2))
        
        # Check response format
        choices = result.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            print(f"\nMessage fields: {list(message.keys())}")
            
            # Check for reasoning_content
            reasoning_content = message.get("reasoning_content")
            if reasoning_content:
                print(f"[FOUND] reasoning_content field")
                print(f"Reasoning content preview: {reasoning_content[:200]}")
            else:
                print("[NOT FOUND] No reasoning_content field found")
            
            # Check for standard content
            content = message.get("content")
            if content:
                print(f"[FOUND] content field")
                print(f"Standard content preview: {content[:200]}")
            else:
                print("[NOT FOUND] No content field found")
        
        return True, result
        
    except Exception as e:
        print(f"[FAILED] Direct API call failed: {e}")
        return False, str(e)

def main():
    """Run comprehensive GLM model tests"""
    print("GLM Model Integration Test")
    print("=" * 60)
    
    # Load environment
    api_url = os.getenv('CHUTES_API_URL', 'https://llm.chutes.ai/v1')
    api_key = os.getenv('CHUTES_API_KEY') or os.getenv('PROVIDER_API_KEY')
    
    if not api_key or api_key.startswith('cpk_your') or api_key.startswith('test-key'):
        print("[ERROR] No valid API key found!")
        print("Please set CHUTES_API_KEY or PROVIDER_API_KEY with a real Chutes API key")
        return
    
    print(f"API URL: {api_url}")
    print(f"API Key: {'*' * (len(api_key) - 4)}{api_key[-4:] if len(api_key) > 4 else ''}")
    
    # Define test prompt
    test_prompt = """Please analyze this statement and provide a brief evaluation:

"Artificial intelligence will completely replace human programmers within 10 years."

Provide a balanced analysis considering current technology trends and limitations."""
    
    # Test models
    models_to_test = [
        "zai-org/GLM-4.6",      # Main GLM-4.6 model
        "zai-org/GLM-4.5-Air",  # GLM-4.5-Air model
    ]
    
    results = {}
    
    for model in models_to_test:
        print(f"\n{'#'*80}")
        print(f"# TESTING {model}")
        print(f"{'#'*80}")
        
        # Test with fixed integration
        success, response = test_glm_model(api_url, api_key, model, test_prompt)
        results[f"{model}_fixed"] = success
        
        # Test direct API call to see raw format
        success_direct, raw_response = test_direct_api_call(api_url, api_key, model, test_prompt)
        results[f"{model}_direct"] = success_direct
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    for test_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{test_name}: {status}")
    
    # Overall result
    all_passed = all(results.values())
    if all_passed:
        print(f"\n[SUCCESS] ALL TESTS PASSED! GLM integration is working correctly.")
    else:
        print(f"\n[WARNING] Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()