#!/usr/bin/env python3
import requests
import json
import os

# Test API connectivity with correct model
api_key = os.getenv('Conjecture_LLM_API_KEY') or os.getenv('CHUTES_API_KEY')
url = "https://llm.chutes.ai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "zai-org/GLM-4.6",
    "messages": [
        {"role": "user", "content": "Hello, respond with just: API_TEST_SUCCESS"}
    ],
    "max_tokens": 50
}

try:
    response = requests.post(url, headers=headers, json=data, timeout=15)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '').strip()
        print(f"Response: {content}")
        print("API Test: PASS" if "API_TEST_SUCCESS" in content else "API Test: FAIL")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Exception: {e}")