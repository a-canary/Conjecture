#!/usr/bin/env python3
"""
Minimal iteration 2 test - just one model and one test case
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv()
    print("[OK] Environment loaded")
except ImportError:
    print("[FAIL] python-dotenv not available")


def make_api_call(prompt: str, model_name: str, max_tokens: int = 500) -> dict:
    """Simple API call to LM Studio"""
    try:
        import requests

        headers = {"Content-Type": "application/json"}
        actual_model = model_name.split(":", 1)[1]  # Remove "lmstudio:" prefix

        data = {
            "model": actual_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }

        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return {"status": "success", "content": content}
        else:
            return {"status": "error", "error": f"API {response.status_code}"}

    except Exception as e:
        return {"status": "error", "error": f"Exception: {str(e)}"}


def test_claim_formats():
    """Test different claim formats with simple example"""
    model = "lmstudio:ibm/granite-4-h-tiny"

    formats = {
        "original": "Use this format: [c1 | claim | / 0.8]",
        "simplified": "Use this format: Claim 1: claim (confidence: 80%)",
        "minimal": "Use this format: C1: claim [80%]",
        "natural": "Use this format: Claim 1: claim (confidence 80%)",
    }

    question = "What is 2+2? Break this into claims and evaluate."

    print("Testing claim formats...")
    print("=" * 50)

    for format_name, format_prompt in formats.items():
        print(f"\nTesting {format_name} format:")
        print(f"Prompt: {format_prompt}")
        print(f"Question: {question}")

        full_prompt = f"{format_prompt}\n\n{question}"

        start_time = time.time()
        result = make_api_call(full_prompt, model)
        response_time = time.time() - start_time

        if result["status"] == "success":
            response = result["content"]
            print(f"Response ({response_time:.1f}s): {response[:200]}...")

            # Check for claim-like patterns
            has_brackets = "[" in response and "]" in response
            has_claim_word = "claim" in response.lower()
            has_confidence = "%" in response or "confidence" in response.lower()

            print(
                f"  Indicators: brackets={has_brackets}, claim_word={has_claim_word}, confidence={has_confidence}"
            )
        else:
            print(f"  ERROR: {result['error']}")

        print("-" * 30)
        time.sleep(1)


if __name__ == "__main__":
    test_claim_formats()
