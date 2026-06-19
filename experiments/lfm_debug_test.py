#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Debug LFM-2.5 connectivity issues.
Test with progressively complex prompts to find breaking point.
"""

import requests
import json

LFM_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL = "liquid/lfm2.5-1.2b"

def test_prompt(prompt, label, max_tokens=200):
    """Test a single prompt."""
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"{'='*60}")

    try:
        response = requests.post(
            LFM_ENDPOINT,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": max_tokens
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data["usage"]["total_tokens"]

        print(f"✓ SUCCESS")
        print(f"Response: {content[:100]}...")
        print(f"Tokens: {tokens}")
        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False

# Test 1: Very simple
test_prompt("What is 2+2?", "Very simple math")

# Test 2: Slightly longer
test_prompt(
    "Solve this problem: If Alice has 3 apples and Bob gives her 2 more, how many apples does Alice have?",
    "Simple word problem"
)

# Test 3: Multiple choice format
test_prompt(
    "Question: Which is larger, 5 or 3?\n(A) 5\n(B) 3\nAnswer:",
    "Multiple choice"
)

# Test 4: Logical reasoning (shorter)
test_prompt(
    """Question: Alice, Bob, and Claire are on the same team.
Alice is to the left of Bob. Bob is to the left of Claire.
Who is in the middle?
(A) Alice
(B) Bob
(C) Claire
Answer:""",
    "Short logical reasoning"
)

# Test 5: Full BBH-style prompt (first problem from dataset)
from datasets import load_dataset
ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects")
first_problem = ds["test"][0]["input"]

print(f"\n{'='*60}")
print(f"Full BBH prompt length: {len(first_problem)} chars")
print(f"First 200 chars: {first_problem[:200]}")
print(f"{'='*60}")

test_prompt(first_problem, "Full BBH problem", max_tokens=400)

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
