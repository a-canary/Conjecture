#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
LFM-2.5 Baseline Simple Test (Synchronous)

Uses synchronous requests library for stability with LM Studio endpoint.
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone

from datasets import load_dataset
import requests


LFM_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL = "liquid/lfm2.5-1.2b"
N_PROBLEMS = 10

def call_lfm(prompt, system="", max_tokens=400):
    """Call LFM-2.5 endpoint."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            LFM_ENDPOINT,
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": max_tokens
            },
            timeout=120  # 2 minute timeout
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data["usage"]["total_tokens"]
        return content, tokens
    except Exception as e:
        print(f"    Error: {e}")
        return None, 0


def extract_answer(text, target):
    """Extract answer from response."""
    if not text:
        return None

    # Try exact match
    if target in text:
        return target

    # Try options
    for opt in ['(A)', '(B)', '(C)']:
        if opt in text:
            return opt

    return None


def main():
    print("\n" + "="*70)
    print("LFM-2.5 BASELINE TEST (Direct Prompting Only)")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Problems: {N_PROBLEMS}")
    print()

    # Load problems
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects")
    data = ds["test"]
    problems = [{"input": data[i]["input"], "target": data[i]["target"]} for i in range(N_PROBLEMS)]

    # Test direct prompting
    print("Running DIRECT baseline...")
    correct = 0
    total_tokens = 0

    for i, problem in enumerate(problems, 1):
        query = problem["input"]
        target = problem["target"]

        system = "You are a reasoning assistant. Think step by step."
        print(f"  [{i}/{N_PROBLEMS}] ", end="", flush=True)

        response, tokens = call_lfm(query, system=system)
        total_tokens += tokens

        if response:
            answer = extract_answer(response, target)
            is_correct = answer == target
            if is_correct:
                correct += 1
                print(f"✓ (answer: {answer})")
            else:
                print(f"✗ (got: {answer}, target: {target})")
        else:
            print(f"✗ (no response)")

    accuracy = 100 * correct / N_PROBLEMS
    print(f"\nRESULTS: {correct}/{N_PROBLEMS} correct ({accuracy:.1f}%)")
    print(f"Total tokens: {total_tokens:,}")

    # Save
    results = {
        "model": MODEL,
        "endpoint": LFM_ENDPOINT,
        "n_problems": N_PROBLEMS,
        "method": "DIRECT",
        "correct": correct,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    results_dir = Path("experiments/results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"lfm_baseline_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {results_file}")

    print("\nNEXT: Run with claim-enhanced prompting to test core thesis")


if __name__ == "__main__":
    main()
