#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
LFM-2.5 Baseline Test Using Curl Wrapper

Workaround for Python HTTP library connection issues.
Uses subprocess curl calls which work reliably with LM Studio.
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime, timezone

from datasets import load_dataset


LFM_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL = "liquid/lfm2.5-1.2b"
N_PROBLEMS = 10  # Quick baseline


def call_lfm(prompt, system="", max_tokens=400):
    """Call LFM via curl subprocess."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens
    }

    cmd = [
        "curl", "-s", "-X", "POST", LFM_ENDPOINT,
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            content = data["choices"][0]["message"]["content"]
            tokens = data["usage"]["total_tokens"]
            return content, tokens
        else:
            return None, 0
    except Exception as e:
        print(f"    Error: {e}")
        return None, 0


def extract_answer(text, target):
    """Extract answer from response."""
    if not text:
        return None

    # Exact match
    if target in text:
        return target

    # Options
    for opt in ['(A)', '(B)', '(C)']:
        if opt in text:
            return opt

    return None


def main():
    print("\n" + "="*70)
    print("LFM-2.5 BASELINE TEST")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Method: curl subprocess wrapper")
    print(f"Problems: {N_PROBLEMS}")
    print()

    # Load BBH problems
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects")
    data = ds["test"]
    problems = [{"input": data[i]["input"], "target": data[i]["target"]} for i in range(N_PROBLEMS)]

    # Direct baseline
    print("Running DIRECT prompting baseline...")
    correct = 0
    total_tokens = 0
    times = []

    for i, problem in enumerate(problems, 1):
        start = time.time()
        query = problem["input"]
        target = problem["target"]

        system = "You are a reasoning assistant. Think step by step and answer clearly."
        print(f"  [{i}/{N_PROBLEMS}] ", end="", flush=True)

        response, tokens = call_lfm(query, system=system, max_tokens=400)
        total_tokens += tokens

        if response:
            answer = extract_answer(response, target)
            is_correct = answer == target
            if is_correct:
                correct += 1
                print(f"✓ (answer: {answer})")
            else:
                print(f"✗ (got: {answer}, expected: {target})")
        else:
            print(f"✗ (no response)")

        times.append(time.time() - start)

    accuracy = 100 * correct / N_PROBLEMS
    avg_time = sum(times) / len(times) if times else 0

    print(f"\n{'='*70}")
    print("BASELINE RESULTS")
    print(f"{'='*70}")
    print(f"Correct: {correct}/{N_PROBLEMS}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Avg time: {avg_time:.2f}s per problem")
    print(f"Total tokens: {total_tokens:,}")
    print(f"{'='*70}\n")

    # Save results
    results = {
        "model": MODEL,
        "endpoint": LFM_ENDPOINT,
        "method": "DIRECT",
        "n_problems": N_PROBLEMS,
        "task": "BBH logical_deduction_three_objects",
        "correct": correct,
        "accuracy": accuracy,
        "avg_time": avg_time,
        "total_tokens": total_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    results_dir = Path("experiments/results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"lfm_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved: {results_file}\n")

    # Interpretation
    print("INTERPRETATION:")
    print(f"LFM-2.5 (1.2B) direct baseline: {accuracy:.1f}%")
    print("\nNEXT: Test with claim-enhanced prompting to validate core thesis:")
    print("  'DB storage + LLM reasoning + semantic indexing = intelligent tiny model'")
    print("\nIf claim enhancement shows +10pp improvement, thesis validated!")

    return results


if __name__ == "__main__":
    main()
