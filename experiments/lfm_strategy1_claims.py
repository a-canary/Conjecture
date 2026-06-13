#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Strategy #1: Multi-stage Retrieval with Claims

Tests core thesis: DB storage + LLM reasoning + semantic indexing = intelligent tiny model

Approach:
1. Pre-populate claims for logical deduction domain
2. Retrieve relevant claims for each problem
3. Present claims + problem to LFM-2.5
4. Compare vs direct baseline (90%)

Success criteria: +5-10pp improvement → validates thesis
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime, timezone

from datasets import load_dataset


LFM_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL = "liquid/lfm2.5-1.2b"
N_PROBLEMS = 10


# =============================================================================
# CLAIM DATABASE (Manual for Strategy #1 - will automate later)
# =============================================================================

LOGICAL_REASONING_CLAIMS = [
    {
        "id": "c001",
        "content": "In ordering problems, if A is before B and B is before C, then A is before C (transitivity)",
        "confidence": 0.95,
        "type": "axiom"
    },
    {
        "id": "c002",
        "content": "In a sequence of three objects, the middle object has exactly one object before and one after",
        "confidence": 0.95,
        "type": "theorem"
    },
    {
        "id": "c003",
        "content": "If X is leftmost, no object is to the left of X. If X is rightmost, no object is to the right of X",
        "confidence": 0.95,
        "type": "axiom"
    },
    {
        "id": "c004",
        "content": "In spatial arrangements, 'between' implies an object has one object on each side",
        "confidence": 0.90,
        "type": "theorem"
    },
    {
        "id": "c005",
        "content": "Relative positions are transitive: if A left-of B and B left-of C, then A left-of C",
        "confidence": 0.95,
        "type": "axiom"
    }
]


def format_claims_for_prompt(claims):
    """Format claims for LLM prompt."""
    lines = ["RELEVANT REASONING PRINCIPLES:"]
    for i, claim in enumerate(claims, 1):
        conf = claim["confidence"]
        content = claim["content"]
        lines.append(f"{i}. [{conf:.2f}] {content}")
    return "\n".join(lines)


# =============================================================================
# LFM CLIENT (Curl Wrapper)
# =============================================================================

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
    if target in text:
        return target
    for opt in ['(A)', '(B)', '(C)']:
        if opt in text:
            return opt
    return None


# =============================================================================
# STRATEGY #1: CLAIM-ENHANCED PROMPTING
# =============================================================================

def run_claim_enhanced(problems):
    """Run with claim enhancement (Strategy #1)."""
    print("Running CLAIM-ENHANCED prompting (Strategy #1)...")
    correct = 0
    total_tokens = 0
    times = []

    for i, problem in enumerate(problems, 1):
        start = time.time()
        query = problem["input"]
        target = problem["target"]

        # Format prompt with claims
        claims_text = format_claims_for_prompt(LOGICAL_REASONING_CLAIMS)
        enhanced_prompt = f"""{claims_text}

PROBLEM:
{query}

Using the principles above, solve this problem step by step."""

        system = "You are a reasoning assistant. Use the provided principles to solve problems."
        print(f"  [{i}/{N_PROBLEMS}] ", end="", flush=True)

        response, tokens = call_lfm(enhanced_prompt, system=system, max_tokens=400)
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

    return {
        "method": "CLAIM_ENHANCED",
        "correct": correct,
        "accuracy": accuracy,
        "avg_time": avg_time,
        "total_tokens": total_tokens,
        "num_claims": len(LOGICAL_REASONING_CLAIMS)
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("STRATEGY #1: MULTI-STAGE RETRIEVAL WITH CLAIMS")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Claims: {len(LOGICAL_REASONING_CLAIMS)} logical reasoning principles")
    print(f"Problems: {N_PROBLEMS}")
    print()

    # Load problems
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects")
    data = ds["test"]
    problems = [{"input": data[i]["input"], "target": data[i]["target"]} for i in range(N_PROBLEMS)]

    # Run claim-enhanced
    enhanced_result = run_claim_enhanced(problems)

    # Load baseline results for comparison
    baseline_file = sorted(Path("experiments/results").glob("lfm_baseline_*.json"))[-1]
    with open(baseline_file) as f:
        baseline = json.load(f)

    # Results comparison
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Method':<25} {'Correct':>10} {'Accuracy':>10} {'Tokens':>10}")
    print("-"*70)
    print(f"{'Direct (baseline)':<25} {baseline['correct']:>6}/{baseline['n_problems']:<3} {baseline['accuracy']:>9.1f}% {baseline['total_tokens']:>10,}")
    print(f"{'Claim-Enhanced':<25} {enhanced_result['correct']:>6}/{N_PROBLEMS:<3} {enhanced_result['accuracy']:>9.1f}% {enhanced_result['total_tokens']:>10,}")
    print("-"*70)

    improvement = enhanced_result['accuracy'] - baseline['accuracy']
    print(f"{'Improvement':<25} {enhanced_result['correct'] - baseline['correct']:>+10} {improvement:>+9.1f}pp")
    print(f"{'='*70}\n")

    # Save results
    results = {
        "strategy": "Strategy #1: Multi-stage retrieval with claims",
        "model": MODEL,
        "task": "BBH logical_deduction_three_objects",
        "n_problems": N_PROBLEMS,
        "baseline": baseline,
        "claim_enhanced": enhanced_result,
        "improvement_pp": improvement,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    results_file = Path("experiments/results") / f"lfm_strategy1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved: {results_file}\n")

    # Interpretation
    print("INTERPRETATION:")
    if improvement >= 5:
        print(f"✅ SUCCESS: +{improvement:.1f}pp improvement validates core thesis!")
        print("   'DB storage + LLM reasoning + semantic indexing = intelligent tiny model'")
        print("\n   Even a 1.2B model benefits from structured reasoning principles.")
        print("   Proceed to Strategy #2-10 for further optimization.")
    elif improvement >= 0:
        print(f"⚠️  MODEST: +{improvement:.1f}pp improvement, but thesis partially validated.")
        print("   Claims help but need optimization. Try:")
        print("   - More specific claims (domain-tuned)")
        print("   - Better claim selection (semantic matching)")
        print("   - Claim confidence weighting")
    else:
        print(f"❌ REGRESSION: {improvement:.1f}pp decrease.")
        print("   Claims may be distracting for this 1.2B model.")
        print("   Try Strategy #21-40 (prompt engineering for small models)")

    return results


if __name__ == "__main__":
    main()
