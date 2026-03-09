#!/usr/bin/env python3
"""
Strategy #11: Dynamic Claim Count Optimization

Finding: 5 claims caused -20pp regression (90% → 70%)
Hypothesis: Optimal claim count for 1.2B models is 0-2, not 5

Test: 0, 1, 2, 3, 5 claims to find sweet spot
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

# Ranked claims by relevance/conciseness
CLAIMS_POOL = [
    "In a sequence of three, the middle position has one object before and one after it.",
    "If A is before B and B is before C, then A is before C (transitivity).",
    "Spatial terms like 'leftmost' mean no object exists to that side.",
    "'Between' means an object has one on each side.",
    "Relative positions are transitive: A left-of B and B left-of C means A left-of C."
]


def call_lfm(prompt, system="", max_tokens=400):
    """Call LFM via curl."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    cmd = [
        "curl", "-s", "-X", "POST", LFM_ENDPOINT,
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "model": MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": max_tokens
        })
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            return data["choices"][0]["message"]["content"], data["usage"]["total_tokens"]
    except:
        pass
    return None, 0


def extract_answer(text, target):
    if not text:
        return None
    if target in text:
        return target
    for opt in ['(A)', '(B)', '(C)']:
        if opt in text:
            return opt
    return None


def test_claim_count(problems, num_claims):
    """Test with specific number of claims."""
    print(f"\nTesting with {num_claims} claim{'s' if num_claims != 1 else ''}...")

    correct = 0
    total_tokens = 0

    for i, problem in enumerate(problems, 1):
        query = problem["input"]
        target = problem["target"]

        # Build prompt with N claims
        if num_claims == 0:
            prompt = f"Problem: {query}\n\nSolve this step by step."
            system = "You are a reasoning assistant."
        else:
            claims = CLAIMS_POOL[:num_claims]
            claims_text = "\n".join([f"{j+1}. {c}" for j, c in enumerate(claims)])
            prompt = f"Key principles:\n{claims_text}\n\nProblem: {query}\n\nUse the principles above to solve."
            system = "You are a reasoning assistant. Apply the principles provided."

        print(f"  [{i}/{N_PROBLEMS}] ", end="", flush=True)

        response, tokens = call_lfm(prompt, system=system, max_tokens=400)
        total_tokens += tokens

        if response:
            answer = extract_answer(response, target)
            if answer == target:
                correct += 1
                print("✓", end="")
            else:
                print("✗", end="")
        else:
            print("E", end="")

    print(f" → {correct}/{N_PROBLEMS} = {100*correct/N_PROBLEMS:.0f}%")

    return {
        "num_claims": num_claims,
        "correct": correct,
        "accuracy": 100 * correct / N_PROBLEMS,
        "total_tokens": total_tokens
    }


def main():
    print("="*70)
    print("STRATEGY #11: DYNAMIC CLAIM COUNT OPTIMIZATION")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Baseline: 90% (9/10) with 0 claims (direct)")
    print(f"Strategy #1: 70% (7/10) with 5 claims (-20pp)")
    print("\nHypothesis: Optimal claim count is 1-2 for 1.2B models")
    print("="*70)

    # Load problems
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects")
    problems = [{"input": ds["test"][i]["input"], "target": ds["test"][i]["target"]}
                for i in range(N_PROBLEMS)]

    # Test different claim counts
    results = []
    for count in [0, 1, 2, 3, 5]:
        result = test_claim_count(problems, count)
        results.append(result)
        time.sleep(1)  # Brief pause between tests

    # Analysis
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Claims':<10} {'Correct':<10} {'Accuracy':<10} {'vs Direct':<10}")
    print("-"*70)

    baseline_acc = 90.0  # From prior test
    for r in results:
        delta = r['accuracy'] - baseline_acc
        print(f"{r['num_claims']:<10} {r['correct']}/{N_PROBLEMS:<6} {r['accuracy']:>6.0f}%    {delta:>+6.1f}pp")

    print("="*70)

    # Find optimal
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\nOPTIMAL: {best['num_claims']} claims → {best['accuracy']:.0f}% accuracy")

    if best['accuracy'] > baseline_acc:
        print(f"✅ IMPROVEMENT: +{best['accuracy']-baseline_acc:.1f}pp over direct baseline")
        print("   Core thesis validated with optimized claim count!")
    elif best['num_claims'] == 0:
        print("⚠️  FINDING: Direct prompting optimal for this 1.2B model")
        print("   Claims consistently hurt performance - model too small for augmentation")
    else:
        print(f"⚠️  FINDING: {best['num_claims']} claims best but not better than direct")
        print("   Try different claim formats (Strategy #21-40: prompt engineering)")

    # Save
    output = {
        "strategy": "Strategy #11: Dynamic claim count",
        "model": MODEL,
        "task": "BBH logical_deduction_three_objects",
        "results": results,
        "optimal": best,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    results_file = Path("experiments/results") / f"lfm_strategy11_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    main()
