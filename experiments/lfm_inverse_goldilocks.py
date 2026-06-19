#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Inverse Goldilocks Hypothesis

Finding: BBH (90% baseline) optimal at 1-3 claims
         MMLU (10% baseline) improved 10% → 20% with 2 claims

Hypothesis: Goldilocks principle INVERTS for weak baselines
- Strong baseline (90%): Less is more (1-3 claims optimal)
- Weak baseline (10%): More is more? (5-10 claims may help)

Test: MMLU with 0, 2, 5, 10 claims on weak baseline
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime, timezone

from datasets import load_dataset


LFM_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL = "liquid/lfm2.5-1.2b"
N_PROBLEMS = 20  # Larger sample for weak baseline


# MMLU knowledge claims - scaling from 2 to 10
MMLU_CLAIMS_POOL = [
    "Read the question carefully to identify the specific knowledge domain being tested",
    "Eliminate obviously incorrect answers before choosing the best option",
    "Look for key words that indicate the type of knowledge needed (dates, definitions, relationships)",
    "Consider common misconceptions - the wrong answers often reflect common errors",
    "Use process of elimination: remove impossible answers first",
    "If uncertain, look for the most specific and complete answer",
    "Domain expertise matters - technical terms have precise meanings",
    "Causal relationships: identify what causes what in the scenario",
    "Compare and contrast: many questions test differences between concepts",
    "Historical context: timing and sequence matter in knowledge questions"
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


def extract_letter(text):
    """Extract multiple choice answer."""
    if not text:
        return None
    for opt in ["(A)", "(B)", "(C)", "(D)"]:
        if opt in text:
            return opt
    return None


def test_mmlu_with_claims(problems, num_claims):
    """Test MMLU with specific claim count."""
    print(f"\nTesting MMLU with {num_claims} claims...")
    correct = 0

    for i, prob in enumerate(problems, 1):
        print(f"  [{i}/{len(problems)}] ", end="", flush=True)

        if num_claims == 0:
            prompt = prob["query"]
        else:
            claims = MMLU_CLAIMS_POOL[:num_claims]
            claims_text = "\n".join([f"{j+1}. {c}" for j, c in enumerate(claims)])
            prompt = f"Knowledge principles:\n{claims_text}\n\nQuestion: {prob['query']}\n\nUse the principles above."

        response, _ = call_lfm(prompt, max_tokens=300)
        answer = extract_letter(response)

        if answer == prob["target"]:
            correct += 1
            print("✓", end="")
        else:
            print("✗", end="")

    accuracy = 100 * correct / len(problems)
    print(f" → {correct}/{len(problems)} = {accuracy:.0f}%")

    return {
        "num_claims": num_claims,
        "correct": correct,
        "accuracy": accuracy
    }


def main():
    print("="*70)
    print("INVERSE GOLDILOCKS HYPOTHESIS")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Task: MMLU (weak baseline ~10-20%)")
    print(f"Problems: {N_PROBLEMS}")
    print("\nHypothesis: Weak baselines need MORE guidance, not less")
    print("  Strong tasks (BBH 90%): 1-3 claims optimal")
    print("  Weak tasks (MMLU 10%): 5-10 claims may be better")
    print("="*70)

    # Load MMLU problems
    print("\nLoading MMLU...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    problems = []
    for i in range(N_PROBLEMS):
        item = ds[i]
        choices = item["choices"]
        options = ["(A)", "(B)", "(C)", "(D)"]
        query = f"{item['question']}\n" + "\n".join([f"{opt} {ch}" for opt, ch in zip(options, choices)])
        problems.append({
            "query": query,
            "target": options[item["answer"]]
        })

    # Test different claim counts
    results = []
    for count in [0, 2, 5, 10]:
        result = test_mmlu_with_claims(problems, count)
        results.append(result)
        time.sleep(1)

    # Analysis
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Claims':<10} {'Correct':<15} {'Accuracy':<10} {'Pattern':<20}")
    print("-"*70)

    baseline_acc = results[0]['accuracy']  # 0 claims
    for r in results:
        delta = r['accuracy'] - baseline_acc
        if r['num_claims'] == 0:
            pattern = "baseline"
        elif r['num_claims'] <= 3:
            pattern = "Goldilocks zone"
        elif r['num_claims'] <= 7:
            pattern = "Beyond Goldilocks"
        else:
            pattern = "High guidance"
        print(f"{r['num_claims']:<10} {r['correct']}/{N_PROBLEMS:<10} {r['accuracy']:>6.0f}%    {pattern:<20}")

    print("="*70)

    # Find optimal
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\nOPTIMAL: {best['num_claims']} claims → {best['accuracy']:.0f}%")

    if best['num_claims'] > 3 and best['accuracy'] > baseline_acc:
        print("\n✅ INVERSE GOLDILOCKS CONFIRMED!")
        print(f"   Weak baseline ({baseline_acc:.0f}%) benefits from MORE claims ({best['num_claims']})")
        print("   Pattern: Strong models need less, weak models need more")
        print("\n   Architectural insight:")
        print("   - High-performing tasks: Use 1-3 claims (avoid overload)")
        print("   - Low-performing tasks: Use 5-10 claims (compensate weakness)")
    elif best['num_claims'] <= 3:
        print("\n⚠️  GOLDILOCKS HOLDS EVEN FOR WEAK BASELINES")
        print(f"   Even at {baseline_acc:.0f}% baseline, 1-3 claims optimal")
        print("   More guidance doesn't help weak models - capacity limit")
    else:
        print("\n⚠️  UNCLEAR: No significant difference across claim counts")
        print("   Weak baseline may be too weak to benefit from claims")

    # Save
    output = {
        "hypothesis": "Inverse Goldilocks - weak baselines need more claims",
        "model": MODEL,
        "task": "MMLU (weak baseline)",
        "n_problems": N_PROBLEMS,
        "results": results,
        "optimal": best,
        "conclusion": "Inverse confirmed" if best['num_claims'] > 3 else "Goldilocks holds",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    results_file = Path("experiments/results") / f"lfm_inverse_goldilocks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    main()
