#!/usr/bin/env python3
"""
Strategy #38: Numeric Constraints - "Use exactly 1 principle"

Finding: 1-3 claims all achieve 100% on BBH (Goldilocks zone)
Question: Is 1 principle actually BETTER than 2-3? Or equivalent?

Test: Compare 0, 1, 2, 3 claims with explicit numeric instruction
- 0 claims: "Solve this problem" (baseline: 90%)
- 1 claim: "Use exactly this 1 principle: [claim]" (validated: 100%)
- 2 claims: "Use exactly these 2 principles: [claims]" (validated: 100%)
- 3 claims: "Use exactly these 3 principles: [claims]" (validated: 100%)

Hypothesis: Explicit numeric instruction focuses attention better
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

from datasets import load_dataset


LFM_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL = "liquid/lfm2.5-1.2b"
N_PROBLEMS = 10


# BBH logical deduction claims (expanded to 3)
CLAIMS = [
    "Use transitivity: if A>B and B>C then A>C",
    "Arrange items in order before comparing",
    "Check each statement against your ordering"
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
            return data["choices"][0]["message"]["content"]
    except:
        pass
    return None


def extract_answer(text, options):
    """Extract answer from response."""
    if not text:
        return None
    for opt in options:
        if opt in text:
            return opt
    return None


def test_with_constraint(problems, num_claims):
    """Test with explicit numeric constraint."""
    print(f"\n{num_claims} claim(s) with explicit instruction:")
    correct = 0

    for i, prob in enumerate(problems, 1):
        print(f"  [{i}/{len(problems)}] ", end="", flush=True)

        if num_claims == 0:
            prompt = f"Problem: {prob['query']}\n\nSolve this."
        elif num_claims == 1:
            prompt = f"Use exactly this 1 principle:\n- {CLAIMS[0]}\n\nProblem: {prob['query']}"
        elif num_claims == 2:
            prompt = f"Use exactly these 2 principles:\n1. {CLAIMS[0]}\n2. {CLAIMS[1]}\n\nProblem: {prob['query']}"
        else:  # 3
            prompt = f"Use exactly these 3 principles:\n1. {CLAIMS[0]}\n2. {CLAIMS[1]}\n3. {CLAIMS[2]}\n\nProblem: {prob['query']}"

        response = call_lfm(prompt, max_tokens=200)
        answer = extract_answer(response, prob["options"])

        if answer == prob["target"]:
            correct += 1
            print("✓", end="")
        else:
            print("✗", end="")

    accuracy = 100 * correct / len(problems)
    print(f" → {accuracy:.0f}%")

    return {
        "num_claims": num_claims,
        "correct": correct,
        "accuracy": accuracy
    }


def main():
    print("="*70)
    print("STRATEGY #38: NUMERIC CONSTRAINTS")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Task: BBH logical deduction")
    print(f"Problems: {N_PROBLEMS}")
    print("\nHypothesis: 'Use exactly N principles' focuses attention better")
    print("="*70)

    # Load BBH
    ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects", split="test")
    problems = []
    for i in range(N_PROBLEMS):
        item = ds[i]
        problems.append({
            "query": item["input"],
            "options": ["(A)", "(B)", "(C)"],
            "target": item["target"]
        })

    # Test 0, 1, 2, 3 claims with explicit numeric instructions
    results = []
    for count in [0, 1, 2, 3]:
        result = test_with_constraint(problems, count)
        results.append(result)

    # Analysis
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Claims':<10} {'Correct':<15} {'Accuracy':<10} {'Pattern':<20}\"")
    print("-"*70)

    baseline_acc = results[0]['accuracy']
    for r in results:
        delta = r['accuracy'] - baseline_acc
        if r['num_claims'] == 0:
            pattern = "baseline"
        elif r['num_claims'] == 1:
            pattern = "single principle"
        else:
            pattern = "Goldilocks zone"
        print(f"{r['num_claims']:<10} {r['correct']}/{N_PROBLEMS:<10} {r['accuracy']:>6.0f}%    {pattern:<20} ({delta:+.0f}pp)")

    print("="*70)

    # Find optimal
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\nOPTIMAL: {best['num_claims']} claim(s) → {best['accuracy']:.0f}%")

    if best['num_claims'] == 1 and best['accuracy'] > baseline_acc:
        print("\n✅ SINGLE PRINCIPLE OPTIMAL!")
        print(f"   Minimal intervention (exactly 1 claim) achieves maximal improvement")
        print(f"   Simpler than 2-3 claims, more effective than 0")
        print("\n   Architectural principle: LESS IS MORE (up to a point)")
    elif best['num_claims'] > 1:
        print(f"\n✅ GOLDILOCKS CONFIRMED: {best['num_claims']} claims optimal")
        print(f"   More than 1 principle needed for full performance")
    else:
        print("\n⚠️  NO IMPROVEMENT: Explicit numeric instruction doesn't help")
        print("   Model already optimally uses available claims")

    # Save
    output = {
        "strategy": "#38 Numeric constraints",
        "model": MODEL,
        "task": "BBH logical deduction",
        "n_problems": N_PROBLEMS,
        "results": results,
        "optimal": best,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    results_file = Path("experiments/results") / f"lfm_single_principle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    main()
