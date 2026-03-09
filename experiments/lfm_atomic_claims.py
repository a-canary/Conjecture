#!/usr/bin/env python3
"""
Strategy #41: Atomic Claims (Single-Fact Statements)

Finding: 1-3 claims optimal for tiny models
Question: Are ATOMIC claims (single fact) better than COMPOUND claims (multiple facts)?

Example compound: "Use transitivity: if A>B and B>C then A>C"
Example atomic: "If A>B and B>C then A>C"

Or even more atomic: "Apply transitivity rule"

Hypothesis: Simpler claim structure = more working memory for problem
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

from datasets import load_dataset


LFM_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL = "liquid/lfm2.5-1.2b"
N_PROBLEMS = 10


# Different atomicity levels for same 2 logical deduction claims
CLAIM_VARIANTS = {
    "compound": [
        "Use transitivity: if A>B and B>C then A>C to determine orderings",
        "Arrange items in sequential order before making comparisons between them"
    ],
    "atomic": [
        "If A>B and B>C then A>C",
        "Arrange items in order before comparing"
    ],
    "ultra_atomic": [
        "Apply transitivity",
        "Order then compare"
    ],
    "single_word": [
        "Transitivity",
        "Ordering"
    ]
}


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


def test_atomicity(problems, level, claims):
    """Test a specific atomicity level."""
    print(f"\\n{level.replace('_', ' ').title()}:")
    print(f"  Claims: '{claims[0]}' / '{claims[1]}'")
    correct = 0

    for i, prob in enumerate(problems, 1):
        print(f"  [{i}/{len(problems)}] ", end="", flush=True)

        prompt = f"Key principles:\\n1. {claims[0]}\\n2. {claims[1]}\\n\\nProblem: {prob['query']}"
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
        "atomicity_level": level,
        "claim_examples": claims,
        "correct": correct,
        "accuracy": accuracy
    }


def main():
    print("="*70)
    print("STRATEGY #41: ATOMIC CLAIMS")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Task: BBH logical deduction (validated 100% with standard claims)")
    print(f"Problems: {N_PROBLEMS}")
    print("\\nHypothesis: Simpler claim structure = better performance")
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

    # Test different atomicity levels
    results = []
    for level in ["compound", "atomic", "ultra_atomic", "single_word"]:
        result = test_atomicity(problems, level, CLAIM_VARIANTS[level])
        results.append(result)

    # Analysis
    print(f"\\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Atomicity Level':<20} {'Example':<30} {'Accuracy':<10}")
    print("-"*70)

    for r in results:
        example = r['claim_examples'][0][:28] + "..." if len(r['claim_examples'][0]) > 28 else r['claim_examples'][0]
        print(f"{r['atomicity_level']:<20} {example:<30} {r['accuracy']:>6.0f}%")

    print("="*70)

    # Find optimal
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\\nOPTIMAL: {best['atomicity_level'].replace('_', ' ').title()} → {best['accuracy']:.0f}%")

    if best['atomicity_level'] == "single_word":
        print("\\n✅ EXTREME ATOMICITY WINS!")
        print("   Single-word hints outperform full explanations")
        print("   Tiny models benefit from minimal claim structure")
        print("\\n   Production insight: 'Transitivity' > 'Use transitivity: if A>B...'")
    elif best['atomicity_level'] == "ultra_atomic":
        print("\\n✅ ULTRA-ATOMIC OPTIMAL!")
        print("   Brief statements without explanation perform best")
        print("   'Order then compare' > 'Arrange items in order...'")
    elif best['atomicity_level'] == "atomic":
        print("\\n✅ ATOMIC WINS!")
        print("   Single-fact claims outperform compound claims")
        print("   Remove prefixes and explanations")
    else:
        print("\\n⚠️  COMPOUND CLAIMS SUFFICIENT")
        print("   Current format already optimal - atomicity doesn't improve")

    # Save
    output = {
        "strategy": "#41 Atomic claims",
        "model": MODEL,
        "task": "BBH logical deduction",
        "n_problems": N_PROBLEMS,
        "results": results,
        "optimal": best,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    results_file = Path("experiments/results") / f"lfm_atomic_claims_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\\nSaved: {results_file}")


if __name__ == "__main__":
    main()
