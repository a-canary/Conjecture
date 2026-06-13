#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Strategy #31: Shorter Prompts

Finding: 1-3 claims optimal, format guidance beats scaffolding
Hypothesis: EXTREME brevity may be even better for tiny models

Test claim presentation styles:
1. Verbose (current): "Key principles: 1. X 2. Y"
2. Terse: "Rules: X. Y."
3. Ultra-short: "X. Y."
4. Single-word hints: "transitivity ordering"
5. No prefix at all: Just claims, then problem
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

from datasets import load_dataset


LFM_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL = "liquid/lfm2.5-1.2b"
N_PROBLEMS = 10


# BBH logical deduction claims (2 claims, Goldilocks zone)
CLAIMS = [
    "Use transitivity: if A>B and B>C then A>C",
    "Arrange items in order before comparing"
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


def test_prompt_style(problems, style_name, prompt_builder):
    """Test a specific prompt style."""
    print(f"\n{style_name}:")
    correct = 0

    for i, prob in enumerate(problems, 1):
        print(f"  [{i}/{len(problems)}] ", end="", flush=True)

        prompt = prompt_builder(prob["query"], CLAIMS)
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
        "style": style_name,
        "correct": correct,
        "accuracy": accuracy
    }


def main():
    print("="*70)
    print("STRATEGY #31: SHORTER PROMPTS")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Task: BBH logical deduction (validated baseline: 100% with 2 claims)")
    print(f"Problems: {N_PROBLEMS}")
    print("\nHypothesis: Extreme brevity > current 'Key principles:' format")
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

    # Baseline: Current verbose format
    s1 = test_prompt_style(
        problems,
        "1. Verbose (current)",
        lambda q, claims: f"Key principles:\n1. {claims[0]}\n2. {claims[1]}\n\nProblem: {q}"
    )

    # Style 2: Terse with "Rules:"
    s2 = test_prompt_style(
        problems,
        "2. Terse 'Rules:'",
        lambda q, claims: f"Rules: {claims[0]}. {claims[1]}.\n\n{q}"
    )

    # Style 3: Ultra-short, no prefix
    s3 = test_prompt_style(
        problems,
        "3. Ultra-short (no prefix)",
        lambda q, claims: f"{claims[0]}. {claims[1]}.\n\n{q}"
    )

    # Style 4: Single-word hints
    s4 = test_prompt_style(
        problems,
        "4. Single-word hints",
        lambda q, claims: f"transitivity ordering\n\n{q}"
    )

    # Style 5: Claims only (minimal)
    s5 = test_prompt_style(
        problems,
        "5. Claims only",
        lambda q, claims: f"{claims[0]} {claims[1]}\n{q}"
    )

    # Results
    results = [s1, s2, s3, s4, s5]

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Style':<35} {'Accuracy':<10} {'vs Verbose'}")
    print("-"*70)

    baseline_acc = s1['accuracy']
    for r in results:
        delta = r['accuracy'] - baseline_acc
        print(f"{r['style']:<35} {r['accuracy']:>6.0f}%    {delta:>+6.0f}pp")

    print("="*70)

    # Find best
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\nBEST: {best['style']} → {best['accuracy']:.0f}%")

    if best['accuracy'] > baseline_acc:
        print(f"\n✅ BREVITY WINS: {best['style']} is {best['accuracy']-baseline_acc:.0f}pp better")
        print(f"   Optimal format: Ultra-concise presentation")
    elif best['style'] == "1. Verbose (current)":
        print(f"\n✅ CURRENT FORMAT OPTIMAL")
        print(f"   'Key principles:' prefix adds clarity without cost")
    else:
        print(f"\n⚠️  EQUIVALENT: All styles ~{baseline_acc:.0f}%")
        print(f"   Format doesn't matter for tiny models - content matters")

    # Save
    output = {
        "strategy": "#31 Shorter prompts",
        "model": MODEL,
        "task": "BBH logical deduction",
        "n_problems": N_PROBLEMS,
        "results": results,
        "best": best,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    results_file = Path("experiments/results") / f"lfm_shorter_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    main()
