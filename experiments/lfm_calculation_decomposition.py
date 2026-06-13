#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Enhanced Math Strategy: Calculation Decomposition

Finding: GSM8K format guidance helps (60% → 70%)
Finding: Claims don't help math (50% → 50%)
Hypothesis: Break calculation into explicit sub-steps with validation

Instead of "solve this", guide model through:
1. Extract numbers from problem
2. Identify operation needed
3. Write equation
4. Calculate result
5. Verify answer makes sense

This tests whether explicit calculation structure beats general format guidance.
"""

import subprocess
import json
import re
from pathlib import Path
from datetime import datetime, timezone

from datasets import load_dataset


LFM_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL = "liquid/lfm2.5-1.2b"
N_PROBLEMS = 10


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


def extract_number(text):
    """Extract final answer number."""
    if not text:
        return None
    if "####" in text:
        return text.split("####")[-1].strip()
    numbers = re.findall(r'-?\\d+(?:\\.\\d+)?', text)
    return numbers[-1] if numbers else None


def test_strategy(problems, name, prompt_builder):
    """Test a specific strategy."""
    print(f"\\n{name}:")
    correct = 0

    for i, prob in enumerate(problems, 1):
        print(f"  [{i}/{len(problems)}] ", end="", flush=True)

        prompt = prompt_builder(prob["query"])
        response = call_lfm(prompt, max_tokens=500)
        answer = extract_number(response)

        if answer == prob["target"]:
            correct += 1
            print("✓", end="")
        else:
            print("✗", end="")

    accuracy = 100 * correct / len(problems)
    print(f" → {accuracy:.0f}%")

    return {
        "strategy": name,
        "correct": correct,
        "accuracy": accuracy
    }


def main():
    print("="*70)
    print("CALCULATION DECOMPOSITION STRATEGY")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Task: GSM8K (math word problems)")
    print(f"Problems: {N_PROBLEMS}")
    print("\\nHypothesis: Explicit calculation steps > general format guidance")
    print("="*70)

    # Load GSM8K
    ds = load_dataset("gsm8k", "main", split="test")
    problems = []
    for i in range(N_PROBLEMS):
        item = ds[i]
        answer = item["answer"].split("####")[-1].strip()
        problems.append({
            "query": item["question"],
            "target": answer
        })

    # Strategy 1: Direct (baseline)
    def direct_prompt(q):
        return f"{q}\n\nSolve this problem."
    s1 = test_strategy(problems, "1. Direct (baseline)", direct_prompt)

    # Strategy 2: Format guidance (validated +10pp)
    def format_prompt(q):
        return f"""{q}

Show your work clearly:
- Write the equation
- Calculate step by step
- Give final answer as: #### [number]"""
    s2 = test_strategy(problems, "2. Format guidance (validated)", format_prompt)

    # Strategy 3: Calculation decomposition (new)
    def decomp_prompt(q):
        return f"""{q}

Solve systematically:
1. Numbers: [extract all numbers from problem]
2. Operation: [what calculation is needed?]
3. Equation: [write the math]
4. Result: [calculate]
5. Answer: #### [number]"""
    s3 = test_strategy(problems, "3. Calculation decomposition", decomp_prompt)

    # Strategy 4: Minimal checklist
    def checklist_prompt(q):
        return f"""{q}

☐ Extract numbers
☐ Write equation
☐ Calculate
☐ Answer: ####"""
    s4 = test_strategy(problems, "4. Minimal checklist", checklist_prompt)

    # Strategy 5: One-sentence guidance
    def oneline_prompt(q):
        return f"""{q}

Extract numbers, write equation, calculate, answer as: ####"""
    s5 = test_strategy(problems, "5. One-sentence guidance", oneline_prompt)

    # Results
    results = [s1, s2, s3, s4, s5]

    print(f"\\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Strategy':<40} {'Accuracy':<10} {'vs Baseline'}")
    print("-"*70)

    baseline_acc = s1['accuracy']
    for r in results:
        delta = r['accuracy'] - baseline_acc
        print(f"{r['strategy']:<40} {r['accuracy']:>6.0f}%    {delta:>+6.0f}pp")

    print("="*70)

    # Find best
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\\nBEST: {best['strategy']} → {best['accuracy']:.0f}%")

    if best['strategy'] == "3. Calculation decomposition":
        print("\\n✅ DECOMPOSITION WINS!")
        print(f"   Explicit step structure beats general format by {best['accuracy']-baseline_acc:.0f}pp")
        print("\\n   Optimal math strategy: Break into numbered substeps")
    elif best['strategy'] == "5. One-sentence guidance":
        print("\\n✅ BREVITY WINS!")
        print("   Minimal guidance sufficient - aligns with Goldilocks Principle")
    elif best['strategy'] == "2. Format guidance (validated)":
        print("\\n✅ VALIDATED APPROACH HOLDS")
        print("   General format guidance remains optimal for GSM8K")
    else:
        print(f"\\n✅ ALTERNATIVE APPROACH: {best['strategy']}")
        print(f"   {best['accuracy']-baseline_acc:.0f}pp improvement")

    # Save
    output = {
        "strategy": "Calculation decomposition variants",
        "model": MODEL,
        "task": "GSM8K",
        "n_problems": N_PROBLEMS,
        "results": results,
        "best": best,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    results_file = Path("experiments/results") / f"lfm_calculation_decomp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\\nSaved: {results_file}")


if __name__ == "__main__":
    main()
