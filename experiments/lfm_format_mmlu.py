#!/usr/bin/env python3
"""
Format-Optimized MMLU

Finding: GSM8K improved from 60% → 70% with format guidance
Finding: MMLU improved from 10% → 20% with 2 claims (reasoning principles)

Hypothesis: Combine both - format guidance + claims for maximal MMLU improvement

Test 4 conditions:
1. Direct (baseline)
2. Claims only (validated: +10pp)
3. Format only (GSM8K pattern)
4. Claims + Format (combined)
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

from datasets import load_dataset


LFM_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL = "liquid/lfm2.5-1.2b"
N_PROBLEMS = 20  # Larger sample for weak baseline


# MMLU claims (validated +10pp improvement)
MMLU_CLAIMS = [
    "Carefully read the question to identify what specific knowledge is being tested",
    "Eliminate obviously incorrect options before choosing the best answer"
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


def extract_letter(text):
    """Extract multiple choice answer."""
    if not text:
        return None
    for opt in ["(A)", "(B)", "(C)", "(D)"]:
        if opt in text:
            return opt
    return None


def test_condition(problems, name, prompt_builder):
    """Test a specific condition."""
    print(f"\n{name}:")
    correct = 0

    for i, prob in enumerate(problems, 1):
        print(f"  [{i}/{len(problems)}] ", end="", flush=True)

        prompt = prompt_builder(prob["query"])
        response = call_lfm(prompt, max_tokens=300)
        answer = extract_letter(response)

        if answer == prob["target"]:
            correct += 1
            print("✓", end="")
        else:
            print("✗", end="")

    accuracy = 100 * correct / len(problems)
    print(f" → {accuracy:.0f}%")

    return {
        "condition": name,
        "correct": correct,
        "accuracy": accuracy
    }


def main():
    print("="*70)
    print("FORMAT-OPTIMIZED MMLU")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Task: MMLU (weak baseline ~10-20%)")
    print(f"Problems: {N_PROBLEMS}")
    print("\nHypothesis: Format guidance + claims > either alone")
    print("="*70)

    # Load MMLU
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

    # Condition 1: Direct (baseline)
    c1 = test_condition(
        problems,
        "1. Direct (baseline)",
        lambda q: q
    )

    # Condition 2: Claims only (validated approach)
    c2 = test_condition(
        problems,
        "2. Claims only",
        lambda q: f"Key principles:\n1. {MMLU_CLAIMS[0]}\n2. {MMLU_CLAIMS[1]}\n\nQuestion: {q}"
    )

    # Condition 3: Format only (GSM8K pattern)
    c3 = test_condition(
        problems,
        "3. Format guidance only",
        lambda q: f"{q}\n\nFormat your answer as:\n- Analysis: [brief reasoning]\n- Answer: [letter only]"
    )

    # Condition 4: Combined (claims + format)
    c4 = test_condition(
        problems,
        "4. Claims + Format (combined)",
        lambda q: f"Key principles:\n1. {MMLU_CLAIMS[0]}\n2. {MMLU_CLAIMS[1]}\n\n{q}\n\nFormat:\n- Analysis: [reasoning]\n- Answer: [letter]"
    )

    # Results
    results = [c1, c2, c3, c4]

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"{'Condition':<35} {'Accuracy':<10} {'vs Baseline'}")
    print("-"*70)

    baseline_acc = c1['accuracy']
    for r in results:
        delta = r['accuracy'] - baseline_acc
        print(f"{r['condition']:<35} {r['accuracy']:>6.0f}%    {delta:>+6.0f}pp")

    print("="*70)

    # Find best
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\nBEST: {best['condition']} → {best['accuracy']:.0f}%")

    if best['condition'] == "4. Claims + Format (combined)":
        print(f"\n✅ SYNERGY: Combined approach wins!")
        print(f"   Format guidance + reasoning claims = maximal MMLU improvement")
        print(f"   Tiny models benefit from BOTH structure and principles")
    elif best['condition'] == "2. Claims only":
        print(f"\n✅ CLAIMS SUFFICIENT: Format adds no value for MMLU")
        print(f"   Knowledge recall benefits from principles, not output structure")
    elif best['condition'] == "3. Format guidance only":
        print(f"\n⚠️  FORMAT ALONE WINS: Unexpected!")
        print(f"   Output structure may force better reasoning even without explicit claims")
    else:
        print(f"\n⚠️  NO IMPROVEMENT: None of the approaches help")
        print(f"   MMLU baseline may be too weak for any prompting strategy")

    # Save
    output = {
        "strategy": "Format-optimized MMLU (claims + format)",
        "model": MODEL,
        "task": "MMLU",
        "n_problems": N_PROBLEMS,
        "results": results,
        "best": best,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    results_file = Path("experiments/results") / f"lfm_format_mmlu_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    main()
