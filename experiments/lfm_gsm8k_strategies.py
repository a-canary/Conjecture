#!/usr/bin/env python3
"""
GSM8K Strategy Exploration

Finding: Claims don't help math (50% → 50%)
Hypothesis: Math needs different strategy than principles

Test 5 approaches:
1. Direct (baseline)
2. Principles (failed earlier)
3. Step-by-step scaffold
4. Worked example
5. Format guidance
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
    # Look for "####" GSM8K format or last number
    if "####" in text:
        return text.split("####")[-1].strip()
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else None


def test_strategy(problems, strategy_name, prompt_builder):
    """Test a specific prompting strategy."""
    print(f"\n{strategy_name}:")
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
        "strategy": strategy_name,
        "correct": correct,
        "accuracy": accuracy
    }


def main():
    print("="*70)
    print("GSM8K STRATEGY EXPLORATION")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Task: GSM8K (math word problems)")
    print(f"Problems: {N_PROBLEMS}")
    print("\nFinding: Principles didn't help (50% → 50%)")
    print("Testing: Alternative strategies for math reasoning")
    print("="*70)

    # Load problems
    ds = load_dataset("gsm8k", "main", split="test")
    problems = []
    for i in range(N_PROBLEMS):
        item = ds[i]
        answer = item["answer"].split("####")[-1].strip()
        problems.append({
            "query": item["question"],
            "target": answer
        })

    # Strategy 1: Direct
    s1 = test_strategy(
        problems,
        "1. Direct (baseline)",
        lambda q: f"{q}\n\nSolve this problem."
    )

    # Strategy 2: Step-by-step scaffold
    s2 = test_strategy(
        problems,
        "2. Step-by-step scaffold",
        lambda q: f"{q}\n\nSolve step-by-step:\n1. Identify what we know\n2. Identify what we need to find\n3. Write equations\n4. Calculate\n5. Final answer"
    )

    # Strategy 3: Worked example
    s3 = test_strategy(
        problems,
        "3. Worked example",
        lambda q: f"""Example: "Sam has 3 apples. He gets 2 more. How many now?"
Solution: 3 + 2 = 5 apples

Now solve: {q}

Follow the same format."""
    )

    # Strategy 4: Format guidance
    s4 = test_strategy(
        problems,
        "4. Format guidance",
        lambda q: f"""{q}

Show your work clearly:
- Write the equation
- Calculate step by step
- Give final answer as: #### [number]"""
    )

    # Strategy 5: Chain-of-thought
    s5 = test_strategy(
        problems,
        "5. Chain-of-thought",
        lambda q: f"{q}\n\nLet's think through this step by step:"
    )

    # Summary
    results = [s1, s2, s3, s4, s5]

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Strategy':<35} {'Accuracy':<10} {'vs Baseline'}")
    print("-"*70)

    baseline_acc = s1['accuracy']
    for r in results:
        delta = r['accuracy'] - baseline_acc
        print(f"{r['strategy']:<35} {r['accuracy']:>6.0f}%    {delta:>+6.0f}pp")

    print("="*70)

    # Find best
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\nBEST: {best['strategy']} → {best['accuracy']:.0f}%")

    if best['accuracy'] > baseline_acc + 10:
        print("\n✅ BREAKTHROUGH: Alternative strategy improves math!")
        print(f"   {best['strategy']} adds {best['accuracy']-baseline_acc:.0f}pp")
        print("\n   Insight: Math needs scaffolding, not principles")
    else:
        print("\n⚠️  LIMITED IMPROVEMENT")
        print("   Math reasoning may require:")
        print("   - Larger models (1.2B too small for multi-step calculation)")
        print("   - Symbolic computation tools (Strategy #71)")
        print("   - Fine-tuning on math problems")

    # Save
    output = {
        "experiment": "GSM8K strategy exploration",
        "model": MODEL,
        "n_problems": N_PROBLEMS,
        "results": results,
        "best": best,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    results_file = Path("experiments/results") / f"lfm_gsm8k_strategies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    main()
