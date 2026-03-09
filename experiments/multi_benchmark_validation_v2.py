#!/usr/bin/env python3
"""
Multi-Benchmark Validation V2

Test refined Goldilocks Principle (5-word claims) across diverse working benchmarks:
- TruthfulQA (truthfulness)
- HellaSwag (commonsense)
- Additional BBH subtasks (diverse reasoning)

Focus on benchmarks with reliable dataset loading.
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

from datasets import load_dataset


LFM_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL = "liquid/lfm2.5-1.2b"
N_PROBLEMS = 20


def call_lfm(prompt, max_tokens=400):
    """Call LFM via curl."""
    messages = [{"role": "user", "content": prompt}]
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
    """Extract answer."""
    if not text:
        return None
    for opt in options:
        if opt in text:
            return opt
    return None


# Ultra-concise 5-word claims (optimal)
CLAIMS_5W = {
    "reasoning": [
        "Think step by step",  # 4w
        "Check each answer carefully"  # 4w
    ],
    "commonsense": [
        "Use common sense knowledge",  # 4w
        "Consider realistic outcomes"  # 3w
    ],
    "truthfulness": [
        "Verify factual accuracy carefully",  # 4w
        "Avoid common misconceptions"  # 3w
    ],
    "causal": [
        "Identify cause effect relationships",  # 4w
        "Consider temporal ordering"  # 3w
    ]
}


def test_bbh_task(task_name, claim_type, with_claims=True):
    """Test BBH subtask."""
    print(f"\nBBH {task_name}:")

    try:
        ds = load_dataset("lukaemon/bbh", task_name, split="test")
    except:
        print(f"  ❌ Failed to load {task_name}")
        return None

    correct = 0
    for i in range(min(N_PROBLEMS, len(ds))):
        item = ds[i]
        query = item["input"]
        target = item["target"]

        if with_claims:
            claims = CLAIMS_5W[claim_type]
            prompt = f"{claims[0]}\n{claims[1]}\n\n{query}"
        else:
            prompt = query

        response = call_lfm(prompt, max_tokens=200)

        # BBH uses various formats, try to match target
        if response and target.strip() in response:
            correct += 1

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{N_PROBLEMS}]", end="", flush=True)

    accuracy = 100 * correct / N_PROBLEMS
    print(f" → {accuracy:.0f}%")

    return {
        "benchmark": f"BBH-{task_name}",
        "with_claims": with_claims,
        "correct": correct,
        "accuracy": accuracy
    }


def test_hellaswag(with_claims=True):
    """HellaSwag commonsense."""
    print(f"\nHellaSwag:")

    ds = load_dataset("hellaswag", split="validation")
    correct = 0

    for i in range(N_PROBLEMS):
        item = ds[i]
        context = item["ctx"]
        endings = item["endings"]
        options = ["(A)", "(B)", "(C)", "(D)"]

        query = f"{context}\n" + "\n".join([f"{opt} {end}" for opt, end in zip(options, endings)])

        if with_claims:
            claims = CLAIMS_5W["commonsense"]
            prompt = f"{claims[0]}\n{claims[1]}\n\n{query}"
        else:
            prompt = query

        response = call_lfm(prompt, max_tokens=100)
        answer = extract_answer(response, options)

        if answer == options[int(item["label"])]:
            correct += 1

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{N_PROBLEMS}]", end="", flush=True)

    accuracy = 100 * correct / N_PROBLEMS
    print(f" → {accuracy:.0f}%")

    return {
        "benchmark": "HellaSwag",
        "with_claims": with_claims,
        "correct": correct,
        "accuracy": accuracy
    }


def test_truthfulqa(with_claims=True):
    """TruthfulQA truthfulness."""
    print(f"\nTruthfulQA:")

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    correct = 0

    for i in range(min(N_PROBLEMS, len(ds))):
        item = ds[i]
        question = item["question"]
        choices = item["mc1_targets"]["choices"]
        labels = item["mc1_targets"]["labels"]

        # Find correct answer
        correct_idx = labels.index(1) if 1 in labels else 0
        options = ["(A)", "(B)", "(C)", "(D)"][:len(choices)]

        query = f"{question}\n" + "\n".join([f"{opt} {ch}" for opt, ch in zip(options, choices)])

        if with_claims:
            claims = CLAIMS_5W["truthfulness"]
            prompt = f"{claims[0]}\n{claims[1]}\n\n{query}"
        else:
            prompt = query

        response = call_lfm(prompt, max_tokens=100)
        answer = extract_answer(response, options[:len(choices)])

        if answer == options[correct_idx]:
            correct += 1

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{N_PROBLEMS}]", end="", flush=True)

    accuracy = 100 * correct / N_PROBLEMS
    print(f" → {accuracy:.0f}%")

    return {
        "benchmark": "TruthfulQA",
        "with_claims": with_claims,
        "correct": correct,
        "accuracy": accuracy
    }


def main():
    print("="*80)
    print("MULTI-BENCHMARK VALIDATION V2")
    print("="*80)
    print(f"Model: {MODEL}")
    print(f"Optimization: 5-word claims (ultra-concise)")
    print(f"Benchmarks: HellaSwag, TruthfulQA, BBH variants")
    print("="*80)

    results = []

    # Test HellaSwag
    print("\n" + "="*80)
    print("1. HELLASWAG (Commonsense)")
    print("="*80)
    results.append(test_hellaswag(with_claims=False))
    results.append(test_hellaswag(with_claims=True))

    # Test TruthfulQA
    print("\n" + "="*80)
    print("2. TRUTHFULQA (Truthfulness)")
    print("="*80)
    results.append(test_truthfulqa(with_claims=False))
    results.append(test_truthfulqa(with_claims=True))

    # Test BBH causal judgment
    print("\n" + "="*80)
    print("3. BBH CAUSAL JUDGMENT")
    print("="*80)
    r1 = test_bbh_task("causal_judgement", "causal", with_claims=False)
    r2 = test_bbh_task("causal_judgement", "causal", with_claims=True)
    if r1 and r2:
        results.extend([r1, r2])

    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Benchmark':<20} {'Method':<15} {'Accuracy':<10} {'Change'}")
    print("-"*80)

    improvements = []
    for i in range(0, len(results), 2):
        if i+1 < len(results):
            baseline = results[i]
            optimized = results[i+1]
            delta = optimized['accuracy'] - baseline['accuracy']
            improvements.append(delta)

            print(f"{baseline['benchmark']:<20} Baseline        {baseline['accuracy']:>6.0f}%")
            print(f"{optimized['benchmark']:<20} 5-word claims   {optimized['accuracy']:>6.0f}%      {delta:>+6.0f}pp")
            print()

    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        success_count = sum(1 for d in improvements if d > 0)

        print("="*80)
        print(f"STATISTICS")
        print(f"  Benchmarks tested: {len(improvements)}")
        print(f"  Average improvement: {avg_improvement:+.1f}pp")
        print(f"  Success rate: {success_count}/{len(improvements)} improved")
        print(f"  Generalization: {'✅ STRONG' if success_count >= 2 else '⚠️ WEAK'}")
        print("="*80)

        # Save
        output = {
            "experiment": "Multi-benchmark validation v2",
            "model": MODEL,
            "optimization": "5-word claims (ultra-concise)",
            "n_per_benchmark": N_PROBLEMS,
            "results": results,
            "statistics": {
                "avg_improvement": avg_improvement,
                "success_rate": f"{success_count}/{len(improvements)}",
                "improvements": improvements
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        results_dir = Path("experiments/results")
        results_file = results_dir / f"multi_benchmark_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_file.write_text(json.dumps(output, indent=2))
        print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    main()
