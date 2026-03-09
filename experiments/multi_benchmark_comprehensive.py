#!/usr/bin/env python3
"""
Multi-Benchmark Comprehensive Validation

Test the refined Goldilocks Principle (5-word claims, 1-3 count) across:
- Code generation (HumanEval)
- Commonsense reasoning (HellaSwag, PIQA)
- Coreference resolution (WinoGrande)
- Social reasoning (SIQA)
- Reading comprehension (RACE)

Validates: Does 5-word claim optimization generalize beyond BBH/MMLU/GSM8K?
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime, timezone

from datasets import load_dataset


LFM_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL = "liquid/lfm2.5-1.2b"
N_PER_BENCHMARK = 20  # Quick validation across many benchmarks


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
    """Extract multiple choice answer."""
    if not text:
        return None
    for opt in options:
        if opt in text:
            return opt
    return None


# Ultra-concise 5-word claims (optimal from word count validation)
CLAIMS_BY_TASK = {
    "commonsense": [
        "Use real-world knowledge",  # 3 words
        "Apply common sense reasoning"  # 4 words
    ],
    "physical": [
        "Consider physical properties matter",  # 4 words
        "Think about real interactions"  # 4 words
    ],
    "social": [
        "Consider social norms behavior",  # 4 words
        "Think about human emotions"  # 4 words
    ],
    "coreference": [
        "Track pronoun references carefully",  # 4 words
        "Match pronouns to nouns"  # 4 words
    ],
    "reading": [
        "Read passage answer accurately",  # 4 words
        "Find evidence in text"  # 4 words
    ]
}


def test_hellaswag(with_claims=True):
    """HellaSwag: Commonsense reasoning."""
    print("\n" + "="*80)
    print("HELLASWAG: Commonsense Reasoning")
    print("="*80)

    ds = load_dataset("hellaswag", split="validation")
    correct = 0

    for i in range(N_PER_BENCHMARK):
        item = ds[i]
        context = item["ctx"]
        endings = item["endings"]
        options = ["(A)", "(B)", "(C)", "(D)"]

        query = f"{context}\n" + "\n".join([f"{opt} {end}" for opt, end in zip(options, endings)])

        if with_claims:
            claims = CLAIMS_BY_TASK["commonsense"]
            prompt = f"{claims[0]}\n{claims[1]}\n\n{query}"
        else:
            prompt = query

        response = call_lfm(prompt, max_tokens=100)
        answer = extract_answer(response, options)

        if answer == options[int(item["label"])]:
            correct += 1

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{N_PER_BENCHMARK}")

    accuracy = 100 * correct / N_PER_BENCHMARK
    print(f"  Result: {correct}/{N_PER_BENCHMARK} = {accuracy:.0f}%")
    return {"benchmark": "HellaSwag", "with_claims": with_claims, "correct": correct, "accuracy": accuracy}


def test_piqa(with_claims=True):
    """PIQA: Physical commonsense."""
    print("\n" + "="*80)
    print("PIQA: Physical Commonsense")
    print("="*80)

    ds = load_dataset("piqa", split="validation")
    correct = 0

    for i in range(N_PER_BENCHMARK):
        item = ds[i]
        goal = item["goal"]
        sol1 = item["sol1"]
        sol2 = item["sol2"]

        query = f"{goal}\n(A) {sol1}\n(B) {sol2}"

        if with_claims:
            claims = CLAIMS_BY_TASK["physical"]
            prompt = f"{claims[0]}\n{claims[1]}\n\n{query}"
        else:
            prompt = query

        response = call_lfm(prompt, max_tokens=100)
        answer = extract_answer(response, ["(A)", "(B)"])

        expected = "(A)" if item["label"] == 0 else "(B)"
        if answer == expected:
            correct += 1

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{N_PER_BENCHMARK}")

    accuracy = 100 * correct / N_PER_BENCHMARK
    print(f"  Result: {correct}/{N_PER_BENCHMARK} = {accuracy:.0f}%")
    return {"benchmark": "PIQA", "with_claims": with_claims, "correct": correct, "accuracy": accuracy}


def test_siqa(with_claims=True):
    """SIQA: Social reasoning."""
    print("\n" + "="*80)
    print("SIQA: Social Reasoning")
    print("="*80)

    ds = load_dataset("social_i_qa", split="validation")
    correct = 0

    for i in range(N_PER_BENCHMARK):
        item = ds[i]
        context = item["context"]
        question = item["question"]
        options = [item["answerA"], item["answerB"], item["answerC"]]

        query = f"{context}\n{question}\n(A) {options[0]}\n(B) {options[1]}\n(C) {options[2]}"

        if with_claims:
            claims = CLAIMS_BY_TASK["social"]
            prompt = f"{claims[0]}\n{claims[1]}\n\n{query}"
        else:
            prompt = query

        response = call_lfm(prompt, max_tokens=100)
        answer = extract_answer(response, ["(A)", "(B)", "(C)"])

        expected = ["(A)", "(B)", "(C)"][int(item["label"]) - 1]
        if answer == expected:
            correct += 1

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{N_PER_BENCHMARK}")

    accuracy = 100 * correct / N_PER_BENCHMARK
    print(f"  Result: {correct}/{N_PER_BENCHMARK} = {accuracy:.0f}%")
    return {"benchmark": "SIQA", "with_claims": with_claims, "correct": correct, "accuracy": accuracy}


def test_winogrande(with_claims=True):
    """WinoGrande: Coreference resolution."""
    print("\n" + "="*80)
    print("WINOGRANDE: Coreference Resolution")
    print("="*80)

    ds = load_dataset("winogrande", "winogrande_xl", split="validation")
    correct = 0

    for i in range(N_PER_BENCHMARK):
        item = ds[i]
        sentence = item["sentence"]
        option1 = item["option1"]
        option2 = item["option2"]

        query = f"{sentence}\n(A) {option1}\n(B) {option2}"

        if with_claims:
            claims = CLAIMS_BY_TASK["coreference"]
            prompt = f"{claims[0]}\n{claims[1]}\n\n{query}"
        else:
            prompt = query

        response = call_lfm(prompt, max_tokens=100)
        answer = extract_answer(response, ["(A)", "(B)"])

        expected = ["(A)", "(B)"][int(item["answer"]) - 1]
        if answer == expected:
            correct += 1

        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{N_PER_BENCHMARK}")

    accuracy = 100 * correct / N_PER_BENCHMARK
    print(f"  Result: {correct}/{N_PER_BENCHMARK} = {accuracy:.0f}%")
    return {"benchmark": "WinoGrande", "with_claims": with_claims, "correct": correct, "accuracy": accuracy}


def main():
    print("="*80)
    print("MULTI-BENCHMARK COMPREHENSIVE VALIDATION")
    print("="*80)
    print(f"Model: {MODEL}")
    print(f"Optimization: 5-word claims (2 claims, ultra-concise)")
    print(f"Benchmarks: 4 new tasks")
    print(f"Sample size: {N_PER_BENCHMARK} per benchmark")
    print("="*80)

    results = []

    # Test each benchmark with and without claims
    for benchmark_fn in [test_hellaswag, test_piqa, test_siqa, test_winogrande]:
        # Baseline (no claims)
        baseline = benchmark_fn(with_claims=False)
        results.append(baseline)

        # Optimized (with 5-word claims)
        optimized = benchmark_fn(with_claims=True)
        results.append(optimized)

    # Summary
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS")
    print("="*80)
    print(f"{'Benchmark':<15} {'Method':<15} {'Accuracy':<10} {'Improvement'}")
    print("-"*80)

    for i in range(0, len(results), 2):
        baseline = results[i]
        optimized = results[i+1]
        delta = optimized['accuracy'] - baseline['accuracy']

        print(f"{baseline['benchmark']:<15} Baseline        {baseline['accuracy']:>6.0f}%")
        print(f"{optimized['benchmark']:<15} 5-word claims   {optimized['accuracy']:>6.0f}%      {delta:>+6.0f}pp")
        print()

    # Overall statistics
    improvements = []
    for i in range(0, len(results), 2):
        baseline = results[i]
        optimized = results[i+1]
        delta = optimized['accuracy'] - baseline['accuracy']
        improvements.append(delta)

    avg_improvement = sum(improvements) / len(improvements)
    success_count = sum(1 for d in improvements if d > 0)

    print("="*80)
    print(f"OVERALL STATISTICS")
    print(f"  Average improvement: {avg_improvement:+.1f}pp")
    print(f"  Success rate: {success_count}/{len(improvements)} benchmarks improved")
    print(f"  Generalization: {'✅ STRONG' if success_count >= 3 else '⚠️ WEAK'}")
    print("="*80)

    # Save
    output = {
        "experiment": "Multi-benchmark comprehensive validation",
        "model": MODEL,
        "optimization": "5-word claims (2 claims, ultra-concise)",
        "n_per_benchmark": N_PER_BENCHMARK,
        "benchmarks_tested": 4,
        "results": results,
        "statistics": {
            "avg_improvement": avg_improvement,
            "success_rate": f"{success_count}/{len(improvements)}",
            "improvements": improvements
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    results_dir = Path("experiments/results")
    results_file = results_dir / f"multi_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    main()
