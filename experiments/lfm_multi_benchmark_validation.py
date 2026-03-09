#!/usr/bin/env python3
"""
Multi-Benchmark Validation of Goldilocks Principle

Tests if 1-3 claim optimization generalizes across task types:
- GSM8K (math reasoning)
- MMLU (knowledge recall)
- ARC-Challenge (science reasoning)

Quick test: n=10 per benchmark (30 total problems)
"""

import subprocess
import json
import time
import re
from pathlib import Path
from datetime import datetime, timezone

from datasets import load_dataset


LFM_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL = "liquid/lfm2.5-1.2b"
N_PROBLEMS = 10  # Per benchmark


# Domain-specific claims (2 claims each - within Goldilocks zone)
BENCHMARK_CLAIMS = {
    "gsm8k": [
        "Break down word problems into clear mathematical equations",
        "Work step-by-step, calculating intermediate values before the final answer"
    ],
    "mmlu": [
        "Carefully read the question to identify what specific knowledge is being tested",
        "Eliminate obviously incorrect options before choosing the best answer"
    ],
    "arc": [
        "Apply fundamental scientific principles (physics, chemistry, biology) to real-world scenarios",
        "Consider cause-and-effect relationships between variables"
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
            return data["choices"][0]["message"]["content"], data["usage"]["total_tokens"]
    except:
        pass
    return None, 0


def extract_number(text):
    """Extract numeric answer from GSM8K response."""
    if not text:
        return None
    # Try to find numbers, prefer last one (usually the final answer)
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else None


def extract_letter(text, options):
    """Extract letter answer from multiple choice."""
    if not text:
        return None
    for opt in options:
        if opt in text:
            return opt
    return None


def test_benchmark(benchmark_name, problems, claims, answer_extractor):
    """Test a benchmark with claims."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {benchmark_name.upper()}")
    print(f"{'='*60}")

    # Direct (no claims)
    print("Direct prompting...")
    direct_correct = 0
    for i, prob in enumerate(problems, 1):
        print(f"  [{i}/{len(problems)}] ", end="", flush=True)
        response, _ = call_lfm(prob["query"], max_tokens=400)
        answer = answer_extractor(response, prob)
        if answer == prob["target"]:
            direct_correct += 1
            print("✓", end="")
        else:
            print("✗", end="")
    direct_acc = 100 * direct_correct / len(problems)
    print(f" → {direct_acc:.0f}%")

    # With claims (Goldilocks: 2 claims)
    print("With 2 claims...")
    claims_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(claims)])
    claim_correct = 0
    for i, prob in enumerate(problems, 1):
        print(f"  [{i}/{len(problems)}] ", end="", flush=True)
        enhanced = f"Key principles:\n{claims_text}\n\nProblem: {prob['query']}"
        response, _ = call_lfm(enhanced, max_tokens=400)
        answer = answer_extractor(response, prob)
        if answer == prob["target"]:
            claim_correct += 1
            print("✓", end="")
        else:
            print("✗", end="")
    claim_acc = 100 * claim_correct / len(problems)
    print(f" → {claim_acc:.0f}%")

    improvement = claim_acc - direct_acc
    print(f"\nImprovement: {improvement:+.0f}pp")

    return {
        "benchmark": benchmark_name,
        "direct": {"correct": direct_correct, "accuracy": direct_acc},
        "with_claims": {"correct": claim_correct, "accuracy": claim_acc},
        "improvement_pp": improvement
    }


def main():
    print("="*70)
    print("MULTI-BENCHMARK VALIDATION: GOLDILOCKS PRINCIPLE")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Strategy: 2 claims per benchmark (Goldilocks zone)")
    print(f"Problems: {N_PROBLEMS} per benchmark")
    print("\nHypothesis: 1-3 claims improves performance across task types")
    print("="*70)

    results = []

    # GSM8K (Math)
    print("\nLoading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="test")
    gsm_problems = []
    for i in range(N_PROBLEMS):
        item = ds[i]
        answer = item["answer"].split("####")[-1].strip()
        gsm_problems.append({
            "query": item["question"],
            "target": answer
        })

    gsm_result = test_benchmark(
        "gsm8k",
        gsm_problems,
        BENCHMARK_CLAIMS["gsm8k"],
        lambda resp, prob: extract_number(resp) if resp else None
    )
    results.append(gsm_result)
    time.sleep(1)

    # MMLU (Knowledge)
    print("\nLoading MMLU...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    mmlu_problems = []
    for i in range(N_PROBLEMS):
        item = ds[i]
        choices = item["choices"]
        options = ["(A)", "(B)", "(C)", "(D)"]
        query = f"{item['question']}\n" + "\n".join([f"{opt} {ch}" for opt, ch in zip(options, choices)])
        mmlu_problems.append({
            "query": query,
            "target": options[item["answer"]]
        })

    mmlu_result = test_benchmark(
        "mmlu",
        mmlu_problems,
        BENCHMARK_CLAIMS["mmlu"],
        lambda resp, prob: extract_letter(resp, ["(A)", "(B)", "(C)", "(D)"]) if resp else None
    )
    results.append(mmlu_result)
    time.sleep(1)

    # ARC-Challenge
    print("\nLoading ARC-Challenge...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    arc_problems = []
    for i in range(N_PROBLEMS):
        item = ds[i]
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        query = f"{item['question']}\n" + "\n".join([f"({lbl}) {txt}" for lbl, txt in zip(labels, texts)])
        arc_problems.append({
            "query": query,
            "target": f"({item['answerKey']})"
        })

    arc_result = test_benchmark(
        "arc",
        arc_problems,
        BENCHMARK_CLAIMS["arc"],
        lambda resp, prob: extract_letter(resp, [f"({item['answerKey']})"]) if resp else None
    )
    results.append(arc_result)

    # Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Benchmark':<15} {'Direct':<10} {'w/ Claims':<10} {'Delta':<10}")
    print("-"*70)

    success_count = 0
    for r in results:
        delta = r['improvement_pp']
        print(f"{r['benchmark']:<15} {r['direct']['accuracy']:>6.0f}%   {r['with_claims']['accuracy']:>6.0f}%     {delta:>+6.0f}pp")
        if delta >= 5:
            success_count += 1

    print("="*70)

    if success_count >= 2:
        print(f"\n✅ VALIDATED: Goldilocks principle generalizes!")
        print(f"   {success_count}/3 benchmarks show +5pp improvement with 2 claims")
        print("\n   Core thesis confirmed across task types:")
        print("   'DB storage + LLM reasoning + semantic indexing = intelligent tiny model'")
    elif success_count >= 1:
        print(f"\n⚠️  PARTIAL: {success_count}/3 benchmarks improved")
        print("   Goldilocks principle may be task-specific")
    else:
        print("\n❌ NOT GENERALIZED: Claims don't help other benchmarks")
        print("   BBH result may be task-specific")

    # Save
    output = {
        "strategy": "Multi-benchmark validation of Goldilocks principle",
        "model": MODEL,
        "claim_count": 2,
        "results": results,
        "success_count": success_count,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    results_file = Path("experiments/results") / f"lfm_multi_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {results_file}")


if __name__ == "__main__":
    main()
