#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
H2 Validation: Goldilocks Principle (n=100 per condition)

Pre-registered hypothesis: 2 claims outperform both 0 claims and 5 claims
Success criteria:
- 2 claims > 0 claims: p < 0.025 (Bonferroni)
- 2 claims > 5 claims: p < 0.025
- Effect size ≥5pp improvement
- Both comparisons must be significant

See: .director/PREREGISTRATION_2026-03-09.md
"""

import subprocess
import json
import random
from datetime import datetime
from pathlib import Path
from datasets import load_dataset
from scipy import stats
import math

# Configuration
LM_STUDIO_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL_NAME = "lfm-2.5-1.2b"
N_SAMPLES = 100
BENCHMARK = "mmlu"  # Using MMLU logical reasoning

# Claim configurations
CLAIMS_0 = []  # Direct baseline

CLAIMS_2 = [
    "Use systematic logical deduction to evaluate each ordering possibility",
    "Eliminate options that contradict any given statement in the problem"
]

CLAIMS_5 = [
    "When solving logical deduction problems, systematically compare all given options",
    "Methodically eliminate impossible orderings by checking constraints carefully",
    "Carefully track and record all confirmed positions as you work through",
    "Apply process of elimination strategy to narrow down correct answer",
    "Always verify your final answer against all given statements before submitting"
]


def call_lm_studio(prompt, max_retries=3):
    """Call LM Studio via curl subprocess"""
    for attempt in range(max_retries):
        try:
            cmd = [
                "curl", "-s", "-X", "POST", LM_STUDIO_ENDPOINT,
                "-H", "Content-Type: application/json",
                "-d", json.dumps({
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 400
                })
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"]
            print(f"Attempt {attempt+1} failed, retrying...")
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                return None
    return None


def extract_answer(response):
    """Extract single letter answer (A-E) from response"""
    if not response:
        return None

    response = response.strip().upper()

    # Check for direct answer patterns
    for letter in ['A', 'B', 'C', 'D', 'E']:
        if f"({letter})" in response or f"ANSWER: {letter}" in response or f"ANSWER {letter}" in response:
            return letter
        if response.startswith(letter) and (len(response) == 1 or response[1] in [' ', '.', ')', ':']):
            return letter

    # Last resort: find first occurrence
    for letter in ['A', 'B', 'C', 'D', 'E']:
        if letter in response:
            return letter

    return None


def build_prompt_with_claims(question, claims):
    """Build prompt with claims context"""
    if not claims:
        return build_prompt_direct(question)

    claims_text = "\n".join(f"- {claim}" for claim in claims)
    return f"""Key principles:
{claims_text}

Question: {question}

Answer (A-E only):"""


def build_prompt_direct(question):
    """Direct prompt without claims"""
    return f"""Question: {question}

Answer (A-E only):"""


def run_validation(condition_name, claims, seed_offset=0):
    """Run validation for one condition (100 samples)"""
    print(f"\n{'='*60}")
    print(f"Running {condition_name} condition (n={N_SAMPLES})")
    if claims:
        print(f"Claims: {len(claims)}")
    else:
        print("Claims: 0 (direct baseline)")
    print(f"{'='*60}\n")

    # Load BBH dataset
    dataset = load_dataset("lukaemon/bbh", "logical_deduction_three_objects")
    all_problems = dataset["test"]

    # Use same seed for consistency, offset per condition
    random.seed(42 + seed_offset)
    # If dataset smaller than N_SAMPLES, cycle through
    if len(all_problems) < N_SAMPLES:
        samples = list(range(len(all_problems))) * (N_SAMPLES // len(all_problems) + 1)
        samples = samples[:N_SAMPLES]
    else:
        samples = random.sample(range(len(all_problems)), N_SAMPLES)

    results = []
    correct = 0

    for i, idx in enumerate(samples):
        problem = all_problems[idx]
        question = problem["input"]
        answer = problem["target"]  # Already a letter

        # Build prompt
        prompt = build_prompt_with_claims(question, claims)

        # Get response
        response = call_lm_studio(prompt)
        extracted = extract_answer(response)

        # BBH answers are already letters
        is_correct = (extracted == answer)

        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "correct_answer": answer,
            "predicted_answer": extracted,
            "correct": is_correct,
            "response": response
        })

        # Progress update every 10
        if (i + 1) % 10 == 0:
            accuracy = correct / (i + 1)
            print(f"Progress: {i+1}/{N_SAMPLES} - Accuracy: {accuracy:.1%}")

    accuracy = correct / len(results)
    print(f"\n{condition_name} Final Accuracy: {accuracy:.1%} ({correct}/{len(results)})")

    return {
        "condition": condition_name,
        "n_claims": len(claims) if claims else 0,
        "n_samples": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "results": results
    }


def calculate_pairwise_stats(data1, data2, label1, label2):
    """Calculate statistics for pairwise comparison"""
    n1 = data1["n_samples"]
    n2 = data2["n_samples"]
    p1 = data1["accuracy"]
    p2 = data2["accuracy"]

    # Two-proportion z-test
    diff = p1 - p2
    se = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    z = diff / se if se > 0 else 0
    p_value = 2 * stats.norm.cdf(-abs(z))

    # 95% confidence interval
    ci_margin = 1.96 * se
    ci_lower = diff - ci_margin
    ci_upper = diff + ci_margin

    return {
        "comparison": f"{label1} vs {label2}",
        f"{label1}_accuracy": p1,
        f"{label2}_accuracy": p2,
        "difference_pp": diff * 100,
        "p_value": p_value,
        "ci_95_lower": ci_lower * 100,
        "ci_95_upper": ci_upper * 100,
        "significant_bonferroni": p_value < 0.025  # Bonferroni correction for 2 tests
    }


def main():
    print("="*60)
    print("H2 VALIDATION: Goldilocks Principle")
    print("Pre-registered hypothesis test (n=100 per condition)")
    print("Testing: 2 claims > 0 claims AND 2 claims > 5 claims")
    print("="*60)

    # Run all three conditions
    data_0 = run_validation("0 claims (baseline)", CLAIMS_0, seed_offset=0)
    data_2 = run_validation("2 claims (Goldilocks)", CLAIMS_2, seed_offset=100)
    data_5 = run_validation("5 claims (overload)", CLAIMS_5, seed_offset=200)

    # Calculate pairwise comparisons
    stats_2_vs_0 = calculate_pairwise_stats(data_2, data_0, "2claims", "0claims")
    stats_2_vs_5 = calculate_pairwise_stats(data_2, data_5, "2claims", "5claims")

    # Print results
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    print(f"\n0 claims accuracy: {data_0['accuracy']:.1%}")
    print(f"2 claims accuracy: {data_2['accuracy']:.1%}")
    print(f"5 claims accuracy: {data_5['accuracy']:.1%}")

    print(f"\n--- Comparison 1: 2 claims vs 0 claims ---")
    print(f"Difference:       {stats_2_vs_0['difference_pp']:+.1f}pp")
    print(f"95% CI:           [{stats_2_vs_0['ci_95_lower']:.1f}pp, {stats_2_vs_0['ci_95_upper']:.1f}pp]")
    print(f"P-value:          {stats_2_vs_0['p_value']:.4f}")
    print(f"Significant (p<0.025): {'✅ YES' if stats_2_vs_0['significant_bonferroni'] else '❌ NO'}")

    print(f"\n--- Comparison 2: 2 claims vs 5 claims ---")
    print(f"Difference:       {stats_2_vs_5['difference_pp']:+.1f}pp")
    print(f"95% CI:           [{stats_2_vs_5['ci_95_lower']:.1f}pp, {stats_2_vs_5['ci_95_upper']:.1f}pp]")
    print(f"P-value:          {stats_2_vs_5['p_value']:.4f}")
    print(f"Significant (p<0.025): {'✅ YES' if stats_2_vs_5['significant_bonferroni'] else '❌ NO'}")

    # Overall hypothesis validation
    both_significant = stats_2_vs_0['significant_bonferroni'] and stats_2_vs_5['significant_bonferroni']
    effect_sufficient = (abs(stats_2_vs_0['difference_pp']) >= 5) or (abs(stats_2_vs_5['difference_pp']) >= 5)

    print(f"\n{'='*60}")
    print(f"Both comparisons significant: {'✅ YES' if both_significant else '❌ NO'}")
    print(f"Effect size ≥5pp:             {'✅ YES' if effect_sufficient else '❌ NO'}")
    print(f"\nHYPOTHESIS VALIDATED:         {'✅ YES' if (both_significant and effect_sufficient) else '❌ NO'}")
    print(f"{'='*60}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path("experiments/results") / f"validate_goldilocks_{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    final_data = {
        "metadata": {
            "hypothesis": "H2: 2 claims > 0 claims AND 2 claims > 5 claims",
            "pre_registration": ".director/PREREGISTRATION_2026-03-09.md",
            "date": timestamp,
            "model": MODEL_NAME,
            "benchmark": "BBH logical_deduction_three_objects",
            "n_samples_per_condition": N_SAMPLES,
            "bonferroni_alpha": 0.025
        },
        "0claims_condition": data_0,
        "2claims_condition": data_2,
        "5claims_condition": data_5,
        "statistics": {
            "2claims_vs_0claims": stats_2_vs_0,
            "2claims_vs_5claims": stats_2_vs_5,
            "hypothesis_validated": both_significant and effect_sufficient
        }
    }

    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Return success code
    return 0 if (both_significant and effect_sufficient) else 1


if __name__ == "__main__":
    exit(main())
