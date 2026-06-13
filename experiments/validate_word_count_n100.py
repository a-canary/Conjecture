#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
H1 Validation: Word Count Optimization (n=100)

Pre-registered hypothesis: Ultra-concise claims (5w) outperform verbose claims (15w) by ≥10pp
Success criteria: p < 0.05, effect ≥10pp, 95% CI lower bound > 0

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
BENCHMARK = "bbh"  # logical_deduction_five_objects
TASK = "logical_deduction_five_objects"

# Example claims for each condition
CLAIMS_5W = [
    "Compare options systematically",
    "Eliminate impossible orderings",
    "Track confirmed positions",
    "Use process of elimination",
    "Verify final answer"
]

CLAIMS_15W = [
    "When solving logical deduction problems, systematically compare all given options",
    "Methodically eliminate impossible orderings by checking each statement's constraints carefully",
    "Carefully track and record all confirmed positions as you work through the problem",
    "Apply process of elimination strategy to narrow down the correct answer step by step",
    "Always verify your final answer against all given statements before submitting response"
]


def call_lm_studio(prompt, max_retries=3):
    """Call LM Studio via curl subprocess (Python HTTP libs fail with connection reset)"""
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
        # Check if it's just the letter at the start
        if response.startswith(letter) and (len(response) == 1 or response[1] in [' ', '.', ')', ':']):
            return letter

    # Last resort: find first occurrence of A-E
    for letter in ['A', 'B', 'C', 'D', 'E']:
        if letter in response:
            return letter

    return None


def build_prompt_with_claims(question, claims):
    """Build prompt with claims context"""
    claims_text = "\n".join(f"- {claim}" for claim in claims)
    return f"""Key principles:
{claims_text}

Question: {question}

Answer (A-E only):"""


def build_prompt_direct(question):
    """Direct prompt without claims"""
    return f"""Question: {question}

Answer (A-E only):"""


def run_validation(condition_name, claims=None):
    """Run validation for one condition (100 samples)"""
    print(f"\n{'='*60}")
    print(f"Running {condition_name} condition (n={N_SAMPLES})")
    print(f"{'='*60}\n")

    # Load BBH dataset
    dataset = load_dataset("lukaemon/bbh", "logical_deduction_five_objects")
    all_problems = dataset["test"]

    # For consistency, use fixed seed for same 100 questions
    random.seed(42)
    # If dataset has fewer than N_SAMPLES, cycle through
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
        answer = problem["target"]  # Already a letter (A-E)

        # Build prompt
        if claims:
            prompt = build_prompt_with_claims(question, claims)
        else:
            prompt = build_prompt_direct(question)

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
        "n_samples": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "results": results
    }


def calculate_statistics(data_5w, data_15w):
    """Calculate statistical significance"""
    n1 = data_5w["n_samples"]
    n2 = data_15w["n_samples"]
    p1 = data_5w["accuracy"]
    p2 = data_15w["accuracy"]

    # Two-proportion z-test
    diff = p1 - p2
    se = math.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    z = diff / se if se > 0 else 0
    p_value = 2 * stats.norm.cdf(-abs(z))

    # 95% confidence interval
    ci_margin = 1.96 * se
    ci_lower = diff - ci_margin
    ci_upper = diff + ci_margin

    # Effect size (Cohen's h)
    phi1 = 2 * math.asin(math.sqrt(p1))
    phi2 = 2 * math.asin(math.sqrt(p2))
    cohens_h = phi1 - phi2

    return {
        "5w_accuracy": p1,
        "15w_accuracy": p2,
        "difference_pp": diff * 100,
        "p_value": p_value,
        "ci_95_lower": ci_lower * 100,
        "ci_95_upper": ci_upper * 100,
        "cohens_h": cohens_h,
        "significant": p_value < 0.05,
        "effect_size_sufficient": abs(diff * 100) >= 10,
        "hypothesis_validated": (p_value < 0.05) and (diff * 100 >= 10) and (ci_lower > 0)
    }


def main():
    print("="*60)
    print("H1 VALIDATION: Word Count Optimization")
    print("Pre-registered hypothesis test (n=100 per condition)")
    print("="*60)

    # Run 5-word condition
    data_5w = run_validation("5-word claims", CLAIMS_5W)

    # Run 15-word condition
    data_15w = run_validation("15-word claims", CLAIMS_15W)

    # Calculate statistics
    stats_result = calculate_statistics(data_5w, data_15w)

    # Print results
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    print(f"5-word accuracy:  {stats_result['5w_accuracy']:.1%}")
    print(f"15-word accuracy: {stats_result['15w_accuracy']:.1%}")
    print(f"Difference:       {stats_result['difference_pp']:+.1f}pp")
    print(f"95% CI:           [{stats_result['ci_95_lower']:.1f}pp, {stats_result['ci_95_upper']:.1f}pp]")
    print(f"P-value:          {stats_result['p_value']:.4f}")
    print(f"Cohen's h:        {stats_result['cohens_h']:.3f}")
    print(f"\nSignificant (p<0.05):     {'✅ YES' if stats_result['significant'] else '❌ NO'}")
    print(f"Effect ≥10pp:              {'✅ YES' if stats_result['effect_size_sufficient'] else '❌ NO'}")
    print(f"CI lower bound > 0:        {'✅ YES' if stats_result['ci_95_lower'] > 0 else '❌ NO'}")
    print(f"\n{'='*60}")
    print(f"HYPOTHESIS VALIDATED:      {'✅ YES' if stats_result['hypothesis_validated'] else '❌ NO'}")
    print(f"{'='*60}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path("experiments/results") / f"validate_word_count_{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    final_data = {
        "metadata": {
            "hypothesis": "H1: 5-word claims > 15-word claims by ≥10pp",
            "pre_registration": ".director/PREREGISTRATION_2026-03-09.md",
            "date": timestamp,
            "model": MODEL_NAME,
            "benchmark": "BBH logical_deduction_five_objects",
            "n_samples_per_condition": N_SAMPLES
        },
        "5w_condition": data_5w,
        "15w_condition": data_15w,
        "statistics": stats_result
    }

    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Return success code
    return 0 if stats_result['hypothesis_validated'] else 1


if __name__ == "__main__":
    exit(main())
