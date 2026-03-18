#!/usr/bin/env python3
"""
H3 Validation: Task-Type Dependency (n=100)

Pre-registered hypothesis: Claims improve BBH reasoning tasks by ≥5pp
Success criteria: p < 0.05, effect ≥5pp, 95% CI lower bound > 0

Note: Commonsense regression (HellaSwag -40pp, p=0.004) already validated in exploration

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
BENCHMARK = "mmlu"  # Using MMLU logical reasoning as proxy for BBH

# Claims for reasoning tasks (concise but not ultra-concise, ~10 words each)
CLAIMS_REASONING = [
    "Use systematic logical deduction to evaluate each ordering possibility carefully",
    "Eliminate options that contradict any given statement in the problem"
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
    claims_text = "\n".join(f"- {claim}" for claim in claims)
    return f"""Key principles:
{claims_text}

Question: {question}

Answer (A-E only):"""


def build_prompt_direct(question):
    """Direct prompt without claims"""
    return f"""Question: {question}

Answer (A-E only):"""


def run_validation(condition_name, use_claims, seed_offset=0):
    """Run validation for one condition (100 samples)"""
    print(f"\n{'='*60}")
    print(f"Running {condition_name} condition (n={N_SAMPLES})")
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
        if use_claims:
            prompt = build_prompt_with_claims(question, CLAIMS_REASONING)
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
        "use_claims": use_claims,
        "n_samples": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "results": results
    }


def calculate_statistics(data_direct, data_claims):
    """Calculate statistical significance"""
    n1 = data_direct["n_samples"]
    n2 = data_claims["n_samples"]
    p1 = data_direct["accuracy"]
    p2 = data_claims["accuracy"]

    # Two-proportion z-test
    diff = p2 - p1  # claims - direct
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
    cohens_h = phi2 - phi1

    return {
        "direct_accuracy": p1,
        "claims_accuracy": p2,
        "difference_pp": diff * 100,
        "p_value": p_value,
        "ci_95_lower": ci_lower * 100,
        "ci_95_upper": ci_upper * 100,
        "cohens_h": cohens_h,
        "significant": p_value < 0.05,
        "effect_size_sufficient": abs(diff * 100) >= 5,
        "hypothesis_validated": (p_value < 0.05) and (diff * 100 >= 5) and (ci_lower > 0)
    }


def main():
    print("="*60)
    print("H3 VALIDATION: Task-Type Dependency (BBH Reasoning)")
    print("Pre-registered hypothesis test (n=100 per condition)")
    print("Testing: Claims improve reasoning tasks by ≥5pp")
    print("="*60)

    # Run direct baseline
    data_direct = run_validation("Direct (no claims)", use_claims=False, seed_offset=0)

    # Run with claims
    data_claims = run_validation("With claims", use_claims=True, seed_offset=100)

    # Calculate statistics
    stats_result = calculate_statistics(data_direct, data_claims)

    # Print results
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    print(f"Direct accuracy:  {stats_result['direct_accuracy']:.1%}")
    print(f"Claims accuracy:  {stats_result['claims_accuracy']:.1%}")
    print(f"Difference:       {stats_result['difference_pp']:+.1f}pp")
    print(f"95% CI:           [{stats_result['ci_95_lower']:.1f}pp, {stats_result['ci_95_upper']:.1f}pp]")
    print(f"P-value:          {stats_result['p_value']:.4f}")
    print(f"Cohen's h:        {stats_result['cohens_h']:.3f}")
    print(f"\nSignificant (p<0.05):     {'✅ YES' if stats_result['significant'] else '❌ NO'}")
    print(f"Effect ≥5pp:              {'✅ YES' if stats_result['effect_size_sufficient'] else '❌ NO'}")
    print(f"CI lower bound > 0:        {'✅ YES' if stats_result['ci_95_lower'] > 0 else '❌ NO'}")
    print(f"\n{'='*60}")
    print(f"HYPOTHESIS VALIDATED:      {'✅ YES' if stats_result['hypothesis_validated'] else '✅ NO'}")
    print(f"{'='*60}")

    print("\n" + "="*60)
    print("TASK-TYPE SUMMARY")
    print("="*60)
    print("Reasoning tasks (BBH):      [Result above]")
    print("Commonsense tasks (HellaSwag): -40pp (p=0.004) ✅ VALIDATED in exploration")
    print("\nConclusion: Task-type routing REQUIRED if both validate")
    print("="*60)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path("experiments/results") / f"validate_task_type_{timestamp}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    final_data = {
        "metadata": {
            "hypothesis": "H3: Claims improve BBH reasoning by ≥5pp",
            "pre_registration": ".director/PREREGISTRATION_2026-03-09.md",
            "date": timestamp,
            "model": MODEL_NAME,
            "benchmark": "BBH logical_deduction_three_objects",
            "n_samples_per_condition": N_SAMPLES,
            "note": "Commonsense regression already validated in exploration (p=0.004)"
        },
        "direct_condition": data_direct,
        "claims_condition": data_claims,
        "statistics": stats_result
    }

    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Return success code
    return 0 if stats_result['hypothesis_validated'] else 1


if __name__ == "__main__":
    exit(main())
