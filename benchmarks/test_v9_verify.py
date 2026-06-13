# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Test v9_verify prompt variant across 5 benchmarks.

Variant template:
- math: "{prompt}\n\nSolve and verify your answer is correct."
- logic: "{prompt}\n\nCheck your reasoning before answering."
- counting: "{prompt}\n\nDouble-check your count."
- truth: "{prompt}\n\nVerify against facts before answering."

Benchmarks: GSM8K, BBH-Logic, BBH-ObjectCounting, LogiQA, TruthfulQA
Model: gpt-oss-20b via Chutes.ai
Samples: 15 per benchmark
"""

import os
import sys
import json
import re
from datetime import datetime

os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

sys.path.insert(0, '/workspace')

from deepeval.benchmarks import GSM8K, LogiQA, TruthfulQA, BigBenchHard
from deepeval.benchmarks.gsm8k.template import GSM8KTemplate
from deepeval.benchmarks.logi_qa.template import LogiQATemplate
from deepeval.benchmarks.truthful_qa.template import TruthfulQATemplate
from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode
from deepeval.benchmarks.big_bench_hard.template import BigBenchHardTemplate
from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask
from deepeval.models import GPTModel


def create_chutes_model(api_key: str = None, model: str = "openai/gpt-oss-20b"):
    """Create DeepEval model using Chutes.ai endpoint"""
    api_key = api_key or os.environ.get("CHUTES_API_KEY")
    if not api_key:
        raise ValueError("CHUTES_API_KEY required")
    return GPTModel(
        model=model,
        api_key=api_key,
        base_url="https://llm.chutes.ai/v1"
    )


def call_model(model, prompt: str) -> str:
    """Call model and return response text."""
    result = model.generate(prompt)
    if isinstance(result, tuple):
        return result[0]
    return str(result)


# Answer extraction functions
def extract_gsm8k_answer(response: str) -> str:
    """Extract numeric answer from GSM8K response."""
    match = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', response)
    if match:
        return match.group(1).replace(',', '')
    match = re.search(r'(?:answer\s+is|=)\s*(-?\d[\d,]*\.?\d*)', response, re.I)
    if match:
        return match.group(1).replace(',', '')
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', response)
    if numbers:
        return numbers[-1].replace(',', '')
    return ""


def extract_logiqa_answer(response: str) -> str:
    """Extract A-D from LogiQA response."""
    match = re.search(r'\b([A-D])\s*[\.\):]', response)
    if match:
        return match.group(1).upper()
    match = re.search(r'answer\s+is\s*[:\s]*([A-D])', response, re.I)
    if match:
        return match.group(1).upper()
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1).upper()
    return ""


def extract_truthfulqa_answer(response: str) -> str:
    """Extract numeric answer from TruthfulQA response."""
    match = re.search(r'answer\s+is\s*[:\s]*(\d+)', response, re.I)
    if match:
        return match.group(1)
    match = re.search(r'\b(\d+)\s*[\.!]?\s*$', response.strip())
    if match:
        return match.group(1)
    match = re.search(r'(?:option|choice)\s*[:\s]*(\d+)', response, re.I)
    if match:
        return match.group(1)
    match = re.search(r'\b([1-9])\b', response)
    if match:
        return match.group(1)
    return ""


def extract_bbh_answer(response: str) -> str:
    """Extract answer from BBH response."""
    # For multiple choice answers (A), (B), etc.
    match = re.search(r'\(([A-E])\)', response)
    if match:
        return f"({match.group(1)})"
    # For numeric answers
    match = re.search(r'answer\s+is\s*[:\s]*(-?\d+)', response, re.I)
    if match:
        return match.group(1)
    match = re.search(r'=\s*(-?\d+)\s*$', response.strip())
    if match:
        return match.group(1)
    match = re.search(r'(-?\d+)\s*[\.!]?\s*$', response.strip())
    if match:
        return match.group(1)
    words = response.strip().split()
    if words:
        return words[-1].strip('.,!?')
    return ""


# V9 Verify Variant Templates
V9_TEMPLATES = {
    "math": "{prompt}\n\nSolve and verify your answer is correct.",
    "logic": "{prompt}\n\nCheck your reasoning before answering.",
    "counting": "{prompt}\n\nDouble-check your count.",
    "truth": "{prompt}\n\nVerify against facts before answering."
}


def run_gsm8k(model, n_samples: int = 15):
    """GSM8K benchmark - math variant."""
    print(f"Running GSM8K ({n_samples} samples)...")

    gsm_bench = GSM8K(n_problems=n_samples, n_shots=5, enable_cot=True)
    goldens = gsm_bench.load_benchmark_dataset()[:n_samples]

    baseline_correct = 0
    variant_correct = 0
    total = len(goldens)

    for i, golden in enumerate(goldens):
        prompt = GSM8KTemplate.generate_output(
            input=golden.input,
            train_set=gsm_bench.shots_dataset,
            n_shots=5,
            enable_cot=True,
        )
        expected = golden.expected_output

        # Baseline
        try:
            baseline_response = call_model(model, prompt)
            extracted = extract_gsm8k_answer(baseline_response)
            try:
                if abs(float(extracted) - float(expected)) < 0.01:
                    baseline_correct += 1
                    print(f"  {i+1}: Baseline CORRECT ({extracted} == {expected})")
                else:
                    print(f"  {i+1}: Baseline wrong ({extracted} != {expected})")
            except ValueError:
                print(f"  {i+1}: Baseline parse error ({extracted})")
        except Exception as e:
            print(f"  {i+1}: Baseline API error: {e}")

        # V9 Variant (math)
        try:
            variant_prompt = V9_TEMPLATES["math"].format(prompt=prompt)
            variant_response = call_model(model, variant_prompt)
            extracted_v = extract_gsm8k_answer(variant_response)
            try:
                if abs(float(extracted_v) - float(expected)) < 0.01:
                    variant_correct += 1
                    print(f"  {i+1}: Variant CORRECT ({extracted_v} == {expected})")
                else:
                    print(f"  {i+1}: Variant wrong ({extracted_v} != {expected})")
            except ValueError:
                print(f"  {i+1}: Variant parse error ({extracted_v})")
        except Exception as e:
            print(f"  {i+1}: Variant API error: {e}")

    baseline_pct = baseline_correct / total * 100
    variant_pct = variant_correct / total * 100

    result = {
        "benchmark": "GSM8K",
        "samples": total,
        "baseline_correct": baseline_correct,
        "variant_correct": variant_correct,
        "baseline_pct": round(baseline_pct, 1),
        "variant_pct": round(variant_pct, 1),
        "delta_pp": round(variant_pct - baseline_pct, 1)
    }
    print(f"GSM8K Result: baseline={baseline_pct:.1f}%, variant={variant_pct:.1f}%, delta={result['delta_pp']:+.1f}pp")
    return result


def run_bbh_logic(model, n_samples: int = 15):
    """BBH-Logic benchmark - logic variant."""
    print(f"Running BBH-Logic ({n_samples} samples)...")

    bbh_bench = BigBenchHard(
        tasks=[BigBenchHardTask.LOGICAL_DEDUCTION_THREE_OBJECTS],
        n_problems_per_task=n_samples,
        n_shots=3,
        enable_cot=True
    )
    task = BigBenchHardTask.LOGICAL_DEDUCTION_THREE_OBJECTS
    goldens = bbh_bench.load_benchmark_dataset(task)[:n_samples]

    baseline_correct = 0
    variant_correct = 0
    total = len(goldens)

    for i, golden in enumerate(goldens):
        prompt = BigBenchHardTemplate.generate_output(
            input=golden.input,
            task=task,
            n_shots=3,
            enable_cot=True,
        )
        expected = golden.expected_output

        # Baseline
        try:
            baseline_response = call_model(model, prompt)
            extracted = extract_bbh_answer(baseline_response)
            if extracted.lower() == expected.lower():
                baseline_correct += 1
                print(f"  {i+1}: Baseline CORRECT ({extracted})")
            else:
                print(f"  {i+1}: Baseline wrong ({extracted} != {expected})")
        except Exception as e:
            print(f"  {i+1}: Baseline error: {e}")

        # V9 Variant (logic)
        try:
            variant_prompt = V9_TEMPLATES["logic"].format(prompt=prompt)
            variant_response = call_model(model, variant_prompt)
            extracted_v = extract_bbh_answer(variant_response)
            if extracted_v.lower() == expected.lower():
                variant_correct += 1
                print(f"  {i+1}: Variant CORRECT ({extracted_v})")
            else:
                print(f"  {i+1}: Variant wrong ({extracted_v} != {expected})")
        except Exception as e:
            print(f"  {i+1}: Variant error: {e}")

    baseline_pct = baseline_correct / total * 100
    variant_pct = variant_correct / total * 100

    result = {
        "benchmark": "BBH-Logic",
        "samples": total,
        "baseline_correct": baseline_correct,
        "variant_correct": variant_correct,
        "baseline_pct": round(baseline_pct, 1),
        "variant_pct": round(variant_pct, 1),
        "delta_pp": round(variant_pct - baseline_pct, 1)
    }
    print(f"BBH-Logic Result: baseline={baseline_pct:.1f}%, variant={variant_pct:.1f}%, delta={result['delta_pp']:+.1f}pp")
    return result


def run_bbh_object_counting(model, n_samples: int = 15):
    """BBH-ObjectCounting benchmark - counting variant."""
    print(f"Running BBH-ObjectCounting ({n_samples} samples)...")

    bbh_bench = BigBenchHard(
        tasks=[BigBenchHardTask.OBJECT_COUNTING],
        n_problems_per_task=n_samples,
        n_shots=3,
        enable_cot=True
    )
    task = BigBenchHardTask.OBJECT_COUNTING
    goldens = bbh_bench.load_benchmark_dataset(task)[:n_samples]

    baseline_correct = 0
    variant_correct = 0
    total = len(goldens)

    for i, golden in enumerate(goldens):
        prompt = BigBenchHardTemplate.generate_output(
            input=golden.input,
            task=task,
            n_shots=3,
            enable_cot=True,
        )
        expected = golden.expected_output

        # Baseline
        try:
            baseline_response = call_model(model, prompt)
            extracted = extract_bbh_answer(baseline_response)
            if extracted == expected:
                baseline_correct += 1
                print(f"  {i+1}: Baseline CORRECT ({extracted})")
            else:
                print(f"  {i+1}: Baseline wrong ({extracted} != {expected})")
        except Exception as e:
            print(f"  {i+1}: Baseline error: {e}")

        # V9 Variant (counting)
        try:
            variant_prompt = V9_TEMPLATES["counting"].format(prompt=prompt)
            variant_response = call_model(model, variant_prompt)
            extracted_v = extract_bbh_answer(variant_response)
            if extracted_v == expected:
                variant_correct += 1
                print(f"  {i+1}: Variant CORRECT ({extracted_v})")
            else:
                print(f"  {i+1}: Variant wrong ({extracted_v} != {expected})")
        except Exception as e:
            print(f"  {i+1}: Variant error: {e}")

    baseline_pct = baseline_correct / total * 100
    variant_pct = variant_correct / total * 100

    result = {
        "benchmark": "BBH-ObjectCounting",
        "samples": total,
        "baseline_correct": baseline_correct,
        "variant_correct": variant_correct,
        "baseline_pct": round(baseline_pct, 1),
        "variant_pct": round(variant_pct, 1),
        "delta_pp": round(variant_pct - baseline_pct, 1)
    }
    print(f"BBH-ObjectCounting Result: baseline={baseline_pct:.1f}%, variant={variant_pct:.1f}%, delta={result['delta_pp']:+.1f}pp")
    return result


def run_logiqa(model, n_samples: int = 15):
    """LogiQA benchmark - logic variant."""
    print(f"Running LogiQA ({n_samples} samples)...")

    logiqa_bench = LogiQA(n_problems_per_task=n_samples, n_shots=3)
    task = logiqa_bench.tasks[0]
    goldens = logiqa_bench.load_benchmark_dataset(task)[:n_samples]

    baseline_correct = 0
    variant_correct = 0
    total = len(goldens)

    for i, golden in enumerate(goldens):
        prompt = LogiQATemplate.generate_output(
            input=golden.input,
            n_shots=3,
        )
        expected = golden.expected_output

        # Baseline
        try:
            baseline_response = call_model(model, prompt)
            extracted = extract_logiqa_answer(baseline_response)
            if extracted == expected:
                baseline_correct += 1
                print(f"  {i+1}: Baseline CORRECT ({extracted})")
            else:
                print(f"  {i+1}: Baseline wrong ({extracted} != {expected})")
        except Exception as e:
            print(f"  {i+1}: Baseline error: {e}")

        # V9 Variant (logic)
        try:
            variant_prompt = V9_TEMPLATES["logic"].format(prompt=prompt)
            variant_response = call_model(model, variant_prompt)
            extracted_v = extract_logiqa_answer(variant_response)
            if extracted_v == expected:
                variant_correct += 1
                print(f"  {i+1}: Variant CORRECT ({extracted_v})")
            else:
                print(f"  {i+1}: Variant wrong ({extracted_v} != {expected})")
        except Exception as e:
            print(f"  {i+1}: Variant error: {e}")

    baseline_pct = baseline_correct / total * 100
    variant_pct = variant_correct / total * 100

    result = {
        "benchmark": "LogiQA",
        "samples": total,
        "baseline_correct": baseline_correct,
        "variant_correct": variant_correct,
        "baseline_pct": round(baseline_pct, 1),
        "variant_pct": round(variant_pct, 1),
        "delta_pp": round(variant_pct - baseline_pct, 1)
    }
    print(f"LogiQA Result: baseline={baseline_pct:.1f}%, variant={variant_pct:.1f}%, delta={result['delta_pp']:+.1f}pp")
    return result


def run_truthfulqa(model, n_samples: int = 15):
    """TruthfulQA benchmark - truth variant."""
    print(f"Running TruthfulQA ({n_samples} samples)...")

    truthqa_bench = TruthfulQA(n_problems_per_task=n_samples, mode=TruthfulQAMode.MC1)
    task = truthqa_bench.tasks[0]
    goldens = truthqa_bench.load_benchmark_dataset(task, TruthfulQAMode.MC1)[:n_samples]

    baseline_correct = 0
    variant_correct = 0
    total = len(goldens)

    for i, golden in enumerate(goldens):
        prompt = TruthfulQATemplate.generate_output(
            input=golden.input,
            mode=TruthfulQAMode.MC1,
        )
        expected = golden.expected_output

        # Baseline
        try:
            baseline_response = call_model(model, prompt)
            extracted = extract_truthfulqa_answer(baseline_response)
            if extracted == expected:
                baseline_correct += 1
                print(f"  {i+1}: Baseline CORRECT ({extracted})")
            else:
                print(f"  {i+1}: Baseline wrong ({extracted} != {expected})")
        except Exception as e:
            print(f"  {i+1}: Baseline error: {e}")

        # V9 Variant (truth)
        try:
            variant_prompt = V9_TEMPLATES["truth"].format(prompt=prompt)
            variant_response = call_model(model, variant_prompt)
            extracted_v = extract_truthfulqa_answer(variant_response)
            if extracted_v == expected:
                variant_correct += 1
                print(f"  {i+1}: Variant CORRECT ({extracted_v})")
            else:
                print(f"  {i+1}: Variant wrong ({extracted_v} != {expected})")
        except Exception as e:
            print(f"  {i+1}: Variant error: {e}")

    baseline_pct = baseline_correct / total * 100
    variant_pct = variant_correct / total * 100

    result = {
        "benchmark": "TruthfulQA",
        "samples": total,
        "baseline_correct": baseline_correct,
        "variant_correct": variant_correct,
        "baseline_pct": round(baseline_pct, 1),
        "variant_pct": round(variant_pct, 1),
        "delta_pp": round(variant_pct - baseline_pct, 1)
    }
    print(f"TruthfulQA Result: baseline={baseline_pct:.1f}%, variant={variant_pct:.1f}%, delta={result['delta_pp']:+.1f}pp")
    return result


def main():
    print("V9 Verify Prompt Variant Test")
    print("=" * 60)
    print("Model: gpt-oss-20b via Chutes.ai")
    print("Samples: 15 per benchmark")
    print()
    print("Variant templates:")
    for k, v in V9_TEMPLATES.items():
        print(f"  {k}: {v}")
    print()

    # Create model
    model = create_chutes_model(model="openai/gpt-oss-20b")
    print("Model initialized.\n")

    n_samples = 15
    results = []
    output_path = "/workspace/benchmarks/v9_verify_results.json"

    # Run all benchmarks, saving progress after each
    benchmarks = [
        ("GSM8K", run_gsm8k),
        ("BBH-Logic", run_bbh_logic),
        ("BBH-ObjectCounting", run_bbh_object_counting),
        ("LogiQA", run_logiqa),
        ("TruthfulQA", run_truthfulqa),
    ]

    for name, func in benchmarks:
        print(f"\n{'='*60}")
        try:
            result = func(model, n_samples)
            results.append(result)

            # Save intermediate results
            output = {
                "variant": "v9_verify",
                "model": "openai/gpt-oss-20b",
                "provider": "chutes.ai",
                "samples_per_benchmark": n_samples,
                "timestamp": datetime.now().isoformat(),
                "templates": V9_TEMPLATES,
                "results": results,
                "status": "in_progress" if len(results) < len(benchmarks) else "complete"
            }
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Progress saved to {output_path}")

        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results.append({
                "benchmark": name,
                "error": str(e)
            })

    # Final summary
    print("\n" + "=" * 60)
    print("V9 VERIFY VARIANT RESULTS")
    print("=" * 60)
    print(f"{'Benchmark':<20} {'Baseline':<10} {'v9_verify':<10} {'Delta':<10}")
    print("-" * 60)

    valid_results = [r for r in results if 'error' not in r]
    for r in results:
        if 'error' in r:
            print(f"{r['benchmark']:<20} ERROR: {r['error']}")
        else:
            print(f"{r['benchmark']:<20} {r['baseline_pct']:>7.1f}%   {r['variant_pct']:>7.1f}%   {r['delta_pp']:>+7.1f}pp")

    print("-" * 60)

    if valid_results:
        # Calculate averages
        avg_baseline = sum(r['baseline_pct'] for r in valid_results) / len(valid_results)
        avg_variant = sum(r['variant_pct'] for r in valid_results) / len(valid_results)
        avg_delta = sum(r['delta_pp'] for r in valid_results) / len(valid_results)

        print(f"{'AVERAGE':<20} {avg_baseline:>7.1f}%   {avg_variant:>7.1f}%   {avg_delta:>+7.1f}pp")

        # Count improvements/regressions
        improvements = sum(1 for r in valid_results if r['delta_pp'] > 0)
        regressions = sum(1 for r in valid_results if r['delta_pp'] < 0)
        no_change = sum(1 for r in valid_results if r['delta_pp'] == 0)

        print(f"\nImprovements: {improvements}, Regressions: {regressions}, No Change: {no_change}")

        # Final output
        output = {
            "variant": "v9_verify",
            "model": "openai/gpt-oss-20b",
            "provider": "chutes.ai",
            "samples_per_benchmark": n_samples,
            "timestamp": datetime.now().isoformat(),
            "templates": V9_TEMPLATES,
            "results": results,
            "summary": {
                "avg_baseline_pct": round(avg_baseline, 1),
                "avg_variant_pct": round(avg_variant, 1),
                "avg_delta_pp": round(avg_delta, 1),
                "improvements": improvements,
                "regressions": regressions,
                "no_change": no_change
            },
            "status": "complete"
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {output_path}")

        # Print JSON output
        print("\n" + "=" * 60)
        print("JSON OUTPUT:")
        print("=" * 60)
        print(json.dumps(output, indent=2))

        return output
    else:
        print("No valid results to summarize.")
        return None


if __name__ == "__main__":
    main()
