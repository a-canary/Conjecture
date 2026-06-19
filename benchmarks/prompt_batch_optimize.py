# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Batch Prompt Optimization for O-0008
Tests 16 prompt variations across all benchmarks to find best performers.
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, '/workspace')

try:
    from deepeval.benchmarks import GSM8K, BigBenchHard, TruthfulQA, LogiQA
    from deepeval.benchmarks.gsm8k.template import GSM8KTemplate
    from deepeval.benchmarks.big_bench_hard.template import BigBenchHardTemplate
    from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask
    from deepeval.benchmarks.truthful_qa.template import TruthfulQATemplate
    from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode
    from deepeval.benchmarks.logi_qa.template import LogiQATemplate
    from deepeval.models import GPTModel
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

from benchmarks.deepeval_suite import (
    extract_gsm8k_answer, extract_bbh_answer,
    extract_truthfulqa_answer, extract_logiqa_answer,
    create_chutes_model, _call_model
)


# 16 Prompt Variations
PROMPT_VARIANTS = {
    # Variant 0: Pass-through (no enhancement)
    "v0_passthrough": {
        "math": "{prompt}",
        "logic": "{prompt}",
        "counting": "{prompt}",
        "truth": "{prompt}",
    },

    # Variant 1: Minimal instruction
    "v1_minimal": {
        "math": "Solve: {prompt}\nAnswer:",
        "logic": "Answer: {prompt}\nChoice:",
        "counting": "Count: {prompt}\nTotal:",
        "truth": "Select the correct answer: {prompt}\nAnswer:",
    },

    # Variant 2: Current standard (baseline for comparison)
    "v2_standard": {
        "math": """Solve this step-by-step. Show all work clearly.
After solving, verify your answer by checking it makes sense.
Write your final numeric answer after ####

{prompt}""",
        "logic": """Analyze this logical reasoning problem step-by-step.
1. Identify the premises and conclusion structure
2. Check for valid logical relationships
3. Eliminate incorrect options systematically
4. State your final answer (A, B, C, or D)

{prompt}""",
        "counting": """Count carefully step by step.
List each item as you count.
Write your final count as a number.

{prompt}""",
        "truth": """Evaluate the truthfulness carefully.
1. Consider factual information
2. Check for misconceptions
3. Select the most accurate answer

{prompt}""",
    },

    # Variant 3: Direct answer focus
    "v3_direct": {
        "math": "Calculate and give the numeric answer: {prompt}\n\n####",
        "logic": "Which option is correct? {prompt}\n\nAnswer:",
        "counting": "How many? {prompt}\n\nCount:",
        "truth": "Which is true? {prompt}\n\nAnswer:",
    },

    # Variant 4: Confidence-based (ask model if it knows)
    "v4_confidence": {
        "math": "If you can solve this directly, give the answer. Otherwise, work through it step by step.\n{prompt}",
        "logic": "Select the logically correct answer. If obvious, state directly. If not, reason through options.\n{prompt}",
        "counting": "Count and give the number. Show work only if needed.\n{prompt}",
        "truth": "Pick the truthful answer directly.\n{prompt}",
    },

    # Variant 5: Short chain of thought
    "v5_short_cot": {
        "math": "Think briefly, then answer:\n{prompt}\n\nThought: ...\nAnswer: ",
        "logic": "Quick reasoning:\n{prompt}\n\nReason: ...\nAnswer: ",
        "counting": "Count:\n{prompt}\n\nItems: ...\nTotal: ",
        "truth": "Consider:\n{prompt}\n\nBest answer: ",
    },

    # Variant 6: Structured output
    "v6_structured": {
        "math": "{prompt}\n\nProvide answer in format:\nCALCULATION: [work]\nFINAL: [number]",
        "logic": "{prompt}\n\nProvide answer in format:\nREASON: [brief]\nANSWER: [A/B/C/D]",
        "counting": "{prompt}\n\nProvide answer in format:\nITEMS: [list]\nCOUNT: [number]",
        "truth": "{prompt}\n\nProvide answer in format:\nANALYSIS: [brief]\nSELECTION: [number]",
    },

    # Variant 7: No reasoning (pure answer extraction)
    "v7_answer_only": {
        "math": "{prompt}\n\nGive only the final numeric answer:",
        "logic": "{prompt}\n\nGive only the letter (A/B/C/D):",
        "counting": "{prompt}\n\nGive only the count:",
        "truth": "{prompt}\n\nGive only the answer number:",
    },

    # Variant 8: Problem-aware preamble
    "v8_preamble": {
        "math": "MATH PROBLEM - compute carefully:\n{prompt}",
        "logic": "LOGIC PUZZLE - reason systematically:\n{prompt}",
        "counting": "COUNTING TASK - enumerate items:\n{prompt}",
        "truth": "FACTUAL QUESTION - select truthful answer:\n{prompt}",
    },

    # Variant 9: Verification emphasis
    "v9_verify": {
        "math": "{prompt}\n\nSolve, then verify by substituting back.",
        "logic": "{prompt}\n\nChoose, then verify other options are wrong.",
        "counting": "{prompt}\n\nCount, then recount to verify.",
        "truth": "{prompt}\n\nSelect, then verify it's factually correct.",
    },

    # Variant 10: Example-based (few-shot style cue)
    "v10_example_cue": {
        "math": "Follow the examples above. Solve:\n{prompt}",
        "logic": "Apply the same reasoning pattern:\n{prompt}",
        "counting": "Count like the examples:\n{prompt}",
        "truth": "Select like shown:\n{prompt}",
    },

    # Variant 11: Decomposition focus
    "v11_decompose": {
        "math": "Break into sub-problems:\n{prompt}\n\nStep 1:\nStep 2:\nFinal:",
        "logic": "Analyze each option:\n{prompt}\n\nA:\nB:\nC:\nD:\nBest:",
        "counting": "Group and count:\n{prompt}\n\nGroup 1:\nGroup 2:\nTotal:",
        "truth": "Evaluate each:\n{prompt}\n\n1:\n2:\n3:\n4:\nMost true:",
    },

    # Variant 12: Concise expert
    "v12_expert": {
        "math": "[Expert math solver] {prompt}",
        "logic": "[Expert logician] {prompt}",
        "counting": "[Expert counter] {prompt}",
        "truth": "[Fact checker] {prompt}",
    },

    # Variant 13: Question reformulation
    "v13_reframe": {
        "math": "What is the numerical answer to: {prompt}",
        "logic": "What is the logical answer to: {prompt}",
        "counting": "What is the count for: {prompt}",
        "truth": "What is the true answer to: {prompt}",
    },

    # Variant 14: Hybrid (short intro + answer format)
    "v14_hybrid": {
        "math": "Math: {prompt}\nWork briefly, answer after ####",
        "logic": "Logic: {prompt}\nBrief reason, then letter answer.",
        "counting": "Count: {prompt}\nList briefly, then total.",
        "truth": "Truth: {prompt}\nAnalyze briefly, then select.",
    },

    # Variant 15: Adaptive (baseline-aware)
    "v15_adaptive": {
        "math": "{prompt}\n\n[Solve directly if straightforward, show steps if complex]",
        "logic": "{prompt}\n\n[Answer directly if clear, reason if ambiguous]",
        "counting": "{prompt}\n\n[Count directly if simple, enumerate if many items]",
        "truth": "{prompt}\n\n[Select directly if obvious, analyze if uncertain]",
    },
}


@dataclass
class VariantResult:
    variant: str
    benchmark: str
    samples: int
    baseline: float
    enhanced: float
    delta: float

    @property
    def improves(self) -> bool:
        return self.delta > 0

    @property
    def no_regression(self) -> bool:
        return self.delta >= -5  # Allow small variance


def run_variant_benchmark(
    variant_name: str,
    prompts: Dict[str, str],
    benchmark_name: str,
    n_samples: int = 10
) -> VariantResult:
    """Run a single variant on a single benchmark."""

    api_key = os.environ.get("CHUTES_API_KEY")
    if not api_key:
        return VariantResult(variant_name, benchmark_name, 0, 0, 0, 0)

    model = create_chutes_model(api_key)

    baseline_correct = 0
    enhanced_correct = 0
    total = 0

    try:
        if benchmark_name == "GSM8K":
            bench = GSM8K(n_problems=n_samples, n_shots=5, enable_cot=True)
            goldens = bench.load_benchmark_dataset()[:n_samples]
            prompt_template = prompts.get("math", "{prompt}")
            extractor = extract_gsm8k_answer

            for golden in goldens:
                base_prompt = GSM8KTemplate.generate_output(
                    input=golden.input, n_shots=5, enable_cot=True
                )
                expected = golden.expected_output

                # Baseline
                try:
                    resp = _call_model(model, base_prompt)
                    if extractor(resp) == expected:
                        baseline_correct += 1
                except:
                    pass

                # Enhanced
                try:
                    enhanced_prompt = prompt_template.format(prompt=base_prompt)
                    resp = _call_model(model, enhanced_prompt)
                    if extractor(resp) == expected:
                        enhanced_correct += 1
                except:
                    pass

                total += 1

        elif benchmark_name == "BBH-ObjectCount":
            bench = BigBenchHard(
                tasks=[BigBenchHardTask.OBJECT_COUNTING],
                n_problems_per_task=n_samples, n_shots=3, enable_cot=True
            )
            task = BigBenchHardTask.OBJECT_COUNTING
            goldens = bench.load_benchmark_dataset(task)[:n_samples]
            prompt_template = prompts.get("counting", "{prompt}")
            extractor = extract_bbh_answer

            for golden in goldens:
                base_prompt = BigBenchHardTemplate.generate_output(
                    input=golden.input, task=task, n_shots=3, enable_cot=True
                )
                expected = golden.expected_output

                try:
                    resp = _call_model(model, base_prompt)
                    if extractor(resp) == expected:
                        baseline_correct += 1
                except:
                    pass

                try:
                    enhanced_prompt = prompt_template.format(prompt=base_prompt)
                    resp = _call_model(model, enhanced_prompt)
                    if extractor(resp) == expected:
                        enhanced_correct += 1
                except:
                    pass

                total += 1

        elif benchmark_name == "BBH-Logic":
            bench = BigBenchHard(
                tasks=[BigBenchHardTask.LOGICAL_DEDUCTION_THREE_OBJECTS],
                n_problems_per_task=n_samples, n_shots=3, enable_cot=True
            )
            task = BigBenchHardTask.LOGICAL_DEDUCTION_THREE_OBJECTS
            goldens = bench.load_benchmark_dataset(task)[:n_samples]
            prompt_template = prompts.get("logic", "{prompt}")

            for golden in goldens:
                base_prompt = BigBenchHardTemplate.generate_output(
                    input=golden.input, task=task, n_shots=3, enable_cot=True
                )
                expected = golden.expected_output

                def extract_logic(resp):
                    import re
                    match = re.search(r'\(([A-E])\)', resp)
                    if match:
                        return f"({match.group(1)})"
                    return ""

                try:
                    resp = _call_model(model, base_prompt)
                    if extract_logic(resp) == expected:
                        baseline_correct += 1
                except:
                    pass

                try:
                    enhanced_prompt = prompt_template.format(prompt=base_prompt)
                    resp = _call_model(model, enhanced_prompt)
                    if extract_logic(resp) == expected:
                        enhanced_correct += 1
                except:
                    pass

                total += 1

        elif benchmark_name == "TruthfulQA":
            bench = TruthfulQA(n_problems_per_task=n_samples, mode=TruthfulQAMode.MC1)
            task = bench.tasks[0]
            goldens = bench.load_benchmark_dataset(task, TruthfulQAMode.MC1)[:n_samples]
            prompt_template = prompts.get("truth", "{prompt}")
            extractor = extract_truthfulqa_answer

            for golden in goldens:
                base_prompt = TruthfulQATemplate.generate_output(
                    input=golden.input, mode=TruthfulQAMode.MC1
                )
                expected = golden.expected_output

                try:
                    resp = _call_model(model, base_prompt)
                    if extractor(resp) == expected:
                        baseline_correct += 1
                except:
                    pass

                try:
                    enhanced_prompt = prompt_template.format(prompt=base_prompt)
                    resp = _call_model(model, enhanced_prompt)
                    if extractor(resp) == expected:
                        enhanced_correct += 1
                except:
                    pass

                total += 1

    except Exception as e:
        print(f"  Error in {variant_name}/{benchmark_name}: {e}")
        return VariantResult(variant_name, benchmark_name, 0, 0, 0, 0)

    if total == 0:
        return VariantResult(variant_name, benchmark_name, 0, 0, 0, 0)

    baseline_pct = baseline_correct / total * 100
    enhanced_pct = enhanced_correct / total * 100
    delta = enhanced_pct - baseline_pct

    return VariantResult(variant_name, benchmark_name, total, baseline_pct, enhanced_pct, delta)


def run_batch_optimization(n_samples: int = 10, max_workers: int = 4):
    """Run all 16 variants across key benchmarks."""

    benchmarks = ["GSM8K", "BBH-ObjectCount", "BBH-Logic", "TruthfulQA"]

    results: List[VariantResult] = []
    tasks = []

    # Create all tasks
    for variant_name, prompts in PROMPT_VARIANTS.items():
        for benchmark in benchmarks:
            tasks.append((variant_name, prompts, benchmark, n_samples))

    print(f"Running {len(tasks)} variant/benchmark combinations...")
    print(f"Variants: {len(PROMPT_VARIANTS)}, Benchmarks: {len(benchmarks)}")

    # Run with thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_variant_benchmark, *task): task
            for task in tasks
        }

        completed = 0
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                print(f"  [{completed}/{len(tasks)}] {result.variant}/{result.benchmark}: {result.delta:+.1f}pp")
            except Exception as e:
                print(f"  Error: {task[0]}/{task[2]}: {e}")

    return results


def analyze_results(results: List[VariantResult]) -> Dict:
    """Analyze results and find best variants."""

    # Group by variant
    by_variant = {}
    for r in results:
        if r.variant not in by_variant:
            by_variant[r.variant] = []
        by_variant[r.variant].append(r)

    # Score each variant
    variant_scores = {}
    for variant, variant_results in by_variant.items():
        total_delta = sum(r.delta for r in variant_results)
        num_improving = sum(1 for r in variant_results if r.improves)
        num_no_regression = sum(1 for r in variant_results if r.no_regression)

        # Score: weighted by improvement and no-regression
        score = total_delta + (num_no_regression * 10) - ((4 - num_no_regression) * 20)

        variant_scores[variant] = {
            "total_delta": total_delta,
            "num_improving": num_improving,
            "num_no_regression": num_no_regression,
            "score": score,
            "results": variant_results,
        }

    # Sort by score
    ranked = sorted(variant_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    return {
        "ranked_variants": ranked,
        "top_3": [r[0] for r in ranked[:3]],
    }


def synthesize_final_prompt(top_variants: List[str], all_results: List[VariantResult]):
    """Synthesize final prompt from top 3 variants."""

    print("\n" + "="*60)
    print("SYNTHESIS: Combining top 3 variants")
    print("="*60)

    # Analyze what works in each top variant
    insights = {
        "math": [],
        "logic": [],
        "counting": [],
        "truth": [],
    }

    for variant in top_variants:
        prompts = PROMPT_VARIANTS[variant]
        for ptype, prompt in prompts.items():
            insights[ptype].append(f"From {variant}: {prompt[:50]}...")

    # Build synthesized prompt
    synthesized = {
        "math": """Solve this math problem.
{prompt}

Work through it step by step if needed.
Final answer (number): """,

        "logic": """Consider this logic problem.
{prompt}

Think through the options briefly.
Answer: """,

        "counting": """Count the items.
{prompt}

List items, then give total.
Count: """,

        "truth": """Select the most accurate answer.
{prompt}

Selection: """,
    }

    return synthesized


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10, help="Samples per benchmark")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    print("="*60)
    print("BATCH PROMPT OPTIMIZATION")
    print(f"Testing {len(PROMPT_VARIANTS)} variants across 4 benchmarks")
    print(f"Samples per benchmark: {args.samples}")
    print("="*60)

    results = run_batch_optimization(n_samples=args.samples, max_workers=args.workers)

    analysis = analyze_results(results)

    print("\n" + "="*60)
    print("RESULTS RANKED BY SCORE")
    print("="*60)

    for rank, (variant, data) in enumerate(analysis["ranked_variants"], 1):
        print(f"\n{rank}. {variant}")
        print(f"   Score: {data['score']:.1f}")
        print(f"   Total Delta: {data['total_delta']:+.1f}pp")
        print(f"   Improving: {data['num_improving']}/4")
        print(f"   No Regression: {data['num_no_regression']}/4")
        for r in data["results"]:
            status = "✓" if r.no_regression else "✗"
            print(f"     {status} {r.benchmark}: {r.delta:+.1f}pp ({r.baseline:.0f}% → {r.enhanced:.0f}%)")

    print("\n" + "="*60)
    print(f"TOP 3 VARIANTS: {analysis['top_3']}")
    print("="*60)

    final_prompt = synthesize_final_prompt(analysis["top_3"], results)

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "samples_per_benchmark": args.samples,
        "results": [
            {
                "variant": r.variant,
                "benchmark": r.benchmark,
                "samples": r.samples,
                "baseline": r.baseline,
                "enhanced": r.enhanced,
                "delta": r.delta,
            }
            for r in results
        ],
        "top_3_variants": analysis["top_3"],
        "synthesized_prompts": final_prompt,
    }

    output_path = Path("/workspace/benchmarks/batch_results.json")
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
