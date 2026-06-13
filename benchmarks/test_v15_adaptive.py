# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Test v15_adaptive prompt variant across 5 benchmarks.

v15_adaptive: Task-adaptive instructions that scale with complexity.
- math: "{prompt}\n\n[Solve directly if straightforward, show steps if complex]"
- logic: "{prompt}\n\n[Answer directly if clear, analyze if ambiguous]"
- counting: "{prompt}\n\n[Count directly if obvious, enumerate if many]"
- truth: "{prompt}\n\n[State directly if factual, evaluate if uncertain]"

Benchmarks: GSM8K, BBH-Logic, BBH-ObjectCounting, LogiQA, TruthfulQA
Samples: 15 per benchmark
Model: gpt-oss-20b via Chutes.ai
"""

import argparse
import asyncio
import json
import os
import sys
import re
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

sys.path.insert(0, '/workspace')

try:
    from deepeval.benchmarks import GSM8K, LogiQA, TruthfulQA, BigBenchHard
    from deepeval.benchmarks.gsm8k.template import GSM8KTemplate
    from deepeval.benchmarks.logi_qa.template import LogiQATemplate
    from deepeval.benchmarks.truthful_qa.template import TruthfulQATemplate
    from deepeval.benchmarks.big_bench_hard.template import BigBenchHardTemplate
    from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask
    from deepeval.models import GPTModel
    DEEPEVAL_AVAILABLE = True
except ImportError as e:
    print(f"DeepEval import error: {e}")
    DEEPEVAL_AVAILABLE = False


# v15_adaptive prompt templates
V15_ADAPTIVE_TEMPLATES = {
    "math": "{prompt}\n\n[Solve directly if straightforward, show steps if complex]",
    "logic": "{prompt}\n\n[Answer directly if clear, analyze if ambiguous]",
    "counting": "{prompt}\n\n[Count directly if obvious, enumerate if many]",
    "truth": "{prompt}\n\n[State directly if factual, evaluate if uncertain]",
}


def create_chutes_model(api_key: str = None, model: str = "openai/gpt-oss-20b"):
    """Create DeepEval model using Chutes.ai endpoint"""
    api_key = api_key or os.environ.get("CHUTES_API_KEY")
    if not api_key:
        raise ValueError("CHUTES_API_KEY required. Set env var or pass api_key")
    return GPTModel(
        model=model,
        api_key=api_key,
        base_url="https://llm.chutes.ai/v1"
    )


def _call_model(model, prompt: str) -> str:
    """Call a model and return the response text."""
    result = model.generate(prompt)
    if isinstance(result, tuple):
        return result[0]
    return str(result)


def apply_v15_adaptive(prompt: str, variant: str) -> str:
    """Apply v15_adaptive template to prompt."""
    template = V15_ADAPTIVE_TEMPLATES.get(variant, "{prompt}")
    return template.format(prompt=prompt)


# Answer extraction functions
def extract_gsm8k_answer(response: str) -> str:
    """Extract numeric answer from GSM8K response."""
    # Pattern 1: #### number
    match = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', response)
    if match:
        return match.group(1).replace(',', '')

    # Pattern 2: "the answer is X" or "= X"
    match = re.search(r'(?:answer\s+is|=)\s*(-?\d[\d,]*\.?\d*)', response, re.I)
    if match:
        return match.group(1).replace(',', '')

    # Pattern 3: Last number in response
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', response)
    if numbers:
        return numbers[-1].replace(',', '')

    return ""


def extract_logiqa_answer(response: str) -> str:
    """Extract multiple choice answer (A-D) from LogiQA response."""
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
    """Extract multiple choice answer from TruthfulQA response."""
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
    """Extract answer from BigBenchHard response."""
    # Pattern 1: "answer is NUMBER" (handles negative)
    match = re.search(r'answer\s+is\s*[:\s]*(-?\d+)', response, re.I)
    if match:
        return match.group(1)

    # Pattern 2: "= NUMBER" at end of calculation
    match = re.search(r'=\s*(-?\d+)\s*$', response.strip())
    if match:
        return match.group(1)

    # Pattern 3: Final number in response (for arithmetic)
    match = re.search(r'(-?\d+)\s*[\.!]?\s*$', response.strip())
    if match:
        return match.group(1)

    # Pattern 4: Multiple choice (A), (B), etc.
    match = re.search(r'\(([A-E])\)', response)
    if match:
        return f"({match.group(1)})"

    # Pattern 5: Last word/phrase
    words = response.strip().split()
    if words:
        return words[-1].strip('.,!?')

    return ""


@dataclass
class BenchmarkResult:
    name: str
    sample_count: int
    baseline_score: float
    variant_score: float
    delta: float
    timestamp: str
    error: Optional[str] = None


class V15AdaptiveTest:
    """Test v15_adaptive prompt variant across benchmarks."""

    def __init__(self, model):
        self.model = model
        self.results: List[BenchmarkResult] = []

    def _get_model_name(self) -> str:
        return getattr(self.model, 'model_name', None) or getattr(self.model, '_model_name', None) or 'unknown'

    def run_gsm8k(self, n_samples: int = 15) -> BenchmarkResult:
        """GSM8K: Grade school math with v15_adaptive math template."""
        if not DEEPEVAL_AVAILABLE:
            return BenchmarkResult("GSM8K", 0, 0.0, 0.0, 0.0, datetime.now().isoformat(), "DeepEval not available")

        try:
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

                # Baseline (direct prompt)
                try:
                    baseline_response = _call_model(self.model, prompt)
                    extracted = extract_gsm8k_answer(baseline_response)
                    try:
                        if abs(float(extracted) - float(expected)) < 0.01:
                            baseline_correct += 1
                    except ValueError:
                        pass
                except Exception as e:
                    print(f"  Baseline error: {e}")

                # v15_adaptive variant
                try:
                    variant_prompt = apply_v15_adaptive(prompt, "math")
                    variant_response = _call_model(self.model, variant_prompt)
                    extracted_v = extract_gsm8k_answer(variant_response)
                    try:
                        if abs(float(extracted_v) - float(expected)) < 0.01:
                            variant_correct += 1
                    except ValueError:
                        pass
                except Exception as e:
                    print(f"  Variant error: {e}")

                if (i + 1) % 5 == 0:
                    print(f"  GSM8K: {i+1}/{total} (baseline {baseline_correct}, variant {variant_correct})")

            baseline_score = baseline_correct / total * 100
            variant_score = variant_correct / total * 100

            return BenchmarkResult(
                "GSM8K", total, baseline_score, variant_score,
                variant_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("GSM8K", n_samples, 0.0, 0.0, 0.0, datetime.now().isoformat(), str(e))

    def run_bbh_logic(self, n_samples: int = 15) -> BenchmarkResult:
        """BBH-Logic: Logical deduction with v15_adaptive logic template."""
        if not DEEPEVAL_AVAILABLE:
            return BenchmarkResult("BBH-Logic", 0, 0.0, 0.0, 0.0, datetime.now().isoformat(), "DeepEval not available")

        try:
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
                    baseline_response = _call_model(self.model, prompt)
                    extracted = extract_bbh_answer(baseline_response)
                    if extracted.lower() == expected.lower():
                        baseline_correct += 1
                except Exception as e:
                    print(f"  Baseline error: {e}")

                # v15_adaptive variant
                try:
                    variant_prompt = apply_v15_adaptive(prompt, "logic")
                    variant_response = _call_model(self.model, variant_prompt)
                    extracted_v = extract_bbh_answer(variant_response)
                    if extracted_v.lower() == expected.lower():
                        variant_correct += 1
                except Exception as e:
                    print(f"  Variant error: {e}")

                if (i + 1) % 5 == 0:
                    print(f"  BBH-Logic: {i+1}/{total} (baseline {baseline_correct}, variant {variant_correct})")

            baseline_score = baseline_correct / total * 100
            variant_score = variant_correct / total * 100

            return BenchmarkResult(
                "BBH-Logic", total, baseline_score, variant_score,
                variant_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("BBH-Logic", n_samples, 0.0, 0.0, 0.0, datetime.now().isoformat(), str(e))

    def run_bbh_object_counting(self, n_samples: int = 15) -> BenchmarkResult:
        """BBH-ObjectCounting: Object counting with v15_adaptive counting template."""
        if not DEEPEVAL_AVAILABLE:
            return BenchmarkResult("BBH-ObjectCounting", 0, 0.0, 0.0, 0.0, datetime.now().isoformat(), "DeepEval not available")

        try:
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
                    baseline_response = _call_model(self.model, prompt)
                    extracted = extract_bbh_answer(baseline_response)
                    if extracted == expected:
                        baseline_correct += 1
                except Exception as e:
                    print(f"  Baseline error: {e}")

                # v15_adaptive variant
                try:
                    variant_prompt = apply_v15_adaptive(prompt, "counting")
                    variant_response = _call_model(self.model, variant_prompt)
                    extracted_v = extract_bbh_answer(variant_response)
                    if extracted_v == expected:
                        variant_correct += 1
                except Exception as e:
                    print(f"  Variant error: {e}")

                if (i + 1) % 5 == 0:
                    print(f"  BBH-ObjectCounting: {i+1}/{total} (baseline {baseline_correct}, variant {variant_correct})")

            baseline_score = baseline_correct / total * 100
            variant_score = variant_correct / total * 100

            return BenchmarkResult(
                "BBH-ObjectCounting", total, baseline_score, variant_score,
                variant_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("BBH-ObjectCounting", n_samples, 0.0, 0.0, 0.0, datetime.now().isoformat(), str(e))

    def run_logiqa(self, n_samples: int = 15) -> BenchmarkResult:
        """LogiQA: Logical reasoning with v15_adaptive logic template."""
        if not DEEPEVAL_AVAILABLE:
            return BenchmarkResult("LogiQA", 0, 0.0, 0.0, 0.0, datetime.now().isoformat(), "DeepEval not available")

        try:
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
                    baseline_response = _call_model(self.model, prompt)
                    extracted = extract_logiqa_answer(baseline_response)
                    if extracted == expected:
                        baseline_correct += 1
                except Exception as e:
                    print(f"  Baseline error: {e}")

                # v15_adaptive variant
                try:
                    variant_prompt = apply_v15_adaptive(prompt, "logic")
                    variant_response = _call_model(self.model, variant_prompt)
                    extracted_v = extract_logiqa_answer(variant_response)
                    if extracted_v == expected:
                        variant_correct += 1
                except Exception as e:
                    print(f"  Variant error: {e}")

                if (i + 1) % 5 == 0:
                    print(f"  LogiQA: {i+1}/{total} (baseline {baseline_correct}, variant {variant_correct})")

            baseline_score = baseline_correct / total * 100
            variant_score = variant_correct / total * 100

            return BenchmarkResult(
                "LogiQA", total, baseline_score, variant_score,
                variant_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("LogiQA", n_samples, 0.0, 0.0, 0.0, datetime.now().isoformat(), str(e))

    def run_truthfulqa(self, n_samples: int = 15) -> BenchmarkResult:
        """TruthfulQA: Truth and factuality with v15_adaptive truth template."""
        if not DEEPEVAL_AVAILABLE:
            return BenchmarkResult("TruthfulQA", 0, 0.0, 0.0, 0.0, datetime.now().isoformat(), "DeepEval not available")

        try:
            from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode
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
                    baseline_response = _call_model(self.model, prompt)
                    extracted = extract_truthfulqa_answer(baseline_response)
                    if extracted == expected:
                        baseline_correct += 1
                except Exception as e:
                    print(f"  Baseline error: {e}")

                # v15_adaptive variant
                try:
                    variant_prompt = apply_v15_adaptive(prompt, "truth")
                    variant_response = _call_model(self.model, variant_prompt)
                    extracted_v = extract_truthfulqa_answer(variant_response)
                    if extracted_v == expected:
                        variant_correct += 1
                except Exception as e:
                    print(f"  Variant error: {e}")

                if (i + 1) % 5 == 0:
                    print(f"  TruthfulQA: {i+1}/{total} (baseline {baseline_correct}, variant {variant_correct})")

            baseline_score = baseline_correct / total * 100
            variant_score = variant_correct / total * 100

            return BenchmarkResult(
                "TruthfulQA", total, baseline_score, variant_score,
                variant_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("TruthfulQA", n_samples, 0.0, 0.0, 0.0, datetime.now().isoformat(), str(e))

    def run_all(self, n_samples: int = 15) -> Dict[str, BenchmarkResult]:
        """Run all 5 benchmarks."""
        results = {}

        print("\n--- GSM8K ---")
        results["GSM8K"] = self.run_gsm8k(n_samples)
        self.results.append(results["GSM8K"])

        print("\n--- BBH-Logic ---")
        results["BBH-Logic"] = self.run_bbh_logic(n_samples)
        self.results.append(results["BBH-Logic"])

        print("\n--- BBH-ObjectCounting ---")
        results["BBH-ObjectCounting"] = self.run_bbh_object_counting(n_samples)
        self.results.append(results["BBH-ObjectCounting"])

        print("\n--- LogiQA ---")
        results["LogiQA"] = self.run_logiqa(n_samples)
        self.results.append(results["LogiQA"])

        print("\n--- TruthfulQA ---")
        results["TruthfulQA"] = self.run_truthfulqa(n_samples)
        self.results.append(results["TruthfulQA"])

        return results

    def to_json(self) -> dict:
        """Export results as JSON-serializable dict."""
        return {
            "variant": "v15_adaptive",
            "model": self._get_model_name(),
            "timestamp": datetime.now().isoformat(),
            "templates": V15_ADAPTIVE_TEMPLATES,
            "results": {r.name: asdict(r) for r in self.results},
            "summary": {
                "benchmarks_tested": len(self.results),
                "avg_baseline": round(sum(r.baseline_score for r in self.results) / len(self.results), 2) if self.results else 0,
                "avg_variant": round(sum(r.variant_score for r in self.results) / len(self.results), 2) if self.results else 0,
                "avg_delta": round(sum(r.delta for r in self.results) / len(self.results), 2) if self.results else 0,
                "positive_deltas": sum(1 for r in self.results if r.delta > 0),
                "negative_deltas": sum(1 for r in self.results if r.delta < 0),
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Test v15_adaptive prompt variant")
    parser.add_argument("--n", type=int, default=15, help="Samples per benchmark (default: 15)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    print("v15_adaptive Prompt Variant Test")
    print("=" * 50)
    print(f"Samples per benchmark: {args.n}")
    print("Benchmarks: GSM8K, BBH-Logic, BBH-ObjectCounting, LogiQA, TruthfulQA")
    print("Model: gpt-oss-20b via Chutes.ai")
    print("=" * 50)

    # Create model
    api_key = os.environ.get("CHUTES_API_KEY")
    if not api_key:
        print("ERROR: CHUTES_API_KEY environment variable not set")
        return

    model = create_chutes_model(api_key, "openai/gpt-oss-20b")
    print(f"Model: {model.model_name if hasattr(model, 'model_name') else 'gpt-oss-20b'}")

    # Run tests
    tester = V15AdaptiveTest(model)
    results = tester.run_all(args.n)

    # Print summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Benchmark':<20} {'Baseline %':<12} {'Variant %':<12} {'Delta':<10}")
    print("-" * 54)

    for name, result in results.items():
        if result.error:
            print(f"{name:<20} ERROR: {result.error}")
        else:
            print(f"{name:<20} {result.baseline_score:>10.1f}% {result.variant_score:>10.1f}% {result.delta:>+8.1f}pp")

    # Export JSON
    json_output = tester.to_json()

    output_file = args.output or "/workspace/benchmarks/v15_adaptive_results.json"
    with open(output_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Print JSON for easy copy
    print("\n" + "=" * 50)
    print("JSON OUTPUT")
    print("=" * 50)
    print(json.dumps(json_output, indent=2))


if __name__ == "__main__":
    main()
