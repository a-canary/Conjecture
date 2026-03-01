"""
DeepEval Benchmark Suite for Conjecture
Benchmarks: DROP (math), ARC (science), BIG-Bench Hard (logic)
Outputs to STATS.yaml
"""

import argparse
import asyncio
import yaml
import os
import sys
sys.path.insert(0, '/workspace')

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from deepeval.benchmarks import DROP, ARC, BigBenchHard
    from deepeval.benchmarks.modes import ARCMode
    from deepeval.benchmarks.drop.task import DROPTask
    from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask
    from deepeval.benchmarks.big_bench_hard.template import BigBenchHardTemplate
    from deepeval.benchmarks.drop.template import DROPTemplate
    from deepeval.benchmarks.arc.template import ARCTemplate
    from deepeval.models import GPTModel
    from deepeval.models.base_model import DeepEvalBaseLLM
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    DeepEvalBaseLLM = object

from benchmarks.answer_extraction import extract_answer, check_answer_match, AnswerType


def create_chutes_model(api_key: str = None, model: str = "Qwen/Qwen3-14B"):
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
    """Call a model and return the response text, handling (text, usage) tuples."""
    result = model.generate(prompt)
    if isinstance(result, tuple):
        return result[0]
    return str(result)


class ConjectureModel:
    """Wrapper that adds Conjecture enhancement to any base model"""

    def __init__(self, base_model):
        self.base_model = base_model
        try:
            from src.agent.prompt_system import PromptSystem
            self.prompt_system = PromptSystem()
        except ImportError:
            self.prompt_system = None

    def generate(self, prompt: str, problem_type: str = None, **kwargs) -> str:
        # Simple enhancement: add step-by-step reasoning instruction
        # More sophisticated: use domain-specific templates via prompt_system
        enhanced = f"""Think step-by-step. Show your reasoning clearly.
After working through the problem, verify your answer makes sense.

{prompt}"""
        return _call_model(self.base_model, enhanced)

    def get_model_name(self) -> str:
        base_name = getattr(self.base_model, 'model_name', None) or getattr(self.base_model, '_model_name', None) or 'unknown'
        return f"{base_name}+Conjecture"


@dataclass
class BenchmarkResult:
    name: str
    sample_count: int
    baseline_score: float
    conjecture_score: float
    delta: float
    timestamp: str
    error: Optional[str] = None


class DeepEvalSuite:
    """Run DeepEval benchmarks comparing baseline vs Conjecture using direct answer extraction"""

    def __init__(self, base_model=None):
        self.base_model = base_model
        self.conjecture_model = ConjectureModel(base_model) if base_model else None
        self.results: List[BenchmarkResult] = []
        self.stats_path = Path(__file__).parent.parent / "STATS.yaml"

    def _get_model_name(self) -> str:
        if not self.base_model:
            return "none"
        return (
            getattr(self.base_model, 'model_name', None)
            or getattr(self.base_model, '_model_name', None)
            or type(self.base_model).__name__
        )

    def run_drop(self, n_samples: int = 20) -> BenchmarkResult:
        """DROP: Discrete reasoning over paragraphs — direct answer extraction"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("DROP", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            # Use a small subset of tasks so we hit exactly n_samples problems
            drop_bench = DROP(n_problems_per_task=n_samples)

            # Load goldens for the first task only (one task = n_samples problems)
            task = drop_bench.tasks[0]
            drop_bench.load_benchmark_dataset(task)
            goldens = drop_bench.load_benchmark_dataset(task)[:n_samples]

            shots_dataset = drop_bench.shots_dataset

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                # Build prompt using DeepEval's DROP template
                prompt = DROPTemplate.generate_output(
                    input=golden.input,
                    train_set=shots_dataset,
                    n_shots=3,
                )
                expected = golden.expected_output

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_answer(baseline_response, expected)
                    # For DROP, expected may be multi-answer (comma-separated); check any match
                    expected_parts = [e.strip() for e in expected.split(",")]
                    if any(check_answer_match(extracted, ep) for ep in expected_parts):
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture
                try:
                    conj_response = self.conjecture_model.generate(prompt)
                    extracted_c = extract_answer(conj_response, expected)
                    if any(check_answer_match(extracted_c, ep) for ep in expected_parts):
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  DROP: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "DROP", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("DROP", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_arc(self, n_samples: int = 20, mode: str = "challenge") -> BenchmarkResult:
        """ARC: AI2 Reasoning Challenge — direct answer extraction"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult(f"ARC-{mode}", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            arc_mode = ARCMode.CHALLENGE if mode == "challenge" else ARCMode.EASY
            arc_bench = ARC(n_problems=n_samples, mode=arc_mode)
            goldens = arc_bench.load_benchmark_dataset(arc_mode)[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                # golden.input already contains the formatted question with choices
                prompt = golden.input + "\n\nOutput 'A', 'B', 'C', or 'D'. Full answer not needed."
                expected = golden.expected_output  # single letter like "A"

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_answer(baseline_response, expected, AnswerType.MULTIPLE_CHOICE)
                    if check_answer_match(extracted, expected, AnswerType.MULTIPLE_CHOICE):
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture
                try:
                    conj_response = self.conjecture_model.generate(prompt)
                    extracted_c = extract_answer(conj_response, expected, AnswerType.MULTIPLE_CHOICE)
                    if check_answer_match(extracted_c, expected, AnswerType.MULTIPLE_CHOICE):
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  ARC-{mode}: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                f"ARC-{mode}", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult(f"ARC-{mode}", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_bbh(self, n_samples: int = 20) -> BenchmarkResult:
        """BIG-Bench Hard: Logic and reasoning — direct answer extraction"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("BBH", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            bbh_bench = BigBenchHard(n_problems_per_task=n_samples)

            # Use a single task to get exactly n_samples problems
            task = BigBenchHardTask.BOOLEAN_EXPRESSIONS
            goldens = bbh_bench.load_benchmark_dataset(task)[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                # Build prompt using BigBenchHard template (3-shot CoT)
                prompt = BigBenchHardTemplate.generate_output(
                    input=golden.input,
                    task=task,
                    n_shots=3,
                    enable_cot=True,
                )
                expected = golden.expected_output  # e.g. "True", "False", "(A)", etc.

                # Determine answer type from expected
                exp_lower = expected.strip().lower()
                if exp_lower in ("true", "false", "yes", "no"):
                    ans_type = AnswerType.CATEGORICAL
                elif exp_lower.startswith("(") and len(exp_lower) == 3:
                    ans_type = AnswerType.MULTIPLE_CHOICE
                else:
                    ans_type = AnswerType.CATEGORICAL

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_answer(baseline_response, expected, ans_type)
                    if check_answer_match(extracted, expected, ans_type):
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture
                try:
                    conj_response = self.conjecture_model.generate(prompt)
                    extracted_c = extract_answer(conj_response, expected, ans_type)
                    if check_answer_match(extracted_c, expected, ans_type):
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  BBH: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "BBH", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("BBH", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_full_suite(self, n_samples: int = 20) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks sequentially"""
        results = {
            "DROP": self.run_drop(n_samples),
            "ARC": self.run_arc(n_samples, "challenge"),
            "BBH": self.run_bbh(n_samples)
        }
        self.results = list(results.values())
        return results

    def update_stats_yaml(self, key: str = "deepeval_benchmarks") -> dict:
        """Update STATS.yaml with results"""
        stats = {}
        if self.stats_path.exists():
            with open(self.stats_path) as f:
                stats = yaml.safe_load(f) or {}

        stats[key] = {
            "last_run": datetime.now().isoformat(),
            "model": self._get_model_name(),
            "benchmarks": {
                r.name: {
                    "sample_count": r.sample_count,
                    "baseline_score": round(r.baseline_score, 2),
                    "conjecture_score": round(r.conjecture_score, 2),
                    "delta": round(r.delta, 2),
                    "error": r.error
                } for r in self.results
            }
        }

        valid = [r for r in self.results if r.error is None]
        if valid:
            stats[key]["summary"] = {
                "avg_baseline": round(sum(r.baseline_score for r in valid) / len(valid), 2),
                "avg_conjecture": round(sum(r.conjecture_score for r in valid) / len(valid), 2),
                "avg_delta": round(sum(r.delta for r in valid) / len(valid), 2),
                "benchmarks_passed": len(valid),
                "benchmarks_failed": len(self.results) - len(valid)
            }

        with open(self.stats_path, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="DeepEval Benchmark Suite — DROP, ARC, BBH via Chutes.ai"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-14B",
        help="Model ID on Chutes.ai (default: Qwen/Qwen3-14B). "
             "Use --model deepseek-ai/DeepSeek-V3-0324 for DeepSeek-V3.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of samples per benchmark (default: 20)",
    )
    parser.add_argument(
        "--stats-key",
        default=None,
        help="Key to use in STATS.yaml (default: deepeval_benchmarks_8b for 8B models, deepeval_benchmarks otherwise)",
    )
    args = parser.parse_args()

    print("DeepEval Benchmark Suite")
    print("=" * 50)
    print(f"Model : {args.model}")
    print(f"N     : {args.n} samples per benchmark")

    api_key = os.environ.get("CHUTES_API_KEY")
    if not api_key:
        print("ERROR: Set CHUTES_API_KEY environment variable")
        print("  export CHUTES_API_KEY=your_key_here")
        return

    model = create_chutes_model(api_key, model=args.model)

    suite = DeepEvalSuite(base_model=model)
    print(f"\nRunning benchmarks ({args.n} samples each)...")

    results = suite.run_full_suite(n_samples=args.n)

    print("\nResults:")
    print("-" * 50)
    for name, r in results.items():
        if r.error:
            print(f"{name}: ERROR - {r.error}")
        else:
            print(f"{name}: baseline={r.baseline_score:.1f}%  conjecture={r.conjecture_score:.1f}%  delta={r.delta:+.1f}pp")

    # Choose stats key: default to 8b-specific key for 8B class models
    stats_key = args.stats_key
    if stats_key is None:
        if "8b" in args.model.lower() or "8B" in args.model:
            stats_key = "deepeval_benchmarks_8b"
        else:
            stats_key = "deepeval_benchmarks"

    suite.update_stats_yaml(key=stats_key)
    print(f"\nSTATS.yaml updated (key: {stats_key})")


if __name__ == "__main__":
    main()
