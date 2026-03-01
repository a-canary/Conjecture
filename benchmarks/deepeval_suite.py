"""
DeepEval Benchmark Suite for Conjecture
Benchmarks: DROP (math), ARC (science), BIG-Bench Hard (logic)
Outputs to STATS.yaml
"""

import asyncio
import yaml
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from deepeval.benchmarks import DROP, ARC, BigBenchHard
    from deepeval.benchmarks.modes import ARCMode
    from deepeval.models import GPTModel
    from deepeval.models.base_model import DeepEvalBaseLLM
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
    DeepEvalBaseLLM = object


def create_chutes_model(api_key: str = None, model: str = "deepseek-ai/DeepSeek-V3-0324"):
    """Create DeepEval model using Chutes.ai endpoint"""
    api_key = api_key or os.environ.get("CHUTES_API_KEY")
    if not api_key:
        raise ValueError("CHUTES_API_KEY required. Set env var or pass api_key")
    return GPTModel(
        model=model,
        api_key=api_key,
        base_url="https://llm.chutes.ai/v1"
    )


class ConjectureModel(DeepEvalBaseLLM):
    """Wrapper that adds Conjecture enhancement to any base model"""

    def __init__(self, base_model: DeepEvalBaseLLM):
        super().__init__()
        self.base_model = base_model
        try:
            from src.agent.prompt_system import PromptSystem
            self.prompt_system = PromptSystem()
        except ImportError:
            self.prompt_system = None

    def generate(self, prompt: str, **kwargs) -> str:
        if self.prompt_system:
            enhanced = self.prompt_system.enhance_prompt(prompt)
        else:
            enhanced = f"Reason step-by-step. Verify your answer.\n\n{prompt}"
        return self.base_model.generate(enhanced, **kwargs)

    async def a_generate(self, prompt: str, **kwargs) -> str:
        if self.prompt_system:
            enhanced = self.prompt_system.enhance_prompt(prompt)
        else:
            enhanced = f"Reason step-by-step. Verify your answer.\n\n{prompt}"
        return await self.base_model.a_generate(enhanced, **kwargs)

    def get_model_name(self) -> str:
        return f"{self.base_model.get_model_name()}+Conjecture"

    def load_model(self):
        return self


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
    """Run DeepEval benchmarks comparing baseline vs Conjecture"""

    def __init__(self, base_model: DeepEvalBaseLLM = None):
        self.base_model = base_model
        self.conjecture_model = ConjectureModel(base_model) if base_model else None
        self.results: List[BenchmarkResult] = []
        self.stats_path = Path(__file__).parent.parent / "STATS.yaml"

    def run_drop(self, n_samples: int = 10) -> BenchmarkResult:
        """DROP: Discrete reasoning over paragraphs"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("DROP", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            benchmark = DROP(n_problems_per_task=n_samples)

            # Baseline
            baseline_result = benchmark.evaluate(self.base_model)
            baseline_score = baseline_result.overall_score * 100

            # With Conjecture
            conj_result = benchmark.evaluate(self.conjecture_model)
            conj_score = conj_result.overall_score * 100

            return BenchmarkResult(
                "DROP", n_samples, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("DROP", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_arc(self, n_samples: int = 10, mode: str = "challenge") -> BenchmarkResult:
        """ARC: AI2 Reasoning Challenge"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult(f"ARC-{mode}", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            arc_mode = ARCMode.CHALLENGE if mode == "challenge" else ARCMode.EASY
            benchmark = ARC(n_problems=n_samples, mode=arc_mode)

            baseline_result = benchmark.evaluate(self.base_model)
            baseline_score = baseline_result.overall_score * 100

            conj_result = benchmark.evaluate(self.conjecture_model)
            conj_score = conj_result.overall_score * 100

            return BenchmarkResult(
                f"ARC-{mode}", n_samples, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult(f"ARC-{mode}", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_bbh(self, n_samples: int = 10) -> BenchmarkResult:
        """BIG-Bench Hard: Logic and reasoning"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("BBH", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            benchmark = BigBenchHard(n_problems_per_task=n_samples)

            baseline_result = benchmark.evaluate(self.base_model)
            baseline_score = baseline_result.overall_score * 100

            conj_result = benchmark.evaluate(self.conjecture_model)
            conj_score = conj_result.overall_score * 100

            return BenchmarkResult(
                "BBH", n_samples, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("BBH", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_full_suite(self, n_samples: int = 10) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks sequentially"""
        results = {
            "DROP": self.run_drop(n_samples),
            "ARC": self.run_arc(n_samples, "challenge"),
            "BBH": self.run_bbh(n_samples)
        }
        self.results = list(results.values())
        return results

    def update_stats_yaml(self) -> dict:
        """Update STATS.yaml with results"""
        stats = {}
        if self.stats_path.exists():
            with open(self.stats_path) as f:
                stats = yaml.safe_load(f) or {}

        stats["deepeval_benchmarks"] = {
            "last_run": datetime.now().isoformat(),
            "model": self.base_model.get_model_name() if self.base_model else "none",
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
            stats["deepeval_benchmarks"]["summary"] = {
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
    """Run benchmarks with Chutes.ai model"""
    print("DeepEval Benchmark Suite")
    print("=" * 50)

    api_key = os.environ.get("CHUTES_API_KEY")
    if not api_key:
        print("ERROR: Set CHUTES_API_KEY environment variable")
        print("  export CHUTES_API_KEY=your_key_here")
        return

    print(f"Using Chutes.ai endpoint...")
    model = create_chutes_model(api_key)

    suite = DeepEvalSuite(base_model=model)
    print(f"\nRunning benchmarks (10 samples each)...")

    results = suite.run_full_suite(n_samples=10)

    print("\nResults:")
    print("-" * 50)
    for name, r in results.items():
        if r.error:
            print(f"{name}: ERROR - {r.error}")
        else:
            print(f"{name}: {r.baseline_score:.1f}% -> {r.conjecture_score:.1f}% ({r.delta:+.1f}pp)")

    suite.update_stats_yaml()
    print("\nSTATS.yaml updated")


if __name__ == "__main__":
    main()
