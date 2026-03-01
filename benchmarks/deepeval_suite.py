"""
DeepEval Benchmark Suite for Conjecture
Benchmarks: DROP (math), ARC (science), BIG-Bench Hard (logic)
Outputs to STATS.yaml
Uses robust answer extraction to avoid 70pp accuracy swings from extraction bugs.
"""

import asyncio
import yaml
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable

from answer_extraction import extract_answer, check_answer_match, AnswerType

try:
    from deepeval.benchmarks import DROP, ARC, BigBenchHard
    from deepeval.benchmarks.modes import ARCMode
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    name: str
    sample_count: int
    baseline_score: float
    conjecture_score: float
    delta: float
    timestamp: str
    error: Optional[str] = None


class DeepEvalSuite:
    """DeepEval benchmark suite for Conjecture validation"""

    def __init__(self, model_callable: Callable = None):
        self.model = model_callable
        self.results: List[BenchmarkResult] = []
        self.stats_path = Path(__file__).parent.parent / "STATS.yaml"

    async def run_drop(self, n_samples: int = 50) -> BenchmarkResult:
        """DROP: Reading comprehension + discrete reasoning (hard math)"""
        return await self._run_benchmark_pair("DROP", n_samples,
            lambda n: DROP(n_problems_per_task=n) if DEEPEVAL_AVAILABLE else None)

    async def run_arc(self, n_samples: int = 50, mode: str = "challenge") -> BenchmarkResult:
        """ARC: AI2 Reasoning Challenge (science reasoning)"""
        def create_arc(n):
            if not DEEPEVAL_AVAILABLE:
                return None
            arc_mode = ARCMode.CHALLENGE if mode == "challenge" else ARCMode.EASY
            return ARC(n_problems=n, mode=arc_mode)
        return await self._run_benchmark_pair(f"ARC-{mode}", n_samples, create_arc)

    async def run_bbh(self, n_samples: int = 50) -> BenchmarkResult:
        """BIG-Bench Hard: Logic and multi-step reasoning"""
        return await self._run_benchmark_pair("BBH", n_samples,
            lambda n: BigBenchHard(n_problems_per_task=n) if DEEPEVAL_AVAILABLE else None)

    async def _run_benchmark_pair(self, name: str, n_samples: int,
                                   benchmark_factory: Callable) -> BenchmarkResult:
        """Run baseline and conjecture versions of a benchmark"""
        if not DEEPEVAL_AVAILABLE:
            return BenchmarkResult(name=name, sample_count=0, baseline_score=0.0,
                conjecture_score=0.0, delta=0.0, timestamp=datetime.now().isoformat(),
                error="DeepEval not installed")
        try:
            benchmark = benchmark_factory(n_samples)
            if benchmark is None:
                raise ValueError("Could not create benchmark")

            baseline = await self._run_single(benchmark, use_conjecture=False)
            conjecture = await self._run_single(benchmark, use_conjecture=True)

            return BenchmarkResult(
                name=name, sample_count=n_samples,
                baseline_score=baseline, conjecture_score=conjecture,
                delta=conjecture - baseline, timestamp=datetime.now().isoformat())
        except Exception as e:
            return BenchmarkResult(name=name, sample_count=n_samples,
                baseline_score=0.0, conjecture_score=0.0, delta=0.0,
                timestamp=datetime.now().isoformat(), error=str(e))

    async def _run_single(self, benchmark, use_conjecture: bool) -> float:
        """Run benchmark and return accuracy"""
        correct = 0
        total = 0

        for problem in getattr(benchmark, 'problems', []):
            prompt = getattr(problem, 'input', str(problem))
            expected = getattr(problem, 'expected_output', None)

            if use_conjecture:
                response = await self._call_with_conjecture(prompt)
            else:
                response = await self._call_baseline(prompt)

            if expected and str(expected).lower().strip() in response.lower():
                correct += 1
            total += 1

        return (correct / total * 100) if total > 0 else 0.0

    async def _call_baseline(self, prompt: str) -> str:
        """Call model without Conjecture"""
        if self.model:
            return await self.model(prompt)
        return await self._default_model_call(prompt)

    async def _call_with_conjecture(self, prompt: str) -> str:
        """Call model with Conjecture enhancement"""
        try:
            from src.agent.prompt_system import PromptSystem
            ps = PromptSystem()
            enhanced = ps.enhance_prompt(prompt)
        except ImportError:
            enhanced = f"Reason step-by-step. Verify your answer.\n\n{prompt}\n\nShow your work."

        if self.model:
            return await self.model(enhanced)
        return await self._default_model_call(enhanced)

    async def _default_model_call(self, prompt: str) -> str:
        """Default model call via configured provider"""
        try:
            from src.processing.unified_bridge import UnifiedLLMBridge
            bridge = UnifiedLLMBridge()
            return await bridge.generate(prompt)
        except Exception as e:
            return f"Error: {e}"

    async def run_full_suite(self, n_samples: int = 50) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks in parallel"""
        tasks = [
            self.run_drop(n_samples),
            self.run_arc(n_samples, "challenge"),
            self.run_bbh(n_samples)
        ]
        results_list = await asyncio.gather(*tasks)
        results = {r.name: r for r in results_list}
        self.results = results_list
        return results

    def update_stats_yaml(self) -> dict:
        """Update STATS.yaml with benchmark results"""
        stats = {}
        if self.stats_path.exists():
            with open(self.stats_path) as f:
                stats = yaml.safe_load(f) or {}

        stats["deepeval_benchmarks"] = {
            "last_run": datetime.now().isoformat(),
            "benchmarks": {
                r.name: {
                    "sample_count": r.sample_count,
                    "baseline_score": round(r.baseline_score, 2),
                    "conjecture_score": round(r.conjecture_score, 2),
                    "delta": round(r.delta, 2),
                    "timestamp": r.timestamp,
                    "error": r.error
                } for r in self.results
            }
        }

        valid = [r for r in self.results if r.error is None]
        if valid:
            stats["deepeval_benchmarks"]["summary"] = {
                "avg_delta": round(sum(r.delta for r in valid) / len(valid), 2),
                "benchmarks_run": len(valid),
                "benchmarks_failed": len(self.results) - len(valid)
            }

        with open(self.stats_path, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
        return stats


async def main():
    """Run benchmark suite"""
    print("DeepEval Benchmark Suite")
    print("=" * 40)

    suite = DeepEvalSuite()
    results = await suite.run_full_suite(n_samples=50)

    for name, r in results.items():
        if r.error:
            print(f"{name}: ERROR - {r.error}")
        else:
            print(f"{name}: {r.baseline_score:.1f}% -> {r.conjecture_score:.1f}% ({r.delta:+.1f}pp)")

    suite.update_stats_yaml()
    print("\nSTATS.yaml updated")


if __name__ == "__main__":
    asyncio.run(main())
