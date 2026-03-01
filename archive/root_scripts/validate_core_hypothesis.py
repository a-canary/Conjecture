#!/usr/bin/env python3
"""
Core Hypothesis Validation: GSM8K-Style Benchmark

Tests the fundamental hypothesis:
"Conjecture provides significant improvement in intelligence and truthfulness"

Compares:
- Baseline: Direct LLM calls (GLM-4.6-FP8)
- Conjecture: Full claim-based reasoning system

Benchmark: GSM8K-style grade school math problems
"""

import asyncio
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.unified_config import UnifiedConfig as Config
from src.processing.llm_bridge import LLMBridge
from src.endpoint.conjecture_endpoint import ConjectureEndpoint


@dataclass
class MathProblem:
    """A math problem with ground truth answer"""

    id: str
    question: str
    ground_truth: str
    expected_answer_value: float
    category: str
    difficulty: str


@dataclass
class BenchmarkResult:
    """Result from evaluating one problem"""

    problem_id: str
    system: str  # "baseline" or "conjecture"
    question: str
    response: str
    extracted_answer: float
    correct_answer: float
    is_correct: bool
    response_time: float
    error: str = None


class CoreHypothesisValidator:
    """Validates the core Conjecture hypothesis with real benchmarks"""

    def __init__(self):
        self.config = Config()
        self.llm_bridge = LLMBridge(self.config)
        self.endpoint = ConjectureEndpoint(self.config)

    def load_math_problems(self) -> List[MathProblem]:
        """Load GSM8K-style math problems from test cases"""
        problems = []

        # Load from research/test_cases
        test_case_dir = Path("research/test_cases")

        # Find all math test cases
        math_files = [
            "math_algebra_20251204_181540.json",
            "math_rate_20251204_181540.json",
            "math_geometry_20251204_181540.json",
        ]

        for filename in math_files:
            filepath = test_case_dir / filename
            if filepath.exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract expected numeric answer from ground truth
                answer_value = self._extract_numeric_answer(
                    data.get("ground_truth", "")
                )

                problem = MathProblem(
                    id=data.get("id", filename),
                    question=data.get("question", ""),
                    ground_truth=data.get("ground_truth", ""),
                    expected_answer_value=answer_value,
                    category=data.get("category", "math"),
                    difficulty=data.get("difficulty", "medium"),
                )
                problems.append(problem)

        # Add some classic GSM8K-style problems
        problems.extend(self._get_classic_gsm8k_problems())

        return problems

    def _extract_numeric_answer(self, text: str) -> float:
        """Extract the numeric answer from ground truth text"""
        # Look for dollar amounts like $2,775.00 or $2775
        dollar_match = re.search(r"\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", text)
        if dollar_match:
            return float(dollar_match.group(1).replace(",", ""))

        # Look for standalone numbers like 124 meters, 34 minutes, etc.
        number_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:meters|hours|minutes|eggs|dollars)?", text
        )
        if number_match:
            return float(number_match.group(1))

        # Look for any number in the text
        any_number = re.findall(r"\d+(?:\.\d+)?", text)
        if any_number:
            return float(any_number[0])

        return 0.0

    def _get_classic_gsm8k_problems(self) -> List[MathProblem]:
        """Get classic GSM8K-style problems"""
        return [
            MathProblem(
                id="gsm8k_janet_eggs",
                question="Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                ground_truth="18 dollars. Solution: 16 - 3 - 4 = 9 eggs. 9 × $2 = $18",
                expected_answer_value=18.0,
                category="arithmetic",
                difficulty="easy",
            ),
            MathProblem(
                id="gsm8k_swimming_pool",
                question="A swimming pool is 50 meters long and 25 meters wide. If the water depth is 2 meters, how many cubic meters of water does the pool contain?",
                ground_truth="2500 cubic meters. Solution: Volume = length × width × depth = 50 × 25 × 2 = 2500",
                expected_answer_value=2500.0,
                category="geometry",
                difficulty="easy",
            ),
            MathProblem(
                id="gsm8k_fruit_stand",
                question="At a fruit stand, apples cost $3 per pound and oranges cost $2 per pound. If John buys 5 pounds of apples and 8 pounds of oranges, how much does he spend in total?",
                ground_truth="31 dollars. Solution: Apples = 5 × $3 = $15. Oranges = 8 × $2 = $16. Total = $15 + $16 = $31",
                expected_answer_value=31.0,
                category="arithmetic",
                difficulty="easy",
            ),
            MathProblem(
                id="gsm8k_bus_problem",
                question="A bus starts with 45 passengers. At the first stop, 12 people get off and 8 people get on. At the second stop, 15 people get off and 20 people get on. How many passengers are on the bus after the second stop?",
                ground_truth="46 passengers. Solution: 45 - 12 + 8 - 15 + 20 = 46",
                expected_answer_value=46.0,
                category="arithmetic",
                difficulty="medium",
            ),
            MathProblem(
                id="gsm8k_reading_challenge",
                question="Sarah wants to read a 480-page book in 8 days. If she reads the same number of pages each day, how many pages must she read per day?",
                ground_truth="60 pages per day. Solution: 480 ÷ 8 = 60",
                expected_answer_value=60.0,
                category="arithmetic",
                difficulty="easy",
            ),
        ]

    async def run_baseline(self, problem: MathProblem) -> BenchmarkResult:
        """Run baseline: Direct LLM call"""
        start_time = time.time()

        try:
            # Simple direct prompt
            prompt = f"Solve this math problem and provide the numeric answer:\n\n{problem.question}\n\nProvide your answer as a number."

            # Direct LLM call using correct method
            response_dict = await self.llm_bridge.generate_response(prompt)
            response = response_dict.get("content", "")
            response_time = time.time() - start_time

            # Extract numeric answer
            extracted = self._extract_numeric_from_response(response)

            # Check correctness (with tolerance for floating point)
            is_correct = abs(extracted - problem.expected_answer_value) < 0.01

            return BenchmarkResult(
                problem_id=problem.id,
                system="baseline",
                question=problem.question,
                response=response,
                extracted_answer=extracted,
                correct_answer=problem.expected_answer_value,
                is_correct=is_correct,
                response_time=response_time,
            )

        except Exception as e:
            return BenchmarkResult(
                problem_id=problem.id,
                system="baseline",
                question=problem.question,
                response="",
                extracted_answer=0.0,
                correct_answer=problem.expected_answer_value,
                is_correct=False,
                response_time=time.time() - start_time,
                error=str(e),
            )

    async def run_conjecture(self, problem: MathProblem) -> BenchmarkResult:
        """Run Conjecture: Full claim-based reasoning"""
        start_time = time.time()

        try:
            # Use Conjecture's request processing
            prompt = f"Solve this math problem step-by-step:\n\n{problem.question}\n\nProvide detailed reasoning and the numeric answer."

            # Process through Conjecture endpoint
            request = {"operation": "analyze_claim", "content": prompt}
            result = await self.endpoint.process_request(request)
            response_time = time.time() - start_time

            # Extract response text
            response = (
                result.get("response", "") if isinstance(result, dict) else str(result)
            )

            # Extract numeric answer
            extracted = self._extract_numeric_from_response(response)

            # Check correctness
            is_correct = abs(extracted - problem.expected_answer_value) < 0.01

            return BenchmarkResult(
                problem_id=problem.id,
                system="conjecture",
                question=problem.question,
                response=response,
                extracted_answer=extracted,
                correct_answer=problem.expected_answer_value,
                is_correct=is_correct,
                response_time=response_time,
            )

        except Exception as e:
            return BenchmarkResult(
                problem_id=problem.id,
                system="conjecture",
                question=problem.question,
                response="",
                extracted_answer=0.0,
                correct_answer=problem.expected_answer_value,
                is_correct=False,
                response_time=time.time() - start_time,
                error=str(e),
            )

    def _extract_numeric_from_response(self, text: str) -> float:
        """Extract numeric answer from model response"""
        # Look for "answer is X" or "= X" patterns
        patterns = [
            r"answer is (\d+(?:\.\d+)?)",
            r"= (\d+(?:\.\d+)?)",
            r"\$(\d+(?:\.\d+)?)",
            r"total[:\s]+(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:dollars|meters|cubic meters|eggs|pages|passengers)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return float(match.group(1))

        # Fallback: get last number in response
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if numbers:
            return float(numbers[-1])

        return 0.0

    async def run_benchmark(self, num_problems: int = None) -> Dict[str, Any]:
        """Run complete benchmark comparison"""
        print("=" * 60)
        print("CORE HYPOTHESIS VALIDATION")
        print("=" * 60)
        print(f"Hypothesis: Conjecture improves intelligence & truthfulness")
        print(f"Benchmark: GSM8K-style math problems")
        print(f"Model: {self.config.get_primary_provider().name}")
        print("=" * 60)
        print()

        # Load problems
        all_problems = self.load_math_problems()
        if num_problems:
            all_problems = all_problems[:num_problems]

        print(f"Loaded {len(all_problems)} problems")
        print()

        baseline_results = []
        conjecture_results = []

        # Run benchmarks
        for i, problem in enumerate(all_problems, 1):
            print(f"[{i}/{len(all_problems)}] {problem.id}")
            print(f"  Question: {problem.question[:80]}...")

            # Baseline
            print(f"  Running baseline...", end="", flush=True)
            baseline_result = await self.run_baseline(problem)
            baseline_results.append(baseline_result)
            status = "PASS" if baseline_result.is_correct else "FAIL"
            print(f" {status} ({baseline_result.response_time:.2f}s)")

            # Conjecture
            print(f"  Running Conjecture...", end="", flush=True)
            conjecture_result = await self.run_conjecture(problem)
            conjecture_results.append(conjecture_result)
            status = "PASS" if conjecture_result.is_correct else "FAIL"
            print(f" {status} ({conjecture_result.response_time:.2f}s)")

            print()

        # Calculate metrics
        baseline_accuracy = sum(r.is_correct for r in baseline_results) / len(
            baseline_results
        )
        conjecture_accuracy = sum(r.is_correct for r in conjecture_results) / len(
            conjecture_results
        )

        baseline_avg_time = statistics.mean(r.response_time for r in baseline_results)
        conjecture_avg_time = statistics.mean(
            r.response_time for r in conjecture_results
        )

        improvement = conjecture_accuracy - baseline_accuracy
        improvement_pct = (
            (improvement / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
        )

        latency_overhead = conjecture_avg_time - baseline_avg_time
        latency_overhead_pct = (
            (latency_overhead / baseline_avg_time * 100) if baseline_avg_time > 0 else 0
        )

        # Build summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model": self.config.get_primary_provider().name,
            "num_problems": len(all_problems),
            "baseline": {
                "accuracy": baseline_accuracy,
                "correct": sum(r.is_correct for r in baseline_results),
                "total": len(baseline_results),
                "avg_latency": baseline_avg_time,
                "results": [asdict(r) for r in baseline_results],
            },
            "conjecture": {
                "accuracy": conjecture_accuracy,
                "correct": sum(r.is_correct for r in conjecture_results),
                "total": len(conjecture_results),
                "avg_latency": conjecture_avg_time,
                "results": [asdict(r) for r in conjecture_results],
            },
            "comparison": {
                "accuracy_improvement": improvement,
                "accuracy_improvement_pct": improvement_pct,
                "latency_overhead": latency_overhead,
                "latency_overhead_pct": latency_overhead_pct,
                "hypothesis_supported": improvement > 0.05,  # 5% improvement threshold
            },
        }

        # Print results
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(
            f"Baseline:    {baseline_accuracy:.1%} accuracy ({baseline_avg_time:.2f}s avg)"
        )
        print(
            f"Conjecture:  {conjecture_accuracy:.1%} accuracy ({conjecture_avg_time:.2f}s avg)"
        )
        print()
        print(f"Improvement: {improvement:+.1%} ({improvement_pct:+.1f}%)")
        print(f"Latency:     {latency_overhead:+.2f}s ({latency_overhead_pct:+.1f}%)")
        print()

        if summary["comparison"]["hypothesis_supported"]:
            print("HYPOTHESIS SUPPORTED")
            print("  Conjecture shows meaningful accuracy improvement (>5%)")
        else:
            print("HYPOTHESIS NOT SUPPORTED")
            print("  Conjecture does not show meaningful improvement")
            print("  Root cause analysis needed")
        print("=" * 60)

        return summary


async def main():
    """Main execution"""
    validator = CoreHypothesisValidator()

    # Run with all available problems (8 total)
    summary = await validator.run_benchmark(num_problems=8)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"hypothesis_validation_{timestamp}.json"

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Results saved to: {result_file}")

    return summary


if __name__ == "__main__":
    asyncio.run(main())
