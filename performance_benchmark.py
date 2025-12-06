#!/usr/bin/env python3
"""
Performance Benchmark for Tiny LLM Prompt Engineering Optimizer

Comprehensive benchmarking to validate performance improvements and
measure effectiveness of optimization strategies.
"""

import asyncio
import time
import json
import statistics
from datetime import datetime
from typing import Dict, List, Any, Tuple
import random

from test_optimizer_standalone import (
    SimplePromptOptimizer,
    TinyModelCapabilityProfiler,
    TaskDescription,
    TaskComplexity,
    OptimizationStrategy,
    TinyModelCapabilities
)


class PerformanceBenchmark:
    """Performance benchmark for prompt optimizer"""

    def __init__(self):
        """Initialize benchmark"""
        self.optimizer = SimplePromptOptimizer()
        self.profiler = TinyModelCapabilityProfiler()
        self.benchmark_results = []

    def create_benchmark_scenarios(self) -> List[Dict[str, Any]]:
        """Create comprehensive benchmark scenarios"""

        scenarios = []

        # Simple tasks
        simple_inputs = [
            "Extract the date from: The meeting is scheduled for December 15, 2024.",
            "Classify this sentiment: I love this product!",
            "Find the number: Call me at 555-123-4567.",
            "Identify the person: John Smith is the project manager.",
            "Extract location: The event takes place in New York City."
        ]

        for i, input_text in enumerate(simple_inputs):
            scenarios.append({
                "name": f"Simple_Extraction_{i+1}",
                "task_type": "extraction",
                "complexity": TaskComplexity.SIMPLE,
                "input": input_text,
                "context": ["General context"],
                "expected_tokens": 50,
                "expected_time_ms": 500
            })

        # Moderate complexity tasks
        moderate_inputs = [
            "Analyze this sales data: Q1 revenue increased by 15% while costs decreased by 8%.",
            "Compare these products: Product A has better features but Product B is cheaper.",
            "Summarize this feedback: The user interface is intuitive but the performance could be improved.",
            "Evaluate this argument: The proposal will increase efficiency but requires significant investment.",
            'Translate this request: "Je veux acheter un ordinateur portable" (French).'
        ]

        for i, input_text in enumerate(moderate_inputs):
            scenarios.append({
                "name": f"Moderate_Analysis_{i+1}",
                "task_type": "analysis",
                "complexity": TaskComplexity.MODERATE,
                "input": input_text,
                "context": ["Business context", "Data analysis context", "Market research"],
                "expected_tokens": 150,
                "expected_time_ms": 1500
            })

        # Complex tasks
        complex_inputs = [
            "Research the impact of artificial intelligence on employment in creative industries.",
            "Analyze the long-term effects of climate change on global food security.",
            "Evaluate the ethical implications of genetic engineering in human medicine.",
            "Compare different renewable energy sources for urban infrastructure development.",
            "Investigate the relationship between social media usage and mental health in adolescents."
        ]

        for i, input_text in enumerate(complex_inputs):
            scenarios.append({
                "name": f"Complex_Research_{i+1}",
                "task_type": "research",
                "complexity": TaskComplexity.COMPLEX,
                "input": input_text,
                "context": [
                    "Academic research papers",
                    "Industry reports",
                    "Government statistics",
                    "Expert opinions",
                    "Case studies"
                ],
                "expected_tokens": 300,
                "expected_time_ms": 3000
            })

        return scenarios

    async def run_optimization_benchmark(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run optimization benchmark on all scenarios"""

        print(f"Running benchmark on {len(scenarios)} scenarios...")

        all_results = []
        models = ["granite-tiny", "llama-3.2-1b", "phi-3-mini"]
        strategies = [OptimizationStrategy.MINIMAL_TOKENS, OptimizationStrategy.ADAPTIVE]

        total_start_time = time.time()

        for scenario in scenarios:
            scenario_results = {
                "scenario_name": scenario["name"],
                "task_type": scenario["task_type"],
                "complexity": scenario["complexity"].value,
                "results": []
            }

            for model in models:
                for strategy in strategies:
                    result = await self.benchmark_single_optimization(
                        scenario, model, strategy
                    )
                    scenario_results["results"].append(result)

            all_results.append(scenario_results)

        total_time = (time.time() - total_start_time) * 1000

        # Calculate overall metrics
        benchmark_summary = self.calculate_benchmark_summary(all_results, total_time)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_scenarios": len(scenarios),
            "total_time_ms": total_time,
            "scenarios": all_results,
            "summary": benchmark_summary
        }

    async def benchmark_single_optimization(
        self,
        scenario: Dict[str, Any],
        model: str,
        strategy: OptimizationStrategy
    ) -> Dict[str, Any]:
        """Benchmark single optimization run"""

        task = TaskDescription(
            task_type=scenario["task_type"],
            complexity=scenario["complexity"],
            required_inputs=[scenario["input"]],
            expected_output_format="structured"
        )

        start_time = time.time()

        try:
            # Run optimization
            result = await self.optimizer.optimize_prompt(
                task=task,
                context_items=scenario["context"],
                model_name=model,
                optimization_strategy=strategy
            )

            optimization_time = (time.time() - start_time) * 1000

            # Calculate metrics
            original_tokens = len(result.original_prompt.split())
            optimized_tokens = len(result.optimized_prompt.split())
            token_efficiency = optimized_tokens / original_tokens if original_tokens > 0 else 1.0

            # Calculate performance score
            performance_score = self.calculate_performance_score(
                result, scenario, model, optimization_time
            )

            return {
                "model": model,
                "strategy": strategy.value,
                "success": True,
                "optimization_time_ms": optimization_time,
                "original_tokens": original_tokens,
                "optimized_tokens": optimized_tokens,
                "token_reduction": result.token_reduction,
                "token_efficiency": token_efficiency,
                "optimization_score": result.optimization_score,
                "performance_score": performance_score,
                "quality_metrics": {
                    "structure_appropriate": self.check_structure_appropriateness(
                        result.optimized_prompt, model
                    ),
                    "clarity_score": self.calculate_clarity_score(result.optimized_prompt),
                    "complexity_match": self.check_complexity_match(
                        scenario["complexity"], result.optimized_prompt
                    )
                }
            }

        except Exception as e:
            return {
                "model": model,
                "strategy": strategy.value,
                "success": False,
                "error": str(e),
                "optimization_time_ms": (time.time() - start_time) * 1000,
                "performance_score": 0.0
            }

    def calculate_performance_score(
        self,
        result,
        scenario: Dict[str, Any],
        model: str,
        optimization_time: float
    ) -> float:
        """Calculate comprehensive performance score"""

        score = 0.0

        # Optimization score (30%)
        score += result.optimization_score * 0.3

        # Token efficiency (25%)
        token_efficiency = result.optimized_tokens / max(result.original_tokens, 1)
        efficiency_score = max(0, 1.0 - abs(1.0 - token_efficiency))  # Prefer optimal reduction
        score += efficiency_score * 0.25

        # Speed performance (20%)
        expected_time = scenario["expected_time_ms"]
        time_score = max(0, 1.0 - (optimization_time / expected_time))
        score += time_score * 0.2

        # Model compatibility (15%)
        model_capabilities = self.profiler.get_model_capabilities(model)
        if model_capabilities:
            structure_match = (
                (model_capabilities.preferred_structure in result.optimized_prompt.lower()) or
                (model_capabilities.preferred_structure == "plain" and
                 "<" not in result.optimized_prompt and "{" not in result.optimized_prompt)
            )
            model_score = 1.0 if structure_match else 0.5
        else:
            model_score = 0.5
        score += model_score * 0.15

        # Task appropriateness (10%)
        expected_tokens = scenario["expected_tokens"]
        token_appropriateness = 1.0 - abs(result.optimized_tokens - expected_tokens) / expected_tokens
        token_appropriateness = max(0, min(1.0, token_appropriateness))
        score += token_appropriateness * 0.1

        return min(1.0, score)

    def check_structure_appropriateness(self, prompt: str, model: str) -> float:
        """Check if prompt structure is appropriate for model"""
        model_capabilities = self.profiler.get_model_capabilities(model)
        if not model_capabilities:
            return 0.5

        if model_capabilities.preferred_structure == "xml":
            return 1.0 if "<" in prompt else 0.3
        elif model_capabilities.preferred_structure == "json":
            return 1.0 if "{" in prompt else 0.3
        else:  # plain
            return 1.0 if "<" not in prompt and "{" not in prompt else 0.7

    def calculate_clarity_score(self, prompt: str) -> float:
        """Calculate prompt clarity score"""
        score = 0.0

        # Check for clear task definition
        if any(keyword in prompt.lower() for keyword in ["task", "input", "extract", "analyze"]):
            score += 0.4

        # Check for structure
        if any(structure in prompt for structure in [":", "\n", "1.", "2."]):
            score += 0.3

        # Check length (not too short, not too long)
        word_count = len(prompt.split())
        if 20 <= word_count <= 200:
            score += 0.3
        elif word_count < 20:
            score += 0.1
        else:
            score += 0.2

        return min(1.0, score)

    def check_complexity_match(self, complexity: TaskComplexity, prompt: str) -> float:
        """Check if prompt complexity matches task complexity"""
        word_count = len(prompt.split())

        if complexity == TaskComplexity.SIMPLE:
            return 1.0 if word_count <= 100 else 0.5
        elif complexity == TaskComplexity.MODERATE:
            return 1.0 if 50 <= word_count <= 200 else 0.7
        elif complexity == TaskComplexity.COMPLEX:
            return 1.0 if word_count >= 100 else 0.6
        else:
            return 0.5

    def calculate_benchmark_summary(
        self,
        all_results: List[Dict[str, Any]],
        total_time: float
    ) -> Dict[str, Any]:
        """Calculate comprehensive benchmark summary"""

        # Flatten all individual results
        individual_results = []
        for scenario in all_results:
            individual_results.extend(scenario["results"])

        # Filter successful results
        successful_results = [r for r in individual_results if r["success"]]

        if not successful_results:
            return {
                "total_tests": len(individual_results),
                "successful_tests": 0,
                "success_rate": 0.0,
                "error": "All tests failed"
            }

        # Calculate statistics
        performance_scores = [r["performance_score"] for r in successful_results]
        optimization_times = [r["optimization_time_ms"] for r in successful_results]
        token_efficiencies = [r["token_efficiency"] for r in successful_results]
        optimization_scores = [r["optimization_score"] for r in successful_results]

        # Model-specific analysis
        model_stats = {}
        for model in ["granite-tiny", "llama-3.2-1b", "phi-3-mini"]:
            model_results = [r for r in successful_results if r["model"] == model]
            if model_results:
                model_scores = [r["performance_score"] for r in model_results]
                model_stats[model] = {
                    "count": len(model_results),
                    "avg_score": statistics.mean(model_scores),
                    "max_score": max(model_scores),
                    "min_score": min(model_scores)
                }

        # Strategy-specific analysis
        strategy_stats = {}
        for strategy in ["minimal_tokens", "adaptive"]:
            strategy_results = [r for r in successful_results if r["strategy"] == strategy]
            if strategy_results:
                strategy_scores = [r["performance_score"] for r in strategy_results]
                strategy_stats[strategy] = {
                    "count": len(strategy_results),
                    "avg_score": statistics.mean(strategy_scores),
                    "avg_token_efficiency": statistics.mean([r["token_efficiency"] for r in strategy_results])
                }

        # Complexity-specific analysis
        complexity_stats = {}
        for complexity in ["simple", "moderate", "complex"]:
            complexity_results = [
                r for scenario in all_results
                if scenario["complexity"] == complexity
                for r in scenario["results"]
                if r["success"]
            ]
            if complexity_results:
                complexity_scores = [r["performance_score"] for r in complexity_results]
                complexity_stats[complexity] = {
                    "count": len(complexity_results),
                    "avg_score": statistics.mean(complexity_scores),
                    "avg_tokens": statistics.mean([r["optimized_tokens"] for r in complexity_results])
                }

        return {
            "total_tests": len(individual_results),
            "successful_tests": len(successful_results),
            "success_rate": len(successful_results) / len(individual_results),
            "total_benchmark_time_ms": total_time,

            "performance_metrics": {
                "avg_performance_score": statistics.mean(performance_scores),
                "max_performance_score": max(performance_scores),
                "min_performance_score": min(performance_scores),
                "median_performance_score": statistics.median(performance_scores),
                "std_dev_performance": statistics.stdev(performance_scores) if len(performance_scores) > 1 else 0
            },

            "efficiency_metrics": {
                "avg_optimization_time_ms": statistics.mean(optimization_times),
                "avg_token_efficiency": statistics.mean(token_efficiencies),
                "avg_optimization_score": statistics.mean(optimization_scores),
                "total_tokens_saved": sum(r["token_reduction"] for r in successful_results)
            },

            "model_performance": model_stats,
            "strategy_performance": strategy_stats,
            "complexity_performance": complexity_stats,

            "quality_metrics": {
                "avg_structure_score": statistics.mean([
                    r["quality_metrics"]["structure_appropriate"] for r in successful_results
                ]),
                "avg_clarity_score": statistics.mean([
                    r["quality_metrics"]["clarity_score"] for r in successful_results
                ]),
                "avg_complexity_match": statistics.mean([
                    r["quality_metrics"]["complexity_match"] for r in successful_results
                ])
            }
        }

    def print_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Print formatted benchmark results"""

        print("\n" + "="*70)
        print("PROMPT ENGINEERING OPTIMIZER BENCHMARK RESULTS")
        print("="*70)

        summary = results["summary"]

        print(f"\nBENCHMARK OVERVIEW:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Successful: {summary['successful_tests']}")
        print(f"  Success Rate: {summary['success_rate']:.1%}")
        print(f"  Total Time: {summary['total_benchmark_time_ms']:.1f}ms")

        print(f"\nPERFORMANCE METRICS:")
        perf = summary["performance_metrics"]
        print(f"  Average Score: {perf['avg_performance_score']:.3f}")
        print(f"  Max Score: {perf['max_performance_score']:.3f}")
        print(f"  Min Score: {perf['min_performance_score']:.3f}")
        print(f"  Median Score: {perf['median_performance_score']:.3f}")
        print(f"  Std Dev: {perf['std_dev_performance']:.3f}")

        print(f"\nEFFICIENCY METRICS:")
        eff = summary["efficiency_metrics"]
        print(f"  Avg Optimization Time: {eff['avg_optimization_time_ms']:.1f}ms")
        print(f"  Avg Token Efficiency: {eff['avg_token_efficiency']:.3f}")
        print(f"  Avg Optimization Score: {eff['avg_optimization_score']:.3f}")
        print(f"  Total Tokens Saved: {eff['total_tokens_saved']}")

        print(f"\nMODEL PERFORMANCE:")
        for model, stats in summary["model_performance"].items():
            print(f"  {model}:")
            print(f"    Count: {stats['count']}")
            print(f"    Avg Score: {stats['avg_score']:.3f}")
            print(f"    Score Range: {stats['min_score']:.3f} - {stats['max_score']:.3f}")

        print(f"\nSTRATEGY PERFORMANCE:")
        for strategy, stats in summary["strategy_performance"].items():
            print(f"  {strategy}:")
            print(f"    Count: {stats['count']}")
            print(f"    Avg Score: {stats['avg_score']:.3f}")
            print(f"    Avg Token Efficiency: {stats['avg_token_efficiency']:.3f}")

        print(f"\nCOMPLEXITY PERFORMANCE:")
        for complexity, stats in summary["complexity_performance"].items():
            print(f"  {complexity.title()}:")
            print(f"    Count: {stats['count']}")
            print(f"    Avg Score: {stats['avg_score']:.3f}")
            print(f"    Avg Tokens: {stats['avg_tokens']:.1f}")

        print(f"\nQUALITY METRICS:")
        quality = summary["quality_metrics"]
        print(f"  Avg Structure Score: {quality['avg_structure_score']:.3f}")
        print(f"  Avg Clarity Score: {quality['avg_clarity_score']:.3f}")
        print(f"  Avg Complexity Match: {quality['avg_complexity_match']:.3f}")

        # Print top and worst performing scenarios
        print(f"\nTOP PERFORMING SCENARIOS:")
        scenario_scores = []
        for scenario in results["scenarios"]:
            for result in scenario["results"]:
                if result["success"]:
                    scenario_scores.append({
                        "name": f"{scenario['scenario_name']} - {result['model']} - {result['strategy']}",
                        "score": result["performance_score"]
                    })

        scenario_scores.sort(key=lambda x: x["score"], reverse=True)
        for i, item in enumerate(scenario_scores[:5]):
            print(f"  {i+1}. {item['name']}: {item['score']:.3f}")

        print(f"\nWORST PERFORMING SCENARIOS:")
        for i, item in enumerate(scenario_scores[-5:]):
            print(f"  {len(scenario_scores)-4+i}. {item['name']}: {item['score']:.3f}")

        # Performance assessment
        avg_score = summary["performance_metrics"]["avg_performance_score"]
        success_rate = summary["success_rate"]

        print(f"\nPERFORMANCE ASSESSMENT:")
        if avg_score >= 0.8 and success_rate >= 0.9:
            print("  [EXCELLENT] Outstanding performance across all metrics")
        elif avg_score >= 0.7 and success_rate >= 0.8:
            print("  [GOOD] Strong performance with room for improvement")
        elif avg_score >= 0.6 and success_rate >= 0.7:
            print("  [ACCEPTABLE] Functional but needs optimization")
        else:
            print("  [NEEDS IMPROVEMENT] Significant performance issues detected")

        print("\n" + "="*70)

    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save benchmark results to file"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prompt_optimizer_benchmark_{timestamp}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nBenchmark results saved to: {filename}")
            return filename
        except Exception as e:
            print(f"Failed to save results: {e}")
            return ""


async def run_comprehensive_benchmark():
    """Run comprehensive performance benchmark"""

    print("Starting comprehensive prompt optimizer benchmark...")

    benchmark = PerformanceBenchmark()

    # Create scenarios
    scenarios = benchmark.create_benchmark_scenarios()
    print(f"Created {len(scenarios)} benchmark scenarios")

    # Run benchmark
    results = await benchmark.run_optimization_benchmark(scenarios)

    # Print results
    benchmark.print_benchmark_results(results)

    # Save results
    benchmark.save_results(results)

    return results


if __name__ == "__main__":
    asyncio.run(run_comprehensive_benchmark())