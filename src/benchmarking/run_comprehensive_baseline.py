#!/usr/bin/env python3
"""
Comprehensive Baseline Runner
Establishes baseline performance across all available benchmarks and models
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.benchmark_framework import BenchmarkRunner, AIME25Benchmark, GPQABenchmark, SWEVerifiedBenchmark, LiveCodeBenchBenchmark
from benchmarking.model_integration import granite_tiny_model, gpt_oss_20b_model, glm_46_model
from benchmarking.lm_studio_integration import granite_tiny_direct

# Simple test cases for faster testing
SIMPLE_TEST_CASES = [
    {
        "task_id": "math_001",
        "prompt": "What is 17 × 24?",
        "expected_answer": "408",
        "metadata": {"category": "basic_math", "difficulty": "easy"}
    },
    {
        "task_id": "logic_001",
        "prompt": "If all cats are animals and some animals are pets, can we conclude that some cats are pets? Answer yes or no.",
        "expected_answer": "no",
        "metadata": {"category": "logic", "difficulty": "easy"}
    },
    {
        "task_id": "reasoning_001",
        "prompt": "Sarah is 5 years older than Tom. In 3 years, Sarah will be twice as old as Tom was 2 years ago. How old is Sarah now?",
        "expected_answer": "13",
        "metadata": {"category": "algebra", "difficulty": "medium"}
    }
]

class SimpleBenchmark:
    """Simple benchmark for quick baseline testing"""

    def __init__(self):
        self.name = "Simple"
        self.tasks = SIMPLE_TEST_CASES

    async def load_tasks(self):
        return self.tasks

    def evaluate_response(self, task, response):
        """Check if expected answer is in response"""
        response_lower = response.lower()
        expected_lower = task.expected_answer.lower()
        return expected_lower in response_lower

async def run_comprehensive_baseline():
    """Run comprehensive baseline tests"""
    print("=" * 80)
    print("COMPREHENSIVE BASELINE EVALUATION")
    print("=" * 80)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test different configurations
    test_configurations = [
        {
            "name": "Quick Baseline (3 simple problems)",
            "models": {
                "GraniteTiny-LMStudio": granite_tiny_direct,
            },
            "benchmarks": {
                "Simple": SimpleBenchmark(),
            }
        },
        {
            "name": "Full Infrastructure Test",
            "models": {
                "GraniteTiny-Conj": granite_tiny_model,
            },
            "benchmarks": {
                "Simple": SimpleBenchmark(),
                "AIME25": AIME25Benchmark(),  # Will use sample tasks
            }
        }
    ]

    all_results = {}

    for config in test_configurations:
        print(f"\n{'='*60}")
        print(f"RUNNING: {config['name']}")
        print(f"{'='*60}")

        runner = BenchmarkRunner()

        # Override with our specific models and benchmarks
        models_to_test = config["models"]
        benchmarks_to_test = config["benchmarks"]

        print(f"Models: {list(models_to_test.keys())}")
        print(f"Benchmarks: {list(benchmarks_to_test.keys())}")
        print()

        for model_name, model_func in models_to_test.items():
            model_results = []

            for bench_name, benchmark in benchmarks_to_test.items():
                print(f"Running {bench_name} with {model_name}...")

                try:
                    # Run direct version
                    result_direct = await benchmark.run_benchmark(model_func, model_name, using_conjecture=False)
                    model_results.append(result_direct)

                    print(f"  Direct: {result_direct.correct_answers}/{result_direct.total_tasks} = {result_direct.accuracy:.1%} ({result_direct.average_time:.1f}s avg)")

                    # Try Conjecture version if available
                    if hasattr(model_func, '__name__') and 'conj' in model_func.__name__.lower():
                        result_conjecture = await benchmark.run_benchmark(model_func, f"{model_name}+Conjecture", using_conjecture=True)
                        model_results.append(result_conjecture)

                        print(f"  Conjecture: {result_conjecture.correct_answers}/{result_conjecture.total_tasks} = {result_conjecture.accuracy:.1%} ({result_conjecture.average_time:.1f}s avg)")

                        # Show comparison
                        improvement = result_conjecture.accuracy - result_direct.accuracy
                        if improvement > 0:
                            print(f"  -> CONJECTURE IMPROVES by {improvement:.1%}")
                        elif improvement < 0:
                            print(f"  -> CONJECTURE HURTS by {abs(improvement):.1%}")
                        else:
                            print(f"  -> NO DIFFERENCE")

                except Exception as e:
                    print(f"  ERROR: {str(e)}")
                    continue

                print()

            all_results[model_name] = model_results

    # Summary
    print(f"\n{'='*80}")
    print("BASELINE SUMMARY")
    print(f"{'='*80}")

    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        for result in results:
            print(f"  {result.benchmark_name}: {result.accuracy:.1%} accuracy, {result.average_time:.1f}s avg")

    # Generate baseline report
    generate_baseline_report(all_results)

def generate_baseline_report(results):
    """Generate baseline report"""
    report_file = Path(__file__).parent / "baseline_report.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Baseline Evaluation Report\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")
        f.write("| Model | Benchmark | Accuracy | Avg Time (s) | Conjecture? |\n")
        f.write("|-------|-----------|----------|--------------|------------|\n")

        for model_name, model_results in results.items():
            for result in model_results:
                conjecture_str = "Yes" if result.using_conjecture else "No"
                f.write(f"| {result.model_name} | {result.benchmark_name} | {result.accuracy:.1%} | {result.average_time:.2f} | {conjecture_str} |\n")

        f.write("\n## Key Findings\n\n")

        # Analyze results
        conjecture_results = [r for r in sum(results.values(), []) if r.using_conjecture]
        direct_results = [r for r in sum(results.values(), []) if not r.using_conjecture]

        if conjecture_results and direct_results:
            conj_avg = sum(r.accuracy for r in conjecture_results) / len(conjecture_results)
            direct_avg = sum(r.accuracy for r in direct_results) / len(direct_results)

            f.write(f"- **Direct models average accuracy**: {direct_avg:.1%}\n")
            f.write(f"- **Conjecture-enhanced average accuracy**: {conj_avg:.1%}\n")
            f.write(f"- **Conjecture impact**: {conj_avg - direct_avg:+.1%}\n\n")

        f.write("## Recommendations\n\n")
        f.write("1. Focus on prompt engineering that actually improves accuracy\n")
        f.write("2. Test with a wider range of problem difficulties\n")
        f.write("3. Optimize for speed without sacrificing accuracy\n")
        f.write("4. Consider model-specific Conjecture strategies\n")

    print(f"\nBaseline report saved to: {report_file}")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_baseline())