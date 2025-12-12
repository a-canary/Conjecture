#!/usr/bin/env python3
"""
Quick AIME 2025 Test - Fixed Unicode Issues
Tests first few problems to get immediate results
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.benchmark_framework import AIME25Benchmark
from benchmarking.lm_studio_integration import granite_tiny_direct, granite_tiny_direct_conjecture

async def run_quick_aime_test():
    """Run quick test of first 5 AIME problems"""
    print("=" * 80)
    print("QUICK AIME 2025 TEST - First 5 Problems")
    print("=" * 80)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize benchmark
    aime_benchmark = AIME25Benchmark()

    # Load tasks
    print("Loading AIME 2025 problems...")
    tasks = await aime_benchmark.load_tasks()
    test_tasks = tasks[:5]  # Test first 5 problems
    print(f"Testing with first {len(test_tasks)} problems")
    print()

    # Test models
    models_to_test = {
        "GraniteTiny-Direct": granite_tiny_direct,
        "GraniteTiny+Conjecture": granite_tiny_direct_conjecture,
    }

    results = {}

    for model_name, model_func in models_to_test.items():
        print(f"Testing {model_name}...")
        print("-" * 60)

        model_results = []
        start_time = time.time()

        for i, task in enumerate(test_tasks):
            try:
                print(f"Problem {task.task_id} ({i+1}/{len(test_tasks)})...", end=" ", flush=True)

                response_start = time.time()
                response = await model_func(task.prompt)
                execution_time = time.time() - response_start

                correct = aime_benchmark.evaluate_response(task, response)

                result = {
                    "task_id": task.task_id,
                    "expected": task.expected_answer,
                    "correct": correct,
                    "time": execution_time
                }

                model_results.append(result)
                print(f"{'CORRECT' if correct else 'INCORRECT'} ({execution_time:.1f}s)")

            except Exception as e:
                print(f"ERROR: {str(e)[:50]}")
                result = {
                    "task_id": task.task_id,
                    "expected": task.expected_answer,
                    "correct": False,
                    "time": 0.0,
                    "error": str(e)
                }
                model_results.append(result)

        total_time = time.time() - start_time
        correct_count = sum(1 for r in model_results if r["correct"])
        accuracy = correct_count / len(model_results)
        avg_time = sum(r["time"] for r in model_results) / len(model_results)

        summary = {
            "model_name": model_name,
            "total_tasks": len(model_results),
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "average_time": avg_time,
            "total_time": total_time,
            "results": model_results
        }

        results[model_name] = summary

        print(f"\n{model_name} Summary:")
        print(f"  Score: {correct_count}/{len(model_results)} = {accuracy:.1%}")
        print(f"  Avg time: {avg_time:.1f}s per problem")
        print()

    # Comparison
    print("PERFORMANCE COMPARISON")
    print("=" * 50)

    if len(results) > 1:
        model_names = list(results.keys())
        baseline = results[model_names[0]]
        enhanced = results[model_names[1]]

        print(f"{model_names[0]}: {baseline['accuracy']:.1%} ({baseline['correct_answers']}/{baseline['total_tasks']})")
        print(f"{model_names[1]}: {enhanced['accuracy']:.1%} ({enhanced['correct_answers']}/{enhanced['total_tasks']})")

        improvement = enhanced['accuracy'] - baseline['accuracy']
        print(f"Conjecture improvement: {improvement:+.1%}")

        # Show problem-by-problem comparison
        print(f"\nProblem-by-problem comparison:")
        for i in range(len(baseline['results'])):
            baseline_result = baseline['results'][i]
            enhanced_result = enhanced['results'][i]

            baseline_status = "CORRECT" if baseline_result['correct'] else "INCORRECT"
            enhanced_status = "CORRECT" if enhanced_result['correct'] else "INCORRECT"

            status_change = ""
            if baseline_result['correct'] != enhanced_result['correct']:
                if enhanced_result['correct']:
                    status_change = " (IMPROVED)"
                else:
                    status_change = " (REGRESSED)"

            print(f"  {baseline_result['task_id']}: {baseline_status} -> {enhanced_status}{status_change}")

if __name__ == "__main__":
    asyncio.run(run_quick_aime_test())