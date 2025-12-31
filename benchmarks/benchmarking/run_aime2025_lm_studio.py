#!/usr/bin/env python3
"""
AIME 2025 Benchmark Runner for GraniteTiny via LM Studio
Direct LM Studio integration for AIME 2025 mathematical reasoning evaluation
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

async def run_aime2025_benchmark():
    """Run AIME 2025 benchmark with GraniteTiny via LM Studio"""
    print("=" * 80)
    print("AIME 2025 BENCHMARK EVALUATION - LM Studio Edition")
    print("=" * 80)
    print(f"Starting evaluation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize benchmark
    aime_benchmark = AIME25Benchmark()

    # Load tasks
    print("Loading AIME 2025 problems...")
    tasks = await aime_benchmark.load_tasks()
    print(f"Loaded {len(tasks)} AIME 2025 problems")
    print()

    # Test models
    models_to_test = {
        "GraniteTiny-Direct": granite_tiny_direct,
        "GraniteTiny+Conjecture": granite_tiny_direct_conjecture,
    }

    results = {}
    detailed_results = []

    for model_name, model_func in models_to_test.items():
        print(f"Running AIME 2025 with {model_name}...")
        print("-" * 60)

        model_results = []
        start_time = time.time()

        for i, task in enumerate(tasks):
            try:
                print(f"Solving {task.task_id} ({i+1}/{len(tasks)})...", end=" ", flush=True)

                response_start = time.time()
                response = await model_func(task.prompt)
                execution_time = time.time() - response_start

                correct = aime_benchmark.evaluate_response(task, response)

                result = {
                    "task_id": task.task_id,
                    "prompt": task.prompt[:100] + "..." if len(task.prompt) > 100 else task.prompt,
                    "expected_answer": task.expected_answer,
                    "model_response": response[:200] + "..." if len(response) > 200 else response,
                    "correct": correct,
                    "execution_time": execution_time,
                    "metadata": task.metadata
                }

                model_results.append(result)
                print(f"{'✓' if correct else '✗'} ({execution_time:.1f}s)")

            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}")
                result = {
                    "task_id": task.task_id,
                    "prompt": task.prompt[:100] + "..." if len(task.prompt) > 100 else task.prompt,
                    "expected_answer": task.expected_answer,
                    "model_response": f"ERROR: {str(e)}",
                    "correct": False,
                    "execution_time": 0.0,
                    "metadata": task.metadata,
                    "error": str(e)
                }
                model_results.append(result)

        total_time = time.time() - start_time
        correct_count = sum(1 for r in model_results if r["correct"])
        accuracy = correct_count / len(model_results)
        avg_time = sum(r["execution_time"] for r in model_results) / len(model_results)

        summary = {
            "model_name": model_name,
            "total_tasks": len(model_results),
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "average_time": avg_time,
            "total_time": total_time,
            "detailed_results": model_results
        }

        results[model_name] = summary
        detailed_results.extend([{
            "model": model_name,
            **r
        } for r in model_results])

        # Print results
        print(f"\nResults for {model_name}:")
        print(f"  Total problems: {summary['total_tasks']}")
        print(f"  Correct answers: {summary['correct_answers']}")
        print(f"  Accuracy: {summary['accuracy']:.1%}")
        print(f"  Average time per problem: {summary['average_time']:.2f}s")
        print(f"  Total evaluation time: {summary['total_time']:.1f}s")
        print()

    # Summary comparison
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'Accuracy':<12} {'Avg Time':<12} {'Correct':<10} {'Total':<10}")
    print("-" * 80)

    for model_name, summary in results.items():
        print(f"{model_name:<25} {summary['accuracy']:<12.1%} {summary['average_time']:<12.2f}s {summary['correct_answers']:<10} {summary['total_tasks']:<10}")

    # Performance comparison
    print(f"\nPERFORMANCE ANALYSIS")
    print("-" * 40)
    if len(results) > 1:
        model_names = list(results.keys())
        baseline = results[model_names[0]]
        enhanced = results[model_names[1]]

        improvement = enhanced['accuracy'] - baseline['accuracy']
        print(f"Accuracy improvement: {improvement:+.1%}")
        print(f"Time per problem change: {enhanced['average_time'] - baseline['average_time']:+.2f}s")

        # Show which problems got better/worse
        baseline_correct = set(r["task_id"] for r in baseline["detailed_results"] if r["correct"])
        enhanced_correct = set(r["task_id"] for r in enhanced["detailed_results"] if r["correct"])

        newly_correct = enhanced_correct - baseline_correct
        newly_incorrect = baseline_correct - enhanced_correct

        if newly_correct:
            print(f"\nNewly correct problems ({len(newly_correct)}): {', '.join(sorted(newly_correct))}")
        if newly_incorrect:
            print(f"Newly incorrect problems ({len(newly_incorrect)}): {', '.join(sorted(newly_incorrect))}")

    # Save detailed results
    results_file = Path(__file__).parent / "aime2025_lm_studio_results.md"
    save_detailed_results_to_markdown(results_file, results, detailed_results)
    print(f"\nDetailed results saved to: {results_file}")

def save_detailed_results_to_markdown(file_path: Path, results: dict, detailed_results: list):
    """Save comprehensive benchmark results to markdown file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("# AIME 2025 Benchmark Results - LM Studio Evaluation\n\n")
        f.write(f"**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model**: IBM Granite-4-H-Tiny via LM Studio\n")
        f.write(f"**Total Problems**: {len(detailed_results) // 2}\n\n")

        # Executive summary
        f.write("## Executive Summary\n\n")
        for model_name, summary in results.items():
            f.write(f"### {model_name}\n")
            f.write(f"- **Accuracy**: {summary['accuracy']:.1%} ({summary['correct_answers']}/{summary['total_tasks']})\n")
            f.write(f"- **Average Time**: {summary['average_time']:.2f}s per problem\n")
            f.write(f"- **Total Time**: {summary['total_time']:.1f}s\n\n")

        # Performance comparison
        if len(results) > 1:
            f.write("## Performance Analysis\n\n")
            model_names = list(results.keys())
            baseline = results[model_names[0]]
            enhanced = results[model_names[1]]

            improvement = enhanced['accuracy'] - baseline['accuracy']
            f.write(f"- **Accuracy Improvement**: {improvement:+.1%}\n")
            f.write(f"- **Time Change**: {enhanced['average_time'] - baseline['average_time']:+.2f}s per problem\n\n")

        # Detailed problem results
        f.write("## Detailed Problem Results\n\n")

        # Group by problem
        problems = {}
        for result in detailed_results:
            task_id = result["task_id"]
            if task_id not in problems:
                problems[task_id] = {
                    "task_id": task_id,
                    "prompt": result["prompt"],
                    "expected_answer": result["expected_answer"],
                    "metadata": result["metadata"],
                    "models": {}
                }
            problems[task_id]["models"][result["model"]] = result

        for task_id in sorted(problems.keys()):
            problem = problems[task_id]
            f.write(f"### {task_id}\n\n")
            f.write(f"**Source**: {problem['metadata'].get('source', 'Unknown')}\n")
            f.write(f"**Expected Answer**: {problem['expected_answer']}\n\n")
            f.write(f"**Problem**: {problem['prompt']}\n\n")

            for model_name, model_result in problem["models"].items():
                status = "✅ CORRECT" if model_result["correct"] else "❌ INCORRECT"
                f.write(f"**{model_name}**: {status} ({model_result['execution_time']:.2f}s)\n")
                f.write(f"- Response: {model_result['model_response']}\n\n")

            f.write("---\n\n")

if __name__ == "__main__":
    asyncio.run(run_aime2025_benchmark())