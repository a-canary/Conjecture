#!/usr/bin/env python3
"""
AIME 2025 Benchmark Runner for GraniteTiny + Conjecture
Evaluates mathematical reasoning capabilities on AIME 2025 problems
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.benchmark_framework import AIME25Benchmark, BenchmarkRunner
from benchmarking.model_integration import granite_tiny_model, granite_tiny_conjecture

async def run_aime2025_benchmark():
    """Run AIME 2025 benchmark with GraniteTiny"""
    print("=" * 80)
    print("AIME 2025 BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"Starting evaluation at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize benchmark
    aime_benchmark = AIME25Benchmark()

    # Load tasks
    print("Loading AIME 2025 problems...")
    tasks = await aime_benchmark.load_tasks()
    print(f"Loaded {len(tasks)} problems")
    print()

    # Test models
    models_to_test = {
        "GraniteTiny": granite_tiny_model,
        "GraniteTiny+Conjecture": granite_tiny_conjecture,
    }

    results = {}

    for model_name, model_func in models_to_test.items():
        print(f"Running AIME 2025 with {model_name}...")
        print("-" * 60)

        start_time = time.time()
        summary = await aime_benchmark.run_benchmark(model_func, model_name, using_conjecture="Conjecture" in model_name)
        elapsed_time = time.time() - start_time

        results[model_name] = summary

        # Print results
        print(f"Results for {model_name}:")
        print(f"  Total problems: {summary.total_tasks}")
        print(f"  Correct answers: {summary.correct_answers}")
        print(f"  Accuracy: {summary.accuracy:.1%}")
        print(f"  Average time per problem: {summary.average_time:.2f}s")
        print(f"  Total evaluation time: {elapsed_time:.2f}s")
        print()

        # Print individual results
        print("Individual problem results:")
        for i, task in enumerate(tasks):
            # We need to run the model again to get individual responses
            # For now, just show which problems we evaluated
            print(f"  {task.task_id}: {task.metadata.get('difficulty', 'unknown')}")
        print()

    # Summary comparison
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Accuracy':<12} {'Avg Time':<12} {'Correct':<10} {'Total':<10}")
    print("-" * 80)

    for model_name, summary in results.items():
        print(f"{model_name:<20} {summary.accuracy:<12.1%} {summary.average_time:<12.2f}s {summary.correct_answers:<10} {summary.total_tasks:<10}")

    print()

    # Save detailed results
    results_file = Path(__file__).parent / "aime2025_results.md"
    save_results_to_markdown(results_file, results, tasks)
    print(f"Detailed results saved to: {results_file}")

def save_results_to_markdown(file_path: Path, results: dict, tasks: list):
    """Save benchmark results to markdown file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("# AIME 2025 Benchmark Results\n\n")
        f.write(f"**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Model | Accuracy | Avg Time (s) | Correct | Total |\n")
        f.write("|-------|----------|--------------|---------|--------|\n")

        for model_name, summary in results.items():
            f.write(f"| {model_name} | {summary.accuracy:.1%} | {summary.average_time:.2f} | {summary.correct_answers} | {summary.total_tasks} |\n")

        f.write("\n")

        # Individual problem details
        f.write("## Problem Details\n\n")
        for i, task in enumerate(tasks):
            f.write(f"### {task.task_id}\n\n")
            f.write(f"**Category**: {task.metadata.get('category', 'unknown')}\n")
            f.write(f"**Difficulty**: {task.metadata.get('difficulty', 'unknown')}\n")
            f.write(f"**Expected Answer**: {task.expected_answer}\n\n")
            f.write(f"**Problem**: {task.prompt[:200]}{'...' if len(task.prompt) > 200 else ''}\n\n")
            f.write("---\n\n")

if __name__ == "__main__":
    asyncio.run(run_aime2025_benchmark())