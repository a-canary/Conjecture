#!/usr/bin/env python3
"""
Simple Performance Benchmark for Tiny LLM Prompt Engineering Optimizer
"""

import asyncio
import time
from datetime import datetime

from test_optimizer_standalone import (
    SimplePromptOptimizer,
    TaskDescription,
    TaskComplexity,
    OptimizationStrategy
)


async def run_simple_benchmark():
    """Run simple performance benchmark"""

    print("="*60)
    print("SIMPLE PROMPT OPTIMIZER BENCHMARK")
    print("="*60)

    optimizer = SimplePromptOptimizer()

    # Test scenarios
    test_cases = [
        {
            "name": "Simple Extraction",
            "task_type": "extraction",
            "complexity": TaskComplexity.SIMPLE,
            "input": "Extract the date: The meeting is on December 15, 2024.",
            "context": ["General context"],
            "model": "granite-tiny"
        },
        {
            "name": "Moderate Analysis",
            "task_type": "analysis",
            "complexity": TaskComplexity.MODERATE,
            "input": "Analyze this: Sales increased by 15% while costs decreased by 8%.",
            "context": ["Business context", "Data context"],
            "model": "llama-3.2-1b"
        },
        {
            "name": "Complex Research",
            "task_type": "research",
            "complexity": TaskComplexity.COMPLEX,
            "input": "Research the impact of AI on creative industries.",
            "context": ["Academic papers", "Industry reports", "Expert opinions"],
            "model": "phi-3-mini"
        }
    ]

    all_results = []
    total_start_time = time.time()

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {test_case['name']}")

        task = TaskDescription(
            task_type=test_case["task_type"],
            complexity=test_case["complexity"],
            required_inputs=[test_case["input"]],
            expected_output_format="structured"
        )

        strategies = [OptimizationStrategy.MINIMAL_TOKENS, OptimizationStrategy.ADAPTIVE]

        for strategy in strategies:
            start_time = time.time()

            try:
                result = await optimizer.optimize_prompt(
                    task=task,
                    context_items=test_case["context"],
                    model_name=test_case["model"],
                    optimization_strategy=strategy
                )

                optimization_time = (time.time() - start_time) * 1000

                # Calculate metrics
                original_tokens = len(result.original_prompt.split())
                optimized_tokens = len(result.optimized_prompt.split())
                token_reduction = original_tokens - optimized_tokens

                print(f"   {strategy.value}:")
                print(f"     Time: {optimization_time:.1f}ms")
                print(f"     Tokens: {original_tokens} -> {optimized_tokens} ({token_reduction:+d})")
                print(f"     Score: {result.optimization_score:.3f}")

                all_results.append({
                    "test": test_case["name"],
                    "model": test_case["model"],
                    "strategy": strategy.value,
                    "success": True,
                    "time_ms": optimization_time,
                    "original_tokens": original_tokens,
                    "optimized_tokens": optimized_tokens,
                    "token_reduction": token_reduction,
                    "score": result.optimization_score
                })

            except Exception as e:
                print(f"   {strategy.value}: ERROR - {e}")
                all_results.append({
                    "test": test_case["name"],
                    "model": test_case["model"],
                    "strategy": strategy.value,
                    "success": False,
                    "error": str(e)
                })

    total_time = (time.time() - total_start_time) * 1000

    # Calculate summary statistics
    successful_results = [r for r in all_results if r["success"]]

    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    print(f"Total tests: {len(all_results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Success rate: {len(successful_results)/len(all_results):.1%}")
    print(f"Total time: {total_time:.1f}ms")

    if successful_results:
        avg_time = sum(r["time_ms"] for r in successful_results) / len(successful_results)
        avg_score = sum(r["score"] for r in successful_results) / len(successful_results)
        total_tokens_saved = sum(r["token_reduction"] for r in successful_results)

        print(f"Average optimization time: {avg_time:.1f}ms")
        print(f"Average score: {avg_score:.3f}")
        print(f"Total tokens saved: {total_tokens_saved}")

        # Strategy comparison
        minimal_results = [r for r in successful_results if r["strategy"] == "minimal_tokens"]
        adaptive_results = [r for r in successful_results if r["strategy"] == "adaptive"]

        print(f"\nStrategy Comparison:")
        if minimal_results:
            avg_minimal_score = sum(r["score"] for r in minimal_results) / len(minimal_results)
            avg_minimal_tokens = sum(r["token_reduction"] for r in minimal_results) / len(minimal_results)
            print(f"  Minimal Tokens:")
            print(f"    Avg Score: {avg_minimal_score:.3f}")
            print(f"    Avg Token Reduction: {avg_minimal_tokens:+.1f}")

        if adaptive_results:
            avg_adaptive_score = sum(r["score"] for r in adaptive_results) / len(adaptive_results)
            avg_adaptive_tokens = sum(r["token_reduction"] for r in adaptive_results) / len(adaptive_results)
            print(f"  Adaptive:")
            print(f"    Avg Score: {avg_adaptive_score:.3f}")
            print(f"    Avg Token Reduction: {avg_adaptive_tokens:+.1f}")

        # Model comparison
        models = {}
        for r in successful_results:
            model = r["model"]
            if model not in models:
                models[model] = []
            models[model].append(r)

        print(f"\nModel Performance:")
        for model, model_results in models.items():
            avg_model_score = sum(r["score"] for r in model_results) / len(model_results)
            avg_model_time = sum(r["time_ms"] for r in model_results) / len(model_results)
            print(f"  {model}:")
            print(f"    Avg Score: {avg_model_score:.3f}")
            print(f"    Avg Time: {avg_model_time:.1f}ms")

    # Performance assessment
    print(f"\nPERFORMANCE ASSESSMENT:")
    if successful_results:
        avg_score = sum(r["score"] for r in successful_results) / len(successful_results)
        if avg_score >= 0.7:
            print("[GOOD] Strong performance - optimizer working well")
        elif avg_score >= 0.5:
            print("[ACCEPTABLE] Moderate performance - some improvements needed")
        else:
            print("[NEEDS IMPROVEMENT] Poor performance - requires optimization")
    else:
        print("[FAILED] All tests failed - critical issues")

    # Get optimizer statistics
    stats = optimizer.get_optimization_stats()
    print(f"\nOptimizer Statistics:")
    print(f"  Optimizations performed: {stats['optimizations_performed']}")
    print(f"  Total tokens saved: {stats['total_tokens_saved']}")

    print("\n" + "="*60)
    print("BENCHMARK COMPLETED")
    print("="*60)

    return len(successful_results) == len(all_results)


if __name__ == "__main__":
    asyncio.run(run_simple_benchmark())