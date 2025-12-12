#!/usr/bin/env python3
"""
Quick Prompt Test - Fast prototype testing without database reset
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompt_prototype_framework import (
    PromptPrototypeFramework,
    MATH_PROBLEMS,
    LOGIC_PROBLEMS
)
from improved_prompts import (
    BASELINE_STRATEGY,
    DIRECT_STRATEGY,
    MATH_SPECIALIZED_STRATEGY,
    LOGIC_SPECIALIZED_STRATEGY,
    MATH_CONTEXT_ENHANCED_STRATEGY,
    ENHANCED_CONJECTURE_MATH_STRATEGY
)
from config_aware_integration import gpt_oss_20b_direct

async def run_quick_test():
    """Run quick test with key strategies"""

    print("=" * 80)
    print("QUICK PROMPT TEST - Key Strategies")
    print("=" * 80)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize framework with GPT-OSS-20B
    framework = PromptPrototypeFramework(gpt_oss_20b_direct, "GPT-OSS-20B")

    # Test key strategies (subset for speed)
    strategies_to_test = [
        BASELINE_STRATEGY,  # Current baseline
        DIRECT_STRATEGY,    # Direct problem solving
        MATH_SPECIALIZED_STRATEGY,  # Specialized math
        LOGIC_SPECIALIZED_STRATEGY, # Specialized logic
        MATH_CONTEXT_ENHANCED_STRATEGY,  # Context enhanced math
        ENHANCED_CONJECTURE_MATH_STRATEGY  # Enhanced Conjecture
    ]

    print("Testing strategies:")
    for strategy in strategies_to_test:
        framework.add_strategy(strategy)
        print(f"  - {strategy.name}: {strategy.description}")
    print()

    # Use smaller problem set for speed
    test_problems = [
        # 2 math problems
        MATH_PROBLEMS[0],  # Basic multiplication
        MATH_PROBLEMS[2],  # Word problem
        # 1 logic problem
        LOGIC_PROBLEMS[0],  # Basic logic
    ]

    print(f"Testing with {len(test_problems)} problems:")
    for problem in test_problems:
        print(f"  - {problem['id']}: {problem['category']}")
    print()

    # Run the test
    start_time = time.time()
    results = await framework.run_competition(test_problems)
    total_time = time.time() - start_time

    # Analyze results
    analysis = framework.analyze_results(results)

    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)

    print(f"Total testing time: {total_time:.1f} seconds")
    print(f"Problems tested: {len(test_problems)}")
    print()

    # Strategy performance
    print("STRATEGY PERFORMANCE:")
    print("-" * 50)

    sorted_strategies = sorted(
        analysis["strategies"].items(),
        key=lambda x: (x[1]["accuracy"], -x[1]["average_time"]),
        reverse=True
    )

    for i, (strategy_name, stats) in enumerate(sorted_strategies, 1):
        print(f"{i}. {strategy_name}")
        print(f"   Accuracy: {stats['accuracy']:.1%} ({stats['correct_answers']}/{len(test_problems)})")
        print(f"   Avg Time: {stats['average_time']:.2f}s")
        print()

    # Baseline comparison
    baseline_acc = analysis["strategies"].get("baseline_current", {}).get("accuracy", 0)
    print("BASELINE COMPARISON:")
    print("-" * 50)
    print(f"Current baseline: {baseline_acc:.1%}")

    improved = [
        (name, stats) for name, stats in analysis["strategies"].items()
        if name != "baseline_current" and stats["accuracy"] > baseline_acc
    ]

    if improved:
        print("Strategies that beat baseline:")
        for name, stats in improved:
            improvement = stats["accuracy"] - baseline_acc
            print(f"  + {name}: {stats['accuracy']:.1%} (+{improvement:.1%})")
    else:
        print("No strategies beat baseline")

    print()

    # Quick recommendations
    if "best_strategy" in analysis:
        best = analysis["best_strategy"]
        print(f"🏆 BEST: {best['name']} ({best['accuracy']:.1%})")

        if best['accuracy'] > baseline_acc:
            print("✅ Improvement found!")
        else:
            print("⚠️  Baseline still best")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"quick_test_results_{timestamp}.json"
    framework.save_results(results, analysis, filename)

    print(f"\nResults saved to: {filename}")

    return analysis

if __name__ == "__main__":
    asyncio.run(run_quick_test())