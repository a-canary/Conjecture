#!/usr/bin/env python3
"""
Run Prompt Competition
Tests multiple prompt strategies against baseline with database reset
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
    LOGIC_PROBLEMS,
    BASELINE_STRATEGY,
    DIRECT_STRATEGY
)
from improved_prompts import (
    ALL_STRATEGIES,
    STRATEGY_CATEGORIES,
    get_strategies_by_category
)
from database_reset import setup_benchmark_environment
from config_aware_integration import gpt_oss_20b_direct

async def run_full_competition():
    """Run comprehensive prompt competition"""

    print("=" * 80)
    print("CONJECTURE PROMPT ENGINEERING COMPETITION")
    print("=" * 80)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup benchmark environment with clean database
    print("Setting up benchmark environment...")
    resetter = setup_benchmark_environment(
        db_path=None,  # Use default
        prime_examples=True,
        verify_state=True
    )
    print()

    # Initialize framework with GPT-OSS-20B
    framework = PromptPrototypeFramework(gpt_oss_20b_direct, "GPT-OSS-20B")

    # Add all strategies to test
    print("Loading prompt strategies...")
    for strategy in ALL_STRATEGIES:
        framework.add_strategy(strategy)
        print(f"  - {strategy.name}: {strategy.description}")
    print()

    # Prepare problem sets
    all_problems = MATH_PROBLEMS + LOGIC_PROBLEMS
    print(f"Testing with {len(all_problems)} problems:")
    print(f"  - Math problems: {len(MATH_PROBLEMS)}")
    print(f"  - Logic problems: {len(LOGIC_PROBLEMS)}")
    print()

    # Run the competition
    start_time = time.time()
    results = await framework.run_competition(all_problems)
    total_time = time.time() - start_time

    # Analyze results
    analysis = framework.analyze_results(results)

    # Print comprehensive results
    print("\n" + "=" * 80)
    print("COMPETITION RESULTS")
    print("=" * 80)

    # Overall summary
    print(f"Total testing time: {total_time:.1f} seconds")
    print(f"Problems tested: {len(all_problems)}")
    print(f"Strategies tested: {len(ALL_STRATEGIES)}")
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
        print(f"{i:2d}. {strategy_name}")
        print(f"     Accuracy: {stats['accuracy']:.1%} ({stats['correct_answers']}/{len(all_problems)})")
        print(f"     Avg Time: {stats['average_time']:.2f}s")
        print(f"     Total: {stats['total_time']:.1f}s")

        # Performance category
        if stats['accuracy'] >= 0.75:
            category = "EXCELLENT"
        elif stats['accuracy'] >= 0.60:
            category = "GOOD"
        elif stats['accuracy'] >= 0.40:
            category = "FAIR"
        else:
            category = "POOR"
        print(f"     Rating: {category}")
        print()

    # Best strategy
    if "best_strategy" in analysis:
        best = analysis["best_strategy"]
        print(f"🏆 BEST STRATEGY: {best['name']}")
        print(f"   Accuracy: {best['accuracy']:.1%}")
        print(f"   Speed: {best['avg_time']:.2f}s per problem")
        print()

    # Category analysis
    print("CATEGORY ANALYSIS:")
    print("-" * 50)

    for category, strategy_names in STRATEGY_CATEGORIES.items():
        category_results = []
        for strategy_name in strategy_names:
            if strategy_name in analysis["strategies"]:
                stats = analysis["strategies"][strategy_name]
                category_results.append((strategy_name, stats))

        if category_results:
            # Find best in category
            best_in_category = max(category_results, key=lambda x: x[1]["accuracy"])
            print(f"{category.upper()}:")
            print(f"  Best: {best_in_category[0]} ({best_in_category[1]['accuracy']:.1%})")
            print(f"  Strategies: {len(category_results)}")

            # Show all in category
            for strategy_name, stats in sorted(category_results, key=lambda x: x[1]["accuracy"], reverse=True):
                print(f"    - {strategy_name}: {stats['accuracy']:.1%}")
            print()

    # Problem type analysis
    print("PROBLEM TYPE ANALYSIS:")
    print("-" * 50)

    # Math problems
    math_results = {}
    logic_results = {}

    for strategy_name, strategy_results in results.items():
        math_correct = sum(1 for r in strategy_results if r.problem_type in ["basic_math", "word_problem", "percentages"] and r.correct)
        math_total = sum(1 for r in strategy_results if r.problem_type in ["basic_math", "word_problem", "percentages"])
        math_accuracy = math_correct / math_total if math_total > 0 else 0

        logic_correct = sum(1 for r in strategy_results if r.problem_type == "logic" and r.correct)
        logic_total = sum(1 for r in strategy_results if r.problem_type == "logic")
        logic_accuracy = logic_correct / logic_total if logic_total > 0 else 0

        math_results[strategy_name] = math_accuracy
        logic_results[strategy_name] = logic_accuracy

    print("Math Problem Performance:")
    for strategy_name, accuracy in sorted(math_results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {strategy_name}: {accuracy:.1%}")

    print("\nLogic Problem Performance:")
    for strategy_name, accuracy in sorted(logic_results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {strategy_name}: {accuracy:.1%}")
    print()

    # Baseline comparison
    baseline_accuracy = analysis["strategies"].get("baseline_current", {}).get("accuracy", 0)
    print("BASELINE COMPARISON:")
    print("-" * 50)
    print(f"Baseline (Current Conjecture): {baseline_accuracy:.1%}")

    improved_strategies = [
        (name, stats) for name, stats in analysis["strategies"].items()
        if name != "baseline_current" and stats["accuracy"] > baseline_accuracy
    ]

    if improved_strategies:
        print(f"Strategies that beat baseline: {len(improved_strategies)}")
        for name, stats in sorted(improved_strategies, key=lambda x: x[1]["accuracy"], reverse=True):
            improvement = stats["accuracy"] - baseline_accuracy
            print(f"  + {name}: {stats['accuracy']:.1%} (+{improvement:.1%})")
    else:
        print("No strategies beat baseline performance")

    print()

    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 50)

    if "best_strategy" in analysis:
        best_name = analysis["best_strategy"]["name"]
        best_acc = analysis["best_strategy"]["accuracy"]

        if best_acc > baseline_accuracy:
            print(f"1. Adopt '{best_name}' as new default - improves accuracy by {(best_acc - baseline_accuracy):.1%}")
        else:
            print(f"1. Current baseline is still optimal - need more prompt engineering")

        print(f"2. Focus on {best_name.split('_')[0]}-type problem solving for best results")

        # Speed vs accuracy tradeoff
        fast_accurate = [
            (name, stats) for name, stats in analysis["strategies"].items()
            if stats["accuracy"] >= baseline_accuracy and stats["average_time"] < analysis["strategies"]["baseline_current"]["average_time"]
        ]

        if fast_accurate:
            print("3. For faster performance with similar accuracy, consider:")
            for name, stats in sorted(fast_accurate, key=lambda x: x[1]["average_time"]):
                speedup = analysis["strategies"]["baseline_current"]["average_time"] / stats["average_time"]
                print(f"   - {name}: {speedup:.1f}x faster")
        else:
            print("3. No faster strategies with similar accuracy found")

    print()

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prompt_competition_{timestamp}.json"
    framework.save_results(results, analysis, filename)

    # Create summary report
    summary_file = Path(__file__).parent / f"competition_summary_{timestamp}.md"
    create_summary_report(analysis, results, summary_file)

    print(f"Detailed results saved to: {filename}")
    print(f"Summary report saved to: {summary_file}")

def create_summary_report(analysis, results, output_file):
    """Create markdown summary report"""

    with open(output_file, 'w') as f:
        f.write("# Prompt Engineering Competition Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model**: {analysis['model_name']}\n")
        f.write(f"**Problems Tested**: {analysis['total_problems']}\n\n")

        f.write("## Strategy Rankings\n\n")
        f.write("| Rank | Strategy | Accuracy | Avg Time | Rating |\n")
        f.write("|------|----------|----------|----------|--------|\n")

        sorted_strategies = sorted(
            analysis["strategies"].items(),
            key=lambda x: (x[1]["accuracy"], -x[1]["average_time"]),
            reverse=True
        )

        for i, (strategy_name, stats) in enumerate(sorted_strategies, 1):
            rating = "🟢" if stats['accuracy'] >= 0.75 else "🟡" if stats['accuracy'] >= 0.50 else "🔴"
            f.write(f"| {i} | {strategy_name} | {stats['accuracy']:.1%} | {stats['average_time']:.2f}s | {rating} |\n")

        f.write("\n## Key Findings\n\n")

        baseline = analysis["strategies"].get("baseline_current", {}).get("accuracy", 0)
        if "best_strategy" in analysis:
            best = analysis["best_strategy"]
            if best["accuracy"] > baseline:
                f.write(f"- **Best strategy**: {best['name']} improves accuracy by {(best['accuracy'] - baseline):.1%}\n")
            else:
                f.write(f"- **Current baseline** remains optimal at {baseline:.1%} accuracy\n")

        f.write(f"- **Total strategies tested**: {len(analysis['strategies'])}\n")
        f.write(f"- **Problems per strategy**: {analysis['total_problems']}\n\n")

        f.write("## Recommendations\n\n")
        f.write("1. **For math problems**: Use specialized math reasoning prompts\n")
        f.write("2. **For logic problems**: Use logic-focused prompts with careful premise analysis\n")
        f.write("3. **Avoid**: Generic prompts that don't match problem domain\n")
        f.write("4. **Context engineering**: Should enhance, not replace, domain-specific knowledge\n\n")

if __name__ == "__main__":
    asyncio.run(run_full_competition())