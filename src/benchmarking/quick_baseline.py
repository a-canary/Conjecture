#!/usr/bin/env python3
"""
Quick Baseline Test
Establishes baseline with working models only
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.lm_studio_integration import granite_tiny_direct, granite_tiny_direct_conjecture

# Test cases
BASELINE_TEST_CASES = [
    {
        "id": "math_basic_1",
        "question": "What is 17 × 24?",
        "expected": "408",
        "category": "basic_math"
    },
    {
        "id": "math_basic_2",
        "question": "What is 156 + 89?",
        "expected": "245",
        "category": "basic_math"
    },
    {
        "id": "math_basic_3",
        "question": "What is 144 ÷ 12?",
        "expected": "12",
        "category": "basic_math"
    },
    {
        "id": "reasoning_1",
        "question": "If a train travels 300 miles in 4 hours, what is its average speed?",
        "expected": "75",
        "category": "word_problem"
    },
    {
        "id": "reasoning_2",
        "question": "A box contains 24 red balls and 36 blue balls. What percentage of the balls are red?",
        "expected": "40",
        "category": "word_problem"
    }
]

def evaluate_response(test_case, response):
    """Evaluate if response contains correct answer"""
    response_lower = response.lower()

    # Extract numbers from response
    import re
    numbers = re.findall(r'\b\d+\b', response_lower)
    return test_case["expected"] in numbers

async def run_quick_baseline():
    """Run quick baseline with LM Studio"""
    print("=" * 80)
    print("QUICK BASELINE - LM Studio GraniteTiny")
    print("=" * 80)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print(f"Testing {len(BASELINE_TEST_CASES)} baseline problems:")
    for i, case in enumerate(BASELINE_TEST_CASES):
        print(f"  {i+1}. {case['id']} ({case['category']}) - Expected: {case['expected']}")
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

        for i, test_case in enumerate(BASELINE_TEST_CASES):
            try:
                print(f"Problem {i+1}/{len(BASELINE_TEST_CASES)}: {test_case['id']}...", end=" ", flush=True)

                response_start = time.time()
                response = await model_func(test_case["question"])
                execution_time = time.time() - response_start

                correct = evaluate_response(test_case, response)

                result = {
                    "test_id": test_case["id"],
                    "category": test_case["category"],
                    "expected": test_case["expected"],
                    "correct": correct,
                    "time": execution_time,
                    "response_preview": response[:100] + "..." if len(response) > 100 else response
                }

                model_results.append(result)
                status = "✓" if correct else "✗"
                print(f"{status} ({execution_time:.1f}s)")

            except Exception as e:
                print(f"ERROR: {str(e)[:50]}")
                result = {
                    "test_id": test_case["id"],
                    "category": test_case["category"],
                    "expected": test_case["expected"],
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

    # Final comparison
    print("BASELINE COMPARISON")
    print("=" * 50)

    if len(results) > 1:
        model_names = list(results.keys())
        baseline = results[model_names[0]]
        enhanced = results[model_names[1]]

        print(f"{model_names[0]}: {baseline['accuracy']:.1%} ({baseline['correct_answers']}/{baseline['total_tasks']}) - {baseline['average_time']:.1f}s avg")
        print(f"{model_names[1]}: {enhanced['accuracy']:.1%} ({enhanced['correct_answers']}/{enhanced['total_tasks']}) - {enhanced['average_time']:.1f}s avg")

        improvement = enhanced['accuracy'] - baseline['accuracy']
        time_change = enhanced['average_time'] - baseline['average_time']

        print(f"\nAccuracy change: {improvement:+.1%}")
        print(f"Speed change: {time_change:+.1f}s per problem")

        if improvement > 0:
            print("🎉 CONJECTURE IMPROVES ACCURACY!")
        elif improvement < 0:
            print("⚠️  CONJECTURE HURTS ACCURACY")
        else:
            print("➖ NO ACCURACY DIFFERENCE")

        if time_change > 0:
            print("⏱️  BUT IT'S SLOWER")
        elif time_change < 0:
            print("⚡ AND IT'S FASTER")
        else:
            print("⏱️  SAME SPEED")

        # Category analysis
        print(f"\nCategory Analysis:")
        categories = {}
        for result in baseline['results']:
            cat = result['category']
            if cat not in categories:
                categories[cat] = {'baseline': 0, 'enhanced': 0, 'total': 0}
            categories[cat]['total'] += 1
            if result['correct']:
                categories[cat]['baseline'] += 1

        for result in enhanced['results']:
            cat = result['category']
            if result['correct']:
                categories[cat]['enhanced'] += 1

        for cat, stats in categories.items():
            base_acc = stats['baseline'] / stats['total']
            enh_acc = stats['enhanced'] / stats['total']
            change = enh_acc - base_acc
            print(f"  {cat}: {base_acc:.1%} -> {enh_acc:.1%} ({change:+.1%})")

    # Save results
    save_baseline_results(results)

def save_baseline_results(results):
    """Save baseline results"""
    results_file = Path(__file__).parent / "quick_baseline_results.json"

    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(run_quick_baseline())