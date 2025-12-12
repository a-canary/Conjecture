#!/usr/bin/env python3
"""
Fast Conjecture Test using GPT-OSS-20B
Quick iteration to test if Conjecture can actually improve performance
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.gpt_oss_integration import gpt_oss_direct, gpt_oss_direct_conjecture

# Simple test cases that should be easy for a 20B model
SIMPLE_TEST_CASES = [
    {
        "id": "math_basic",
        "question": "What is 17 × 24?",
        "expected": "408",
        "category": "basic_math"
    },
    {
        "id": "logic_simple",
        "question": "If all cats are animals and some animals are pets, can we conclude that some cats are pets? Answer yes or no and explain.",
        "expected": "no",
        "category": "logic"
    },
    {
        "id": "reasoning_chain",
        "question": "Sarah is 5 years older than Tom. In 3 years, Sarah will be twice as old as Tom was 2 years ago. How old is Sarah now?",
        "expected": "13",
        "category": "algebra"
    },
    {
        "id": "pattern_recognition",
        "question": "What is the next number in this sequence: 2, 6, 12, 20, 30, ?",
        "expected": "42",
        "category": "pattern"
    }
]

def evaluate_response(test_case, response):
    """Simple evaluation - check if expected answer is in response"""
    response_lower = response.lower()
    expected_lower = test_case["expected"].lower()

    # Handle multiple number formats
    if test_case["category"] in ["basic_math", "algebra", "pattern"]:
        # Extract numbers from response
        import re
        numbers = re.findall(r'\b\d+\b', response_lower)
        return test_case["expected"] in numbers

    # For other categories, look for the expected text
    return expected_lower in response_lower

async def run_fast_conjecture_test():
    """Run fast test with simple problems"""
    print("=" * 80)
    print("FAST CONJECTURE TEST - GPT-OSS-20B")
    print("=" * 80)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check if API key is set
    print("Testing GPT-OSS connection...")
    try:
        test_response = await gpt_oss_direct("Hello, respond with 'ready' to confirm connection.")
        if "ready" in test_response.lower():
            print("✓ GPT-OSS connection successful")
        else:
            print("✗ GPT-OSS connection issue - check API key")
            return
    except Exception as e:
        print(f"✗ GPT-OSS connection failed: {e}")
        print("Please set your OpenRouter API key in the config or gpt_oss_integration.py")
        return

    print(f"\nTesting {len(SIMPLE_TEST_CASES)} simple problems:")
    for i, case in enumerate(SIMPLE_TEST_CASES):
        print(f"  {i+1}. {case['id']} ({case['category']}) - Expected: {case['expected']}")
    print()

    # Test models
    models_to_test = {
        "GPT-OSS-Direct": gpt_oss_direct,
        "GPT-OSS+Conjecture": gpt_oss_direct_conjecture,
    }

    results = {}

    for model_name, model_func in models_to_test.items():
        print(f"Testing {model_name}...")
        print("-" * 60)

        model_results = []
        start_time = time.time()

        for i, test_case in enumerate(SIMPLE_TEST_CASES):
            try:
                print(f"Problem {i+1}/{len(SIMPLE_TEST_CASES)}: {test_case['id']}...", end=" ", flush=True)

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
                    "response_preview": response[:150] + "..." if len(response) > 150 else response
                }

                model_results.append(result)
                status = "✓" if correct else "✗"
                print(f"{status} ({execution_time:.1f}s)")

            except Exception as e:
                print(f"✗ ERROR: {str(e)[:50]}")
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
    print("FINAL RESULTS")
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

        # Show detailed comparison
        print(f"\nDetailed comparison:")
        for i in range(len(baseline['results'])):
            baseline_result = baseline['results'][i]
            enhanced_result = enhanced['results'][i]

            baseline_status = "✓" if baseline_result['correct'] else "✗"
            enhanced_status = "✓" if enhanced_result['correct'] else "✗"

            status_change = ""
            if baseline_result['correct'] != enhanced_result['correct']:
                if enhanced_result['correct']:
                    status_change = " (IMPROVED!)"
                else:
                    status_change = " (REGRESSED)"

            print(f"  {baseline_result['test_id']}: {baseline_status} -> {enhanced_status}{status_change}")

if __name__ == "__main__":
    asyncio.run(run_fast_conjecture_test())