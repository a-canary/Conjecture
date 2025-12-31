#!/usr/bin/env python3
"""
Quick Conjecture vs Direct Test Using Existing Test Cases
Tests with easier, more suitable problems than AIME
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarking.lm_studio_integration import granite_tiny_direct, granite_tiny_direct_conjecture

def load_test_cases():
    """Load a subset of existing test cases"""
    test_cases = []

    # Use absolute path from project root
    project_root = Path(__file__).parent.parent.parent
    selected_cases = [
        project_root / "research" / "test_cases" / "math_algebra_20251204_181540.json",  # Medium algebra
        project_root / "research" / "test_cases" / "logic_puzzle_20251204_181540.json",   # Hard but structured
        project_root / "research" / "test_cases" / "context_qa_20251204_181540.json",     # Easy context
    ]

    for case_file in selected_cases:
        if case_file.exists():
            with open(case_file, "r") as f:
                case_data = json.load(f)
                test_cases.append({
                    "file": case_file.name,
                    "data": case_data
                })
        else:
            print(f"Warning: Test case not found: {case_file}")

    return test_cases

def evaluate_response(test_case, response):
    """Simple evaluation of response correctness"""
    ground_truth = test_case["ground_truth"].lower()
    response_lower = response.lower()

    # Extract key information from ground truth
    if "$" in ground_truth:
        # Financial problem - look for dollar amount
        import re
        amounts = re.findall(r'\$[\d,]+\.?\d*', ground_truth)
        if amounts:
            target_amount = amounts[0]
            return target_amount.lower() in response_lower

    # Look for key phrases from ground truth
    key_phrases = []
    if "owns the red house" in ground_truth:
        key_phrases.append("red house")
    if "elderberry" in ground_truth:
        key_phrases.append("elderberry")
    if "engineer" in ground_truth:
        key_phrases.append("engineer")

    # Math problem - look for final answer
    if "cost" in ground_truth and "$" in ground_truth:
        return any(phrase in response_lower for phrase in ["$2,130", "2130", "2,130"])

    # Check if any key phrases are present
    if key_phrases:
        return any(phrase in response_lower for phrase in key_phrases)

    # Default: check if ground truth is contained in response
    return ground_truth[:50] in response_lower

async def run_quick_conjecture_test():
    """Run quick test of Conjecture vs Direct"""
    print("=" * 80)
    print("QUICK CONJECTURE VS DIRECT TEST")
    print("=" * 80)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load test cases
    test_cases = load_test_cases()
    print(f"Loaded {len(test_cases)} test cases:")
    for i, case in enumerate(test_cases):
        print(f"  {i+1}. {case['data']['id']} ({case['data']['difficulty']}) - {case['data']['description'][:60]}...")
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

        for i, test_case in enumerate(test_cases):
            try:
                case_data = test_case["data"]
                print(f"Problem {i+1}/{len(test_cases)}: {case_data['id']}...", end=" ", flush=True)

                response_start = time.time()
                response = await model_func(case_data["question"])
                execution_time = time.time() - response_start

                correct = evaluate_response(case_data, response)

                result = {
                    "test_id": case_data["id"],
                    "difficulty": case_data["difficulty"],
                    "correct": correct,
                    "time": execution_time,
                    "response_preview": response[:100] + "..." if len(response) > 100 else response
                }

                model_results.append(result)
                print(f"{'CORRECT' if correct else 'INCORRECT'} ({execution_time:.1f}s)")

            except Exception as e:
                print(f"ERROR: {str(e)[:50]}")
                result = {
                    "test_id": test_case["data"]["id"],
                    "difficulty": test_case["data"]["difficulty"],
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

        if improvement > 0:
            print("✅ CONJECTURE HELPS!")
        elif improvement < 0:
            print("❌ CONJECTURE HURTS")
        else:
            print("➖ NO DIFFERENCE")

        # Show problem-by-problem comparison
        print(f"\nProblem-by-problem comparison:")
        for i in range(len(baseline['results'])):
            baseline_result = baseline['results'][i]
            enhanced_result = enhanced['results'][i]

            baseline_status = "✅" if baseline_result['correct'] else "❌"
            enhanced_status = "✅" if enhanced_result['correct'] else "❌"

            status_change = ""
            if baseline_result['correct'] != enhanced_result['correct']:
                if enhanced_result['correct']:
                    status_change = " (IMPROVED)"
                else:
                    status_change = " (REGRESSED)"

            print(f"  {baseline_result['test_id']} ({baseline_result['difficulty']}): {baseline_status} -> {enhanced_status}{status_change}")

if __name__ == "__main__":
    asyncio.run(run_quick_conjecture_test())