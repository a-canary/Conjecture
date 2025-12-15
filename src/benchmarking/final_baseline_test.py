#!/usr/bin/env python3
"""
Final Baseline Test
Test all configured models with simple problems to establish baseline
"""

import asyncio
import sys
import time
import pytest
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Simple test cases
BASELINE_PROBLEMS = [
    {
        "id": "math_1",
        "question": "What is 17 × 24?",
        "expected": "408",
        "category": "basic_math"
    },
    {
        "id": "math_2",
        "question": "What is 156 + 89?",
        "expected": "245",
        "category": "basic_math"
    },
    {
        "id": "logic_1",
        "question": "If all cats are animals and some animals are pets, can we conclude that some cats are pets? Answer yes or no.",
        "expected": "no",
        "category": "logic"
    },
    {
        "id": "reasoning_1",
        "question": "A train travels 300 miles in 4 hours. What is its average speed?",
        "expected": "75",
        "category": "word_problem"
    }
]

@pytest.mark.asyncio
async def test_gpt_oss_20b():
    """Test GPT-OSS-20B directly"""
    try:
        # Import here to avoid circular imports
        import aiohttp
        import json

        # Read config to get API key
        config_path = Path(__file__).parent.parent.parent / ".conjecture" / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Find GPT-OSS-20B provider
        gpt_provider = None
        for provider in config.get('providers', []):
            if provider.get('name') == 'gpt-oss-20b':
                gpt_provider = provider
                break

        if not gpt_provider:
            return None, "GPT-OSS-20B provider not found"

        api_key = gpt_provider.get('key') or gpt_provider.get('api')
        if not api_key:
            return None, "No API key found"

        # Test with simple request
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "openai/gpt-oss-20b",
                "messages": [
                    {"role": "user", "content": "What is 2+2? Just give the number."}
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"], "Success"
                else:
                    error = await response.text()
                    return None, f"API Error {response.status}: {error}"

    except Exception as e:
        return None, f"Exception: {str(e)}"

@pytest.mark.asyncio
async def test_granite_tiny():
    """Test GraniteTiny via LM Studio"""
    try:
        from .lm_studio_integration import granite_tiny_direct
        response = await granite_tiny_direct("What is 2+2? Just give the number.")
        return response, "Success"
    except Exception as e:
        return None, f"Exception: {str(e)}"

async def run_final_baseline():
    """Run final baseline test"""
    print("=" * 80)
    print("FINAL BASELINE TEST")
    print("=" * 80)
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test model connections
    print("Testing model connections...")
    print("-" * 40)

    # Test GPT-OSS-20B
    print("GPT-OSS-20B:", end=" ")
    gpt_response, gpt_status = await test_gpt_oss_20b()
    if gpt_response:
        print("✓ CONNECTED")
        print(f"  Test response: {gpt_response[:50]}...")
    else:
        print(f"✗ FAILED - {gpt_status}")

    # Test GraniteTiny
    print("GraniteTiny:", end=" ")
    granite_response, granite_status = await test_granite_tiny()
    if granite_response:
        print("✓ CONNECTED")
        print(f"  Test response: {granite_response[:50]}...")
    else:
        print(f"✗ FAILED - {granite_status}")

    print()

    # Determine which models to test
    working_models = {}
    if gpt_response:
        working_models["GPT-OSS-20B"] = test_gpt_oss_20b
    if granite_response:
        working_models["GraniteTiny"] = test_granite_tiny

    if not working_models:
        print("No working models found!")
        return

    print(f"Running baseline with {len(working_models)} models on {len(BASELINE_PROBLEMS)} problems:")
    for model_name in working_models.keys():
        print(f"  - {model_name}")
    print()

    results = {}

    for model_name, test_func in working_models.items():
        print(f"Testing {model_name}...")
        print("-" * 60)

        model_results = []
        start_time = time.time()

        for i, problem in enumerate(BASELINE_PROBLEMS):
            try:
                print(f"Problem {i+1}/{len(BASELINE_PROBLEMS)}: {problem['id']}...", end=" ", flush=True)

                response_start = time.time()
                response = await test_func(problem['question'])
                execution_time = time.time() - response_start

                # Simple evaluation
                expected = problem['expected']
                correct = expected.lower() in response.lower()

                result = {
                    "problem_id": problem['id'],
                    "category": problem['category'],
                    "expected": expected,
                    "correct": correct,
                    "time": execution_time,
                    "response": response[:100] + "..." if len(response) > 100 else response
                }

                model_results.append(result)
                status = "✓" if correct else "✗"
                print(f"{status} ({execution_time:.1f}s)")

            except Exception as e:
                print(f"ERROR: {str(e)[:50]}")

        total_time = time.time() - start_time
        correct_count = sum(1 for r in model_results if r["correct"])
        accuracy = correct_count / len(model_results)
        avg_time = sum(r["time"] for r in model_results) / len(model_results)

        summary = {
            "model_name": model_name,
            "total_problems": len(model_results),
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
    print("BASELINE RESULTS")
    print("=" * 50)

    for model_name, summary in results.items():
        print(f"{model_name}: {summary['accuracy']:.1%} accuracy, {summary['average_time']:.1f}s avg")

    if len(results) > 1:
        print("\nSpeed Comparison:")
        sorted_models = sorted(results.items(), key=lambda x: x[1]['average_time'])
        for model_name, summary in sorted_models:
            print(f"  {model_name}: {summary['average_time']:.1f}s")

    # Save results
    save_baseline_results(results)

def save_baseline_results(results):
    """Save baseline results"""
    results_file = Path(__file__).parent / "final_baseline_results.json"

    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(run_final_baseline())