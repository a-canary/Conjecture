#!/usr/bin/env python3
"""
LM Studio Research Runner
Tests Conjecture hypothesis with local models: granite-4-h-tiny vs glm-z1-9b
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] Environment loaded")
except ImportError:
    print("[FAIL] python-dotenv not available")

# Test cases for evaluation
TEST_CASES = [
    {
        "id": "logic_puzzle_001",
        "category": "complex_reasoning",
        "question": "In a small town, there are five houses in a row, each painted a different color: red, blue, green, yellow, and white. Each house is owned by a person with a different profession: doctor, teacher, engineer, artist, and baker. Each person has a different favorite fruit: apple, banana, cherry, date, and elderberry. Using the following clues, determine who owns the red house and what is their favorite fruit?\n\nClues:\n1. The doctor lives in the middle house.\n2. The artist lives next to the person who likes apples.\n3. The engineer lives in the green house.\n4. The teacher likes bananas.\n5. The baker lives in the first house.\n6. The person who likes cherries lives next to the white house.\n7. The red house is somewhere to the left of the blue house.\n8. The artist does not live in the yellow house.\n9. The person who likes dates lives next to the doctor.\n10. The person who likes elderberries lives in the last house.",
        "expected": "The teacher owns the red house and their favorite fruit is banana."
    },
    {
        "id": "math_reasoning_001",
        "category": "mathematical_reasoning",
        "question": "A train travels from City A to City B at 60 mph and returns at 40 mph. What is the average speed for the entire round trip?",
        "expected": "48 mph"
    },
    {
        "id": "evidence_eval_001",
        "category": "evidence_evaluation",
        "question": "Evaluate whether this new drug should be approved based on: 1) Two large studies show statistically significant blood pressure reduction, 2) Two smaller studies show no significant effect, 3) Side effects include mild headaches in 5% of patients and dizziness in 2%, 4) The drug costs 3x more than existing treatments, 5) All studies were funded by the pharmaceutical company. Should this drug be approved?",
        "expected": "Insufficient evidence - need independent studies and long-term safety data"
    }
]

def make_lm_studio_call(prompt: str, model: str, max_tokens: int = 1500) -> Dict[str, Any]:
    """Make API call to LM Studio"""
    try:
        import requests

        api_url = "http://localhost:1234"
        headers = {"Content-Type": "application/json"}

        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }

        endpoint = f"{api_url}/v1/chat/completions"

        start_time = time.time()
        response = requests.post(endpoint, headers=headers, json=data, timeout=300)
        response.raise_for_status()
        end_time = time.time()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        return {
            "content": content,
            "response_time": end_time - start_time,
            "status": "success"
        }

    except Exception as e:
        return {
            "content": f"Error: {str(e)}",
            "response_time": 0,
            "status": "error",
            "error": str(e)
        }

def generate_direct_prompt(test_case: Dict[str, Any]) -> str:
    """Generate direct baseline prompt"""
    return f"""Answer the following question to the best of your ability:

{test_case['question']}

Provide a clear, accurate, and complete answer."""

def generate_conjecture_prompt(test_case: Dict[str, Any]) -> str:
    """Generate True Conjecture prompt with claim generation"""
    return f"""You are tasked with solving a complex problem using Conjecture's approach of breaking down the problem into smaller, manageable claims.

**Problem:**
{test_case['question']}

**Instructions:**
1. First, generate 3-5 specific claims about the problem in this exact format:
   [c1 | claim content | / confidence]
   [c2 | claim content | / confidence]
   etc.

2. Then, evaluate each claim and provide a final answer based on your analysis.

Format your response with:
- Claims section (using the exact format above)
- Analysis section (evaluating each claim)
- Final Answer section (your conclusion)"""

def run_single_test(model: str, test_case: Dict[str, Any], approach: str) -> Dict[str, Any]:
    """Run a single test case"""
    print(f"  Testing {model} with {approach} approach...")

    if approach == "direct":
        prompt = generate_direct_prompt(test_case)
    else:
        prompt = generate_conjecture_prompt(test_case)

    result = make_lm_studio_call(prompt, model)

    return {
        "model": model,
        "approach": approach,
        "test_case_id": test_case["id"],
        "prompt": prompt,
        "response": result["content"],
        "response_time": result["response_time"],
        "status": result["status"],
        "timestamp": datetime.now().isoformat()
    }

async def run_all_experiments():
    """Run complete experiment suite"""
    print("=" * 70)
    print("CONJECTURE HYPOTHESIS TEST: LM STUDIO MODELS")
    print("=" * 70)
    print(f"Testing: ibm/granite-4-h-tiny (tiny LLM) vs glm-z1-9b-0414 (mid-size)")
    print(f"Hypothesis: Conjecture enables tiny LLM to match larger model performance")
    print("=" * 70)

    models = ["ibm/granite-4-h-tiny", "glm-z1-9b-0414"]
    approaches = ["direct", "true_conjecture"]

    results = []

    # Test connectivity first
    print("\n[1/3] Testing LM Studio connectivity...")
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json().get("data", [])
            available_models = [m["id"] for m in models_data]
            print(f"[OK] LM Studio connected. Available models: {len(available_models)}")
            for model in models:
                status = "[OK]" if model in available_models else "[MISSING]"
                print(f"  {status} {model}")
        else:
            print(f"[FAIL] LM Studio returned status {response.status_code}")
            return
    except Exception as e:
        print(f"[FAIL] Cannot connect to LM Studio: {e}")
        return

    # Run experiments
    print("\n[2/3] Running experiments...")
    total_tests = len(models) * len(approaches) * len(TEST_CASES)
    current_test = 0

    for model in models:
        for approach in approaches:
            for test_case in TEST_CASES:
                current_test += 1
                print(f"\n  [{current_test}/{total_tests}] {model} - {approach} - {test_case['id']}")

                try:
                    result = run_single_test(model, test_case, approach)
                    results.append(result)

                    if result["status"] == "success":
                        print(f"    [OK] Completed in {result['response_time']:.1f}s")
                    else:
                        print(f"    [FAIL] Failed: {result.get('error', 'Unknown error')}")

                    # Small delay between requests
                    await asyncio.sleep(1)

                except Exception as e:
                    print(f"    [ERROR] Exception: {e}")
                    results.append({
                        "model": model,
                        "approach": approach,
                        "test_case_id": test_case["id"],
                        "status": "error",
                        "error": str(e)
                    })

    # Save results
    print("\n[3/3] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("research/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"lm_studio_experiment_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "experiment_id": f"lm_studio_{timestamp}",
            "models_tested": models,
            "approaches_tested": approaches,
            "test_cases": TEST_CASES,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"[OK] Results saved to: {results_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    successful_results = [r for r in results if r["status"] == "success"]
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(results) - len(successful_results)}")

    if successful_results:
        # Group by model and approach
        summary = {}
        for result in successful_results:
            key = f"{result['model']} | {result['approach']}"
            if key not in summary:
                summary[key] = []
            summary[key].append(result['response_time'])

        print("\nAverage response times:")
        for key, times in summary.items():
            avg_time = sum(times) / len(times)
            print(f"  {key}: {avg_time:.1f}s average")

    print("\n" + "=" * 70)
    print("Next steps:")
    print("1. Review detailed results in the JSON file")
    print("2. Analyze if Conjecture improved tiny LLM performance")
    print("3. Check if granite-4-h-tiny with Conjecture matches glm-z1-9b direct")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_all_experiments())
