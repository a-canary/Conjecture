#!/usr/bin/env python3
"""
Test Cloud Models Only (Chutes API)
Run this after the LM Studio models have been tested
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

# Cloud model configurations only
CLOUD_MODELS = [
    {
        "name": "openai/gpt-oss-20b",
        "type": "large",
        "provider": "chutes",
        "url": "https://llm.chutes.ai/v1",
        "api_key": os.getenv("CHUTES_API_KEY", ""),
        "description": "Large open-source model (20B parameters)"
    },
    {
        "name": "zai-org/GLM-4.6",
        "type": "sota",
        "provider": "chutes",
        "url": "https://llm.chutes.ai/v1",
        "api_key": os.getenv("CHUTES_API_KEY", ""),
        "description": "State-of-the-art model (benchmark)"
    }
]

# Test cases
TEST_CASES = [
    {
        "id": "logic_puzzle_complex",
        "category": "complex_reasoning",
        "question": "In a small town, there are five houses in a row, each painted a different color: red, blue, green, yellow, and white. Each house is owned by a person with a different profession: doctor, teacher, engineer, artist, and baker. Each person has a different favorite fruit: apple, banana, cherry, date, and elderberry. Using the following clues, determine who owns the red house and what is their favorite fruit?\n\nClues:\n1. The doctor lives in the middle house.\n2. The artist lives next to the person who likes apples.\n3. The engineer lives in the green house.\n4. The teacher likes bananas.\n5. The baker lives in the first house.\n6. The person who likes cherries lives next to the white house.\n7. The red house is somewhere to the left of the blue house.\n8. The artist does not live in the yellow house.\n9. The person who likes dates lives next to the doctor.\n10. The person who likes elderberries lives in the last house.",
    },
    {
        "id": "math_reasoning_multi_step",
        "category": "mathematical_reasoning",
        "question": "A train travels from City A to City B at 60 mph and returns at 40 mph. What is the average speed for the entire round trip? Explain why the answer is not 50 mph.",
    },
    {
        "id": "evidence_eval_conflicting",
        "category": "evidence_evaluation",
        "question": "Evaluate whether this new drug should be approved based on: 1) Two large studies show statistically significant blood pressure reduction, 2) Two smaller studies show no significant effect, 3) Side effects include mild headaches in 5% of patients and dizziness in 2%, 4) The drug costs 3x more than existing treatments, 5) All studies were funded by the pharmaceutical company. Provide a structured analysis with claims and confidence scores.",
    }
]

def make_chutes_api_call(prompt: str, model_config: Dict[str, Any], max_tokens: int = 2000) -> Dict[str, Any]:
    """Make API call to Chutes"""
    try:
        import requests

        url = model_config["url"]
        api_key = model_config["api_key"]
        model_name = model_config["name"]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }

        # Fix endpoint - don't double /v1
        endpoint = f"{url}/chat/completions"

        print(f"      Calling {endpoint} with model {model_name}")

        start_time = time.time()
        response = requests.post(endpoint, headers=headers, json=data, timeout=600)
        response.raise_for_status()
        end_time = time.time()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        return {
            "content": content,
            "response_time": end_time - start_time,
            "status": "success",
            "response_length": len(content)
        }

    except Exception as e:
        return {
            "content": f"Error: {str(e)}",
            "response_time": 0,
            "status": "error",
            "response_length": 0,
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
1. First, generate 3-7 specific claims about the problem in this exact format:
   [c1 | claim content | / confidence]
   [c2 | claim content | / confidence]
   etc.

2. Then, evaluate each claim and provide a final answer based on your analysis.

Format your response with:
- Claims section (using the exact format above)
- Analysis section (evaluating each claim)
- Final Answer section (your conclusion)"""

async def test_cloud_model(model_config: Dict[str, Any]):
    """Test a single cloud model"""
    print(f"\n{'=' * 70}")
    print(f"TESTING: {model_config['name']}")
    print(f"{'=' * 70}")

    if not model_config["api_key"]:
        print(f"[SKIP] No API key available")
        return []

    results = []
    approaches = ["direct", "true_conjecture"]

    for approach in approaches:
        print(f"\n[Approach: {approach.upper()}]")

        for i, test_case in enumerate(TEST_CASES):
            print(f"\n[{i+1}/{len(TEST_CASES)}] {test_case['id']}")

            prompt = generate_direct_prompt(test_case) if approach == "direct" else generate_conjecture_prompt(test_case)
            result = make_chutes_api_call(prompt, model_config)

            results.append({
                "model": model_config["name"],
                "type": model_config["type"],
                "approach": approach,
                "test_case_id": test_case["id"],
                "prompt": prompt,
                "response": result["content"],
                "response_time": result["response_time"],
                "response_length": result["response_length"],
                "status": result["status"],
                "error": result.get("error"),
                "timestamp": datetime.now().isoformat()
            })

            if result["status"] == "success":
                print(f"  [OK] {result['response_time']:.1f}s | {result['response_length']} chars")
            else:
                print(f"  [FAIL] {result.get('error')}")

            await asyncio.sleep(1)

    return results

async def run_cloud_experiments():
    """Run experiments for cloud models only"""
    print("=" * 70)
    print("CLOUD MODEL TESTING (Chutes API)")
    print("=" * 70)

    all_results = []

    for model in CLOUD_MODELS:
        model_results = await test_cloud_model(model)
        all_results.extend(model_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("research/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"cloud_models_experiment_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "experiment_id": f"cloud_models_{timestamp}",
            "timestamp": datetime.now().isoformat(),
            "models_tested": [m["name"] for m in CLOUD_MODELS],
            "test_cases": TEST_CASES,
            "results": all_results
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {results_file}")
    print(f"{'=' * 70}")

    # Summary
    successful = [r for r in all_results if r["status"] == "success"]
    failed = [r for r in all_results if r["status"] != "success"]

    print(f"\nTotal tests: {len(all_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\nAverage response times:")
        for model in CLOUD_MODELS:
            model_results = [r for r in successful if r["model"] == model["name"]]
            if model_results:
                avg_time = sum(r["response_time"] for r in model_results) / len(model_results)
                print(f"  {model['name']}: {avg_time:.1f}s")

if __name__ == "__main__":
    asyncio.run(run_cloud_experiments())
