#!/usr/bin/env python3
"""
Simplified comprehensive experiment runner
Focus on LM Studio models only for now
"""

import os
import sys
import json
import time
import asyncio
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv()
    print("[OK] Environment loaded")
except ImportError:
    print("[FAIL] python-dotenv not available")


@dataclass
class TestResult:
    """Complete test result with quality metrics"""

    model: str
    model_type: str
    approach: str
    test_case_id: str
    test_category: str
    prompt: str
    response: str
    response_time: float
    response_length: int
    status: str
    error: str = None
    claims_generated: List[Dict[str, Any]] = None
    has_claim_format: bool = False
    reasoning_steps: int = 0
    correctness_score: float = None
    completeness_score: float = None
    coherence_score: float = None
    reasoning_quality_score: float = None
    agentic_capability_score: float = None
    timestamp: str = None

    def __post_init__(self):
        if self.claims_generated is None:
            self.claims_generated = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


# Simplified test cases
TEST_CASES = [
    {
        "id": "logic_puzzle_simple",
        "category": "complex_reasoning",
        "question": """Five people live in five houses of different colors. Each person has a different profession and favorite fruit.

Clues:
1. The doctor lives in the middle house
2. The baker lives in the first house  
3. The teacher likes bananas
4. The engineer lives in the green house
5. The person who likes elderberries lives in the last house

Who lives in the red house and what fruit do they like? Think step by step.""",
        "expected": "Need to analyze step by step",
    },
    {
        "id": "planning_simple",
        "category": "agentic_planning",
        "question": """Plan a 2-hour team meeting for 5 people to discuss a project launch.

Requirements:
- Review project status (30 min)
- Brainstorm marketing ideas (45 min)
- Assign action items (30 min)
- Q&A session (15 min)

Create a detailed agenda with timing and responsibilities.""",
        "expected": "Structured plan with time allocation",
    },
    {
        "id": "evidence_simple",
        "category": "evidence_evaluation",
        "question": """A new software update shows mixed results:
- 70% of users report improved performance
- 30% report crashes
- No change in overall satisfaction scores

Should you recommend this update? Provide reasoning with claims and confidence levels.""",
        "expected": "Balanced analysis with pros/cons",
    },
]

# LM Studio models only
MODELS = {
    "lmstudio:ibm/granite-4-h-tiny": {
        "type": "tiny",
        "provider": "lmstudio",
        "url": "http://localhost:1234",
        "api_key": "",
    },
    "lmstudio:glm-z1-9b-0414": {
        "type": "medium",
        "provider": "lmstudio",
        "url": "http://localhost:1234",
        "api_key": "",
    },
    "lmstudio:qwen3-4b-thinking-2507": {
        "type": "tiny",
        "provider": "lmstudio",
        "url": "http://localhost:1234",
        "api_key": "",
    },
}

APPROACHES = ["direct", "true_conjecture"]


def make_api_call(
    prompt: str, model_config: Dict, max_tokens: int = 1500
) -> Dict[str, Any]:
    """Make API call to LM Studio"""
    try:
        import requests

        headers = {"Content-Type": "application/json"}

        model_name = model_config["name"].split(":", 1)[1]  # Remove "lmstudio:" prefix

        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }

        response = requests.post(
            f"{model_config['url']}/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=120,
        )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return {
                "status": "success",
                "content": content,
                "usage": result.get("usage", {}),
            }
        else:
            return {
                "status": "error",
                "error": f"API error {response.status_code}: {response.text}",
            }

    except Exception as e:
        return {"status": "error", "error": f"Exception: {str(e)}"}


def create_prompt(test_case: Dict, approach: str) -> str:
    """Create prompt based on approach"""
    base_question = test_case["question"]

    if approach == "direct":
        return f"""Please answer this question accurately and clearly:

{base_question}

Your answer:"""

    elif approach == "true_conjecture":
        return f"""Use the Conjecture approach:

1. Break this into specific claims
2. Format each claim as: [c{{id}} | claim | / confidence]
3. Evaluate claims systematically
4. Give final answer

Problem:
{base_question}

Generate claims first:"""


def parse_claims(response: str) -> List[Dict[str, Any]]:
    """Parse claims from response"""
    claims = []
    pattern = r"\[c(\d+)\s*\|\s*([^|]+)\s*\|\s*/\s*([0-9.]+)\]"

    matches = re.findall(pattern, response)
    for match in matches:
        claims.append(
            {"id": match[0], "content": match[1].strip(), "confidence": float(match[2])}
        )

    return claims


def count_reasoning_steps(response: str) -> int:
    """Count reasoning indicators"""
    step_indicators = [
        "step",
        "first",
        "second",
        "third",
        "next",
        "then",
        "finally",
        "because",
        "since",
    ]
    response_lower = response.lower()
    count = sum(1 for indicator in step_indicators if indicator in response_lower)
    return min(count, 8)


def simple_evaluation(result: TestResult) -> TestResult:
    """Simple evaluation without LLM judge"""
    # Basic quality metrics
    response = result.response.lower()

    # Correctness (based on response completeness)
    result.correctness_score = min(len(result.response) / 200, 1.0)

    # Completeness (addresses question)
    result.completeness_score = 1.0 if len(result.response) > 100 else 0.5

    # Coherence (logical flow)
    coherence_indicators = [
        "because",
        "therefore",
        "however",
        "first",
        "second",
        "finally",
    ]
    coherence_count = sum(
        1 for indicator in coherence_indicators if indicator in response
    )
    result.coherence_score = min(coherence_count / 3, 1.0)

    # Reasoning quality
    result.reasoning_quality_score = min(result.reasoning_steps / 4, 1.0)

    # Agentic capability (planning, structure)
    agentic_indicators = ["plan", "step", "agenda", "first", "next", "finally"]
    agentic_count = sum(1 for indicator in agentic_indicators if indicator in response)
    result.agentic_capability_score = min(agentic_count / 3, 1.0)

    return result


async def run_single_test(
    model_key: str, model_config: Dict, test_case: Dict, approach: str
) -> TestResult:
    """Run single test"""
    print(f"    Testing {approach} on {test_case['id']}...")

    start_time = time.time()
    prompt = create_prompt(test_case, approach)

    api_result = make_api_call(prompt, {**model_config, "name": model_key})
    response_time = time.time() - start_time

    if api_result["status"] == "success":
        response = api_result["content"]

        claims = []
        has_claim_format = False
        if approach == "true_conjecture":
            claims = parse_claims(response)
            has_claim_format = len(claims) > 0

        reasoning_steps = count_reasoning_steps(response)

        result = TestResult(
            model=model_key,
            model_type=model_config["type"],
            approach=approach,
            test_case_id=test_case["id"],
            test_category=test_case["category"],
            prompt=prompt,
            response=response,
            response_time=response_time,
            response_length=len(response),
            status="success",
            claims_generated=claims,
            has_claim_format=has_claim_format,
            reasoning_steps=reasoning_steps,
        )

        # Simple evaluation
        result = simple_evaluation(result)

        print(
            f"      OK {response_time:.1f}s, Quality: {result.reasoning_quality_score:.2f}"
        )
        return result
    else:
        print(f"      FAIL {api_result['error']}")
        return TestResult(
            model=model_key,
            model_type=model_config["type"],
            approach=approach,
            test_case_id=test_case["id"],
            test_category=test_case["category"],
            prompt=prompt,
            response="",
            response_time=response_time,
            response_length=0,
            status="failed",
            error=api_result["error"],
        )


async def run_model_tests(model_key: str, model_config: Dict) -> List[TestResult]:
    """Run all tests for a single model"""
    print(f"\n{'=' * 50}")
    print(f"MODEL: {model_key} ({model_config['type']})")
    print(f"{'=' * 50}")

    results = []

    for approach in APPROACHES:
        print(f"\n  Approach: {approach}")

        for test_case in TEST_CASES:
            result = await run_single_test(model_key, model_config, test_case, approach)
            results.append(result)
            await asyncio.sleep(0.5)

    return results


async def run_simplified_experiment():
    """Run simplified experiment"""
    print("=" * 60)
    print("SIMPLIFIED CONJECTURE EXPERIMENT")
    print("LM Studio models only - focus on reasoning quality")
    print("=" * 60)

    print(f"\nModels: {len(MODELS)} LM Studio models")
    print(f"Approaches: {', '.join(APPROACHES)}")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Total evaluations: {len(MODELS) * len(APPROACHES) * len(TEST_CASES)}")

    # Test connectivity
    print("\nTesting connectivity...")
    for model_key, config in MODELS.items():
        test_prompt = "Say 'OK' if you can read this."
        result = make_api_call(
            test_prompt, {**config, "name": model_key}, max_tokens=10
        )
        if result["status"] == "success":
            print(f"  {model_key}: OK")
        else:
            print(f"  {model_key}: FAIL - {result['error']}")

    # Run experiments
    print("\nRunning experiments...")
    all_results = []

    for model_key, model_config in MODELS.items():
        model_results = await run_model_tests(model_key, model_config)
        all_results.extend(model_results)
        await asyncio.sleep(2)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simplified_conjecture_{timestamp}.json"
    filepath = Path("research/results") / filename

    data = {
        "experiment_id": f"simplified_conjecture_{timestamp}",
        "timestamp": datetime.now().isoformat(),
        "models_tested": list(MODELS.keys()),
        "approaches_tested": APPROACHES,
        "test_cases": [
            {"id": tc["id"], "category": tc["category"]} for tc in TEST_CASES
        ],
        "results": [asdict(result) for result in all_results],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[OK] Results saved to: {filepath}")

    # Simple analysis
    print("\n" + "=" * 60)
    print("QUICK ANALYSIS")
    print("=" * 60)

    # Group by model and approach
    for model_key in MODELS.keys():
        print(f"\n{model_key}:")
        for approach in APPROACHES:
            approach_results = [
                r
                for r in all_results
                if r.model == model_key
                and r.approach == approach
                and r.status == "success"
            ]
            if approach_results:
                avg_quality = sum(
                    r.reasoning_quality_score or 0 for r in approach_results
                ) / len(approach_results)
                avg_correctness = sum(
                    r.correctness_score or 0 for r in approach_results
                ) / len(approach_results)
                avg_time = sum(r.response_time for r in approach_results) / len(
                    approach_results
                )
                claim_success = sum(
                    1 for r in approach_results if r.has_claim_format
                ) / len(approach_results)

                print(f"  {approach}:")
                print(
                    f"    Quality: {avg_quality:.3f}, Correctness: {avg_correctness:.3f}"
                )
                print(f"    Time: {avg_time:.1f}s, Claims: {claim_success:.2f} success")

    return all_results


if __name__ == "__main__":
    asyncio.run(run_simplified_experiment())
