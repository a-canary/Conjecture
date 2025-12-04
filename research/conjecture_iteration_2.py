#!/usr/bin/env python3
"""
Conjecture Iteration 2: Improved Claim Format Testing
Tests multiple claim format variations to improve compliance
Adds coding tasks and better evaluation
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
    claim_format_type: str = None
    reasoning_steps: int = 0
    correctness_score: float = None
    completeness_score: float = None
    coherence_score: float = None
    reasoning_quality_score: float = None
    agentic_capability_score: float = None
    coding_score: float = None
    timestamp: str = None

    def __post_init__(self):
        if self.claims_generated is None:
            self.claims_generated = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


# Enhanced test cases with coding tasks
TEST_CASES = [
    {
        "id": "logic_puzzle_simple",
        "category": "complex_reasoning",
        "question": """Five people live in five houses of different colors. Each person has a different profession and favorite fruit.

Clues:
1. The doctor lives in middle house
2. The baker lives in first house  
3. The teacher likes bananas
4. The engineer lives in green house
5. The person who likes elderberries lives in the last house

Who lives in the red house and what fruit do they like? Think step by step.""",
        "expected": "Logical deduction with clear reasoning",
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
    {
        "id": "coding_debug",
        "category": "coding_task",
        "question": """Debug this Python function that should find the median of a list:

```python
def find_median(numbers):
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    if n % 2 == 1:
        return sorted_nums[n//2]
    else:
        return (sorted_nums[n//2] + sorted_nums[n//2 - 1]) / 2
```

The function fails on empty lists. Identify the bug and provide a corrected version with proper error handling.

Break this down into claims about what's wrong and how to fix it.""",
        "expected": "Corrected code with edge case handling",
    },
    {
        "id": "coding_algorithm",
        "category": "coding_task",
        "question": """Write a function to check if a string is a palindrome.

Requirements:
- Ignore case and non-alphanumeric characters
- Return True if palindrome, False otherwise
- Include test cases

Break this into claims about the algorithm design, then implement.""",
        "expected": "Working palindrome function with tests",
    },
]

# LM Studio models
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

# Multiple claim format variations to test
CLAIM_FORMATS = {
    "original": {
        "description": "Original format: [c{id} | content | / confidence]",
        "pattern": r"\[c(\d+)\s*\|\s*([^|]+)\s*\|\s*/\s*([0-9.]+)\]",
        "prompt": """Use Conjecture approach:

1. Break this into specific claims
2. Generate claims in this exact format: [c{{id}} | claim content | / confidence_level]
3. Evaluate each claim systematically
4. Provide your final answer

Problem:
{question}

Generate claims first:""",
    },
    "simplified": {
        "description": "Simplified format: Claim {id}: content (confidence: X%)",
        "pattern": r"Claim\s+(\d+):\s*([^()]+)\s*\(confidence:\s*([0-9.]+)%\)",
        "prompt": """Use Conjecture approach with simplified format:

1. Break this into specific claims
2. Generate claims in this format: Claim {{id}}: content (confidence: X%)
3. Evaluate each claim systematically
4. Provide your final answer

Problem:
{question}

Generate claims first:""",
    },
    "minimal": {
        "description": "Minimal format: C{id}: content [X%]",
        "pattern": r"C(\d+):\s*([^\[]+)\s*\[([0-9.]+)%\]",
        "prompt": """Use Conjecture approach with minimal format:

1. Break this into specific claims
2. Generate claims in this format: C{{id}}: content [X%]
3. Evaluate each claim systematically
4. Provide your final answer

Problem:
{question}

Generate claims first:""",
    },
    "natural": {
        "description": "Natural language: Claim X: content (confidence X%)",
        "pattern": r"Claim\s+(\d+):\s*([^()]+)\s*\(confidence\s+([0-9.]+)%\)",
        "prompt": """Use Conjecture approach with natural language:

1. Break this into specific claims
2. Generate claims in this format: Claim X: content (confidence X%)
3. Evaluate each claim systematically
4. Provide your final answer

Problem:
{question}

Generate claims first:""",
    },
}

APPROACHES = [
    "direct",
    "conjecture_original",
    "conjecture_simplified",
    "conjecture_minimal",
    "conjecture_natural",
]


def make_api_call(
    prompt: str, model_config: Dict, max_tokens: int = 2000
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
    elif approach.startswith("conjecture_"):
        format_type = approach.replace("conjecture_", "")
        if format_type in CLAIM_FORMATS:
            format_config = CLAIM_FORMATS[format_type]
            return format_config["prompt"].format(question=base_question)
        else:
            return f"Use Conjecture approach: {base_question}"
    else:
        return f"Answer this question: {base_question}"


def parse_claims(response: str, format_type: str) -> Tuple[List[Dict[str, Any]], bool]:
    """Parse claims from response using specific format pattern"""
    if format_type not in CLAIM_FORMATS:
        return [], False

    pattern = CLAIM_FORMATS[format_type]["pattern"]
    matches = re.findall(pattern, response, re.IGNORECASE)

    claims = []
    for match in matches:
        claims.append(
            {"id": match[0], "content": match[1].strip(), "confidence": float(match[2])}
        )

    return claims, len(claims) > 0


def count_reasoning_steps(response: str) -> int:
    """Count reasoning indicators"""
    step_indicators = [
        "step",
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "next",
        "then",
        "finally",
        "because",
        "since",
        "however",
        "although",
        "moreover",
        "therefore",
        "thus",
        "hence",
    ]
    response_lower = response.lower()
    count = sum(1 for indicator in step_indicators if indicator in response_lower)
    return min(count, 10)


def evaluate_coding_response(response: str, test_case_id: str) -> float:
    """Evaluate coding task responses"""
    if not test_case_id.startswith("coding_"):
        return None

    # Check for code elements
    has_function_def = "def " in response
    has_error_handling = any(
        keyword in response.lower()
        for keyword in ["try:", "except", "if not", "error", "empty"]
    )
    has_test_cases = any(
        keyword in response.lower() for keyword in ["test", "assert", "example"]
    )

    score = 0.0
    if has_function_def:
        score += 0.4
    if has_error_handling:
        score += 0.3
    if has_test_cases:
        score += 0.3

    return min(score, 1.0)


def enhanced_evaluation(result: TestResult) -> TestResult:
    """Enhanced evaluation with coding assessment"""
    response = result.response.lower()

    # Basic quality metrics
    result.correctness_score = min(len(result.response) / 300, 1.0)
    result.completeness_score = 1.0 if len(result.response) > 150 else 0.6

    # Coherence (logical flow)
    coherence_indicators = [
        "because",
        "therefore",
        "however",
        "first",
        "second",
        "finally",
        "step",
        "next",
    ]
    coherence_count = sum(
        1 for indicator in coherence_indicators if indicator in response
    )
    result.coherence_score = min(coherence_count / 4, 1.0)

    # Reasoning quality
    result.reasoning_quality_score = min(result.reasoning_steps / 5, 1.0)

    # Agentic capability (planning, structure)
    agentic_indicators = [
        "plan",
        "step",
        "agenda",
        "first",
        "next",
        "finally",
        "structure",
        "organize",
    ]
    agentic_count = sum(1 for indicator in agentic_indicators if indicator in response)
    result.agentic_capability_score = min(agentic_count / 4, 1.0)

    # Coding evaluation
    result.coding_score = evaluate_coding_response(response, result.test_case_id)

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

        # Parse claims if Conjecture approach
        claims = []
        has_claim_format = False
        claim_format_type = None

        if approach.startswith("conjecture_"):
            format_type = approach.replace("conjecture_", "")
            claims, has_claim_format = parse_claims(response, format_type)
            claim_format_type = format_type if has_claim_format else None

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
            claim_format_type=claim_format_type,
            reasoning_steps=reasoning_steps,
        )

        # Enhanced evaluation
        result = enhanced_evaluation(result)

        quality_score = result.reasoning_quality_score or 0
        if result.coding_score is not None:
            quality_score = (quality_score + result.coding_score) / 2

        print(
            f"      OK {response_time:.1f}s, Quality: {quality_score:.2f}, Claims: {len(claims)}"
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
    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_key} ({model_config['type']})")
    print(f"{'=' * 60}")

    results = []

    for approach in APPROACHES:
        print(f"\n  Approach: {approach}")

        for test_case in TEST_CASES:
            result = await run_single_test(model_key, model_config, test_case, approach)
            results.append(result)
            await asyncio.sleep(0.5)

    return results


async def run_iteration_2():
    """Run iteration 2 with improved claim formats"""
    print("=" * 80)
    print("CONJECTURE ITERATION 2: IMPROVED FORMAT TESTING")
    print("Testing multiple claim format variations + coding tasks")
    print("=" * 80)

    print(f"\nModels: {len(MODELS)} LM Studio models")
    print(f"Approaches: {len(APPROACHES)} (1 direct + 4 claim formats)")
    print(f"Test cases: {len(TEST_CASES)} (including coding tasks)")
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
    filename = f"conjecture_iteration_2_{timestamp}.json"
    filepath = Path("research/results") / filename

    data = {
        "experiment_id": f"conjecture_iteration_2_{timestamp}",
        "timestamp": datetime.now().isoformat(),
        "models_tested": list(MODELS.keys()),
        "approaches_tested": APPROACHES,
        "claim_formats": {
            name: config["description"] for name, config in CLAIM_FORMATS.items()
        },
        "test_cases": [
            {"id": tc["id"], "category": tc["category"]} for tc in TEST_CASES
        ],
        "results": [asdict(result) for result in all_results],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[OK] Results saved to: {filepath}")

    # Analysis
    await generate_iteration_analysis(all_results)

    return all_results


async def generate_iteration_analysis(results: List[TestResult]):
    """Generate analysis for iteration 2"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"iteration_2_analysis_{timestamp}.md"
    filepath = Path("research/results") / filename

    # Claim format compliance analysis
    format_compliance = {}
    for approach in APPROACHES:
        if approach.startswith("conjecture_"):
            approach_results = [
                r for r in results if r.approach == approach and r.status == "success"
            ]
            if approach_results:
                compliance_rate = sum(
                    1 for r in approach_results if r.has_claim_format
                ) / len(approach_results)
                format_compliance[approach] = compliance_rate

    # Model performance by approach
    model_performance = {}
    for model_key in MODELS.keys():
        model_performance[model_key] = {}
        for approach in APPROACHES:
            approach_results = [
                r
                for r in results
                if r.model == model_key
                and r.approach == approach
                and r.status == "success"
            ]
            if approach_results:
                avg_quality = sum(
                    r.reasoning_quality_score or 0 for r in approach_results
                ) / len(approach_results)
                avg_coding = sum(
                    r.coding_score or 0
                    for r in approach_results
                    if r.coding_score is not None
                ) / len([r for r in approach_results if r.coding_score is not None])
                model_performance[model_key][approach] = {
                    "avg_quality": avg_quality,
                    "avg_coding": avg_coding,
                    "count": len(approach_results),
                }

    report = f"""# Conjecture Iteration 2 Analysis
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

Tested {len(MODELS)} models with {len(APPROACHES)} approaches on {len(TEST_CASES)} test cases (including coding).
Key focus: **Claim format compliance** and **coding task evaluation**.

## Claim Format Compliance Results

"""

    for approach, compliance in format_compliance.items():
        format_name = approach.replace("conjecture_", "")
        format_desc = CLAIM_FORMATS.get(format_name, {}).get("description", "Unknown")
        report += f"""### {format_name.title()} Format
- Description: {format_desc}
- Compliance Rate: {compliance:.1%}
- Status: {"✅ SUCCESS" if compliance > 0.5 else "❌ NEEDS IMPROVEMENT"}

"""

    report += """
## Model Performance by Approach

"""

    for model_key, approaches in model_performance.items():
        report += f"### {model_key}\n\n"
        for approach, metrics in approaches.items():
            report += f"**{approach}**:\n"
            report += f"- Quality: {metrics['avg_quality']:.3f}\n"
            if metrics["avg_coding"] > 0:
                report += f"- Coding: {metrics['avg_coding']:.3f}\n"
            report += f"- Tests: {metrics['count']}\n\n"

    # Best format recommendation
    best_format = (
        max(format_compliance.items(), key=lambda x: x[1])
        if format_compliance
        else None
    )
    if best_format:
        best_format_name = best_format[0].replace("conjecture_", "")
        report += f"""## Recommendation

**Best Claim Format**: {best_format_name.title()}

Reason: {format_compliance[best_format[0]]:.1%} compliance rate - highest among tested formats.

## Next Steps

1. **Adopt best format** for future experiments
2. **Test on more models** with proven format
3. **Expand coding tasks** for better agentic evaluation
4. **Compare against SOTA** benchmarks

---
*Iteration 2 Analysis Complete*
"""

    with open(filepath, "w") as f:
        f.write(report)

    print(f"[OK] Analysis saved to: {filepath}")


if __name__ == "__main__":
    asyncio.run(run_iteration_2())
