#!/usr/bin/env python3
"""
Comprehensive Conjecture Research Runner
Tests reasoning and agentic capabilities with quality evaluation
Model-by-model execution to prevent LM Studio reloading
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
    model_type: str  # "tiny", "medium", "large", "sota"
    approach: str
    test_case_id: str
    test_category: str
    prompt: str
    response: str
    response_time: float
    response_length: int
    status: str
    error: str = None
    # Quality metrics
    claims_generated: List[Dict[str, Any]] = None
    has_claim_format: bool = False
    reasoning_steps: int = 0
    # Evaluation scores (populated by LLM judge)
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

# Enhanced test cases focusing on reasoning and agentic capabilities
TEST_CASES = [
    {
        "id": "logic_puzzle_complex",
        "category": "complex_reasoning",
        "difficulty": "hard",
        "question": """In a small town, there are five houses in a row, each painted a different color: red, blue, green, yellow, and white. 
Each house is owned by a person with a different profession: doctor, teacher, engineer, artist, and baker. 
Each person has a different favorite fruit: apple, banana, cherry, date, and elderberry.

Using these clues, determine who owns the red house and what is their favorite fruit:
1. The doctor lives in the middle house.
2. The artist lives next to the person who likes apples.
3. The engineer lives in the green house.
4. The teacher likes bananas.
5. The baker lives in the first house.
6. The person who likes cherries lives next to the white house.
7. The red house is somewhere to the left of the blue house.
8. The artist does not live in the yellow house.
9. The person who likes dates lives next to the doctor.
10. The person who likes elderberries lives in the last house.

Think step by step and show your reasoning.""",
        "expected_approach": "systematic_deduction",
        "evaluation_focus": ["logical_consistency", "step_by_step_reasoning"],
    },
    {
        "id": "planning_task",
        "category": "agentic_planning",
        "difficulty": "medium",
        "question": """You need to organize a 3-day technical conference for 100 developers. 
Requirements:
- Budget: $10,000
- Venue must accommodate 100 people with tech setup
- Need 6 speakers (3 keynotes, 3 workshops)
- Include catering for lunch all 3 days
- Provide networking opportunities
- Create a detailed timeline and action plan

Break this down into specific, actionable steps with priorities and dependencies.""",
        "expected_approach": "task_decomposition",
        "evaluation_focus": [
            "planning_capability",
            "task_breakdown",
            "priority_setting",
        ],
    },
    {
        "id": "evidence_synthesis",
        "category": "evidence_evaluation",
        "difficulty": "hard",
        "question": """Evaluate whether this new drug should be approved based on conflicting evidence:

Positive evidence:
- Two large studies (n=2000 each) show statistically significant blood pressure reduction (p<0.01)
- Meta-analysis of 8 studies shows overall positive effect
- Drug mechanism is well-understood and scientifically plausible

Negative evidence:
- Two smaller studies (n=300 each) show no significant effect
- Side effects: mild headaches in 5% of patients, dizziness in 2%
- Drug costs 3x more than existing treatments
- All studies were funded by the pharmaceutical company

Provide a balanced evaluation with specific claims and confidence levels for each claim.""",
        "expected_approach": "evidence_balancing",
        "evaluation_focus": [
            "critical_thinking",
            "evidence_weighting",
            "confidence_calibration",
        ],
    },
    {
        "id": "debugging_challenge",
        "category": "technical_reasoning",
        "difficulty": "medium",
        "question": """A Python function is supposed to find the median of a list of numbers, but it's failing on edge cases.

Current code:
```python
def find_median(numbers):
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    if n % 2 == 1:
        return sorted_nums[n//2]
    else:
        return (sorted_nums[n//2] + sorted_nums[n//2 - 1]) / 2
```

Test cases that fail:
- find_median([]) -> should return None
- find_median([5]) -> works
- find_median([1, 2, 3, 4]) -> works
- find_median([1, 1, 1, 1]) -> works

Identify the bug, explain why it fails, and provide a corrected version with edge case handling.""",
        "expected_approach": "systematic_debugging",
        "evaluation_focus": [
            "technical_reasoning",
            "edge_case_analysis",
            "problem_solving",
        ],
    },
    {
        "id": "causal_reasoning",
        "category": "causal_inference",
        "difficulty": "hard",
        "question": """A company noticed that after implementing a new training program, employee productivity increased by 15%. 
The CEO wants to claim the training program caused the improvement.

Consider these factors:
- The program was implemented during the company's busiest season
- A new performance bonus system was introduced at the same time
- Employee turnover was low during this period
- The previous quarter had unusually low productivity

Analyze whether the training program actually caused the productivity increase. 
Consider alternative explanations and what additional data would be needed to establish causality.""",
        "expected_approach": "causal_analysis",
        "evaluation_focus": [
            "causal_reasoning",
            "alternative_explanations",
            "critical_analysis",
        ],
    },
]

# Model configurations
MODELS = {
    # LM Studio models (local)
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
    # Cloud benchmark models (using available models)
    "cerebras:llama3.1-8b": {
        "type": "medium",
        "provider": "cerebras",
        "url": "https://api.cerebras.ai/v1",
        "api_key": "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8",
    },
}

APPROACHES = ["direct", "true_conjecture", "chain_of_thought"]

def make_api_call(
    prompt: str, model_config: Dict, max_tokens: int = 2000
) -> Dict[str, Any]:
    """Make API call to specified model"""
    try:
        import requests

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model_config['api_key']}"
            if model_config["api_key"]
            else "",
        }

        data = {
            "model": model_config["name"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }

        # Handle different API formats
        if model_config["provider"] == "lmstudio":
            # LM Studio uses OpenAI-compatible format
            data["model"] = model_config["name"].split(":", 1)[
                1
            ]  # Remove "lmstudio:" prefix
            response = requests.post(
                f"{model_config['url']}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=120,
            )
        elif model_config["provider"] == "cerebras":
            # Cerebras API uses OpenAI-compatible format
            data["model"] = model_config["name"].split(":", 1)[
                1
            ]  # Remove "cerebras:" prefix
            response = requests.post(
                f"{model_config['url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=120,
            )
        else:
            # Default API
            response = requests.post(
                f"{model_config['url']}/chat/completions",
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
        return f"""Please answer the following question directly and accurately:

{base_question}

Provide your complete answer below:"""

    elif approach == "true_conjecture":
        return f"""Use the Conjecture approach to solve this problem. Follow these steps:

1. Break down the problem into specific claims
2. Generate claims in this exact format: [c{{id}} | claim content | / confidence_level]
3. Evaluate each claim systematically
4. Provide your final answer based on claim analysis

Problem:
{base_question}

First, generate your claims, then evaluate them:"""

    elif approach == "chain_of_thought":
        return f"""Think step by step to solve this problem. Show your reasoning process clearly.

{base_question}

Work through this step by step:"""

def parse_claims(response: str) -> List[Dict[str, Any]]:
    """Parse claims from Conjecture response"""
    claims = []
    pattern = r"\[c(\d+)\s*\|\s*([^|]+)\s*\|\s*/\s*([0-9.]+)\]"

    matches = re.findall(pattern, response)
    for match in matches:
        claims.append(
            {"id": match[0], "content": match[1].strip(), "confidence": float(match[2])}
        )

    return claims

def count_reasoning_steps(response: str) -> int:
    """Count reasoning indicators in response"""
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
        "therefore",
        "because",
        "since",
        "however",
        "although",
        "moreover",
        "furthermore",
    ]

    response_lower = response.lower()
    count = sum(1 for indicator in step_indicators if indicator in response_lower)
    return min(count, 10)  # Cap at 10 for normalization

async def evaluate_with_llm_judge(test_result: TestResult) -> TestResult:
    """Evaluate test result using LLM-as-a-Judge"""
    try:
        # Import judge system
        sys.path.insert(0, str(Path(__file__).parent / "experiments"))
        from llm_judge import LLMJudge
        from processing.llm.llm_manager import LLMManager
        from config.common import ProviderConfig

        # Setup judge
        judge_config = ProviderConfig(
            url="https://api.cerebras.ai/v1",
            api_key="csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8",
            model="zai-glm-4.6",
        )

        llm_manager = LLMManager()
        await llm_manager.add_provider("chutes", judge_config)

        judge = LLMJudge(llm_manager)

        # Evaluate based on test case category
        evaluation_prompt = f"""Evaluate this model response on a scale of 0.0 to 1.0 for each criterion:

Test Case: {test_result.test_case_id} ({test_result.test_category})
Question: {test_result.prompt[:500]}...
Model Response: {test_result.response[:1000]}...

Criteria:
1. Correctness: Factual accuracy and correctness
2. Completeness: Addresses all aspects of the question  
3. Coherence: Logical flow and consistency
4. Reasoning Quality: Quality of logical reasoning and argumentation
5. Agentic Capability: Planning, task decomposition, goal-directed behavior

Provide scores in JSON format:
{{"correctness": 0.0-1.0, "completeness": 0.0-1.0, "coherence": 0.0-1.0, "reasoning_quality": 0.0-1.0, "agentic_capability": 0.0-1.0}}"""

        # Get evaluation
        eval_result = await judge.evaluate_response(
            response=test_result.response,
            question=test_result.prompt,
            criteria=[
                "correctness",
                "completeness",
                "coherence",
                "reasoning_quality",
                "agentic_capability",
            ],
        )

        # Update test result with scores
        if eval_result and len(eval_result) > 0:
            evaluation = eval_result[0]  # Take first evaluation
            test_result.correctness_score = evaluation.get("correctness", 0.0)
            test_result.completeness_score = evaluation.get("completeness", 0.0)
            test_result.coherence_score = evaluation.get("coherence", 0.0)
            test_result.reasoning_quality_score = evaluation.get(
                "reasoning_quality", 0.0
            )
            test_result.agentic_capability_score = evaluation.get(
                "agentic_capability", 0.0
            )

    except Exception as e:
        print(f"[WARN] LLM Judge evaluation failed: {e}")
        # Set default scores
        test_result.correctness_score = 0.5
        test_result.completeness_score = 0.5
        test_result.coherence_score = 0.5
        test_result.reasoning_quality_score = 0.5
        test_result.agentic_capability_score = 0.5

    return test_result

async def run_single_test(
    model_key: str, model_config: Dict, test_case: Dict, approach: str
) -> TestResult:
    """Run single test with specified model and approach"""
    print(f"  Testing {model_key} with {approach} approach on {test_case['id']}...")

    start_time = time.time()

    # Create prompt
    prompt = create_prompt(test_case, approach)

    # Make API call
    api_result = make_api_call(prompt, {**model_config, "name": model_key})

    response_time = time.time() - start_time

    if api_result["status"] == "success":
        response = api_result["content"]

        # Parse claims if Conjecture approach
        claims = []
        has_claim_format = False
        if approach == "true_conjecture":
            claims = parse_claims(response)
            has_claim_format = len(claims) > 0

        # Count reasoning steps
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

        # Evaluate with LLM judge
        result = await evaluate_with_llm_judge(result)

        print(
            f"    [OK] Completed in {response_time:.1f}s, Quality: {result.reasoning_quality_score:.2f}"
        )
        return result
    else:
        print(f"    [FAIL] {api_result['error']}")
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
    """Run all tests for a single model (prevents reloading)"""
    print(f"\n{'=' * 60}")
    print(f"TESTING MODEL: {model_key} ({model_config['type']})")
    print(f"{'=' * 60}")

    results = []

    for approach in APPROACHES:
        print(
            f"\n[{APPROACHES.index(approach) + 1}/{len(APPROACHES)}] Approach: {approach}"
        )

        for test_case in TEST_CASES:
            result = await run_single_test(model_key, model_config, test_case, approach)
            results.append(result)

            # Small delay between requests
            await asyncio.sleep(1)

    return results

async def run_comprehensive_experiment():
    """Run comprehensive experiment with all models"""
    print("=" * 80)
    print("COMPREHENSIVE CONJECTURE RESEARCH EXPERIMENT")
    print("Testing reasoning and agentic capabilities with quality evaluation")
    print("=" * 80)

    print(f"\nModels to test: {len(MODELS)}")
    print(f"Approaches: {', '.join(APPROACHES)}")
    print(f"Test cases: {len(TEST_CASES)}")
    print(f"Total evaluations: {len(MODELS) * len(APPROACHES) * len(TEST_CASES)}")

    # Test connectivity first
    print("\n[1/2] Testing model connectivity...")
    await test_model_connectivity()

    # Run experiments model-by-model
    print("\n[2/2] Running experiments...")
    all_results = []

    for model_key, model_config in MODELS.items():
        model_results = await run_model_tests(model_key, model_config)
        all_results.extend(model_results)

        # Longer delay between models (allows LM Studio to load new model if needed)
        await asyncio.sleep(3)

    # Save results
    await save_results(all_results)

    # Generate analysis
    await generate_analysis(all_results)

    return all_results

async def test_model_connectivity():
    """Test connectivity to all models"""
    for model_key, config in MODELS.items():
        print(f"  Testing {model_key}...")

        test_prompt = "Respond with 'OK' if you can read this."
        result = make_api_call(
            test_prompt, {**config, "name": model_key}, max_tokens=10
        )

        if result["status"] == "success":
            print(f"    [OK] Connected")
        else:
            print(f"    [FAIL] {result['error']}")

async def save_results(results: List[TestResult]):
    """Save results to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_conjecture_{timestamp}.json"
    filepath = Path("research/results") / filename

    # Convert to serializable format
    data = {
        "experiment_id": f"comprehensive_conjecture_{timestamp}",
        "timestamp": datetime.now().isoformat(),
        "models_tested": list(MODELS.keys()),
        "approaches_tested": APPROACHES,
        "test_cases": [
            {"id": tc["id"], "category": tc["category"]} for tc in TEST_CASES
        ],
        "results": [asdict(result) for result in results],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n[OK] Results saved to: {filepath}")

async def generate_analysis(results: List[TestResult]):
    """Generate analysis report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_analysis_{timestamp}.md"
    filepath = Path("research/results") / filename

    # Group results by model and approach
    model_performance = {}
    for result in results:
        if result.status != "success":
            continue

        key = f"{result.model} | {result.approach}"
        if key not in model_performance:
            model_performance[key] = []
        model_performance[key].append(result)

    # Calculate statistics
    report = f"""# Comprehensive Conjecture Research Analysis
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This experiment evaluated {len(MODELS)} models using {len(APPROACHES)} approaches on {len(TEST_CASES)} diverse test cases focusing on reasoning and agentic capabilities.

## Model Performance by Approach

"""

    for key, result_list in model_performance.items():
        if not result_list:
            continue

        avg_quality = sum(r.reasoning_quality_score or 0 for r in result_list) / len(
            result_list
        )
        avg_correctness = sum(r.correctness_score or 0 for r in result_list) / len(
            result_list
        )
        avg_agentic = sum(r.agentic_capability_score or 0 for r in result_list) / len(
            result_list
        )
        avg_time = sum(r.response_time for r in result_list) / len(result_list)

        report += f"""### {key}
- Average Reasoning Quality: {avg_quality:.3f}
- Average Correctness: {avg_correctness:.3f}
- Average Agentic Capability: {avg_agentic:.3f}
- Average Response Time: {avg_time:.1f}s
- Tests Completed: {len(result_list)}

"""

    # Add detailed analysis
    report += """
## Key Findings

### Reasoning Quality Analysis
[Analysis of reasoning quality across models and approaches]

### Agentic Capability Assessment  
[Analysis of planning, task decomposition, and goal-directed behavior]

### Conjecture Effectiveness
[Analysis of whether Conjecture approach improves tiny model performance]

### Model Size vs Approach Effectiveness
[Analysis of how different approaches work across model sizes]

## Recommendations

[Specific recommendations based on experimental results]

---
*Report generated by Comprehensive Conjecture Research Runner*
"""

    with open(filepath, "w") as f:
        f.write(report)

    print(f"[OK] Analysis report saved to: {filepath}")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_experiment())
