#!/usr/bin/env python3
"""
Conjecture Iteration 3: Proper JSON Tool Calls Implementation
Tests original Conjecture specification with JSON tool calls format
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
    # Tool call specific metrics
    tool_calls_found: int = 0
    valid_tool_calls: int = 0
    tool_call_names: List[str] = None
    reasoning_steps: int = 0
    correctness_score: float = None
    completeness_score: float = None
    coherence_score: float = None
    reasoning_quality_score: float = None
    agentic_capability_score: float = None
    coding_score: float = None
    timestamp: str = None

    def __post_init__(self):
        if self.tool_call_names is None:
            self.tool_call_names = []
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

Should you recommend this update? Provide reasoning with systematic analysis.""",
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

Break this down into systematic analysis steps.""",
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

Break this into systematic design and implementation steps.""",
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
    "cerebras:openai/gpt-oss-20b": {
        "type": "large",
        "provider": "cerebras",
        "url": "https://api.cerebras.ai/v1",
        "api_key": "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8",
    },
    "cerebras:zai-glm-4.6": {
        "type": "sota",
        "provider": "cerebras",
        "url": "https://api.cerebras.ai/v1",
        "api_key": "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8",
    },
}

APPROACHES = ["direct", "conjecture_tool_calls"]

def make_api_call(
    prompt: str, model_config: Dict, max_tokens: int = 2000
) -> Dict[str, Any]:
    """Make API call to specified model"""
    try:
        import requests

        headers = {"Content-Type": "application/json"}

        if model_config["provider"] == "lmstudio":
            # LM Studio uses OpenAI-compatible format
            model_name = model_config["name"].split(":", 1)[
                1
            ]  # Remove "lmstudio:" prefix
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
        elif model_config["provider"] == "cerebras":
            # Cerebras API uses OpenAI-compatible format
            model_name = model_config["name"].split(":", 1)[
                1
            ]  # Remove "cerebras:" prefix
            data = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.3,
            }
            response = requests.post(
                f"{model_config['url']}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {model_config['api_key']}",
                },
                json=data,
                timeout=120,
            )
        else:
            return {
                "status": "error",
                "error": f"Unknown provider: {model_config['provider']}",
            }

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

    elif approach == "conjecture_tool_calls":
        # Build the prompt step by step to avoid f-string issues
        prompt_parts = [
            "Use the Conjecture approach with JSON tool calls to solve this problem systematically.",
            "",
            "Break down your reasoning into specific tool calls:",
            "- CreateClaim: Make a specific claim with confidence",
            "- EvaluateClaim: Assess a claim's validity",
            "- UpdateClaim: Modify a claim based on new information",
            "- FinalAnswer: Provide the final solution",
            "",
            "Available tools:",
            "1. CreateClaim(content, confidence)",
            "2. EvaluateClaim(claim_id, assessment)",
            "3. UpdateClaim(claim_id, new_content, new_confidence)",
            "4. FinalAnswer(answer, reasoning)",
            "",
            "Please respond with ONLY JSON tool calls in this format:",
            "```json",
            '{"tool_calls": [',
            '{"name": "CreateClaim", "arguments": {"content": "your claim", "confidence": 0.8}},',
            '{"name": "EvaluateClaim", "arguments": {"claim_id": "c1", "assessment": "valid because..."}},',
            '{"name": "CreateClaim", "arguments": {"content": "another claim", "confidence": 0.9}}',
            "]}",
            "```",
            "",
            "Problem to solve:",
            base_question,
            "",
            "Think step by step and use tool calls to structure your reasoning.",
        ]
        return "\n".join(prompt_parts)

    else:
        return f"Answer this question: {base_question}"

def parse_tool_calls(response: str) -> Tuple[List[Dict[str, Any]], bool]:
    """Parse tool calls from JSON response"""
    tool_calls = []

    # Try to extract JSON from response
    json_patterns = [
        r"```json\s*\n(.*?)\n```",  # Markdown JSON blocks
        r'\{[^{}]*"tool_calls"[^{}]*\}',  # Direct JSON in text
        r'"tool_calls"\s*:\s*\[.*?\]',  # Simplified tool_calls array
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                # Extract the JSON part
                json_str = match if isinstance(match, str) else match[0]
                json_str = json_str.strip()

                # Parse JSON
                data = json.loads(json_str)

                if isinstance(data, dict) and "tool_calls" in data:
                    tool_calls_data = data["tool_calls"]
                    if isinstance(tool_calls_data, list):
                        for tool_call in tool_calls_data:
                            if isinstance(tool_call, dict) and "name" in tool_call:
                                tool_calls.append(tool_call)

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                continue  # Skip invalid JSON

    return tool_calls, len(tool_calls) > 0

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
        "analyze",
        "evaluate",
        "assess",
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
        for keyword in ["try:", "except", "if not", "error", "empty", "len"]
    )
    has_test_cases = any(
        keyword in response.lower()
        for keyword in ["test", "assert", "example", "print("]
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
    result.correctness_score = min(len(result.response) / 400, 1.0)
    result.completeness_score = 1.0 if len(result.response) > 200 else 0.7

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
        "analyze",
        "evaluate",
    ]
    coherence_count = sum(
        1 for indicator in coherence_indicators if indicator in response
    )
    result.coherence_score = min(coherence_count / 5, 1.0)

    # Reasoning quality
    result.reasoning_quality_score = min(result.reasoning_steps / 6, 1.0)

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
        "tool_calls",
        "createclaim",
        "evaluateclaim",
    ]
    agentic_count = sum(1 for indicator in agentic_indicators if indicator in response)
    result.agentic_capability_score = min(agentic_count / 5, 1.0)

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

        # Parse tool calls if Conjecture approach
        tool_calls = []
        has_tool_calls = False
        tool_call_names = []

        if approach == "conjecture_tool_calls":
            tool_calls, has_tool_calls = parse_tool_calls(response)
            tool_call_names = [call.get("name", "") for call in tool_calls]

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
            tool_calls_found=len(tool_calls),
            valid_tool_calls=len(
                [tc for tc in tool_calls if "name" in tc and "arguments" in tc]
            ),
            tool_call_names=tool_call_names,
            reasoning_steps=reasoning_steps,
        )

        # Enhanced evaluation
        result = enhanced_evaluation(result)

        quality_score = result.reasoning_quality_score or 0
        if result.coding_score is not None:
            quality_score = (quality_score + result.coding_score) / 2

        tool_success_rate = result.valid_tool_calls / max(result.tool_calls_found, 1)

        print(
            f"      OK {response_time:.1f}s, Quality: {quality_score:.2f}, Tools: {result.tool_calls_found}/{result.valid_tool_calls} ({tool_success_rate:.1%})"
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

async def run_iteration_3():
    """Run iteration 3 with proper JSON tool calls"""
    print("=" * 80)
    print("CONJECTURE ITERATION 3: PROPER JSON TOOL CALLS")
    print("Testing original Conjecture specification with JSON tool calls")
    print("=" * 80)

    print(f"\nModels: {len(MODELS)} models (3 LM Studio + 2 Cerebras)")
    print(f"Approaches: {len(APPROACHES)} (direct + JSON tool calls)")
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
    filename = f"conjecture_iteration_3_{timestamp}.json"
    filepath = Path("research/results") / filename

    data = {
        "experiment_id": f"conjecture_iteration_3_{timestamp}",
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

    # Analysis
    await generate_iteration_analysis(all_results)

    return all_results

async def generate_iteration_analysis(results: List[TestResult]):
    """Generate analysis for iteration 3"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"iteration_3_analysis_{timestamp}.md"
    filepath = Path("research/results") / filename

    # Tool call compliance analysis
    tool_call_compliance = {}
    for model_key in MODELS.keys():
        model_results = [
            r for r in results if r.model == model_key and r.status == "success"
        ]
        direct_results = [r for r in model_results if r.approach == "direct"]
        tool_results = [
            r for r in model_results if r.approach == "conjecture_tool_calls"
        ]

        if direct_results and tool_results:
            direct_quality = sum(
                r.reasoning_quality_score or 0 for r in direct_results
            ) / len(direct_results)
            tool_quality = sum(
                r.reasoning_quality_score or 0 for r in tool_results
            ) / len(tool_results)
            tool_compliance_rate = sum(r.valid_tool_calls for r in tool_results) / sum(
                r.tool_calls_found for r in tool_results
            )

            tool_call_compliance[model_key] = {
                "direct_quality": direct_quality,
                "tool_quality": tool_quality,
                "improvement": (tool_quality - direct_quality) / direct_quality * 100
                if direct_quality > 0
                else 0,
                "compliance_rate": tool_compliance_rate,
                "direct_count": len(direct_results),
                "tool_count": len(tool_results),
            }

    report = f"""# Conjecture Iteration 3 Analysis
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

Tested {len(MODELS)} models with proper JSON tool calls format on {len(TEST_CASES)} diverse test cases.
**Key Finding**: Does original Conjecture JSON tool calls specification work better than claim formats?

## Tool Call Compliance Results

"""

    for model_key, metrics in tool_call_compliance.items():
        report += f"""### {model_key}
- Direct Approach Quality: {metrics["direct_quality"]:.3f}
- Tool Calls Quality: {metrics["tool_quality"]:.3f}
- Improvement: {metrics["improvement"]:+.1f}%
- Tool Call Compliance: {metrics["compliance_rate"]:.1%}
- Tests: {metrics["direct_count"]} direct, {metrics["tool_count"]} tool calls

"""

    # Overall assessment
    avg_improvement = sum(
        m["improvement"] for m in tool_call_compliance.values()
    ) / len(tool_call_compliance)
    avg_compliance = sum(
        m["compliance_rate"] for m in tool_call_compliance.values()
    ) / len(tool_call_compliance)

    report += f"""
## Overall Assessment

**Average Improvement with Tool Calls**: {avg_improvement:+.1f}%
**Average Tool Call Compliance**: {avg_compliance:.1%}

### Key Findings

1. **Tool Call Format**: {"✅ SUCCESSFUL" if avg_compliance > 0.5 else "❌ NEEDS WORK"}
2. **Model Performance**: {"✅ IMPROVED" if avg_improvement > 0 else "❌ NO IMPROVEMENT"}
3. **Specification Compliance**: {"✅ ORIGINAL CONJECTURE WORKS" if avg_compliance > 0.7 else "❌ SPECIFICATION NEEDS REVISION"}

## Recommendations

"""

    if avg_compliance > 0.7 and avg_improvement > 0:
        report += """✅ **ADOPT ORIGINAL CONJECTURE SPECIFICATION**

- JSON tool calls format shows high compliance
- Models demonstrate improved reasoning quality
- Proceed with full implementation and testing"""
    elif avg_compliance > 0.5:
        report += """⚠️ **REFINE TOOL CALL FORMAT**

- Moderate compliance suggests format is understood
- Consider simplifying tool call structure
- Provide more examples in prompts"""
    else:
        report += """❌ **RECONSIDER APPROACH**

- Low compliance indicates format issues
- Models may need different prompting strategy
- Consider hybrid approaches"""

    report += f"""
## Next Steps

1. **Based on results**: {"Proceed with tool calls" if avg_compliance > 0.5 else "Revise approach"}
2. **Expand testing**: More models and test cases
3. **Compare approaches**: Tool calls vs direct vs claim formats
4. **Final validation**: Large-scale testing with proven approach

---
*Iteration 3 Analysis Complete*
"""

    with open(filepath, "w") as f:
        f.write(report)

    print(f"[OK] Analysis saved to: {filepath}")

if __name__ == "__main__":
    asyncio.run(run_iteration_3())
