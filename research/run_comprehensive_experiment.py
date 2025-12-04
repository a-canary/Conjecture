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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

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
    self_consistency_score: float = 0.0
    # Evaluation scores (populated by LLM judge)
    correctness_score: float = None
    completeness_score: float = None
    coherence_score: float = None
    reasoning_quality_score: float = None
    depth_score: float = None
    agentic_capability_score: float = None
    timestamp: str = None

    def __post_init__(self):
        if self.claims_generated is None:
            self.claims_generated = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

# Test cases focusing on reasoning and agentic capabilities
TEST_CASES = [
    {
        "id": "logic_puzzle_complex",
        "category": "complex_reasoning",
        "difficulty": "hard",
        "question": "In a small town, there are five houses in a row, each painted a different color: red, blue, green, yellow, and white. Each house is owned by a person with a different profession: doctor, teacher, engineer, artist, and baker. Each person has a different favorite fruit: apple, banana, cherry, date, and elderberry. Using the following clues, determine who owns the red house and what is their favorite fruit?\n\nClues:\n1. The doctor lives in the middle house.\n2. The artist lives next to the person who likes apples.\n3. The engineer lives in the green house.\n4. The teacher likes bananas.\n5. The baker lives in the first house.\n6. The person who likes cherries lives next to the white house.\n7. The red house is somewhere to the left of the blue house.\n8. The artist does not live in the yellow house.\n9. The person who likes dates lives next to the doctor.\n10. The person who likes elderberries lives in the last house.",
        "expected_answer": "The teacher owns the red house and their favorite fruit is banana.",
        "reasoning_requirements": ["spatial_reasoning", "constraint_satisfaction", "deductive_logic"],
        "agentic_capabilities": ["problem_decomposition", "systematic_analysis"]
    },
    {
        "id": "math_reasoning_multi_step",
        "category": "mathematical_reasoning",
        "difficulty": "medium",
        "question": "A train travels from City A to City B at 60 mph and returns at 40 mph. What is the average speed for the entire round trip? Explain why the answer is not 50 mph.",
        "expected_answer": "48 mph. The average speed is not 50 mph because more time is spent traveling at the slower speed, so the harmonic mean must be used: (2 * 60 * 40) / (60 + 40) = 48 mph.",
        "reasoning_requirements": ["mathematical_reasoning", "conceptual_understanding", "explanation"],
        "agentic_capabilities": ["calculation", "concept_explanation"]
    },
    {
        "id": "evidence_eval_conflicting",
        "category": "evidence_evaluation",
        "difficulty": "hard",
        "question": "Evaluate whether this new drug should be approved based on: 1) Two large studies show statistically significant blood pressure reduction, 2) Two smaller studies show no significant effect, 3) Side effects include mild headaches in 5% of patients and dizziness in 2%, 4) The drug costs 3x more than existing treatments, 5) All studies were funded by the pharmaceutical company. Provide a structured analysis with claims and confidence scores.",
        "expected_answer": "Insufficient evidence for approval. Need: independent replication studies, long-term safety data, cost-effectiveness analysis.",
        "reasoning_requirements": ["evidence_assessment", "bias_recognition", "risk_benefit_analysis"],
        "agentic_capabilities": ["critical_thinking", "uncertainty_quantification"]
    },
    {
        "id": "planning_multi_step",
        "category": "planning",
        "difficulty": "hard",
        "question": "You need to plan a 3-day software development sprint to implement a user authentication system with OAuth, 2FA, and role-based access control. Break this down into tasks, estimate time for each, identify dependencies, and create a timeline. What are the key risks and how would you mitigate them?",
        "expected_answer": "Should include: task breakdown (UI, backend, testing), time estimates, dependency mapping, risk identification (security, integration, scope creep), mitigation strategies.",
        "reasoning_requirements": ["task_decomposition", "dependency_analysis", "risk_assessment"],
        "agentic_capabilities": ["project_planning", "risk_management", "resource_allocation"]
    },
    {
        "id": "causal_inference",
        "category": "causal_reasoning",
        "difficulty": "hard",
        "question": "A city implemented a congestion pricing program and traffic decreased by 15% in the first month. However, a new subway line opened the same month, and there was a 10% increase in gas prices. How would you determine whether the congestion pricing was effective? What additional data would you need?",
        "expected_answer": "Need to control for confounding variables: compare to control areas, analyze timing patterns, gather data on subway ridership, gas price elasticity, economic indicators, seasonal patterns.",
        "reasoning_requirements": ["causal_inference", "confounder_identification", "counterfactual_reasoning"],
        "agentic_capabilities": ["experimental_design", "data_requirements_analysis"]
    },
    {
        "id": "code_analysis_debug",
        "category": "code_reasoning",
        "difficulty": "medium",
        "question": "Analyze this Python function for potential bugs and edge cases:\n\ndef calculate_discount(price, discount_percent, user_type):\n    if user_type == 'premium':\n        discount_percent += 10\n    final_price = price - (price * discount_percent / 100)\n    return final_price\n\nWhat are the issues? How would you fix them? Provide the corrected code.",
        "expected_answer": "Issues: discount can exceed 100%, negative prices possible, no input validation, no handling of None values. Fixes: clamp discount, validate inputs, handle edge cases.",
        "reasoning_requirements": ["code_analysis", "edge_case_identification", "bug_detection"],
        "agentic_capabilities": ["static_analysis", "correctness_reasoning"]
    }
]

# Model configurations
MODEL_CONFIGS = [
    {
        "name": "ibm/granite-4-h-tiny",
        "type": "tiny",
        "provider": "lm_studio",
        "url": "http://localhost:1234",
        "api_key": "",
        "description": "Tiny LLM (~3B parameters)"
    },
    {
        "name": "glm-z1-9b-0414",
        "type": "medium",
        "provider": "lm_studio",
        "url": "http://localhost:1234",
        "api_key": "",
        "description": "Medium LLM (9B parameters)"
    },
    {
        "name": "qwen3-4b-thinking-2507",
        "type": "medium",
        "provider": "lm_studio",
        "url": "http://localhost:1234",
        "api_key": "",
        "description": "Qwen thinking model (4B parameters)"
    },
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

def make_api_call(prompt: str, model_config: Dict[str, Any], max_tokens: int = 2000) -> Dict[str, Any]:
    """Make API call to either LM Studio or Chutes"""
    try:
        import requests

        provider = model_config["provider"]
        url = model_config["url"]
        api_key = model_config["api_key"]
        model_name = model_config["name"]

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3
        }

        # Fix Chutes API endpoint (don't double /v1)
        if "chutes.ai" in url and url.endswith("/v1"):
            endpoint = f"{url}/chat/completions"
        else:
            endpoint = f"{url}/v1/chat/completions"

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

3. For agentic/planning tasks, also include action steps.

Format your response with:
- Claims section (using the exact format above)
- Analysis section (evaluating each claim)
- Final Answer section (your conclusion)
- Action Steps (if applicable)"""

def extract_claims(response: str) -> List[Dict[str, Any]]:
    """Extract claims from response using regex"""
    import re

    claims = []
    # Pattern: [c1 | content | / confidence]
    pattern = r'\[c(\d+)\s*\|\s*([^|]+)\s*\|\s*/\s*([0-9.]+)\s*\]'
    matches = re.findall(pattern, response, re.IGNORECASE)

    for match in matches:
        claim_id, content, confidence = match
        claims.append({
            "id": claim_id,
            "content": content.strip(),
            "confidence": float(confidence)
        })

    return claims

def analyze_reasoning_steps(response: str) -> int:
    """Count reasoning steps in response"""
    # Simple heuristic: count numbered/bulleted steps, logical connectors
    import re

    patterns = [
        r'\d+\.',  # Numbered steps
        r'•',      # Bullet points
        r'First,|Second,|Third,|Then,|Next,|Finally,',  # Sequence words
        r'Because|Therefore|However|Thus|Hence',  # Logical connectors
    ]

    steps = 0
    for pattern in patterns:
        steps += len(re.findall(pattern, response, re.IGNORECASE))

    return steps

def evaluate_self_consistency(response: str) -> float:
    """Basic self-consistency check (0.0 to 1.0)"""
    # Check for contradictions, confidence calibration
    import re

    score = 1.0

    # Check for contradictory statements
    contradiction_patterns = [
        (r'\b(not|never|no)\b', r'\b(always|yes|is)\b'),
    ]

    # Check if confidence scores are reasonable
    confidence_pattern = r'/\s*([0-9.]+)\s*\]'
    confidences = re.findall(confidence_pattern, response)

    for conf in confidences:
        try:
            confidence_val = float(conf)
            if confidence_val < 0.0 or confidence_val > 1.0:
                score -= 0.1
        except:
            score -= 0.1

    return max(0.0, score)

def run_test_for_model(model_config: Dict[str, Any], test_case: Dict[str, Any], approach: str) -> TestResult:
    """Run a single test for a specific model and approach"""
    print(f"    Testing {approach} approach...")

    if approach == "direct":
        prompt = generate_direct_prompt(test_case)
    else:
        prompt = generate_conjecture_prompt(test_case)

    result = make_api_call(prompt, model_config, max_tokens=2500)

    # Analyze response
    claims = extract_claims(result["content"]) if result["status"] == "success" else []
    reasoning_steps = analyze_reasoning_steps(result["content"]) if result["status"] == "success" else 0
    consistency = evaluate_self_consistency(result["content"]) if result["status"] == "success" else 0.0

    return TestResult(
        model=model_config["name"],
        model_type=model_config["type"],
        approach=approach,
        test_case_id=test_case["id"],
        test_category=test_case["category"],
        prompt=prompt,
        response=result["content"],
        response_time=result["response_time"],
        response_length=result["response_length"],
        status=result["status"],
        error=result.get("error"),
        claims_generated=claims,
        has_claim_format=len(claims) > 0,
        reasoning_steps=reasoning_steps,
        self_consistency_score=consistency
    )

async def evaluate_with_llm_judge(results: List[TestResult]) -> List[TestResult]:
    """Evaluate results using LLM-as-a-Judge (GLM-4.6)"""
    print("\n[4/5] Evaluating results with LLM judge...")

    judge_config = {
        "name": "zai-org/GLM-4.6",
        "type": "sota",
        "provider": "chutes",
        "url": "https://llm.chutes.ai/v1",
        "api_key": os.getenv("CHUTES_API_KEY", ""),
        "description": "Judge model"
    }

    if not judge_config["api_key"]:
        print("[WARN] No CHUTES_API_KEY found, skipping LLM judge evaluation")
        return results

    evaluated_results = []

    for i, result in enumerate(results):
        if result.status != "success":
            evaluated_results.append(result)
            continue

        print(f"  Evaluating {i+1}/{len(results)}: {result.model} | {result.test_case_id}")

        # Create evaluation prompt
        eval_prompt = f"""You are an expert evaluator assessing AI responses on reasoning and agentic capabilities.

**Task:** {result.test_category}
**Question:** {result.prompt[:500]}...

**Model Response:**
{result.response[:1500]}...

**Evaluation Criteria:**

1. **Correctness (0-10):** Is the answer factually accurate?
2. **Completeness (0-10):** Does it address all aspects of the question?
3. **Coherence (0-10):** Is the reasoning logical and well-structured?
4. **Reasoning Quality (0-10):** Depth and validity of logical steps
5. **Depth (0-10):** Insightfulness and thoroughness
6. **Agentic Capability (0-10):** For planning/agent tasks - quality of action steps

**Scoring:** Provide scores 0-10 for each criterion, where 10 is perfect.

**Format:**
Correctness: X/10
Completeness: X/10
Coherence: X/10
Reasoning Quality: X/10
Depth: X/10
Agentic Capability: X/10

**Brief Justification:** [1-2 sentences]"""

        try:
            # Fix judge config endpoint too
            judge_result = make_api_call(eval_prompt, judge_config, max_tokens=500)

            if judge_result["status"] == "success":
                content = judge_result["content"]

                # Extract scores
                import re
                score_pattern = r'(\w+):\s*(\d+(?:\.\d+)?)\s*/\s*10'
                scores = dict(re.findall(score_pattern, content))

                # Update result with scores
                result.correctness_score = float(scores.get("Correctness", 0)) / 10.0
                result.completeness_score = float(scores.get("Completeness", 0)) / 10.0
                result.coherence_score = float(scores.get("Coherence", 0)) / 10.0
                result.reasoning_quality_score = float(scores.get("Reasoning Quality", 0)) / 10.0
                result.depth_score = float(scores.get("Depth", 0)) / 10.0
                result.agentic_capability_score = float(scores.get("Agentic Capability", 0)) / 10.0

                print(f"    Scores: C={result.correctness_score:.2f}, RQ={result.reasoning_quality_score:.2f}, AC={result.agentic_capability_score:.2f}")
            else:
                print(f"    [WARN] Judge evaluation failed: {judge_result.get('error')}")

        except Exception as e:
            print(f"    [WARN] Judge evaluation error: {e}")

        evaluated_results.append(result)
        await asyncio.sleep(0.5)  # Rate limiting

    return evaluated_results

async def run_all_experiments():
    """Run complete experiment suite model-by-model"""
    print("=" * 80)
    print("COMPREHENSIVE CONJECTURE RESEARCH")
    print("Testing Reasoning & Agentic Capabilities with Quality Evaluation")
    print("=" * 80)
    print(f"Models: {len(MODEL_CONFIGS)} | Test Cases: {len(TEST_CASES)} | Approaches: 2")
    print("=" * 80)

    # Filter available models
    available_models = []
    for model in MODEL_CONFIGS:
        if model["provider"] == "chutes" and not model["api_key"]:
            print(f"[SKIP] {model['name']} - No API key")
            continue
        available_models.append(model)

    print(f"\nAvailable models: {len(available_models)}")
    for model in available_models:
        print(f"  - {model['name']} ({model['type']})")

    approaches = ["direct", "true_conjecture"]
    total_tests = len(available_models) * len(approaches) * len(TEST_CASES)
    print(f"\nTotal tests to run: {total_tests}")

    all_results = []
    current_test = 0

    # Run model-by-model to prevent LM Studio reloading
    for model in available_models:
        print(f"\n{'=' * 80}")
        print(f"TESTING MODEL: {model['name']}")
        print(f"{'=' * 80}")

        # Test all approaches for this model
        for approach in approaches:
            print(f"\n[Approach: {approach.upper()}]")

            for test_case in TEST_CASES:
                current_test += 1
                print(f"\n[{current_test}/{total_tests}] {test_case['id']} ({test_case['category']})")

                try:
                    result = run_test_for_model(model, test_case, approach)
                    all_results.append(result)

                    if result.status == "success":
                        print(f"  [OK] {result.response_time:.1f}s | {result.response_length} chars | {len(result.claims_generated)} claims | {result.reasoning_steps} reasoning steps")
                    else:
                        print(f"  [FAIL] {result.error}")

                    # Brief pause between requests
                    await asyncio.sleep(0.5)

                except Exception as e:
                    print(f"  [ERROR] {e}")
                    error_result = TestResult(
                        model=model["name"],
                        model_type=model["type"],
                        approach=approach,
                        test_case_id=test_case["id"],
                        test_category=test_case["category"],
                        prompt="",
                        response=f"Error: {str(e)}",
                        response_time=0,
                        response_length=0,
                        status="error",
                        error=str(e)
                    )
                    all_results.append(error_result)

    # Evaluate with LLM judge
    print(f"\n{'=' * 80}")
    print("EVALUATING RESULTS WITH LLM JUDGE")
    print(f"{'=' * 80}")
    evaluated_results = await evaluate_with_llm_judge(all_results)

    # Save results
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("research/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"conjecture_comprehensive_{timestamp}.json"

    # Convert dataclass to dict for JSON serialization
    results_data = {
        "experiment_id": f"conjecture_comprehensive_{timestamp}",
        "timestamp": datetime.now().isoformat(),
        "models_tested": [m["name"] for m in available_models],
        "approaches_tested": approaches,
        "test_cases": TEST_CASES,
        "results": [asdict(r) for r in evaluated_results]
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"[OK] Results saved to: {results_file}")

    # Generate summary
    print(f"\n{'=' * 80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")

    successful = [r for r in evaluated_results if r.status == "success"]
    failed = [r for r in evaluated_results if r.status != "success"]

    print(f"Total tests: {len(evaluated_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    # Group by model and approach
    summary = {}
    for result in successful:
        key = f"{result.model} | {result.approach}"
        if key not in summary:
            summary[key] = []
        summary[key].append(result)

    print(f"\nPerformance by model and approach:")
    for key, result_list in summary.items():
        avg_time = sum(r.response_time for r in result_list) / len(result_list)
        avg_claims = sum(len(r.claims_generated) for r in result_list) / len(result_list)
        avg_reasoning = sum(r.reasoning_steps for r in result_list) / len(result_list)

        # Quality scores (if available)
        quality_scores = []
        for r in result_list:
            if r.correctness_score is not None:
                quality_scores.append(r.correctness_score)

        print(f"\n  {key}:")
        print(f"    Tests: {len(result_list)}")
        print(f"    Avg time: {avg_time:.1f}s")
        print(f"    Avg claims: {avg_claims:.1f}")
        print(f"    Avg reasoning steps: {avg_reasoning:.1f}")

        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            print(f"    Avg quality: {avg_quality:.2f}")

    print(f"\n{'=' * 80}")
    print("Next steps:")
    print("1. Review detailed results in the JSON file")
    print("2. Run analysis script to generate statistical report")
    print("3. Evaluate if Conjecture improves reasoning quality")
    print("4. Compare tiny LLM performance to SOTA benchmarks")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    asyncio.run(run_all_experiments())
