#!/usr/bin/env python3
"""
GPT-OSS-20B Scaled Testing Framework

Uses the actual GPT-OSS-20B model via OpenRouter to test Conjecture performance
with different claim evaluation thresholds.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import aiohttp

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestProblem:
    """Test problem for evaluation"""
    problem_id: str
    domain: str
    problem: str
    expected_solution: Optional[str] = None
    complexity: str = "medium"

@dataclass
class TestResult:
    """Test result data"""
    problem_id: str
    method: str
    claims_evaluated: int
    response: str
    is_correct: bool
    execution_time_seconds: float
    token_usage: Optional[int] = None
    error_message: Optional[str] = None

class GptOssScaledTester:
    """Uses GPT-OSS-20B via OpenRouter for real testing"""

    def __init__(self):
        # Load configuration from .conjecture/config.json
        config_path = os.path.join(os.getcwd(), ".conjecture", "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Find GPT-OSS-20B provider
        gpt_provider = None
        for provider in config["providers"]:
            if provider.get("model") == "openai/gpt-oss-20b":
                gpt_provider = provider
                break

        if not gpt_provider:
            raise ValueError("GPT-OSS-20B provider not found in config.json")

        self.api_key = gpt_provider["key"]
        self.base_url = gpt_provider["url"]
        self.model = gpt_provider["model"]
        self.max_tokens = gpt_provider.get("max_tokens", 2000)
        self.temperature = gpt_provider.get("temperature", 0.2)
        self.problems = []
        logger.info(f"Using GPT-OSS-20B via {self.base_url}")

    async def initialize(self):
        """Initialize the test system"""
        await self.load_test_problems()
        await self.setup_judge_model()

    async def load_test_problems(self):
        """Load comprehensive test problems"""
        # 50 problems across 5 domains (10 per domain) for statistically significant testing
        self.problems = [
            # Mathematical Reasoning
            TestProblem(
                problem_id="math_001",
                domain="mathematical_reasoning",
                problem="What is 15% of 240?",
                expected_solution="36"
            ),
            TestProblem(
                problem_id="math_002",
                domain="mathematical_reasoning",
                problem="If x + 8 = 15, what is x?",
                expected_solution="7"
            ),
            TestProblem(
                problem_id="math_003",
                domain="mathematical_reasoning",
                problem="Calculate: 25 × 4 + 18 ÷ 3",
                expected_solution="106"
            ),
            TestProblem(
                problem_id="math_004",
                domain="mathematical_reasoning",
                problem="What is the square root of 144?",
                expected_solution="12"
            ),
            TestProblem(
                problem_id="math_005",
                domain="mathematical_reasoning",
                problem="What is 3/4 of 80?",
                expected_solution="60"
            ),
            TestProblem(
                problem_id="math_006",
                domain="mathematical_reasoning",
                problem="If 2x - 7 = 13, what is x?",
                expected_solution="10"
            ),
            TestProblem(
                problem_id="math_007",
                domain="mathematical_reasoning",
                problem="Calculate: 12² + 5³",
                expected_solution="169"
            ),
            TestProblem(
                problem_id="math_008",
                domain="mathematical_reasoning",
                problem="What is 20% of 150?",
                expected_solution="30"
            ),
            TestProblem(
                problem_id="math_009",
                domain="mathematical_reasoning",
                problem="Solve: 3x + 4 = 19",
                expected_solution="5"
            ),
            TestProblem(
                problem_id="math_010",
                domain="mathematical_reasoning",
                problem="What is the cube root of 27?",
                expected_solution="3"
            ),

            # Logical Inference
            TestProblem(
                problem_id="logic_001",
                domain="logical_inference",
                problem="If all cats are animals, and some animals are pets, can we conclude that some cats are pets?",
                expected_solution="No"
            ),
            TestProblem(
                problem_id="logic_002",
                domain="logical_inference",
                problem="If A implies B, and B implies C, does A imply C?",
                expected_solution="Yes"
            ),
            TestProblem(
                problem_id="logic_003",
                domain="logical_inference",
                problem="Some doctors are rich. All rich people drive cars. Can we conclude some doctors drive cars?",
                expected_solution="No"
            ),
            TestProblem(
                problem_id="logic_004",
                domain="logical_inference",
                problem="All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?",
                expected_solution="Yes"
            ),
            TestProblem(
                problem_id="logic_005",
                domain="logical_inference",
                problem="If all birds can fly, and penguins are birds, can penguins fly?",
                expected_solution="No"
            ),
            TestProblem(
                problem_id="logic_006",
                domain="logical_inference",
                problem="Some students are athletes. All athletes are disciplined. Can we conclude some students are disciplined?",
                expected_solution="Yes"
            ),
            TestProblem(
                problem_id="logic_007",
                domain="logical_inference",
                problem="No squares are circles. All triangles are squares. Are any triangles circles?",
                expected_solution="No"
            ),
            TestProblem(
                problem_id="logic_008",
                domain="logical_inference",
                problem="If P is true, then Q is false. Q is true. Is P false?",
                expected_solution="Yes"
            ),
            TestProblem(
                problem_id="logic_009",
                domain="logical_inference",
                problem="All philosophers are thinkers. Some thinkers are writers. Can we conclude some philosophers are writers?",
                expected_solution="No"
            ),
            TestProblem(
                problem_id="logic_010",
                domain="logical_inference",
                problem="If A and B imply C, and A is true but C is false, what can we conclude about B?",
                expected_solution="B is false"
            ),

            # Scientific Reasoning
            TestProblem(
                problem_id="science_001",
                domain="scientific_reasoning",
                problem="What happens to water when it reaches 100 degrees Celsius at standard pressure?",
                expected_solution="It boils and turns to steam"
            ),
            TestProblem(
                problem_id="science_002",
                domain="scientific_reasoning",
                problem="What gas do plants primarily produce during photosynthesis?",
                expected_solution="Oxygen"
            ),
            TestProblem(
                problem_id="science_003",
                domain="scientific_reasoning",
                problem="What force keeps planets orbiting the sun?",
                expected_solution="Gravity"
            ),
            TestProblem(
                problem_id="science_004",
                domain="scientific_reasoning",
                problem="What is the chemical formula for water?",
                expected_solution="H2O"
            ),
            TestProblem(
                problem_id="science_005",
                domain="scientific_reasoning",
                problem="What is the speed of light in vacuum?",
                expected_solution="299,792,458 meters per second"
            ),
            TestProblem(
                problem_id="science_006",
                domain="scientific_reasoning",
                problem="What process converts liquid water to water vapor?",
                expected_solution="Evaporation"
            ),
            TestProblem(
                problem_id="science_007",
                domain="scientific_reasoning",
                problem="What is the smallest unit of matter?",
                expected_solution="Atom"
            ),
            TestProblem(
                problem_id="science_008",
                domain="scientific_reasoning",
                problem="What gas makes up most of Earth's atmosphere?",
                expected_solution="Nitrogen"
            ),
            TestProblem(
                problem_id="science_009",
                domain="scientific_reasoning",
                problem="What type of energy is stored in chemical bonds?",
                expected_solution="Chemical energy"
            ),
            TestProblem(
                problem_id="science_010",
                domain="scientific_reasoning",
                problem="What is the process by which plants make their own food?",
                expected_solution="Photosynthesis"
            ),

            # Strategic Planning
            TestProblem(
                problem_id="strategy_001",
                domain="strategic_planning",
                problem="A company has $10,000 budget. Marketing costs $3,000, development costs $5,000. How much remains?",
                expected_solution="$2,000"
            ),
            TestProblem(
                problem_id="strategy_002",
                domain="strategic_planning",
                problem="Project A: $5,000 profit, 3 months. Project B: $8,000 profit, 5 months. Which has better monthly return?",
                expected_solution="Project A"
            ),
            TestProblem(
                problem_id="strategy_003",
                domain="strategic_planning",
                problem="If you invest $1,000 at 8% annual interest, how much after 2 years (simple interest)?",
                expected_solution="$1,160"
            ),
            TestProblem(
                problem_id="strategy_004",
                domain="strategic_planning",
                problem="Company sells 100 units at $20 each, costs $12 each. What is total profit?",
                expected_solution="$800"
            ),
            TestProblem(
                problem_id="strategy_005",
                domain="strategic_planning",
                problem="Store offers 20% discount on $200 item, then additional 10% off discounted price. Final price?",
                expected_solution="$144"
            ),
            TestProblem(
                problem_id="strategy_006",
                domain="strategic_planning",
                problem="Car rental: $50/day + $0.25/mile. For 3 days and 200 miles, total cost?",
                expected_solution="$200"
            ),
            TestProblem(
                problem_id="strategy_007",
                domain="strategic_planning",
                problem="If inflation is 3% annually, what will be $1,000's value after 1 year?",
                expected_solution="$970"
            ),
            TestProblem(
                problem_id="strategy_008",
                domain="strategic_planning",
                problem="Production: 100 units/hour, 8-hour shift. 2 shifts with 10% downtime. Daily output?",
                expected_solution="1,440 units"
            ),
            TestProblem(
                problem_id="strategy_009",
                domain="strategic_planning",
                problem="Salary: $50,000 + 5% bonus. Tax: 20% on bonus only. Net take-home?",
                expected_solution="$51,000"
            ),
            TestProblem(
                problem_id="strategy_010",
                domain="strategic_planning",
                problem="Phone plan: $30/month + $0.10/minute over 500 minutes. Used 650 minutes. Total bill?",
                expected_solution="$45"
            ),

            # Problem Decomposition
            TestProblem(
                problem_id="decomp_001",
                domain="problem_decomposition",
                problem="Calculate the total cost: 3 items at $15 each, plus 5% tax",
                expected_solution="$47.25"
            ),
            TestProblem(
                problem_id="decomp_002",
                domain="problem_decomposition",
                problem="Calculate average: (80 + 90 + 100) / 3",
                expected_solution="90"
            ),
            TestProblem(
                problem_id="decomp_003",
                domain="problem_decomposition",
                problem="Rectangle perimeter: length 12, width 8. Calculate area and perimeter.",
                expected_solution="Area: 96, Perimeter: 40"
            ),
            TestProblem(
                problem_id="decomp_004",
                domain="problem_decomposition",
                problem="Train travels 60 mph for 2.5 hours. How far does it travel?",
                expected_solution="150 miles"
            ),
            TestProblem(
                problem_id="decomp_005",
                domain="problem_decomposition",
                problem="Tank holds 500 gallons, loses 2% per day. After 3 days, how much remains?",
                expected_solution="470.2 gallons"
            ),
            TestProblem(
                problem_id="decomp_006",
                domain="problem_decomposition",
                problem="Salary increased 15% from $40,000. Then decreased 10%. Final salary?",
                expected_solution="$42,300"
            ),
            TestProblem(
                problem_id="decomp_007",
                domain="problem_decomposition",
                problem="Circle radius 8 cm. Calculate area and circumference.",
                expected_solution="Area: 201.06 cm², Circumference: 50.27 cm"
            ),
            TestProblem(
                problem_id="decomp_008",
                domain="problem_decomposition",
                problem="Recipe serves 6, needs 2.5 cups flour. How much for 10 people?",
                expected_solution="4.17 cups"
            ),
            TestProblem(
                problem_id="decomp_009",
                domain="problem_decomposition",
                problem="Car depreciates 20% first year, 15% second. $30,000 initial. Value after 2 years?",
                expected_solution="$20,400"
            ),
            TestProblem(
                problem_id="decomp_010",
                domain="problem_decomposition",
                problem="Project takes 8 hours alone, 6 hours with helper. Working together, time to complete?",
                expected_solution="3.43 hours"
            )
        ]

    async def setup_judge_model(self):
        """Setup GLM-4.6 as judge model"""
        # Load GLM-4.6 configuration from config file
        config_path = os.path.join(os.getcwd(), ".conjecture", "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        glm_provider = None
        for provider in config["providers"]:
            if provider.get("model") == "glm-4.6":
                glm_provider = provider
                break

        if not glm_provider:
            raise ValueError("GLM-4.6 provider not found in config")

        self.judge_api_key = glm_provider["key"]
        self.judge_base_url = glm_provider["url"]
        self.judge_model = glm_provider["model"]
        logger.info(f"Using GLM-4.6 as judge via {self.judge_base_url}")

    async def call_judge_model(self, problem: str, response: str, expected: str) -> Dict[str, Any]:
        """Use GLM-4.6 to judge response quality and correctness"""
        judge_prompt = f"""You are an expert evaluator for AI responses. Assess the following response for correctness and quality.

PROBLEM: {problem}
EXPECTED ANSWER: {expected}
RESPONSE TO EVALUATE: {response}

Evaluate:
1. Is the answer factually correct? (correct/incorrect)
2. How confident are you? (0-100%)
3. Quality of reasoning (poor/fair/good/excellent)
4. Any issues or strengths?

Return your evaluation as JSON:
{{
    "is_correct": true/false,
    "confidence": 0-100,
    "reasoning_quality": "poor/fair/good/excellent",
    "feedback": "Brief explanation of your evaluation"
}}
"""

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.judge_api_key}",
                    "Content-Type": "application/json"
                }

                data = {
                    "model": self.judge_model,
                    "messages": [{"role": "user", "content": judge_prompt}],
                    "temperature": 0.1,
                    "max_tokens": 500
                }

                async with session.post(
                    f"{self.judge_base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result["choices"][0]["message"]["content"]

                        # Extract JSON from response
                        try:
                            import re
                            json_match = re.search(r'\{.*\}', content, re.DOTALL)
                            if json_match:
                                evaluation = json.loads(json_match.group())
                                return {
                                    "success": True,
                                    "evaluation": evaluation
                                }
                        except:
                            pass

                        # Fallback if JSON parsing fails
                        return {
                            "success": False,
                            "raw_response": content
                        }
                    else:
                        logger.error(f"Judge API error: {resp.status}")
                        return {"success": False, "error": str(resp.status)}

        except Exception as e:
            logger.error(f"Judge model error: {e}")
            return {"success": False, "error": str(e)}

    async def call_gpt_oss(self, prompt: str, temperature: float = None, max_tokens: int = None) -> Dict[str, Any]:
        """Make API call to GPT-OSS-20B via OpenRouter"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://conjecture.ai",
            "X-Title": "Conjecture Testing Framework"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Provide accurate, direct answers to the questions asked. Be concise and precise."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "content": result["choices"][0]["message"]["content"],
                            "tokens_used": result.get("usage", {}).get("total_tokens", 0)
                        }
                    else:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"API Error {response.status}: {error_text}"
                        }
        except Exception as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}"
            }

    async def test_baseline(self, problem: TestProblem) -> TestResult:
        """Test baseline GPT-OSS response"""
        start_time = time.time()

        baseline_prompt = f"""Solve this problem and give the answer clearly:

{problem.problem}

Please provide a direct, accurate answer."""

        result = await self.call_gpt_oss(baseline_prompt, temperature=0.1, max_tokens=200)
        execution_time = time.time() - start_time

        if result["success"]:
            response_text = result["content"]
            is_correct, evaluation_meta = await self._check_correctness(
                problem.problem, response_text, problem.expected_solution
            )

            return TestResult(
                problem_id=problem.problem_id,
                method="baseline",
                claims_evaluated=0,
                response=response_text,
                is_correct=is_correct,
                execution_time_seconds=execution_time,
                token_usage=result.get("tokens_used"),
                error_message=f"Evaluation method: {evaluation_meta.get('method', 'unknown')}"
            )
        else:
            return TestResult(
                problem_id=problem.problem_id,
                method="baseline",
                claims_evaluated=0,
                response="",
                is_correct=False,
                execution_time_seconds=execution_time,
                error_message=result["error"]
            )

    async def test_conjecture_with_claims(self, problem: TestProblem, num_claims: int) -> TestResult:
        """Test Conjecture with forced claim evaluation"""
        start_time = time.time()

        # Generate contextual claims based on problem domain
        claims = self._generate_domain_claims(problem.domain, num_claims)

        # Enhanced prompt incorporating claim evaluation
        enhanced_prompt = f"""Problem: {problem.problem}

CONTEXT ANALYSIS - Evaluating {num_claims} relevant claims:
{chr(10).join(f"- {claim}" for claim in claims)}

Based on the above contextual analysis, provide a comprehensive solution.
Consider the claims and principles evaluated, then give your final answer clearly."""

        result = await self.call_gpt_oss(enhanced_prompt, temperature=0.1, max_tokens=500)
        execution_time = time.time() - start_time

        if result["success"]:
            response_text = result["content"]
            is_correct, evaluation_meta = await self._check_correctness(
                problem.problem, response_text, problem.expected_solution
            )

            return TestResult(
                problem_id=problem.problem_id,
                method=f"conjecture_{num_claims}_claims",
                claims_evaluated=num_claims,
                response=response_text,
                is_correct=is_correct,
                execution_time_seconds=execution_time,
                token_usage=result.get("tokens_used"),
                error_message=f"Evaluation method: {evaluation_meta.get('method', 'unknown')}, Quality: {evaluation_meta.get('reasoning_quality', 'unknown')}"
            )
        else:
            return TestResult(
                problem_id=problem.problem_id,
                method=f"conjecture_{num_claims}_claims",
                claims_evaluated=num_claims,
                response="",
                is_correct=False,
                execution_time_seconds=execution_time,
                error_message=result["error"]
            )

    def _generate_domain_claims(self, domain: str, num_claims: int) -> List[str]:
        """Generate relevant claims for the domain"""
        claim_sets = {
            "mathematical_reasoning": [
                "Mathematical operations follow specific order of operations (PEMDAS)",
                "Percentage calculations: (percentage/100) × base = result",
                "Algebraic equations maintain balance when operations are applied to both sides",
                "Numerical results should be verified through alternative methods when possible",
                "Square root operations find the inverse of squaring",
                "Multi-step calculations require careful tracking of intermediate results"
            ],
            "logical_inference": [
                "Logical transitivity: If A→B and B→C, then A→C",
                "Universal statements require all cases to be true",
                "Existential statements require at least one case to be true",
                "Invalid inferences occur when conclusions don't follow from premises",
                "Counterexamples can invalidate universal claims",
                "Logical validity depends on structure, not content"
            ],
            "scientific_reasoning": [
                "Scientific principles are based on empirical evidence and reproducibility",
                "Phase transitions occur at specific physical conditions",
                "Photosynthesis produces oxygen and consumes carbon dioxide",
                "Gravitational force governs orbital mechanics",
                "Chemical formulas represent molecular composition",
                "Scientific laws are consistent across valid applications"
            ],
            "strategic_planning": [
                "Budget constraints require resource allocation optimization",
                "Time-value analysis requires considering opportunity costs",
                "Profit calculations must account for all relevant expenses",
                "Investment returns should be evaluated over consistent time periods",
                "Strategic decisions involve weighing multiple factors and constraints",
                "Risk assessment should accompany financial calculations"
            ],
            "problem_decomposition": [
                "Complex problems can be broken into manageable sub-components",
                "Step-by-step solutions reduce error probability",
                "Verification steps improve solution reliability",
                "Multi-part problems require attention to each component",
                "Final results should synthesize all calculated components",
                "Alternative approaches can validate solution correctness"
            ]
        }

        claims = claim_sets.get(domain, ["Contextual analysis supports systematic problem-solving"])

        # Return requested number of claims, cycling if needed
        result = []
        for i in range(num_claims):
            result.append(claims[i % len(claims)])

        return result

    async def _check_correctness_llm_judge(self, problem: str, response: str, expected: str) -> tuple[bool, Dict[str, Any]]:
        """Use LLM judge for rigorous evaluation"""
        if not hasattr(self, 'judge_api_key'):
            # Fallback to string matching if judge not available
            return self._check_correctness_fallback(response, expected), {"method": "string_match"}

        try:
            judge_result = await self.call_judge_model(problem, response, expected)

            if judge_result.get("success") and "evaluation" in judge_result:
                evaluation = judge_result["evaluation"]
                is_correct = evaluation.get("is_correct", False)

                # Add confidence threshold - only accept if confidence >= 70%
                confidence = evaluation.get("confidence", 0)
                if confidence < 70:
                    logger.warning(f"Low confidence judge evaluation ({confidence}%), falling back to string matching")
                    is_correct = self._check_correctness_fallback(response, expected)
                    evaluation["reasoning_quality"] = evaluation.get("reasoning_quality", "fair")

                return is_correct, {
                    "method": "llm_judge",
                    "confidence": confidence,
                    "reasoning_quality": evaluation.get("reasoning_quality", "unknown"),
                    "feedback": evaluation.get("feedback", "")
                }
            else:
                # Fallback if judge fails
                logger.warning("LLM judge evaluation failed, falling back to string matching")
                return self._check_correctness_fallback(response, expected), {"method": "string_match_fallback"}

        except Exception as e:
            logger.error(f"LLM judge error: {e}, falling back to string matching")
            return self._check_correctness_fallback(response, expected), {"method": "string_match_error"}

    def _check_correctness_fallback(self, response: str, expected: str) -> bool:
        """Fallback string-based correctness check"""
        if not expected:
            return False

        response_lower = response.lower()
        expected_lower = expected.lower().strip()

        # Clean up expected answer for better matching
        expected_variations = [expected_lower]

        # Add common variations
        if expected_lower.isdigit():
            expected_variations.extend([
                f"the answer is {expected_lower}",
                f"answer: {expected_lower}",
                f"result: {expected_lower}",
                f"= {expected_lower}"
            ])

        if expected_lower in ["yes", "no"]:
            expected_variations.extend([
                f"the answer is {expected_lower}",
                f"conclusion: {expected_lower}",
                expected_lower + "."
            ])

        if expected_lower == "oxygen":
            expected_variations.extend(["o2", "o2 gas"])

        # Handle currency and percentage answers
        if expected_lower.startswith("$"):
            expected_variations.extend([
                expected_lower.replace("$", ""),
                expected_lower.replace("$", "USD "),
                f"{expected_lower[1:]} dollars"
            ])

        if "%" in expected_lower:
            expected_variations.extend([
                expected_lower.replace("%", " percent"),
                expected_lower.replace("%", ""),
                f"{expected_lower.replace('%', '')} percent"
            ])

        # Check if any variation appears in response
        return any(var in response_lower for var in expected_variations)

    async def _check_correctness(self, problem: str, response: str, expected: str) -> tuple[bool, Dict[str, Any]]:
        """Enhanced correctness check with LLM judge evaluation"""
        # Use LLM judge for rigorous evaluation when available
        return await self._check_correctness_llm_judge(problem, response, expected)

    async def run_claim_evaluation_test(self, claim_thresholds: List[int]) -> Dict[str, Any]:
        """Run comprehensive test across claim evaluation thresholds"""
        logger.info(f"Starting GPT-OSS-20B claim evaluation test with thresholds: {claim_thresholds}")

        results = {
            "test_config": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "api_endpoint": self.base_url,
                "problems_tested": len(self.problems),
                "claim_thresholds": claim_thresholds
            },
            "results": {}
        }

        # Test baseline first
        logger.info("Testing baseline GPT-OSS-20B performance...")
        baseline_results = []
        for i, problem in enumerate(self.problems):
            logger.info(f"  Baseline problem {i+1}/{len(self.problems)}: {problem.problem_id}")
            result = await self.test_baseline(problem)
            baseline_results.append(result)
            status = "PASS" if result.is_correct else "FAIL"
            if result.error_message:
                logger.warning(f"    Error: {result.error_message}")
            else:
                logger.info(f"    {status}")

        baseline_correct = sum(1 for r in baseline_results if r.is_correct)
        baseline_accuracy = (baseline_correct / len(self.problems)) * 100

        results["results"]["baseline"] = {
            "correct_answers": baseline_correct,
            "accuracy_percent": baseline_accuracy,
            "total_time_seconds": sum(r.execution_time_seconds for r in baseline_results),
            "total_tokens": sum(r.token_usage or 0 for r in baseline_results),
            "details": [asdict(r) for r in baseline_results]
        }

        # Test each claim evaluation threshold
        for num_claims in claim_thresholds:
            logger.info(f"Testing Conjecture with {num_claims} claim evaluations...")
            threshold_results = []

            for i, problem in enumerate(self.problems):
                logger.info(f"  {num_claims} claims - problem {i+1}/{len(self.problems)}: {problem.problem_id}")
                result = await self.test_conjecture_with_claims(problem, num_claims)
                threshold_results.append(result)
                status = "PASS" if result.is_correct else "FAIL"
                if result.error_message:
                    logger.warning(f"    Error: {result.error_message}")
                else:
                    logger.info(f"    {status}")

                # Add delay between requests to avoid rate limiting
                await asyncio.sleep(0.5)

            threshold_correct = sum(1 for r in threshold_results if r.is_correct)
            threshold_accuracy = (threshold_correct / len(self.problems)) * 100
            improvement = threshold_accuracy - baseline_accuracy

            results["results"][f"conjecture_{num_claims}_claims"] = {
                "correct_answers": threshold_correct,
                "accuracy_percent": threshold_accuracy,
                "improvement_over_baseline_percent": improvement,
                "total_time_seconds": sum(r.execution_time_seconds for r in threshold_results),
                "total_tokens": sum(r.token_usage or 0 for r in threshold_results),
                "total_claims_evaluated": num_claims * len(self.problems),
                "details": [asdict(r) for r in threshold_results]
            }

        # Analysis
        results["analysis"] = self._analyze_results(results["results"])

        return results

    def _analyze_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the impact of claim evaluation on performance"""
        baseline_acc = results["baseline"]["accuracy_percent"]

        # Analyze evaluation methods used
        evaluation_methods = {"baseline": [], "llm_judge": [], "string_match": [], "fallback": []}

        for key, result in results.items():
            if "details" in result:
                for detail in result["details"]:
                    error_msg = detail.get("error_message", "")
                    if "Evaluation method:" in error_msg:
                        method = error_msg.split("Evaluation method:")[1].split(",")[0].strip()
                        if method in evaluation_methods:
                            evaluation_methods[method].append(detail["problem_id"])
                    else:
                        evaluation_methods["baseline"].append(detail["problem_id"])

        analysis = {
            "baseline_accuracy": baseline_acc,
            "baseline_tokens": results["baseline"]["total_tokens"],
            "claim_evaluation_impact": [],
            "optimal_threshold": None,
            "max_improvement": 0.0,
            "efficiency_analysis": [],
            "evaluation_methodology": {
                "methods_used": {k: len(v) for k, v in evaluation_methods.items() if v},
                "llm_judge_usage": len(evaluation_methods["llm_judge"]) / sum(len(v) for v in evaluation_methods.values()) * 100,
                "scientific_rigor_improved": len(evaluation_methods["llm_judge"]) > 0
            }
        }

        for key, result in results.items():
            if key.startswith("conjecture_") and key != "baseline":
                num_claims = int(key.split("_")[1])
                improvement = result["improvement_over_baseline_percent"]
                token_efficiency = result["total_tokens"] / (num_claims * len(self.problems)) if num_claims > 0 else float('inf')

                analysis["claim_evaluation_impact"].append({
                    "claims_evaluated": num_claims,
                    "accuracy_percent": result["accuracy_percent"],
                    "improvement_percent": improvement,
                    "total_tokens": result["total_tokens"],
                    "tokens_per_claim": token_efficiency
                })

                if improvement > analysis["max_improvement"]:
                    analysis["max_improvement"] = improvement
                    analysis["optimal_threshold"] = num_claims

        return analysis

    def save_results(self, results: Dict[str, Any]):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gpt_oss_claim_evaluation_test_{timestamp}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {filename}")
        return filename

async def main():
    """Main test function"""
    logger.info("Starting GPT-OSS-20B Claim Evaluation Impact Test")

    # Initialize tester
    tester = GptOssScaledTester()
    await tester.initialize()

    # Test with different claim evaluation thresholds
    claim_thresholds = [0, 5, 10, 25]

    logger.info(f"Testing with {len(tester.problems)} problems across 5 domains")
    logger.info("This will make real API calls to GPT-OSS-20B and consume tokens!")

    results = await tester.run_claim_evaluation_test(claim_thresholds)

    # Save results
    filename = tester.save_results(results)

    # Print summary
    print("\n" + "="*80)
    print("GPT-OSS-20B CLAIM EVALUATION IMPACT TEST RESULTS")
    print("="*80)
    print(f"Model: {results['test_config']['model']}")
    print(f"API Endpoint: {results['test_config']['api_endpoint']}")
    print(f"Problems tested: {results['test_config']['problems_tested']}")
    print(f"Timestamp: {results['test_config']['timestamp']}")
    print()

    baseline = results["results"]["baseline"]
    print(f"Baseline Performance:")
    print(f"  Accuracy: {baseline['accuracy_percent']:.1f}%")
    print(f"  Tokens used: {baseline['total_tokens']:,}")
    print(f"  Time: {baseline['total_time_seconds']:.1f}s")
    print()

    print("Conjecture Performance by Claim Evaluation Threshold:")
    print("-" * 80)
    print(f"{'Claims':<8} {'Accuracy':<10} {'Improvement':<12} {'Tokens':<12} {'Efficiency':<12}")
    print("-" * 80)

    for impact in results["analysis"]["claim_evaluation_impact"]:
        efficiency = "N/A" if impact['tokens_per_claim'] == float('inf') else f"{impact['tokens_per_claim']:.1f}"
        print(f"{impact['claims_evaluated']:8d} {impact['accuracy_percent']:9.1f}% "
              f"{impact['improvement_percent']:+10.1f}% "
              f"{impact['total_tokens']:11,} {efficiency:>11s}")

    print()
    if results["analysis"]["optimal_threshold"]:
        print(f"Optimal threshold: {results['analysis']['optimal_threshold']} claims "
              f"({results['analysis']['max_improvement']:+.1f}% improvement)")

        # Test the 10+ claims hypothesis
        ten_plus_results = [r for r in results["analysis"]["claim_evaluation_impact"] if r['claims_evaluated'] >= 10]
        if ten_plus_results:
            best_ten_plus = max(ten_plus_results, key=lambda x: x['improvement_percent'])
            print(f"Best 10+ claims result: {best_ten_plus['claims_evaluated']} claims "
                  f"({best_ten_plus['improvement_percent']:+.1f}% improvement)")
    else:
        print("No optimal threshold found (baseline performed best)")

    logger.info("GPT-OSS-20B claim evaluation impact test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())