#!/usr/bin/env python3
"""
Fast Prototype Testing Framework for Rapid Prompt Iteration
Enables quick testing of different prompt strategies against baseline
"""

import asyncio
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

@dataclass
class PromptTestResult:
    """Result of a single prompt test"""
    problem_id: str
    problem_type: str
    prompt_strategy: str
    response: str
    expected: str
    correct: bool
    execution_time: float
    confidence: float = 0.0
    error: Optional[str] = None

@dataclass
class PromptStrategy:
    """A prompt strategy to test"""
    name: str
    system_prompt: str
    description: str
    context_instructions: str = ""

class PromptPrototypeFramework:
    """Framework for rapid prompt prototype testing"""

    def __init__(self, model_func: Callable, model_name: str):
        self.model_func = model_func
        self.model_name = model_name
        self.results: List[PromptTestResult] = []

    def add_strategy(self, strategy: PromptStrategy):
        """Add a prompt strategy to test"""
        if not hasattr(self, 'strategies'):
            self.strategies = []
        self.strategies.append(strategy)

    async def test_strategy(self, strategy: PromptStrategy, problems: List[Dict]) -> List[PromptTestResult]:
        """Test a single strategy against problems"""
        print(f"\nTesting strategy: {strategy.name}")
        print(f"Description: {strategy.description}")
        print("-" * 60)

        strategy_results = []

        for i, problem in enumerate(problems):
            try:
                print(f"Problem {i+1}/{len(problems)}: {problem['id']}... ", end="", flush=True)

                # Construct prompt with strategy
                full_prompt = self._build_prompt(strategy, problem)

                # Get response
                start_time = time.time()
                response = await self.model_func(full_prompt)
                execution_time = time.time() - start_time

                # Evaluate response
                expected = problem['expected']
                correct = self._evaluate_response(response, expected)

                result = PromptTestResult(
                    problem_id=problem['id'],
                    problem_type=problem.get('category', 'unknown'),
                    prompt_strategy=strategy.name,
                    response=response[:200] + "..." if len(response) > 200 else response,
                    expected=expected,
                    correct=correct,
                    execution_time=execution_time
                )

                strategy_results.append(result)
                status = "✓" if correct else "✗"
                print(f"{status} ({execution_time:.1f}s)")

            except Exception as e:
                error_result = PromptTestResult(
                    problem_id=problem['id'],
                    problem_type=problem.get('category', 'unknown'),
                    prompt_strategy=strategy.name,
                    response="",
                    expected=problem['expected'],
                    correct=False,
                    execution_time=0.0,
                    error=str(e)
                )
                strategy_results.append(error_result)
                print(f"ERROR: {str(e)[:50]}")

        return strategy_results

    def _build_prompt(self, strategy: PromptStrategy, problem: Dict) -> str:
        """Build full prompt for strategy"""
        prompt_parts = []

        # System prompt
        if strategy.system_prompt:
            prompt_parts.append(strategy.system_prompt)
            prompt_parts.append("")

        # Context instructions
        if strategy.context_instructions:
            prompt_parts.append(strategy.context_instructions)
            prompt_parts.append("")

        # Problem
        prompt_parts.append("PROBLEM:")
        prompt_parts.append(problem['question'])
        prompt_parts.append("")

        # Request for answer
        prompt_parts.append("Please provide the answer to this problem.")

        return "\n".join(prompt_parts)

    def _evaluate_response(self, response: str, expected: str) -> bool:
        """Evaluate if response is correct"""
        # Simple string matching (can be enhanced)
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()

        # Extract numbers from response
        import re
        response_numbers = re.findall(r'-?\d+\.?\d*', response_lower)
        expected_numbers = re.findall(r'-?\d+\.?\d*', expected_lower)

        if expected_numbers:
            # Check if any number in response matches expected
            for exp_num in expected_numbers:
                for resp_num in response_numbers:
                    if abs(float(resp_num) - float(exp_num)) < 0.001:
                        return True
            return False

        # For non-numeric answers, check substring
        return expected_lower in response_lower

    async def run_competition(self, problems: List[Dict]) -> Dict[str, List[PromptTestResult]]:
        """Run all strategies in competition"""
        if not hasattr(self, 'strategies') or not self.strategies:
            print("No strategies added!")
            return {}

        print(f"Starting prompt competition with {len(self.strategies)} strategies")
        print(f"Model: {self.model_name}")
        print(f"Problems: {len(problems)}")
        print("=" * 80)

        all_results = {}

        for strategy in self.strategies:
            strategy_results = await self.test_strategy(strategy, problems)
            all_results[strategy.name] = strategy_results
            self.results.extend(strategy_results)

        return all_results

    def analyze_results(self, results: Dict[str, List[PromptTestResult]]) -> Dict[str, Any]:
        """Analyze competition results"""
        analysis = {
            "model_name": self.model_name,
            "total_problems": len(next(iter(results.values()))),
            "strategies": {}
        }

        for strategy_name, strategy_results in results.items():
            correct_count = sum(1 for r in strategy_results if r.correct)
            accuracy = correct_count / len(strategy_results) if strategy_results else 0
            avg_time = sum(r.execution_time for r in strategy_results) / len(strategy_results) if strategy_results else 0

            analysis["strategies"][strategy_name] = {
                "correct_answers": correct_count,
                "accuracy": accuracy,
                "average_time": avg_time,
                "total_time": sum(r.execution_time for r in strategy_results)
            }

        # Find best strategy
        if analysis["strategies"]:
            best_strategy = max(analysis["strategies"].items(),
                              key=lambda x: (x[1]["accuracy"], -x[1]["average_time"]))
            analysis["best_strategy"] = {
                "name": best_strategy[0],
                "accuracy": best_strategy[1]["accuracy"],
                "avg_time": best_strategy[1]["average_time"]
            }

        return analysis

    def save_results(self, results: Dict[str, List[PromptTestResult]],
                    analysis: Dict[str, Any], filename: str = None):
        """Save results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prompt_test_results_{timestamp}.json"

        results_file = Path(__file__).parent / filename

        save_data = {
            "analysis": analysis,
            "detailed_results": {
                strategy_name: [asdict(r) for r in strategy_results]
                for strategy_name, strategy_results in results.items()
            }
        }

        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"\nResults saved to: {results_file}")

# Problem sets for testing
MATH_PROBLEMS = [
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
        "id": "math_3",
        "question": "If a train travels 300 miles in 4 hours, what is its average speed?",
        "expected": "75",
        "category": "word_problem"
    },
    {
        "id": "math_4",
        "question": "What is 15% of 80?",
        "expected": "12",
        "category": "percentages"
    }
]

LOGIC_PROBLEMS = [
    {
        "id": "logic_1",
        "question": "If all cats are animals and some animals are pets, can we conclude that some cats are pets? Answer yes or no.",
        "expected": "no",
        "category": "logic"
    },
    {
        "id": "logic_2",
        "question": "If it rains tomorrow, I will stay home. I did not stay home today. Can we conclude it did not rain today? Answer yes or no.",
        "expected": "no",  # This is tricky - the statement is about tomorrow, not today
        "category": "logic"
    }
]

# Baseline strategy (current Conjecture)
BASELINE_STRATEGY = PromptStrategy(
    name="baseline_current",
    system_prompt="""You are Conjecture, an AI assistant that helps with research, coding, and knowledge management. You have access to tools for gathering information and creating structured knowledge claims.

CRITICAL PRINCIPLE: Claims are NOT facts. Claims are impressions, assumptions, observations, and conjectures that have a variable or unknown amount of truth. All claims are provisional and subject to revision based on new evidence.

Your core approach is to:
1. Understand the user's request clearly
2. Use relevant skills to guide your thinking process
3. Use available tools to gather information and create solutions
4. Create claims to capture important knowledge as impressions, assumptions, observations, or conjectures
5. Always include uncertainty estimates and acknowledge limitations
6. Support claims with evidence while recognizing evidence may be incomplete""",

    description="Current baseline Conjecture prompt focused on claim creation"
)

# Direct problem-solving strategy
DIRECT_STRATEGY = PromptStrategy(
    name="direct_problem_solving",
    system_prompt="""You are an expert problem-solving assistant. Your goal is to provide accurate, clear answers to mathematical and logical problems.

APPROACH:
1. Carefully read and understand the problem
2. Identify the key information and what's being asked
3. Work through the problem step-by-step
4. Provide the final answer clearly and directly
5. Be confident in your mathematical reasoning

Focus on accuracy and clarity rather than creating claims or using tools.""",

    description="Direct problem-solving without Conjecture overhead"
)

if __name__ == "__main__":
    # Example usage
    async def test_framework():
        from config_aware_integration import gpt_oss_20b_direct

        framework = PromptPrototypeFramework(gpt_oss_20b_direct, "GPT-OSS-20B")

        # Add strategies
        framework.add_strategy(BASELINE_STRATEGY)
        framework.add_strategy(DIRECT_STRATEGY)

        # Test on math problems
        problems = MATH_PROBLEMS + LOGIC_PROBLEMS

        # Run competition
        results = await framework.run_competition(problems)

        # Analyze results
        analysis = framework.analyze_results(results)

        # Print summary
        print("\n" + "=" * 80)
        print("COMPETITION RESULTS")
        print("=" * 80)

        for strategy_name, stats in analysis["strategies"].items():
            print(f"{strategy_name}:")
            print(f"  Accuracy: {stats['accuracy']:.1%}")
            print(f"  Avg Time: {stats['average_time']:.1f}s")
            print()

        if "best_strategy" in analysis:
            best = analysis["best_strategy"]
            print(f"Best strategy: {best['name']} ({best['accuracy']:.1%} accuracy)")

        # Save results
        framework.save_results(results, analysis)

    # asyncio.run(test_framework())