#!/usr/bin/env python3
"""
Three-prompt architecture with REAL LLM provider

Tests the split-prompt approach on actual benchmark problems:
1. Update claim confidence
2. Create claim or SKIP
3. Final response (when confidence > 0.7 and SKIP)
"""
import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.three_prompt_test import ThreePromptSystem, Claim
from src.config.unified_config import UnifiedConfig
from datetime import datetime


class RealLLMProvider:
    """Real LLM provider using Conjecture's backend"""

    def __init__(self):
        self.config = UnifiedConfig()
        # Import here to avoid circular dependencies
        from src.agent.llm_providers import get_provider

        self.provider = get_provider(self.config)

    async def complete(self, prompt: str) -> str:
        """Call real LLM"""
        try:
            # Use the actual provider
            response = await self.provider.complete(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3  # Lower temp for more consistent JSON
            )

            return response
        except Exception as e:
            print(f"LLM Error: {e}")
            return "{}"


async def test_real_problems():
    """Test on actual benchmark problems"""

    test_cases = [
        {
            "query": "If Alice has 3 apples and Bob has twice as many, and they share equally with Carol, how many does each person get?",
            "expected": "3",
            "initial_claims": [
                Claim("c001", "Alice has 3 apples", 0.5),
                Claim("c002", "Bob has twice as many as Alice", 0.5),
                Claim("c003", "They share equally with Carol", 0.5),
                Claim("c004", "Equal sharing means divide total by number of people", 0.8),
                Claim("c005", "Multiplication: 2 × 3 = 6", 0.9),
            ]
        },
        {
            "query": "What is 15% of 80?",
            "expected": "12",
            "initial_claims": [
                Claim("c001", "Percentage means divide by 100", 0.9),
                Claim("c002", "15% = 15/100 = 0.15", 0.9),
                Claim("c003", "To find X% of Y, multiply Y × (X/100)", 0.85),
            ]
        },
        {
            "query": "In a sequence 2, 4, 6, 8, what is the next number?",
            "expected": "10",
            "initial_claims": [
                Claim("c001", "This appears to be an arithmetic sequence", 0.7),
                Claim("c002", "Difference between consecutive terms: 4-2=2, 6-4=2, 8-6=2", 0.8),
                Claim("c003", "In arithmetic sequence, add common difference to last term", 0.85),
            ]
        }
    ]

    # Initialize with real LLM
    print("Initializing real LLM provider...")
    llm = RealLLMProvider()
    system = ThreePromptSystem(llm, max_iterations=4)

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'#'*60}")
        print(f"TEST CASE {i}/{len(test_cases)}")
        print(f"{'#'*60}")

        try:
            result = await system.process_query(
                query=test_case["query"],
                initial_claims=test_case["initial_claims"]
            )

            result["expected"] = test_case["expected"]
            results.append(result)

            # Check correctness
            answer = result["final_result"].get("answer", "")
            expected = test_case["expected"]
            correct = expected.lower() in answer.lower()

            print(f"\n{'='*60}")
            print(f"Expected: {expected}")
            print(f"Got: {answer[:100]}...")
            print(f"Correct: {'✓' if correct else '✗'}")
            print(f"{'='*60}")

        except Exception as e:
            print(f"\n✗ Test case failed: {e}")
            import traceback
            traceback.print_exc()

    # Save all results
    output_file = f"experiments/results/three_prompt_real_{datetime.now().isoformat()}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'#'*60}")

    # Summary
    print("\nSUMMARY:")
    correct_count = 0
    for i, result in enumerate(results, 1):
        answer = result["final_result"].get("answer", "")
        expected = result.get("expected", "")
        correct = expected.lower() in answer.lower()
        if correct:
            correct_count += 1

        iterations = len(result["iterations"])
        claims = result["final_result"]["final_claim_count"]
        confidence = result["final_result"].get("confidence", 0)

        print(f"\nTest {i}:")
        print(f"  Iterations: {iterations}")
        print(f"  Claims: {claims}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Correct: {'✓' if correct else '✗'}")

    print(f"\nAccuracy: {correct_count}/{len(results)} ({100*correct_count/len(results):.1f}%)")


if __name__ == "__main__":
    asyncio.run(test_real_problems())
