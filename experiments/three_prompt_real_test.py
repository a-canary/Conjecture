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

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import time
import re


@dataclass
class Claim:
    """Simplified claim for testing"""
    id: str
    content: str
    confidence: float
    dirty: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class PromptResult:
    """Result from a single prompt execution"""
    prompt_type: str
    response: str
    tokens: int
    time_ms: int
    parsed_output: Optional[Dict[str, Any]] = None


class ThreePromptSystem:
    """
    Three-prompt architecture with shared context.

    Design principles:
    - Same 50-claim context for all prompts
    - Same dirty claim selection
    - Iterative: loop until confidence > 0.7 AND SKIP
    - Each prompt has one job
    """

    def __init__(self, llm_provider, max_iterations: int = 5):
        self.llm = llm_provider
        self.max_iterations = max_iterations

    def build_context(self, query: str, claims: List[Claim]) -> str:
        """Build shared context used by all 3 prompts"""
        context = f"QUERY: {query}\n\n"
        context += "RELEVANT CLAIMS (top 50):\n"
        for i, claim in enumerate(claims[:50], 1):
            dirty_flag = " [DIRTY]" if claim.dirty else ""
            context += f"{i}. [{claim.confidence:.2f}] {claim.content}{dirty_flag}\n"
        return context

    def prompt1_update_confidence(self, context: str) -> str:
        return f"""{context}

TASK: Update claim confidence scores (0.0 to 1.0)

Review the claims above. For each claim, assess:
- Does it directly support answering the query?
- How certain are you about this claim?

Output JSON:
{{
  "updates": [
    {{"id": "c001", "confidence": 0.85, "reason": "Directly relevant"}}
  ]
}}

Respond with JSON only:"""

    def prompt2_create_claim(self, context: str, iteration: int) -> str:
        return f"""{context}

ITERATION: {iteration}

TASK: Create ONE new claim to help answer the query, or say SKIP if no more needed.

If you have high confidence (>0.7) and no more claims needed, respond with:
{{"action": "SKIP"}}

Otherwise:
{{
  "action": "CREATE",
  "claim": {{
    "content": "The specific claim text",
    "confidence": 0.5,
    "type": "question"
  }}
}}

Respond with JSON only:"""

    def prompt3_final_response(self, context: str) -> str:
        return f"""{context}

TASK: Provide final answer to the query.

Output JSON:
{{
  "answer": "Your complete answer here",
  "supporting_claims": ["c001"],
  "confidence": 0.9
}}

Respond with JSON only:"""

    async def execute_prompt(self, prompt: str, prompt_type: str) -> PromptResult:
        start = time.time()
        response = await self.llm.complete(prompt)
        elapsed_ms = int((time.time() - start) * 1000)
        parsed = None
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
        return PromptResult(
            prompt_type=prompt_type,
            response=response,
            tokens=len(prompt.split()) + len(response.split()),
            time_ms=elapsed_ms,
            parsed_output=parsed
        )

    async def process_query(self, query: str, initial_claims: List[Claim]) -> Dict[str, Any]:
        start_time = time.time()
        context = self.build_context(query, initial_claims)
        trace = {
            "query": query,
            "initial_claims": len(initial_claims),
            "context_size": len(context),
            "iterations": [],
            "final_result": None,
            "total_time_ms": 0,
            "total_tokens": 0
        }
        claims = initial_claims.copy()
        for iteration in range(1, self.max_iterations + 1):
            iteration_trace = {"iteration": iteration, "prompts": []}
            p1 = self.prompt1_update_confidence(context)
            r1 = await self.execute_prompt(p1, "update_confidence")
            iteration_trace["prompts"].append(asdict(r1))
            if r1.parsed_output and "updates" in r1.parsed_output:
                for update in r1.parsed_output["updates"]:
                    for claim in claims:
                        if claim.id == update["id"]:
                            claim.confidence = update["confidence"]
            context = self.build_context(query, claims)
            p2 = self.prompt2_create_claim(context, iteration)
            r2 = await self.execute_prompt(p2, "create_claim")
            iteration_trace["prompts"].append(asdict(r2))
            should_skip = False
            if r2.parsed_output:
                action = r2.parsed_output.get("action")
                if action == "SKIP":
                    should_skip = True
                elif action == "CREATE" and "claim" in r2.parsed_output:
                    new_claim_data = r2.parsed_output["claim"]
                    claims.append(Claim(
                        id=f"c{len(claims)+1:03d}",
                        content=new_claim_data["content"],
                        confidence=new_claim_data.get("confidence", 0.5),
                        dirty=True
                    ))
                    context = self.build_context(query, claims)
            max_confidence = max([c.confidence for c in claims], default=0.0)
            iteration_trace["max_confidence"] = max_confidence
            iteration_trace["should_skip"] = should_skip
            iteration_trace["claim_count"] = len(claims)
            trace["iterations"].append(iteration_trace)
            if should_skip and max_confidence > 0.7:
                break
            elif iteration == self.max_iterations:
                break
        p3 = self.prompt3_final_response(context)
        r3 = await self.execute_prompt(p3, "final_response")
        trace["final_result"] = {
            "prompt": asdict(r3),
            "final_claim_count": len(claims),
            "answer": r3.parsed_output.get("answer") if r3.parsed_output else None,
            "confidence": r3.parsed_output.get("confidence") if r3.parsed_output else None
        }
        trace["total_time_ms"] = int((time.time() - start_time) * 1000)
        trace["total_tokens"] = sum(
            p["tokens"]
            for it in trace["iterations"]
            for p in it["prompts"]
        ) + r3.tokens
        return trace


class RealLLMProvider:
    """Real LLM provider using Conjecture's backend"""

    def __init__(self):
        # Use SimplifiedLLMManager
        from src.processing.simplified_llm_manager import SimplifiedLLMManager

        self.manager = SimplifiedLLMManager()

    async def complete(self, prompt: str) -> str:
        """Call real LLM"""
        try:
            # Use generate_text() which returns raw string response
            text = self.manager.generate_text(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3
            )
            return text if text else "{}"
        except Exception as e:
            print(f"LLM Error: {e}")
            import traceback
            traceback.print_exc()
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
