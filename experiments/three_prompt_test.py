#!/usr/bin/env python3
"""
Test 3-prompt architecture with shared context:
1. Update claim confidence (0-1.0)
2. Create claim or SKIP
3. [if confidence > 0.7 and SKIP] Final response

All prompts use same 50-claim context and dirty claim selection.
"""
import asyncio
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


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
        """Prompt 1: Update claim confidence based on evidence"""
        return f"""{context}

TASK: Update claim confidence scores (0.0 to 1.0)

Review the claims above. For each claim, assess:
- Does it directly support answering the query?
- How certain are you about this claim?
- Does it conflict with other claims?

Output JSON:
{{
  "updates": [
    {{"id": "c001", "confidence": 0.85, "reason": "Directly relevant"}},
    {{"id": "c003", "confidence": 0.45, "reason": "Partially related"}}
  ]
}}

Only include claims that need confidence updates.
Respond with JSON only:"""

    def prompt2_create_claim(self, context: str, iteration: int) -> str:
        """Prompt 2: Create new claim or SKIP"""
        return f"""{context}

ITERATION: {iteration}

TASK: Create ONE new claim to help answer the query, or say SKIP if no more needed.

New claims can be:
- Question to explore
- Assumption to verify
- Intermediate calculation
- Generalization from evidence
- Next reasoning step

If you have high confidence (>0.7) in your understanding and no more claims would help, respond with:
{{"action": "SKIP"}}

Otherwise, create ONE claim:
{{
  "action": "CREATE",
  "claim": {{
    "content": "The specific claim text",
    "confidence": 0.5,
    "type": "question|assumption|calculation|generalization|step"
  }}
}}

Respond with JSON only:"""

    def prompt3_final_response(self, context: str) -> str:
        """Prompt 3: Generate final answer when confidence > 0.7 and SKIP"""
        return f"""{context}

TASK: Provide final answer to the query

Based on the claims above, generate a complete answer.
Include:
- Direct answer to the query
- Key supporting claims (by ID)
- Confidence in your answer (0.0 to 1.0)

Output JSON:
{{
  "answer": "Your complete answer here",
  "supporting_claims": ["c001", "c005", "c012"],
  "confidence": 0.9
}}

Respond with JSON only:"""

    async def execute_prompt(self, prompt: str, prompt_type: str) -> PromptResult:
        """Execute a single prompt and track metrics"""
        start = time.time()

        # Mock LLM call for now - replace with actual provider
        response = await self.llm.complete(prompt)

        elapsed_ms = int((time.time() - start) * 1000)

        # Try to parse JSON response
        parsed = None
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            # Extract JSON from markdown if needed
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))

        return PromptResult(
            prompt_type=prompt_type,
            response=response,
            tokens=len(prompt.split()) + len(response.split()),  # Rough estimate
            time_ms=elapsed_ms,
            parsed_output=parsed
        )

    async def process_query(self, query: str, initial_claims: List[Claim]) -> Dict[str, Any]:
        """
        Process a query through the 3-prompt system.

        Returns complete trace with all iterations and final result.
        """
        start_time = time.time()

        # Build shared context once
        context = self.build_context(query, initial_claims)

        # Track all prompts executed
        trace = {
            "query": query,
            "initial_claims": len(initial_claims),
            "context_size": len(context),
            "iterations": [],
            "final_result": None,
            "total_time_ms": 0,
            "total_tokens": 0
        }

        # Working set of claims
        claims = initial_claims.copy()

        # Iteration loop
        for iteration in range(1, self.max_iterations + 1):
            iteration_trace = {
                "iteration": iteration,
                "prompts": []
            }

            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}")
            print(f"{'='*60}")

            # PROMPT 1: Update confidence
            print("\n[1/3] Updating claim confidence...")
            p1 = self.prompt1_update_confidence(context)
            r1 = await self.execute_prompt(p1, "update_confidence")
            iteration_trace["prompts"].append(asdict(r1))

            if r1.parsed_output and "updates" in r1.parsed_output:
                print(f"  Updated {len(r1.parsed_output['updates'])} claims")
                # Apply updates to claims
                for update in r1.parsed_output["updates"]:
                    for claim in claims:
                        if claim.id == update["id"]:
                            claim.confidence = update["confidence"]
                            print(f"    {claim.id}: {claim.confidence:.2f} - {update.get('reason', '')}")

            # Rebuild context with updated confidences
            context = self.build_context(query, claims)

            # PROMPT 2: Create claim or SKIP
            print("\n[2/3] Creating new claim or skipping...")
            p2 = self.prompt2_create_claim(context, iteration)
            r2 = await self.execute_prompt(p2, "create_claim")
            iteration_trace["prompts"].append(asdict(r2))

            should_skip = False
            if r2.parsed_output:
                action = r2.parsed_output.get("action")
                print(f"  Action: {action}")

                if action == "SKIP":
                    should_skip = True
                    print("  No more claims needed")
                elif action == "CREATE" and "claim" in r2.parsed_output:
                    new_claim_data = r2.parsed_output["claim"]
                    new_claim = Claim(
                        id=f"c{len(claims)+1:03d}",
                        content=new_claim_data["content"],
                        confidence=new_claim_data.get("confidence", 0.5),
                        dirty=True
                    )
                    claims.append(new_claim)
                    print(f"  Created: {new_claim.content[:60]}...")
                    print(f"  Confidence: {new_claim.confidence:.2f}")

                    # Rebuild context with new claim
                    context = self.build_context(query, claims)

            # Check stopping condition
            max_confidence = max([c.confidence for c in claims], default=0.0)
            iteration_trace["max_confidence"] = max_confidence
            iteration_trace["should_skip"] = should_skip
            iteration_trace["claim_count"] = len(claims)

            trace["iterations"].append(iteration_trace)

            if should_skip and max_confidence > 0.7:
                print(f"\n✓ Stopping: confidence {max_confidence:.2f} > 0.7 and SKIP")
                break
            elif iteration == self.max_iterations:
                print(f"\n⚠ Max iterations reached ({self.max_iterations})")
                break
            else:
                print(f"\n→ Continue: confidence {max_confidence:.2f}, need more claims")

        # PROMPT 3: Final response
        print(f"\n{'='*60}")
        print("GENERATING FINAL RESPONSE")
        print(f"{'='*60}")
        p3 = self.prompt3_final_response(context)
        r3 = await self.execute_prompt(p3, "final_response")

        trace["final_result"] = {
            "prompt": asdict(r3),
            "final_claim_count": len(claims),
            "answer": r3.parsed_output.get("answer") if r3.parsed_output else None,
            "confidence": r3.parsed_output.get("confidence") if r3.parsed_output else None
        }

        # Calculate totals
        trace["total_time_ms"] = int((time.time() - start_time) * 1000)
        trace["total_tokens"] = sum(
            p["tokens"]
            for it in trace["iterations"]
            for p in it["prompts"]
        ) + r3.tokens

        print(f"\n{'='*60}")
        print(f"COMPLETE")
        print(f"{'='*60}")
        print(f"Total iterations: {len(trace['iterations'])}")
        print(f"Total claims: {len(claims)}")
        print(f"Total time: {trace['total_time_ms']}ms")
        print(f"Total tokens: {trace['total_tokens']}")
        if trace["final_result"]["answer"]:
            print(f"\nAnswer: {trace['final_result']['answer'][:100]}...")
            print(f"Confidence: {trace['final_result']['confidence']}")

        return trace


# Mock LLM provider for testing
class MockLLMProvider:
    """Mock LLM for testing the architecture without API calls"""

    async def complete(self, prompt: str) -> str:
        """Return mock responses based on prompt type"""
        await asyncio.sleep(0.1)  # Simulate latency

        if "Update claim confidence" in prompt:
            return json.dumps({
                "updates": [
                    {"id": "c001", "confidence": 0.85, "reason": "Highly relevant"},
                    {"id": "c002", "confidence": 0.65, "reason": "Partially relevant"}
                ]
            })

        elif "Create ONE new claim" in prompt:
            if "ITERATION: 1" in prompt:
                return json.dumps({
                    "action": "CREATE",
                    "claim": {
                        "content": "Need to verify the calculation method",
                        "confidence": 0.6,
                        "type": "question"
                    }
                })
            else:
                return json.dumps({"action": "SKIP"})

        elif "Provide final answer" in prompt:
            return json.dumps({
                "answer": "Based on the analysis, the answer is 42.",
                "supporting_claims": ["c001", "c002"],
                "confidence": 0.9
            })

        return "{}"


async def main():
    """Run test cases"""

    # Test case 1: Simple math problem
    test_query = "If Alice has 3 apples and Bob has twice as many, how many total?"

    initial_claims = [
        Claim("c001", "Alice has 3 apples", 0.5),
        Claim("c002", "Bob has twice Alice's apples", 0.5),
        Claim("c003", "Total = sum of individual amounts", 0.8, dirty=True),
    ]

    print(f"\n{'#'*60}")
    print("THREE-PROMPT ARCHITECTURE TEST")
    print(f"{'#'*60}")
    print(f"\nQuery: {test_query}")
    print(f"Initial claims: {len(initial_claims)}")

    # Initialize system with mock provider
    llm = MockLLMProvider()
    system = ThreePromptSystem(llm, max_iterations=3)

    # Process query
    result = await system.process_query(test_query, initial_claims)

    # Save detailed trace
    output_file = f"experiments/results/three_prompt_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'#'*60}")
    print(f"Detailed trace saved to: {output_file}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    asyncio.run(main())
