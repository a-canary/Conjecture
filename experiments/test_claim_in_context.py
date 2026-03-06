#!/usr/bin/env python3
"""
Test optimal prompt with claims in context.
Model should either:
1. Update confidence on existing claim, OR
2. Write exactly 1 missing claim

Uses the winner configuration: detailed + interrogative + natural
"""

import asyncio
import os
import json
from datetime import datetime
from openai import AsyncOpenAI

# Winner prompt from 32-variant batch
SYSTEM_PROMPT = """You are a careful reasoning assistant that builds knowledge through claims.

A claim is a single, atomic statement that captures one piece of reasoning.
Each claim must have: content (what you believe), type (observation/assertion/assumption), and confidence (0.0-1.0).

You have access to existing claims. For each turn, you may EITHER:
1. UPDATE confidence on an existing claim if you have new evidence
2. CREATE exactly ONE new claim that fills a gap in reasoning

Output format:
- To update: "UPDATE claim_id: new_confidence (reason)"
- To create: "CREATE: [content] (type, confidence: X.X)"

Only ONE action per turn. Be precise."""


def build_prompt_with_context(problem: str, existing_claims: list) -> str:
    """Build prompt with claims in context."""
    claims_text = "\n".join([
        f"  [{c['id']}] {c['content']} ({c['type']}, conf: {c['confidence']})"
        for c in existing_claims
    ])

    return f"""EXISTING CLAIMS:
{claims_text}

PROBLEM: {problem}

What is the single most important action: update a claim's confidence OR create one missing claim?

Your action:"""


async def test_with_context():
    """Test the optimal prompt with claims in context."""

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    # Test scenarios
    scenarios = [
        {
            "name": "Math with partial claims",
            "problem": "A store sells apples for $3 each. If I buy 7 apples with a 10% discount, how much do I pay?",
            "claims": [
                {"id": "c001", "content": "Each apple costs $3", "type": "observation", "confidence": 1.0},
                {"id": "c002", "content": "I am buying 7 apples", "type": "observation", "confidence": 1.0},
                {"id": "c003", "content": "There is a 10% discount", "type": "observation", "confidence": 1.0},
            ]
        },
        {
            "name": "Logic with uncertain claim",
            "problem": "All birds can fly. Penguins are birds. Can penguins fly?",
            "claims": [
                {"id": "c010", "content": "All birds can fly", "type": "assumption", "confidence": 0.9},
                {"id": "c011", "content": "Penguins are birds", "type": "observation", "confidence": 1.0},
            ]
        },
        {
            "name": "Reasoning gap",
            "problem": "Why might a remote worker be more productive than an office worker?",
            "claims": [
                {"id": "c020", "content": "Remote workers avoid commute time", "type": "assertion", "confidence": 0.95},
                {"id": "c021", "content": "Remote workers have fewer interruptions", "type": "conjecture", "confidence": 0.7},
            ]
        },
    ]

    print("\n" + "="*70)
    print("TESTING CLAIMS IN CONTEXT")
    print("="*70)
    print(f"Model: meta-llama/llama-3.1-8b-instruct")
    print(f"Prompt: detailed + interrogative (winner from 32-variant batch)")
    print()

    results = []

    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        print(f"Problem: {scenario['problem'][:60]}...")
        print(f"Existing claims: {len(scenario['claims'])}")

        prompt = build_prompt_with_context(scenario['problem'], scenario['claims'])

        try:
            response = await client.chat.completions.create(
                model="meta-llama/llama-3.1-8b-instruct",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )

            output = response.choices[0].message.content
            print(f"\nOutput:\n{output}")

            # Evaluate
            is_update = "UPDATE" in output.upper()
            is_create = "CREATE" in output.upper()
            has_confidence = any(c in output for c in ["0.", "1.0", "conf"])
            is_single_action = (is_update and not is_create) or (is_create and not is_update)
            is_atomic = len(output.split('\n')) <= 3 and len(output) < 300

            result = {
                "scenario": scenario['name'],
                "is_update": is_update,
                "is_create": is_create,
                "is_single_action": is_single_action,
                "has_confidence": has_confidence,
                "is_atomic": is_atomic,
                "output_length": len(output),
                "success": is_single_action and has_confidence and is_atomic
            }
            results.append(result)

            status = "✅" if result['success'] else "❌"
            action = "UPDATE" if is_update else ("CREATE" if is_create else "UNCLEAR")
            print(f"\n{status} Action: {action}, Single: {is_single_action}, Confidence: {has_confidence}, Atomic: {is_atomic}")

        except Exception as e:
            print(f"Error: {e}")
            results.append({"scenario": scenario['name'], "success": False, "error": str(e)})

        await asyncio.sleep(0.5)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    successes = sum(1 for r in results if r.get('success'))
    print(f"Success rate: {successes}/{len(results)} ({100*successes/len(results):.0f}%)")

    for r in results:
        status = "✅" if r.get('success') else "❌"
        print(f"  {status} {r['scenario']}")

    # Save results
    results_file = f"experiments/results/claim_context_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == "__main__":
    asyncio.run(test_with_context())
