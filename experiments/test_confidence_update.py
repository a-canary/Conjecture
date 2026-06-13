#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Test confidence UPDATE scenario.
Model should recognize when to update an existing claim's confidence
rather than create a new one.
"""

import asyncio
import os
import json
from datetime import datetime
from openai import AsyncOpenAI

SYSTEM_PROMPT = """You are a careful reasoning assistant that builds knowledge through claims.

A claim is a single, atomic statement that captures one piece of reasoning.
Each claim must have: content (what you believe), type (observation/assertion/assumption), and confidence (0.0-1.0).

You have access to existing claims. For each turn, you may EITHER:
1. UPDATE confidence on an existing claim if evidence changes your belief
2. CREATE exactly ONE new claim that fills a gap

Output format:
- To update: "UPDATE [claim_id]: confidence X.X → Y.Y (reason)"
- To create: "CREATE: [content] (type, confidence: X.X)"

Only ONE action per turn. Prefer UPDATE when new evidence affects existing claims."""


def build_prompt(problem: str, claims: list, new_evidence: str = None) -> str:
    claims_text = "\n".join([
        f"  [{c['id']}] {c['content']} ({c['type']}, conf: {c['confidence']})"
        for c in claims
    ])

    evidence_text = f"\nNEW EVIDENCE: {new_evidence}" if new_evidence else ""

    return f"""EXISTING CLAIMS:
{claims_text}
{evidence_text}
PROBLEM: {problem}

Should you UPDATE an existing claim's confidence or CREATE a new claim?

Your action:"""


async def test_confidence_updates():
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    scenarios = [
        {
            "name": "Disconfirming evidence",
            "problem": "Can penguins fly?",
            "claims": [
                {"id": "c001", "content": "All birds can fly", "type": "assumption", "confidence": 0.9},
                {"id": "c002", "content": "Penguins are birds", "type": "observation", "confidence": 1.0},
            ],
            "new_evidence": "Penguins are flightless birds that swim instead of fly.",
            "expected_action": "UPDATE",
            "expected_claim": "c001"
        },
        {
            "name": "Confirming evidence",
            "problem": "Is this calculation correct: 7 × 3 = 21?",
            "claims": [
                {"id": "c010", "content": "7 times 3 equals 21", "type": "assertion", "confidence": 0.8},
            ],
            "new_evidence": "Double-checked: 7 + 7 + 7 = 21 ✓",
            "expected_action": "UPDATE",
            "expected_claim": "c010"
        },
        {
            "name": "Missing claim (should CREATE)",
            "problem": "What is the total cost?",
            "claims": [
                {"id": "c020", "content": "Item costs $10", "type": "observation", "confidence": 1.0},
                {"id": "c021", "content": "Quantity is 5", "type": "observation", "confidence": 1.0},
            ],
            "new_evidence": None,
            "expected_action": "CREATE",
            "expected_claim": None
        },
    ]

    print("\n" + "="*70)
    print("TESTING CONFIDENCE UPDATES")
    print("="*70)
    print(f"Model: meta-llama/llama-3.1-8b-instruct")
    print()

    results = []

    for s in scenarios:
        print(f"\n--- {s['name']} ---")
        print(f"Expected: {s['expected_action']} {s['expected_claim'] or '(new)'}")

        prompt = build_prompt(s['problem'], s['claims'], s['new_evidence'])

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
            correct_action = (
                (s['expected_action'] == "UPDATE" and is_update) or
                (s['expected_action'] == "CREATE" and is_create)
            )
            has_confidence = any(c in output for c in ["0.", "1.0", "conf", "→"])

            result = {
                "scenario": s['name'],
                "expected": s['expected_action'],
                "actual": "UPDATE" if is_update else ("CREATE" if is_create else "UNCLEAR"),
                "correct_action": correct_action,
                "has_confidence": has_confidence,
                "success": correct_action and has_confidence
            }
            results.append(result)

            status = "✅" if result['success'] else "❌"
            print(f"\n{status} Expected: {s['expected_action']}, Got: {result['actual']}, Correct: {correct_action}")

        except Exception as e:
            print(f"Error: {e}")
            results.append({"scenario": s['name'], "success": False, "error": str(e)})

        await asyncio.sleep(0.5)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    successes = sum(1 for r in results if r.get('success'))
    correct_actions = sum(1 for r in results if r.get('correct_action'))
    print(f"Correct action: {correct_actions}/{len(results)} ({100*correct_actions/len(results):.0f}%)")
    print(f"Full success: {successes}/{len(results)} ({100*successes/len(results):.0f}%)")

    for r in results:
        status = "✅" if r.get('success') else "❌"
        print(f"  {status} {r['scenario']}: expected {r.get('expected')}, got {r.get('actual')}")

    return results


if __name__ == "__main__":
    asyncio.run(test_confidence_updates())
