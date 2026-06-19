#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic: Why does Conjecture hurt strong model performance?

Hypothesis: The decomposition step loses critical information that the
direct prompt retains. This script captures side-by-side responses to
identify the failure pattern.
"""

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv("/workspace/.env")


class LLMClient:
    def __init__(self):
        self.url = "https://llm.chutes.ai/v1/chat/completions"
        self.api_key = os.getenv("CHUTES_API_KEY")
        self.model = "deepseek-ai/DeepSeek-V3"

    async def generate(self, prompt: str, max_tokens: int = 400) -> str:
        async with httpx.AsyncClient() as client:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        self.url,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": 0.1
                        },
                        timeout=120.0
                    )
                    if resp.status_code == 429:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    await asyncio.sleep(2)
        return ""


def load_gsm8k_sample(n: int = 10):
    """Load a small sample for detailed analysis"""
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    problems = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        solution = item["answer"]
        match = re.search(r'####\s*(\-?[\d,\.]+)', solution)
        if match:
            answer = match.group(1).replace(",", "")
            problems.append({
                "id": i,
                "question": item["question"],
                "answer": answer,
                "full_solution": solution
            })
    return problems


def extract_answer(response: str) -> str:
    """Extract answer from response"""
    if not response:
        return ""

    # #### pattern
    match = re.search(r'####\s*(\-?[\d,\.]+)', response)
    if match:
        return match.group(1).replace(",", "")

    # boxed pattern
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        return match.group(1).replace(",", "")

    # "answer is X" pattern
    match = re.search(r'answer\s*(?:is|:)\s*\$?(\-?[\d,\.]+)', response, re.I)
    if match:
        return match.group(1).replace(",", "")

    # Last number
    numbers = re.findall(r'\-?[\d,]+\.?\d*', response)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def check_answer(pred: str, expected: str) -> bool:
    try:
        pn = float(pred.replace(",", ""))
        en = float(expected.replace(",", ""))
        return abs(pn - en) < 0.01
    except:
        return pred.strip() == expected.strip()


async def diagnose():
    llm = LLMClient()
    problems = load_gsm8k_sample(20)

    results = []

    print("=" * 80)
    print("DIAGNOSTIC: Why Conjecture Hurts Strong Models")
    print("=" * 80)

    for p in problems:
        print(f"\n--- Problem {p['id']} ---")
        print(f"Q: {p['question'][:100]}...")
        print(f"Expected: {p['answer']}")

        # Method 1: Direct
        direct_prompt = f"""Solve this math problem. Show your work and end with #### followed by the answer.

Problem: {p['question']}

Solution:"""

        direct_response = await llm.generate(direct_prompt)
        direct_answer = extract_answer(direct_response)
        direct_correct = check_answer(direct_answer, p['answer'])

        # Method 2: Conjecture (decompose then solve)
        decompose_prompt = f"""Analyze this problem. What are the key facts and what steps are needed?

Problem: {p['question']}

Analysis:"""

        decomposition = await llm.generate(decompose_prompt, max_tokens=200)

        solve_prompt = f"""Analysis: {decomposition[:250]}

Problem: {p['question']}

Solve step by step. End with #### and the answer."""

        conjecture_response = await llm.generate(solve_prompt)
        conjecture_answer = extract_answer(conjecture_response)
        conjecture_correct = check_answer(conjecture_answer, p['answer'])

        result = {
            "id": p['id'],
            "question": p['question'],
            "expected": p['answer'],
            "direct": {
                "response": direct_response[:500],
                "answer": direct_answer,
                "correct": direct_correct
            },
            "conjecture": {
                "decomposition": decomposition[:300],
                "response": conjecture_response[:500],
                "answer": conjecture_answer,
                "correct": conjecture_correct
            },
            "direct_only": direct_correct and not conjecture_correct,
            "conjecture_only": conjecture_correct and not direct_correct,
        }
        results.append(result)

        # Print comparison
        status_d = "OK" if direct_correct else "FAIL"
        status_c = "OK" if conjecture_correct else "FAIL"
        print(f"  Direct: {direct_answer} [{status_d}]")
        print(f"  Conjecture: {conjecture_answer} [{status_c}]")

        if direct_correct and not conjecture_correct:
            print("  >>> CONJECTURE LOST INFORMATION <<<")
            print(f"  Decomposition: {decomposition[:150]}...")

        await asyncio.sleep(0.5)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    direct_correct = sum(1 for r in results if r['direct']['correct'])
    conjecture_correct = sum(1 for r in results if r['conjecture']['correct'])
    direct_only = sum(1 for r in results if r['direct_only'])
    conjecture_only = sum(1 for r in results if r['conjecture_only'])

    print(f"Direct accuracy: {direct_correct}/{len(results)} ({100*direct_correct/len(results):.1f}%)")
    print(f"Conjecture accuracy: {conjecture_correct}/{len(results)} ({100*conjecture_correct/len(results):.1f}%)")
    print(f"Direct solves but Conjecture fails: {direct_only}")
    print(f"Conjecture solves but Direct fails: {conjecture_only}")

    # Analyze failure patterns
    print("\n" + "-" * 80)
    print("FAILURE PATTERN ANALYSIS")
    print("-" * 80)

    failures = [r for r in results if r['direct_only']]
    for f in failures[:5]:
        print(f"\nProblem {f['id']}: {f['question'][:80]}...")
        print(f"Expected: {f['expected']}")
        print(f"Direct got: {f['direct']['answer']} (CORRECT)")
        print(f"Conjecture got: {f['conjecture']['answer']} (WRONG)")
        print(f"Decomposition: {f['conjecture']['decomposition'][:200]}...")

    # Save results
    output_dir = Path("/workspace/data/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"conjecture_loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    return results


if __name__ == "__main__":
    asyncio.run(diagnose())
