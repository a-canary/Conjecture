#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Conjecture V2: Verification-Only Approach

Key insight: Strong models already reason well. Don't use Conjecture for generation.
Instead, use it for VERIFICATION of the direct answer.

Flow:
1. Get direct answer from model (fast, accurate)
2. IF confidence is low OR problem is complex: verify with decomposition
3. Return verified answer

This should give us the best of both worlds:
- Speed and accuracy of direct generation
- Error catching from Conjecture verification
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
        self.total_tokens = 0

    async def generate(self, prompt: str, max_tokens: int = 400) -> tuple[str, int]:
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
                        data = resp.json()
                        tokens = data.get("usage", {}).get("total_tokens", 0)
                        self.total_tokens += tokens
                        return data["choices"][0]["message"]["content"], tokens
                except Exception as e:
                    await asyncio.sleep(2)
        return "", 0


def load_gsm8k_sample(n: int = 30):
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
            })
    return problems


def extract_answer(response: str) -> str:
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


def problem_complexity(question: str) -> str:
    """Estimate problem complexity based on text features"""
    # Count numbers (more numbers = more complex)
    numbers = len(re.findall(r'\d+', question))
    # Count clauses (more commas/periods = more complex)
    clauses = question.count(',') + question.count('.')
    # Count conditional words
    conditionals = len(re.findall(r'\b(if|then|but|however|because|after|before)\b', question, re.I))

    score = numbers * 1 + clauses * 0.5 + conditionals * 2

    if score > 10:
        return "high"
    elif score > 5:
        return "medium"
    return "low"


async def method_direct(llm: LLMClient, question: str) -> tuple[str, str]:
    """Direct generation"""
    prompt = f"""Solve this math problem. Show your work and end with #### followed by the answer.

Problem: {question}

Solution:"""
    response, _ = await llm.generate(prompt)
    return extract_answer(response), response


async def method_verify(llm: LLMClient, question: str, direct_answer: str) -> tuple[str, bool]:
    """Verification step - check if direct answer is correct"""
    prompt = f"""Verify this answer to the math problem.

Problem: {question}

Proposed answer: {direct_answer}

Check the work step by step. Is this answer correct? Reply with:
- "VERIFIED" if the answer is correct
- "INCORRECT: [correct_answer]" if wrong, providing the right answer
"""
    response, _ = await llm.generate(prompt, max_tokens=300)

    if "VERIFIED" in response.upper():
        return direct_answer, True

    # Extract corrected answer
    match = re.search(r'INCORRECT[:\s]*(\-?[\d,\.]+)', response, re.I)
    if match:
        return match.group(1).replace(",", ""), False

    # Try to extract any number from the response
    corrected = extract_answer(response)
    if corrected:
        return corrected, False

    return direct_answer, True  # Fall back to direct if verification fails


async def method_conjecture_v2(llm: LLMClient, question: str, complexity: str) -> tuple[str, str]:
    """
    Conjecture V2: Verify-only approach

    1. Always get direct answer first
    2. For complex problems, verify with second call
    3. Return verified answer
    """
    # Step 1: Direct answer
    direct_answer, direct_response = await method_direct(llm, question)

    # Step 2: For complex problems, verify
    if complexity in ["high", "medium"]:
        verified_answer, was_correct = await method_verify(llm, question, direct_answer)
        return verified_answer, f"[direct={direct_answer}, verified={verified_answer}, was_correct={was_correct}]"

    return direct_answer, direct_response


async def benchmark():
    llm = LLMClient()
    problems = load_gsm8k_sample(30)

    results = {
        "direct": {"correct": 0, "total": 0},
        "conjecture_v2": {"correct": 0, "total": 0},
        "conjecture_v2_verification_helped": 0,
        "conjecture_v2_verification_hurt": 0,
    }

    print("=" * 80)
    print("CONJECTURE V2: VERIFICATION-ONLY APPROACH")
    print("=" * 80)

    for p in problems:
        complexity = problem_complexity(p["question"])
        print(f"\n[{p['id']}] Complexity: {complexity}")
        print(f"Q: {p['question'][:80]}...")

        # Method 1: Direct
        direct_answer, _ = await method_direct(llm, p["question"])
        direct_correct = check_answer(direct_answer, p["answer"])
        results["direct"]["total"] += 1
        if direct_correct:
            results["direct"]["correct"] += 1

        # Method 2: Conjecture V2 (verify-only)
        v2_answer, v2_trace = await method_conjecture_v2(llm, p["question"], complexity)
        v2_correct = check_answer(v2_answer, p["answer"])
        results["conjecture_v2"]["total"] += 1
        if v2_correct:
            results["conjecture_v2"]["correct"] += 1

        # Track verification impact
        if v2_correct and not direct_correct:
            results["conjecture_v2_verification_helped"] += 1
            print(f"  >>> VERIFICATION HELPED: {direct_answer} -> {v2_answer}")
        elif direct_correct and not v2_correct:
            results["conjecture_v2_verification_hurt"] += 1
            print(f"  >>> VERIFICATION HURT: {direct_answer} -> {v2_answer}")

        status_d = "OK" if direct_correct else "FAIL"
        status_v = "OK" if v2_correct else "FAIL"
        print(f"  Direct: {direct_answer} [{status_d}] | V2: {v2_answer} [{status_v}] | Expected: {p['answer']}")

        await asyncio.sleep(0.3)

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    direct_acc = 100 * results["direct"]["correct"] / results["direct"]["total"]
    v2_acc = 100 * results["conjecture_v2"]["correct"] / results["conjecture_v2"]["total"]

    print(f"Direct:          {results['direct']['correct']}/{results['direct']['total']} ({direct_acc:.1f}%)")
    print(f"Conjecture V2:   {results['conjecture_v2']['correct']}/{results['conjecture_v2']['total']} ({v2_acc:.1f}%)")
    print(f"Verification helped: {results['conjecture_v2_verification_helped']}")
    print(f"Verification hurt:   {results['conjecture_v2_verification_hurt']}")
    print(f"Net impact: {results['conjecture_v2_verification_helped'] - results['conjecture_v2_verification_hurt']:+d}")
    print(f"Tokens used: {llm.total_tokens}")

    # Save results
    output_dir = Path("/workspace/data/benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"conjecture_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    asyncio.run(benchmark())
