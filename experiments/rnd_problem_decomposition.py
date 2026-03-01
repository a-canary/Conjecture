#!/usr/bin/env python3
"""
R&D: Problem Decomposition Strategies Experiment

Hypothesis: The way we decompose problems affects claim quality.
Test: Compare different decomposition strategies.

Research basis:
- Chain-of-Thought hurts small models (Phase 4 finding)
- Problem decomposition (Cycle 12: +9% improvement)
"""

import asyncio
import os
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
import httpx
from dotenv import load_dotenv

load_dotenv("/workspace/.env")


@dataclass
class DecompExperiment:
    """Result of decomposition experiment"""
    strategy: str
    correct: int
    total: int
    accuracy: float
    avg_tokens: float


async def generate(prompt: str, max_tokens: int = 300) -> Tuple[str, int]:
    """Generate response using Chutes API"""
    url = "https://llm.chutes.ai/v1/chat/completions"
    api_key = os.getenv("CHUTES_API_KEY")

    async with httpx.AsyncClient() as client:
        for attempt in range(3):
            try:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "deepseek-ai/DeepSeek-V3",
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
                    return data["choices"][0]["message"]["content"], tokens
            except Exception:
                await asyncio.sleep(2)
    return "", 0


def extract_answer(response: str) -> str:
    """Extract numerical answer"""
    if not response:
        return ""

    patterns = [
        r'####\s*(\-?[\d,\.]+)',
        r'\\boxed\{([^}]+)\}',
        r'answer\s*(?:is|:)\s*\$?(\-?[\d,\.]+)',
        r'=\s*\$?(\-?[\d,\.]+)\s*$'
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.I | re.M)
        if match:
            return match.group(1).replace(",", "")

    numbers = re.findall(r'\-?[\d,]+\.?\d*', response)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def check_answer(pred: str, expected: str) -> bool:
    """Check numerical equality"""
    try:
        p = float(pred.replace(",", ""))
        e = float(expected.replace(",", ""))
        return abs(p - e) < 0.01
    except:
        return str(pred).strip() == str(expected).strip()


def load_gsm8k_sample(n: int = 50) -> List[Dict]:
    """Load GSM8K sample"""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")

        problems = []
        for i, item in enumerate(ds):
            if i >= n:
                break
            match = re.search(r'####\s*(\-?[\d,\.]+)', item["answer"])
            if match:
                problems.append({
                    "question": item["question"],
                    "answer": match.group(1).replace(",", "")
                })
        return problems
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        return []


async def solve_direct(question: str) -> Tuple[str, int]:
    """Strategy 1: Direct solving"""
    prompt = f"""Solve this math problem and give the final numerical answer.

Problem: {question}

Answer:"""
    return await generate(prompt, max_tokens=200)


async def solve_identify_then_solve(question: str) -> Tuple[str, int]:
    """Strategy 2: Identify key information, then solve"""
    prompt = f"""Problem: {question}

Step 1: What are the key numbers and what operation is needed?
Step 2: Calculate the answer.

Final answer (number only):"""
    return await generate(prompt, max_tokens=250)


async def solve_backward(question: str) -> Tuple[str, int]:
    """Strategy 3: Work backward from goal"""
    prompt = f"""Problem: {question}

What is being asked for?
What information do we have?
Calculate the answer.

Answer:"""
    return await generate(prompt, max_tokens=250)


async def solve_formula(question: str) -> Tuple[str, int]:
    """Strategy 4: Identify formula first"""
    prompt = f"""Problem: {question}

1. What formula or relationship applies here?
2. Plug in the numbers.
3. Calculate.

Answer:"""
    return await generate(prompt, max_tokens=250)


async def solve_structured(question: str) -> Tuple[str, int]:
    """Strategy 5: Structured decomposition"""
    prompt = f"""Problem: {question}

Given:
- [List the given quantities]

Find:
- [What we need to calculate]

Solution:
- [One calculation step]

Answer:"""
    return await generate(prompt, max_tokens=300)


STRATEGIES = {
    "direct": solve_direct,
    "identify_solve": solve_identify_then_solve,
    "backward": solve_backward,
    "formula": solve_formula,
    "structured": solve_structured
}


async def run_strategy_experiment(strategy: str, problems: List[Dict]) -> DecompExperiment:
    """Run experiment with specific strategy"""
    print(f"\n{'='*50}")
    print(f"STRATEGY: {strategy.upper()}")
    print(f"{'='*50}")

    solver = STRATEGIES[strategy]
    correct = 0
    total_tokens = 0

    for i, p in enumerate(problems):
        response, tokens = await solver(p["question"])
        total_tokens += tokens

        pred = extract_answer(response)
        is_correct = check_answer(pred, p["answer"])

        if is_correct:
            correct += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):.1f}%")

        await asyncio.sleep(0.1)

    return DecompExperiment(
        strategy=strategy,
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        avg_tokens=round(total_tokens / len(problems), 1)
    )


async def main():
    print("="*60)
    print("R&D: PROBLEM DECOMPOSITION STRATEGIES")
    print("="*60)
    print("\nTesting different approaches to problem decomposition.\n")

    # Load real GSM8K problems
    problems = load_gsm8k_sample(50)
    if not problems:
        print("Failed to load GSM8K, using generated problems")
        import random
        random.seed(42)
        problems = []
        for i in range(50):
            a, b = random.randint(10, 50), random.randint(10, 50)
            problems.append({
                "question": f"A store has {a} apples and {b} oranges. How many fruits total?",
                "answer": str(a + b)
            })

    print(f"Loaded {len(problems)} problems\n")

    results = []
    for strategy in STRATEGIES.keys():
        result = await run_strategy_experiment(strategy, problems)
        results.append(result)
        await asyncio.sleep(2)

    # Summary
    print("\n" + "="*70)
    print("DECOMPOSITION STRATEGY RESULTS")
    print("="*70)
    print(f"{'Strategy':<15} {'Accuracy':>10} {'Avg Tokens':>12}")
    print("-"*70)

    for r in results:
        print(f"{r.strategy:<15} {r.accuracy:>9.1f}% {r.avg_tokens:>11.0f}")

    print("="*70)

    # Analysis
    print("\nANALYSIS:")
    best_acc = max(results, key=lambda x: x.accuracy)
    most_efficient = min(results, key=lambda x: x.avg_tokens)

    print(f"- Best accuracy: {best_acc.strategy} ({best_acc.accuracy}%)")
    print(f"- Most token-efficient: {most_efficient.strategy} ({most_efficient.avg_tokens} tokens)")

    # Efficiency score
    for r in results:
        # Score = accuracy / (tokens / 100)
        efficiency = r.accuracy / (r.avg_tokens / 100) if r.avg_tokens > 0 else 0
        r.efficiency = efficiency

    best_eff = max(results, key=lambda x: x.efficiency)
    print(f"- Best efficiency (acc/tokens): {best_eff.strategy}")

    direct = next((r for r in results if r.strategy == "direct"), None)
    if direct and best_acc.strategy != "direct":
        delta = best_acc.accuracy - direct.accuracy
        print(f"\n✓ Decomposition improves over direct: {delta:+.1f}pp")
    elif direct:
        print(f"\n→ Direct solving is optimal for these problems")


if __name__ == "__main__":
    asyncio.run(main())
