#!/usr/bin/env python3
"""
R&D: Claim Window Size Experiment

Hypothesis: Claim accumulation degrades at scale due to "lost in the middle" effect.
Test: Compare different claim window sizes (last N vs all claims).

Research basis:
- Lost in the Middle (Liu et al. 2023): LLMs perform worse on middle context
- Context Rot: Performance degrades as context window grows
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
class WindowExperiment:
    """Result of a window size experiment"""
    window_size: int  # 0 = all claims
    correct: int
    total: int
    accuracy: float
    q1_acc: float
    q4_acc: float
    learning_delta: float


class ClaimMemory:
    """Memory with configurable window"""

    def __init__(self, window_size: int = 0):
        self.claims: List[Dict] = []
        self.window_size = window_size  # 0 = unlimited

    def add(self, content: str, confidence: float, is_correct: bool):
        self.claims.append({
            "content": content[:200],
            "confidence": confidence,
            "is_correct": is_correct
        })

    def get_hints(self, n: int = 3) -> str:
        """Get hints from windowed claims"""
        # Apply window: only use last N claims if window_size > 0
        claims = self.claims
        if self.window_size > 0:
            claims = claims[-self.window_size:]

        correct = [c for c in claims if c["is_correct"]]
        top = sorted(correct, key=lambda x: x["confidence"], reverse=True)[:n]

        if not top:
            return ""

        return "Patterns from similar problems:\n" + "\n".join(
            f"- {c['content'][:80]}" for c in top
        ) + "\n\n"


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
            except Exception as e:
                await asyncio.sleep(2)
    return "", 0


def extract_answer(response: str) -> str:
    """Extract numerical answer"""
    if not response:
        return ""

    # GSM8K #### pattern
    match = re.search(r'####\s*(\-?[\d,\.]+)', response)
    if match:
        return match.group(1).replace(",", "")

    # Boxed pattern
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
    """Check numerical equality"""
    try:
        p = float(pred.replace(",", ""))
        e = float(expected.replace(",", ""))
        return abs(p - e) < 0.01
    except:
        return str(pred).strip() == str(expected).strip()


def generate_problems(n: int = 100) -> List[Dict]:
    """Generate math problems"""
    import random
    random.seed(42)

    problems = []
    for i in range(n):
        # Vary difficulty
        if i % 4 == 0:
            a, b = random.randint(10, 50), random.randint(10, 50)
            problems.append({
                "question": f"A store sells {a} apples at $2 each and {b} oranges at $3 each. What is the total revenue?",
                "answer": str(a * 2 + b * 3)
            })
        elif i % 4 == 1:
            speed, time = random.randint(40, 80), random.randint(2, 6)
            problems.append({
                "question": f"A car travels at {speed} mph for {time} hours. How many miles does it travel?",
                "answer": str(speed * time)
            })
        elif i % 4 == 2:
            total, part = random.randint(100, 500), random.randint(20, 80)
            problems.append({
                "question": f"A library has {total} books. If {part}% are fiction, how many fiction books are there?",
                "answer": str(int(total * part / 100))
            })
        else:
            items, people = random.randint(24, 120), random.randint(3, 8)
            problems.append({
                "question": f"If {items} candies are shared equally among {people} children, how many candies does each child get?",
                "answer": str(items // people)
            })

    return problems


async def run_window_experiment(window_size: int, problems: List[Dict]) -> WindowExperiment:
    """Run experiment with specific window size"""
    print(f"\n{'='*50}")
    print(f"WINDOW SIZE: {window_size if window_size > 0 else 'UNLIMITED'}")
    print(f"{'='*50}")

    memory = ClaimMemory(window_size=window_size)
    correct = 0
    q1_correct, q4_correct = 0, 0
    quartile_size = len(problems) // 4

    for i, p in enumerate(problems):
        # Get hints from memory
        hints = memory.get_hints(n=3)

        # Simple prompt with hints
        prompt = f"""{hints}Solve this problem. Give the numerical answer.

Problem: {p['question']}

Answer:"""

        response, _ = await generate(prompt, max_tokens=150)
        pred = extract_answer(response)
        is_correct = check_answer(pred, p['answer'])

        if is_correct:
            correct += 1
            if i < quartile_size:
                q1_correct += 1
            elif i >= 3 * quartile_size:
                q4_correct += 1

        # Store claim
        memory.add(
            content=f"{p['question'][:50]}... → {pred}",
            confidence=0.9 if is_correct else 0.3,
            is_correct=is_correct
        )

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):.1f}%")

        await asyncio.sleep(0.1)

    q1_acc = 100 * q1_correct / quartile_size
    q4_acc = 100 * q4_correct / quartile_size

    return WindowExperiment(
        window_size=window_size,
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        q1_acc=round(q1_acc, 2),
        q4_acc=round(q4_acc, 2),
        learning_delta=round(q4_acc - q1_acc, 2)
    )


async def main():
    print("="*60)
    print("R&D: CLAIM WINDOW SIZE EXPERIMENT")
    print("="*60)
    print("\nHypothesis: Limiting claim window size improves performance")
    print("by avoiding 'lost in the middle' and context pollution.\n")

    # Generate problems
    problems = generate_problems(100)
    print(f"Generated {len(problems)} problems\n")

    # Test different window sizes
    window_sizes = [0, 5, 10, 20, 50]  # 0 = unlimited
    results = []

    for ws in window_sizes:
        result = await run_window_experiment(ws, problems)
        results.append(result)
        await asyncio.sleep(2)

    # Summary
    print("\n" + "="*70)
    print("WINDOW SIZE EXPERIMENT RESULTS")
    print("="*70)
    print(f"{'Window':<10} {'Accuracy':>10} {'Q1 Acc':>10} {'Q4 Acc':>10} {'Learning':>10}")
    print("-"*70)

    for r in results:
        ws = str(r.window_size) if r.window_size > 0 else "ALL"
        print(f"{ws:<10} {r.accuracy:>9.1f}% {r.q1_acc:>9.1f}% {r.q4_acc:>9.1f}% {r.learning_delta:>+9.1f}pp")

    print("="*70)

    # Analysis
    print("\nANALYSIS:")
    best = max(results, key=lambda x: x.accuracy)
    best_learning = max(results, key=lambda x: x.learning_delta)

    ws = str(best.window_size) if best.window_size > 0 else "ALL"
    print(f"- Best accuracy: {ws} window ({best.accuracy}%)")

    ws = str(best_learning.window_size) if best_learning.window_size > 0 else "ALL"
    print(f"- Best learning effect: {ws} window ({best_learning.learning_delta:+.1f}pp)")

    # Check for "lost in the middle" effect
    unlimited = next((r for r in results if r.window_size == 0), None)
    limited = [r for r in results if r.window_size > 0]

    if unlimited and limited:
        better_limited = [r for r in limited if r.accuracy > unlimited.accuracy]
        if better_limited:
            print(f"\n✓ CONFIRMED: Windowing improves accuracy over unlimited")
            print(f"  Unlimited: {unlimited.accuracy}%")
            print(f"  Best limited: {max(limited, key=lambda x: x.accuracy).accuracy}%")
        else:
            print(f"\n✗ NOT CONFIRMED: Unlimited ({unlimited.accuracy}%) beats all windowed")


if __name__ == "__main__":
    asyncio.run(main())
