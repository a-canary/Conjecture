#!/usr/bin/env python3
"""
R&D: Prompt Position Primacy Experiment

Based on research finding "Lost in the Middle":
- LLMs have U-shaped attention curve
- Beginning (primacy) and end (recency) positions get more attention
- Middle positions see >30% performance degradation

Hypothesis: Moving claims from middle to START of prompt improves performance.
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
class PositionResult:
    """Result of position experiment"""
    position: str  # start, middle, end
    correct: int
    total: int
    accuracy: float
    q1_acc: float
    q4_acc: float


class ClaimMemory:
    def __init__(self):
        self.claims: List[Dict] = []

    def add(self, content: str, confidence: float, is_correct: bool):
        self.claims.append({
            "content": content[:200],
            "confidence": confidence,
            "is_correct": is_correct
        })

    def get_top_correct(self, n: int = 3) -> List[Dict]:
        correct = [c for c in self.claims if c["is_correct"]]
        return sorted(correct, key=lambda x: x["confidence"], reverse=True)[:n]


async def generate(prompt: str, max_tokens: int = 200) -> Tuple[str, int]:
    """Generate response"""
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
            except:
                await asyncio.sleep(2)
    return "", 0


def extract_answer(response: str) -> str:
    if not response:
        return ""
    match = re.search(r'####\s*(\-?[\d,\.]+)', response)
    if match:
        return match.group(1).replace(",", "")
    match = re.search(r'answer\s*(?:is|:)\s*\$?(\-?[\d,\.]+)', response, re.I)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r'\-?[\d,]+\.?\d*', response)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def check_answer(pred: str, expected: str) -> bool:
    try:
        return abs(float(pred.replace(",", "")) - float(expected.replace(",", ""))) < 0.01
    except:
        return str(pred).strip() == str(expected).strip()


def build_prompt_start(question: str, claims: List[Dict]) -> str:
    """Claims at START (testing primacy bias)"""
    if claims:
        hints = "KEY PATTERNS FROM SIMILAR PROBLEMS:\n"
        for c in claims:
            hints += f"• {c['content'][:80]}\n"
        hints += "\n"
    else:
        hints = ""

    return f"""{hints}Problem: {question}

Solve and give the answer."""


def build_prompt_middle(question: str, claims: List[Dict]) -> str:
    """Claims in MIDDLE (current approach)"""
    if claims:
        hints = "\nPatterns from similar problems:\n"
        for c in claims:
            hints += f"- {c['content'][:80]}\n"
    else:
        hints = ""

    return f"""Problem: {question}
{hints}
Solve step by step and give the answer."""


def build_prompt_end(question: str, claims: List[Dict]) -> str:
    """Claims at END (testing recency bias)"""
    if claims:
        hints = "\n\nReference patterns:\n"
        for c in claims:
            hints += f"- {c['content'][:80]}\n"
    else:
        hints = ""

    return f"""Problem: {question}

Solve and give the answer.{hints}"""


BUILDERS = {
    "start": build_prompt_start,
    "middle": build_prompt_middle,
    "end": build_prompt_end
}


def generate_problems(n: int = 80) -> List[Dict]:
    import random
    random.seed(42)
    problems = []
    for i in range(n):
        if i % 4 == 0:
            a, b = random.randint(10, 50), random.randint(10, 50)
            problems.append({
                "question": f"A store sells {a} items at $2 each and {b} items at $3 each. Total revenue?",
                "answer": str(a * 2 + b * 3)
            })
        elif i % 4 == 1:
            speed, time = random.randint(40, 80), random.randint(2, 6)
            problems.append({
                "question": f"A car travels at {speed} mph for {time} hours. Distance?",
                "answer": str(speed * time)
            })
        elif i % 4 == 2:
            total, pct = random.randint(100, 500), random.randint(20, 80)
            problems.append({
                "question": f"What is {pct}% of {total}?",
                "answer": str(int(total * pct / 100))
            })
        else:
            items, groups = random.randint(24, 120), random.randint(3, 8)
            problems.append({
                "question": f"Divide {items} equally among {groups}. How many each?",
                "answer": str(items // groups)
            })
    return problems


async def run_position_experiment(position: str, problems: List[Dict]) -> PositionResult:
    print(f"\n{'='*50}")
    print(f"POSITION: {position.upper()} (claims at {position})")
    print(f"{'='*50}")

    builder = BUILDERS[position]
    memory = ClaimMemory()
    correct = 0
    q1_correct, q4_correct = 0, 0
    quartile = len(problems) // 4

    for i, p in enumerate(problems):
        claims = memory.get_top_correct(3)
        prompt = builder(p["question"], claims)

        response, _ = await generate(prompt, max_tokens=150)
        pred = extract_answer(response)
        is_correct = check_answer(pred, p["answer"])

        if is_correct:
            correct += 1
            if i < quartile:
                q1_correct += 1
            elif i >= 3 * quartile:
                q4_correct += 1

        memory.add(
            content=f"{p['question'][:50]}... → {pred}",
            confidence=0.9 if is_correct else 0.3,
            is_correct=is_correct
        )

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):.1f}%")

        await asyncio.sleep(0.1)

    q1_acc = 100 * q1_correct / quartile
    q4_acc = 100 * q4_correct / quartile

    return PositionResult(
        position=position,
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        q1_acc=round(q1_acc, 2),
        q4_acc=round(q4_acc, 2)
    )


async def main():
    print("="*60)
    print("R&D: PROMPT POSITION PRIMACY EXPERIMENT")
    print("="*60)
    print("\nHypothesis: Claims at START leverage primacy bias")
    print("(Based on 'Lost in the Middle' research)\n")

    problems = generate_problems(80)
    print(f"Generated {len(problems)} problems\n")

    results = []
    for pos in ["middle", "start", "end"]:
        result = await run_position_experiment(pos, problems)
        results.append(result)
        await asyncio.sleep(2)

    print("\n" + "="*70)
    print("POSITION EXPERIMENT RESULTS")
    print("="*70)
    print(f"{'Position':<10} {'Accuracy':>10} {'Q1 Acc':>10} {'Q4 Acc':>10} {'Learning':>10}")
    print("-"*70)

    for r in results:
        learning = r.q4_acc - r.q1_acc
        print(f"{r.position:<10} {r.accuracy:>9.1f}% {r.q1_acc:>9.1f}% {r.q4_acc:>9.1f}% {learning:>+9.1f}pp")

    print("="*70)

    middle = next((r for r in results if r.position == "middle"), None)
    start = next((r for r in results if r.position == "start"), None)

    if middle and start:
        delta = start.accuracy - middle.accuracy
        if delta > 0:
            print(f"\n✓ PRIMACY CONFIRMED: START position {delta:+.1f}pp better than MIDDLE")
        else:
            print(f"\n✗ PRIMACY NOT CONFIRMED: MIDDLE ({middle.accuracy}%) >= START ({start.accuracy}%)")


if __name__ == "__main__":
    asyncio.run(main())
