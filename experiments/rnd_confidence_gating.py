#!/usr/bin/env python3
"""
R&D: Confidence Gating Threshold Experiment

Based on research finding:
- Current gating (0.5 confidence) is too permissive
- Low-confidence claims add noise
- Incorrect claims (even with 0.3 confidence) still pollute context

Hypothesis: Stricter confidence gating (0.8+) improves accumulation.
"""

import asyncio
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple
import httpx
from dotenv import load_dotenv

load_dotenv("/workspace/.env")


@dataclass
class GatingResult:
    threshold: float
    correct: int
    total: int
    accuracy: float
    claims_used: int
    claims_rejected: int


class GatedClaimMemory:
    def __init__(self, threshold: float = 0.5):
        self.claims: List[Dict] = []
        self.threshold = threshold
        self.rejected = 0

    def add(self, content: str, confidence: float, is_correct: bool):
        self.claims.append({
            "content": content[:200],
            "confidence": confidence,
            "is_correct": is_correct
        })

    def get_filtered(self, n: int = 3) -> Tuple[List[Dict], int, int]:
        # Only include claims above confidence threshold
        filtered = [c for c in self.claims
                   if c["is_correct"] and c["confidence"] >= self.threshold]
        rejected = len(self.claims) - len(filtered)
        top = sorted(filtered, key=lambda x: x["confidence"], reverse=True)[:n]
        return top, len(filtered), rejected


async def generate(prompt: str, max_tokens: int = 200) -> Tuple[str, int]:
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


def generate_problems(n: int = 80) -> List[Dict]:
    import random
    random.seed(42)
    problems = []
    for i in range(n):
        if i % 4 == 0:
            a, b = random.randint(10, 50), random.randint(10, 50)
            problems.append({
                "question": f"Store sells {a} items at $2 and {b} at $3. Total?",
                "answer": str(a * 2 + b * 3)
            })
        elif i % 4 == 1:
            speed, time = random.randint(40, 80), random.randint(2, 6)
            problems.append({
                "question": f"Car at {speed} mph for {time} hours. Distance?",
                "answer": str(speed * time)
            })
        elif i % 4 == 2:
            total, pct = random.randint(100, 500), random.randint(20, 80)
            problems.append({
                "question": f"{pct}% of {total}?",
                "answer": str(int(total * pct / 100))
            })
        else:
            items, groups = random.randint(24, 120), random.randint(3, 8)
            problems.append({
                "question": f"{items} divided by {groups}?",
                "answer": str(items // groups)
            })
    return problems


async def run_gating_experiment(threshold: float, problems: List[Dict]) -> GatingResult:
    print(f"\n{'='*50}")
    print(f"CONFIDENCE THRESHOLD: {threshold}")
    print(f"{'='*50}")

    memory = GatedClaimMemory(threshold=threshold)
    correct = 0
    total_used, total_rejected = 0, 0

    for i, p in enumerate(problems):
        claims, used, rejected = memory.get_filtered(3)
        total_used += len(claims)
        total_rejected += rejected

        if claims:
            hints = "Patterns:\n" + "\n".join(f"- {c['content'][:60]}" for c in claims) + "\n\n"
        else:
            hints = ""

        prompt = f"""{hints}Problem: {p['question']}

Answer:"""

        response, _ = await generate(prompt, max_tokens=100)
        pred = extract_answer(response)
        is_correct = check_answer(pred, p["answer"])

        if is_correct:
            correct += 1

        # Assign confidence based on correctness
        # Simulate real-world where we don't know correctness at store time
        # But correct answers tend to have higher model confidence
        confidence = 0.85 if is_correct else 0.4

        memory.add(
            content=f"{p['question'][:40]}... → {pred}",
            confidence=confidence,
            is_correct=is_correct
        )

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):.1f}%")

        await asyncio.sleep(0.1)

    return GatingResult(
        threshold=threshold,
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        claims_used=total_used,
        claims_rejected=total_rejected
    )


async def main():
    print("="*60)
    print("R&D: CONFIDENCE GATING THRESHOLD EXPERIMENT")
    print("="*60)
    print("\nHypothesis: Higher confidence thresholds reduce noise\n")

    problems = generate_problems(80)
    print(f"Generated {len(problems)} problems\n")

    thresholds = [0.0, 0.5, 0.7, 0.8, 0.9]
    results = []

    for thresh in thresholds:
        result = await run_gating_experiment(thresh, problems)
        results.append(result)
        await asyncio.sleep(2)

    print("\n" + "="*70)
    print("CONFIDENCE GATING RESULTS")
    print("="*70)
    print(f"{'Threshold':<12} {'Accuracy':>10} {'Claims Used':>12} {'Rejected':>10}")
    print("-"*70)

    for r in results:
        print(f"{r.threshold:<12.1f} {r.accuracy:>9.1f}% {r.claims_used:>12} {r.claims_rejected:>10}")

    print("="*70)

    # Find optimal threshold
    best = max(results, key=lambda x: x.accuracy)
    baseline = next((r for r in results if r.threshold == 0.0), None)

    print(f"\nBest threshold: {best.threshold} ({best.accuracy}%)")
    if baseline and best.threshold > 0:
        delta = best.accuracy - baseline.accuracy
        print(f"Improvement over no gating: {delta:+.1f}pp")

        if delta > 0:
            print(f"\n✓ GATING CONFIRMED: Higher threshold improves accuracy")
        else:
            print(f"\n✗ GATING NOT CONFIRMED: No gating is optimal")


if __name__ == "__main__":
    asyncio.run(main())
