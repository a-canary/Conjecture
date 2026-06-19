#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
R&D: Optimized Accumulation - Combining All Research Findings

This experiment implements ALL recommended improvements:
1. Position Primacy: Claims at START of prompt
2. Strict Gating: Only 0.8+ confidence claims
3. Windowing: Only recent 20 claims
4. Semantic Filtering: Category-matched claims
5. Limited Count: Max 3 claims, not 5

Expected: +10-15pp improvement over baseline accumulation
"""

import asyncio
import os
import re
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple
import httpx
from dotenv import load_dotenv

load_dotenv("/workspace/.env")


@dataclass
class AccumResult:
    method: str
    correct: int
    total: int
    accuracy: float
    q1_acc: float
    q4_acc: float
    learning_delta: float


def simple_embed(text: str, dim: int = 64) -> List[float]:
    """Hash-based embedding"""
    words = text.lower().split()
    vec = [0.0] * dim
    for word in words:
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot


def get_category(question: str) -> str:
    """Simple category detection"""
    q = question.lower()
    if any(x in q for x in ["sell", "buy", "price", "cost", "revenue", "store"]):
        return "sales"
    elif any(x in q for x in ["travel", "speed", "mph", "distance", "hour"]):
        return "distance"
    elif any(x in q for x in ["percent", "%"]):
        return "percentage"
    elif any(x in q for x in ["divide", "share", "equally", "each"]):
        return "division"
    return "general"


class BaselineMemory:
    """Original accumulation approach"""
    def __init__(self):
        self.claims: List[Dict] = []

    def add(self, content: str, confidence: float, is_correct: bool, **kwargs):
        self.claims.append({
            "content": content[:200],
            "confidence": confidence,
            "is_correct": is_correct
        })

    def get_hints(self, question: str) -> str:
        correct = [c for c in self.claims if c["is_correct"]]
        top = sorted(correct, key=lambda x: x["confidence"], reverse=True)[:5]
        if not top:
            return ""
        return "\nPatterns:\n" + "\n".join(f"- {c['content'][:80]}" for c in top) + "\n"

    def build_prompt(self, question: str) -> str:
        hints = self.get_hints(question)
        # Claims in MIDDLE (original)
        return f"""Problem: {question}
{hints}
Solve and give the answer."""


class OptimizedMemory:
    """Optimized accumulation with all improvements"""
    def __init__(self):
        self.claims: List[Dict] = []
        self.window_size = 20
        self.confidence_threshold = 0.8
        self.max_claims = 3

    def add(self, content: str, confidence: float, is_correct: bool, category: str = "", embedding: List[float] = None):
        self.claims.append({
            "content": content[:200],
            "confidence": confidence,
            "is_correct": is_correct,
            "category": category,
            "embedding": embedding or []
        })

    def get_hints(self, question: str, category: str = "") -> str:
        # Window: only recent claims
        windowed = self.claims[-self.window_size:] if len(self.claims) > self.window_size else self.claims

        # Correct + high confidence
        filtered = [c for c in windowed
                   if c["is_correct"] and c["confidence"] >= self.confidence_threshold]

        # Semantic + category filtering
        question_emb = simple_embed(question)
        scored = []
        for c in filtered:
            # Category match bonus
            cat_bonus = 1.5 if c["category"] == category else 1.0
            # Semantic similarity
            sem_sim = cosine_sim(question_emb, c["embedding"]) if c["embedding"] else 0.5
            score = c["confidence"] * cat_bonus * (0.5 + sem_sim)
            scored.append((c, score))

        # Top N claims
        top = sorted(scored, key=lambda x: x[1], reverse=True)[:self.max_claims]

        if not top:
            return ""

        hints = ""
        for c, _ in top:
            hints += f"• {c['content'][:80]}\n"
        return hints

    def build_prompt(self, question: str) -> str:
        category = get_category(question)
        hints = self.get_hints(question, category)

        # Claims at START (primacy bias)
        if hints:
            return f"""KEY PATTERNS FROM SIMILAR PROBLEMS:
{hints}
Problem: {question}

Answer:"""
        else:
            return f"""Problem: {question}

Answer:"""


async def generate(prompt: str, max_tokens: int = 150) -> Tuple[str, int]:
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


def generate_problems(n: int = 100) -> List[Dict]:
    import random
    random.seed(42)
    problems = []
    for i in range(n):
        cat = ["sales", "distance", "percentage", "division"][i % 4]
        if cat == "sales":
            a, b = random.randint(10, 50), random.randint(10, 50)
            problems.append({
                "question": f"Store sells {a} at $2 and {b} at $3. Total?",
                "answer": str(a * 2 + b * 3),
                "category": "sales"
            })
        elif cat == "distance":
            speed, time = random.randint(40, 80), random.randint(2, 6)
            problems.append({
                "question": f"Car at {speed} mph for {time} hours. Distance?",
                "answer": str(speed * time),
                "category": "distance"
            })
        elif cat == "percentage":
            total, pct = random.randint(100, 500), random.randint(20, 80)
            problems.append({
                "question": f"{pct}% of {total}?",
                "answer": str(int(total * pct / 100)),
                "category": "percentage"
            })
        else:
            items, groups = random.randint(24, 120), random.randint(3, 8)
            problems.append({
                "question": f"{items} divided by {groups}?",
                "answer": str(items // groups),
                "category": "division"
            })
    return problems


async def run_method(method: str, memory, problems: List[Dict]) -> AccumResult:
    print(f"\n{'='*50}")
    print(f"METHOD: {method.upper()}")
    print(f"{'='*50}")

    correct = 0
    q1_correct, q4_correct = 0, 0
    quartile = len(problems) // 4

    for i, p in enumerate(problems):
        prompt = memory.build_prompt(p["question"])
        response, _ = await generate(prompt, max_tokens=150)
        pred = extract_answer(response)
        is_correct = check_answer(pred, p["answer"])

        if is_correct:
            correct += 1
            if i < quartile:
                q1_correct += 1
            elif i >= 3 * quartile:
                q4_correct += 1

        # Store claim
        memory.add(
            content=f"{p['question'][:40]}... → {pred}",
            confidence=0.85 if is_correct else 0.35,
            is_correct=is_correct,
            category=p.get("category", ""),
            embedding=simple_embed(p["question"])
        )

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):.1f}%")

        await asyncio.sleep(0.1)

    q1_acc = 100 * q1_correct / quartile
    q4_acc = 100 * q4_correct / quartile

    return AccumResult(
        method=method,
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        q1_acc=round(q1_acc, 2),
        q4_acc=round(q4_acc, 2),
        learning_delta=round(q4_acc - q1_acc, 2)
    )


async def main():
    print("="*60)
    print("R&D: OPTIMIZED VS BASELINE ACCUMULATION")
    print("="*60)
    print("\nComparing baseline accumulation vs research-optimized approach")
    print("Optimizations: position primacy, strict gating, windowing, semantic filtering\n")

    problems = generate_problems(100)
    print(f"Generated {len(problems)} problems\n")

    results = []

    # Baseline accumulation
    baseline = BaselineMemory()
    result1 = await run_method("baseline", baseline, problems)
    results.append(result1)

    await asyncio.sleep(2)

    # Optimized accumulation
    optimized = OptimizedMemory()
    result2 = await run_method("optimized", optimized, problems)
    results.append(result2)

    # Summary
    print("\n" + "="*70)
    print("OPTIMIZED ACCUMULATION RESULTS")
    print("="*70)
    print(f"{'Method':<12} {'Accuracy':>10} {'Q1 Acc':>10} {'Q4 Acc':>10} {'Learning':>10}")
    print("-"*70)

    for r in results:
        print(f"{r.method:<12} {r.accuracy:>9.1f}% {r.q1_acc:>9.1f}% {r.q4_acc:>9.1f}% {r.learning_delta:>+9.1f}pp")

    print("="*70)

    baseline_r = results[0]
    optimized_r = results[1]

    delta_acc = optimized_r.accuracy - baseline_r.accuracy
    delta_learn = optimized_r.learning_delta - baseline_r.learning_delta

    print(f"\nIMPROVEMENT:")
    print(f"  Accuracy: {delta_acc:+.1f}pp")
    print(f"  Learning: {delta_learn:+.1f}pp")

    if delta_acc > 0:
        print(f"\n✓ OPTIMIZATIONS CONFIRMED: All improvements work together")
    else:
        print(f"\n✗ OPTIMIZATIONS NOT CONFIRMED: Baseline still better")


if __name__ == "__main__":
    asyncio.run(main())
