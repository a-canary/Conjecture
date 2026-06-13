#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
R&D: Semantic Relevance Filtering Experiment

Hypothesis: Claims should be filtered by semantic relevance to current problem,
not just recency or correctness.

Test: Compare random selection vs semantic similarity selection.

Research basis:
- Cluster-based Adaptive Retrieval (CAR): Dynamic context selection
- Context-Aware RAG: Semantic document clustering
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
class FilterExperiment:
    """Result of a filtering experiment"""
    method: str
    correct: int
    total: int
    accuracy: float
    q1_acc: float
    q4_acc: float


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Simple cosine similarity"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)


def simple_embed(text: str, dim: int = 64) -> List[float]:
    """Simple word-frequency based embedding (no external deps)"""
    import hashlib

    # Tokenize and hash
    words = text.lower().split()
    vec = [0.0] * dim

    for word in words:
        # Hash word to bucket
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0

    # Normalize
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]

    return vec


class SemanticClaimMemory:
    """Memory with semantic filtering"""

    def __init__(self, filter_method: str = "none"):
        self.claims: List[Dict] = []
        self.filter_method = filter_method

    def add(self, content: str, question: str, confidence: float, is_correct: bool):
        embedding = simple_embed(question)
        self.claims.append({
            "content": content[:200],
            "question": question[:100],
            "embedding": embedding,
            "confidence": confidence,
            "is_correct": is_correct
        })

    def get_hints(self, current_question: str, n: int = 3) -> str:
        """Get hints using selected filter method"""
        correct = [c for c in self.claims if c["is_correct"]]

        if not correct:
            return ""

        if self.filter_method == "none":
            # No filtering - just take top confidence
            top = sorted(correct, key=lambda x: x["confidence"], reverse=True)[:n]

        elif self.filter_method == "recency":
            # Take most recent correct claims
            top = correct[-n:]

        elif self.filter_method == "semantic":
            # Filter by semantic similarity to current question
            current_emb = simple_embed(current_question)
            for c in correct:
                c["similarity"] = cosine_similarity(current_emb, c["embedding"])
            top = sorted(correct, key=lambda x: x["similarity"], reverse=True)[:n]

        elif self.filter_method == "category":
            # Simple keyword-based category matching
            current_keywords = set(current_question.lower().split())
            for c in correct:
                c_keywords = set(c["question"].lower().split())
                c["overlap"] = len(current_keywords & c_keywords)
            top = sorted(correct, key=lambda x: x["overlap"], reverse=True)[:n]

        else:
            top = correct[:n]

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

    match = re.search(r'####\s*(\-?[\d,\.]+)', response)
    if match:
        return match.group(1).replace(",", "")

    match = re.search(r'\\boxed\{([^}]+)\}', response)
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
    """Check numerical equality"""
    try:
        p = float(pred.replace(",", ""))
        e = float(expected.replace(",", ""))
        return abs(p - e) < 0.01
    except:
        return str(pred).strip() == str(expected).strip()


def generate_categorized_problems(n: int = 100) -> List[Dict]:
    """Generate categorized math problems"""
    import random
    random.seed(42)

    problems = []
    categories = ["sales", "distance", "percentage", "division"]

    for i in range(n):
        cat = categories[i % 4]

        if cat == "sales":
            a, b = random.randint(10, 50), random.randint(10, 50)
            price_a, price_b = random.randint(2, 5), random.randint(2, 5)
            problems.append({
                "question": f"A store sells {a} items at ${price_a} each and {b} items at ${price_b} each. Total revenue?",
                "answer": str(a * price_a + b * price_b),
                "category": "sales"
            })
        elif cat == "distance":
            speed = random.randint(40, 80)
            time = random.randint(2, 6)
            problems.append({
                "question": f"A vehicle travels at {speed} mph for {time} hours. Total distance?",
                "answer": str(speed * time),
                "category": "distance"
            })
        elif cat == "percentage":
            total = random.randint(100, 500)
            pct = random.randint(20, 80)
            problems.append({
                "question": f"If {pct}% of {total} items are selected, how many items?",
                "answer": str(int(total * pct / 100)),
                "category": "percentage"
            })
        else:  # division
            items = random.randint(24, 120)
            groups = random.randint(3, 8)
            problems.append({
                "question": f"Divide {items} items equally among {groups} groups. Items per group?",
                "answer": str(items // groups),
                "category": "division"
            })

    return problems


async def run_filter_experiment(filter_method: str, problems: List[Dict]) -> FilterExperiment:
    """Run experiment with specific filter method"""
    print(f"\n{'='*50}")
    print(f"FILTER METHOD: {filter_method.upper()}")
    print(f"{'='*50}")

    memory = SemanticClaimMemory(filter_method=filter_method)
    correct = 0
    q1_correct, q4_correct = 0, 0
    quartile_size = len(problems) // 4

    for i, p in enumerate(problems):
        hints = memory.get_hints(p["question"], n=3)

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

        memory.add(
            content=f"{p['question'][:50]}... → {pred}",
            question=p["question"],
            confidence=0.9 if is_correct else 0.3,
            is_correct=is_correct
        )

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):.1f}%")

        await asyncio.sleep(0.1)

    q1_acc = 100 * q1_correct / quartile_size
    q4_acc = 100 * q4_correct / quartile_size

    return FilterExperiment(
        method=filter_method,
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        q1_acc=round(q1_acc, 2),
        q4_acc=round(q4_acc, 2)
    )


async def main():
    print("="*60)
    print("R&D: SEMANTIC RELEVANCE FILTERING EXPERIMENT")
    print("="*60)
    print("\nHypothesis: Semantic filtering of claims improves performance")
    print("by selecting the most relevant prior examples.\n")

    problems = generate_categorized_problems(100)
    print(f"Generated {len(problems)} categorized problems\n")

    # Test different filter methods
    methods = ["none", "recency", "semantic", "category"]
    results = []

    for method in methods:
        result = await run_filter_experiment(method, problems)
        results.append(result)
        await asyncio.sleep(2)

    # Summary
    print("\n" + "="*70)
    print("SEMANTIC FILTERING EXPERIMENT RESULTS")
    print("="*70)
    print(f"{'Method':<12} {'Accuracy':>10} {'Q1 Acc':>10} {'Q4 Acc':>10} {'Learning':>10}")
    print("-"*70)

    for r in results:
        learning = r.q4_acc - r.q1_acc
        print(f"{r.method:<12} {r.accuracy:>9.1f}% {r.q1_acc:>9.1f}% {r.q4_acc:>9.1f}% {learning:>+9.1f}pp")

    print("="*70)

    # Analysis
    print("\nANALYSIS:")
    best = max(results, key=lambda x: x.accuracy)
    print(f"- Best overall accuracy: {best.method} ({best.accuracy}%)")

    baseline = next((r for r in results if r.method == "none"), None)
    semantic = next((r for r in results if r.method == "semantic"), None)

    if baseline and semantic:
        delta = semantic.accuracy - baseline.accuracy
        if delta > 0:
            print(f"\n✓ CONFIRMED: Semantic filtering improves over baseline")
            print(f"  Baseline (none): {baseline.accuracy}%")
            print(f"  Semantic: {semantic.accuracy}% ({delta:+.1f}pp)")
        else:
            print(f"\n✗ NOT CONFIRMED: Semantic ({semantic.accuracy}%) doesn't beat baseline ({baseline.accuracy}%)")


if __name__ == "__main__":
    asyncio.run(main())
