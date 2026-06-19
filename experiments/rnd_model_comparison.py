#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
R&D: Model Size/Provider Comparison

Hypothesis: Accumulation effectiveness varies by model size.
- Small models (8B): More sensitive to context pollution
- Large models (70B+): Better able to filter relevant info

Test: Compare Cerebras llama3.1-8b vs Chutes DeepSeek-V3
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
class ModelResult:
    provider: str
    model: str
    method: str  # direct, accumulated
    correct: int
    total: int
    accuracy: float


class LLMClient:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model

        if provider == "cerebras":
            self.url = "https://api.cerebras.ai/v1/chat/completions"
            self.api_key = os.getenv("CEREBRAS_API_KEY")
        elif provider == "chutes":
            self.url = "https://llm.chutes.ai/v1/chat/completions"
            self.api_key = os.getenv("CHUTES_API_KEY")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def generate(self, prompt: str, max_tokens: int = 150) -> str:
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
                    print(f"  Error: {e}")
                    await asyncio.sleep(2)
        return ""


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


def generate_problems(n: int = 50) -> List[Dict]:
    import random
    random.seed(42)
    problems = []
    for i in range(n):
        if i % 4 == 0:
            a, b = random.randint(10, 50), random.randint(10, 50)
            problems.append({"question": f"Store: {a} at $2, {b} at $3. Total?", "answer": str(a*2 + b*3)})
        elif i % 4 == 1:
            s, t = random.randint(40, 80), random.randint(2, 6)
            problems.append({"question": f"Car: {s} mph, {t} hours. Miles?", "answer": str(s*t)})
        elif i % 4 == 2:
            total, pct = random.randint(100, 500), random.randint(20, 80)
            problems.append({"question": f"{pct}% of {total}?", "answer": str(int(total*pct/100))})
        else:
            items, groups = random.randint(24, 120), random.randint(3, 8)
            problems.append({"question": f"{items}/{groups}?", "answer": str(items//groups)})
    return problems


async def run_test(llm: LLMClient, problems: List[Dict], use_accumulation: bool) -> ModelResult:
    method = "accumulated" if use_accumulation else "direct"
    print(f"\n{'='*50}")
    print(f"MODEL: {llm.provider}/{llm.model} ({method})")
    print(f"{'='*50}")

    memory: List[Dict] = []
    correct = 0

    for i, p in enumerate(problems):
        if use_accumulation and memory:
            correct_claims = [c for c in memory if c["is_correct"]][-3:]
            if correct_claims:
                hints = "Patterns:\n" + "\n".join(f"- {c['content'][:60]}" for c in correct_claims) + "\n\n"
            else:
                hints = ""
        else:
            hints = ""

        prompt = f"{hints}Problem: {p['question']}\n\nAnswer:"

        response = await llm.generate(prompt, max_tokens=100)
        pred = extract_answer(response)
        is_correct = check_answer(pred, p["answer"])

        if is_correct:
            correct += 1

        if use_accumulation:
            memory.append({
                "content": f"{p['question'][:40]}... → {pred}",
                "is_correct": is_correct
            })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):.1f}%")

        await asyncio.sleep(0.2)  # Rate limiting

    return ModelResult(
        provider=llm.provider,
        model=llm.model,
        method=method,
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2)
    )


async def main():
    print("="*60)
    print("R&D: MODEL SIZE/PROVIDER COMPARISON")
    print("="*60)

    problems = generate_problems(50)
    print(f"Generated {len(problems)} problems\n")

    models = [
        ("cerebras", "llama3.1-8b"),
        ("chutes", "deepseek-ai/DeepSeek-V3"),
    ]

    results = []

    for provider, model in models:
        llm = LLMClient(provider, model)

        # Direct (no accumulation)
        r1 = await run_test(llm, problems, use_accumulation=False)
        results.append(r1)
        await asyncio.sleep(2)

        # With accumulation
        r2 = await run_test(llm, problems, use_accumulation=True)
        results.append(r2)
        await asyncio.sleep(2)

    # Summary
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    print(f"{'Provider':<12} {'Model':<25} {'Method':<12} {'Accuracy':>10}")
    print("-"*70)

    for r in results:
        model_short = r.model[-20:] if len(r.model) > 20 else r.model
        print(f"{r.provider:<12} {model_short:<25} {r.method:<12} {r.accuracy:>9.1f}%")

    print("="*70)

    # Analysis
    print("\nACCUMULATION EFFECT BY MODEL:")
    for provider, model in models:
        direct = next((r for r in results if r.provider == provider and r.method == "direct"), None)
        accum = next((r for r in results if r.provider == provider and r.method == "accumulated"), None)
        if direct and accum:
            delta = accum.accuracy - direct.accuracy
            print(f"  {provider}/{model[-15:]}: {delta:+.1f}pp")


if __name__ == "__main__":
    asyncio.run(main())
