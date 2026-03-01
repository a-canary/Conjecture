#!/usr/bin/env python3
"""
Hybrid Accumulation Test

Key insight: Fresh reasoning is reliable, accumulated claims add value as SUPPLEMENT.

Approach:
1. Always do fresh analysis (reliable baseline)
2. ADD accumulated claims as supplemental context (bonus knowledge)
3. Combine both for final answer

This should get the best of both worlds.
"""
import asyncio
import json
import time
import os
import re
import httpx
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict
from collections import defaultdict

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "llama3.1-8b"

QUESTIONS = [
    {"q": "A store sells apples for $2 each. If you buy 5 apples and pay with a $20 bill, how much change do you get?", "a": "10", "domain": "math"},
    {"q": "A train travels 60 miles per hour. How far does it travel in 2.5 hours?", "a": "150", "domain": "math"},
    {"q": "If 3 workers can build a wall in 12 days, how many days would it take 6 workers?", "a": "6", "domain": "math"},
    {"q": "A rectangle has length 8 and width 5. What is its area?", "a": "40", "domain": "math"},
    {"q": "You have $100. You spend 30% on food and 20% on transport. How much do you have left?", "a": "50", "domain": "math"},
    {"q": "A baker makes 24 cookies. He gives 1/3 to his neighbor and 1/4 to his friend. How many does he have left?", "a": "10", "domain": "math"},
    {"q": "If a car uses 8 gallons of gas to travel 240 miles, how many miles per gallon does it get?", "a": "30", "domain": "math"},
    {"q": "A shirt costs $40. It's on sale for 25% off. What is the sale price?", "a": "30", "domain": "math"},
    {"q": "If you read 30 pages per day, how many days to finish a 450-page book?", "a": "15", "domain": "math"},
    {"q": "A pool fills at 50 gallons per minute. How long to fill a 3000 gallon pool?", "a": "60", "domain": "math"},
    {"q": "You invest $1000 at 5% annual interest. How much interest do you earn in 2 years (simple interest)?", "a": "100", "domain": "math"},
    {"q": "A pizza is cut into 8 slices. If 3 people each eat 2 slices, how many slices are left?", "a": "2", "domain": "math"},
    {"q": "A factory produces 500 units per day. How many units in a 5-day work week?", "a": "2500", "domain": "math"},
    {"q": "If a dozen eggs costs $3.60, how much does one egg cost in cents?", "a": "30", "domain": "math"},
    {"q": "A room is 12 feet by 15 feet. How many square feet of carpet do you need?", "a": "180", "domain": "math"},
    {"q": "All dogs are animals. All animals need water. Do dogs need water? (Yes/No)", "a": "Yes", "domain": "logic"},
    {"q": "If it rains, the ground gets wet. The ground is wet. Did it rain? (Yes/No/Cannot determine)", "a": "Cannot determine", "domain": "logic"},
    {"q": "A is taller than B. B is taller than C. Is A taller than C? (Yes/No)", "a": "Yes", "domain": "logic"},
    {"q": "Some birds can fly. Penguins are birds. Can all penguins fly? (Yes/No)", "a": "No", "domain": "logic"},
    {"q": "If X then Y. Not Y. Therefore? (X / Not X / Cannot determine)", "a": "Not X", "domain": "logic"},
    {"q": "All roses are flowers. Some flowers are red. Are all roses red? (Yes/No/Cannot determine)", "a": "Cannot determine", "domain": "logic"},
    {"q": "Monday comes before Tuesday. Wednesday comes after Tuesday. What comes first? (Monday/Tuesday/Wednesday)", "a": "Monday", "domain": "logic"},
    {"q": "If A=B and B=C, does A=C? (Yes/No)", "a": "Yes", "domain": "logic"},
    {"q": "No fish are mammals. Some mammals live in water. Do any fish live in water? (Yes/No/Cannot determine)", "a": "Cannot determine", "domain": "logic"},
    {"q": "All squares are rectangles. Is a rectangle always a square? (Yes/No)", "a": "No", "domain": "logic"},
    {"q": "If it's Saturday, then I don't work. I'm working. Is it Saturday? (Yes/No)", "a": "No", "domain": "logic"},
    {"q": "Some cats are black. Some black things are chairs. Are some cats chairs? (Yes/No/Cannot determine)", "a": "Cannot determine", "domain": "logic"},
    {"q": "All prime numbers greater than 2 are odd. Is 9 prime? (Yes/No)", "a": "No", "domain": "logic"},
    {"q": "If P implies Q, and Q implies R, does P imply R? (Yes/No)", "a": "Yes", "domain": "logic"},
    {"q": "The opposite of 'always' is? (never/sometimes/rarely)", "a": "never", "domain": "logic"},
    {"q": "What is the chemical formula for water?", "a": "H2O", "domain": "science"},
    {"q": "How many chromosomes do humans have?", "a": "46", "domain": "science"},
    {"q": "What is the speed of light in km/s (approximately)?", "a": "300000", "domain": "science"},
    {"q": "What is the atomic number of Carbon?", "a": "6", "domain": "science"},
    {"q": "What is the boiling point of water in Celsius?", "a": "100", "domain": "science"},
    {"q": "How many planets are in our solar system?", "a": "8", "domain": "science"},
    {"q": "What is the pH of pure water?", "a": "7", "domain": "science"},
    {"q": "What is the chemical symbol for gold?", "a": "Au", "domain": "science"},
    {"q": "How many bones are in the adult human body?", "a": "206", "domain": "science"},
    {"q": "What is absolute zero in Celsius?", "a": "-273", "domain": "science"},
    {"q": "If you flip a fair coin 3 times, what's the probability of getting all heads? Express as a fraction.", "a": "1/8", "domain": "math"},
    {"q": "What comes next: 2, 6, 12, 20, 30, ?", "a": "42", "domain": "math"},
    {"q": "A clock shows 3:15. What is the angle between the hour and minute hands?", "a": "7.5", "domain": "math"},
    {"q": "If TODAY is Wednesday, what day was it 100 days ago?", "a": "Sunday", "domain": "math"},
    {"q": "How many prime numbers are there between 1 and 20?", "a": "8", "domain": "math"},
    {"q": "What is 15% of 80?", "a": "12", "domain": "math"},
    {"q": "If a=2 and b=3, what is a^b + b^a?", "a": "17", "domain": "math"},
    {"q": "What is the next prime number after 29?", "a": "31", "domain": "math"},
    {"q": "A hexagon has how many sides?", "a": "6", "domain": "math"},
    {"q": "What is the sum of angles in a triangle?", "a": "180", "domain": "math"},
]


@dataclass
class Claim:
    content: str
    domain: str
    confidence: float
    is_correct: bool = False

    def quality_score(self) -> float:
        return self.confidence + (0.5 if self.is_correct else 0.0)


@dataclass
class ClaimMemory:
    claims: List[Claim] = field(default_factory=list)
    domain_pools: Dict[str, List[Claim]] = field(default_factory=lambda: defaultdict(list))

    def add(self, claim: Claim):
        if claim.confidence >= 0.5:
            self.claims.append(claim)
            self.domain_pools[claim.domain].append(claim)

    def get_top_claims(self, domain: str, n: int = 3) -> List[Claim]:
        """Get top N correct claims from same domain"""
        correct_claims = [c for c in self.domain_pools[domain] if c.is_correct]
        return sorted(correct_claims, key=lambda c: c.quality_score(), reverse=True)[:n]


class CerebrasLLM:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.total_tokens = 0

    async def generate(self, prompt: str, max_tokens: int = 200) -> tuple[str, int]:
        for attempt in range(3):
            try:
                resp = await self.client.post(
                    CEREBRAS_URL,
                    headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
                    json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": 0.1},
                    timeout=60.0
                )
                if resp.status_code == 429:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    return "", 0
                data = resp.json()
                tokens = data.get("usage", {}).get("total_tokens", 0)
                self.total_tokens += tokens
                return data["choices"][0]["message"]["content"], tokens
            except:
                await asyncio.sleep(1)
        return "", 0


def extract_answer(response: str, expected: str) -> str:
    if not response:
        return ""
    response = response.strip()
    if expected.lower() in response.lower()[:50]:
        return expected
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        for num in numbers:
            try:
                if abs(float(num) - float(expected)) < 0.1:
                    return num
            except:
                pass
        return numbers[0]
    if expected.lower() in ['yes', 'no']:
        if 'yes' in response.lower()[:20]:
            return 'Yes'
        if 'no' in response.lower()[:20]:
            return 'No'
    return response.split()[0] if response.split() else ""


def check_answer(predicted: str, expected: str) -> bool:
    pred = str(predicted).lower().strip().rstrip('.')
    exp = str(expected).lower().strip()
    if pred == exp or pred.startswith(exp) or exp in pred[:20]:
        return True
    try:
        return abs(float(pred.replace(',', '')) - float(exp.replace(',', ''))) < 0.1
    except:
        return False


async def run_fresh(llm: CerebrasLLM, questions: List[Dict]) -> Dict:
    print("\n" + "="*70)
    print("FRESH CONJECTURE (baseline)")
    print("="*70)

    correct = 0
    results = []

    for i, q in enumerate(questions):
        prompt1 = f"Question: {q['q']}\n\nWhat key facts or steps are needed? Be brief."
        resp1, _ = await llm.generate(prompt1, 100)

        prompt2 = f"Analysis: {resp1[:150] if resp1 else 'solve'}\n\nQuestion: {q['q']}\n\nAnswer with just the value."
        resp2, _ = await llm.generate(prompt2, 50)

        answer = extract_answer(resp2, q['a'])
        is_correct = check_answer(answer, q['a'])
        if is_correct:
            correct += 1
        results.append(is_correct)

        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:2d}/50] {status} {q['domain']:7s} | acc={100*correct/(i+1):5.1f}%", flush=True)
        await asyncio.sleep(0.2)

    return {"method": "fresh", "correct": correct, "accuracy": round(100*correct/50, 1), "results": results}


async def run_hybrid(llm: CerebrasLLM, questions: List[Dict]) -> Dict:
    """Hybrid: Fresh reasoning + accumulated claims as supplement"""
    print("\n" + "="*70)
    print("HYBRID (Fresh + Accumulated supplement)")
    print("="*70)

    memory = ClaimMemory()
    correct = 0
    results = []

    for i, q in enumerate(questions):
        domain = q['domain']

        # Get prior CORRECT claims as hints (not replacement)
        prior_hints = memory.get_top_claims(domain, n=2)
        hints = ""
        if prior_hints:
            hints = "Hints from similar problems:\n"
            for c in prior_hints:
                hints += f"- {c.content[:60]}...\n"
            hints += "\n"

        # Step 1: Fresh analysis (always do this)
        prompt1 = f"{hints}Question: {q['q']}\n\nWhat key facts or steps are needed? Be specific."
        resp1, _ = await llm.generate(prompt1, 120)

        # Step 2: Solve with fresh analysis
        prompt2 = f"Analysis: {resp1[:150] if resp1 else 'solve carefully'}\n\nQuestion: {q['q']}\n\nAnswer with just the value."
        resp2, _ = await llm.generate(prompt2, 50)

        answer = extract_answer(resp2, q['a'])
        is_correct = check_answer(answer, q['a'])
        if is_correct:
            correct += 1
        results.append(is_correct)

        # Store claim with correctness
        memory.add(Claim(
            content=f"[{domain}] {q['q'][:30]}... → {answer}",
            domain=domain,
            confidence=0.9 if is_correct else 0.3,
            is_correct=is_correct
        ))

        n_hints = len(prior_hints)
        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:2d}/50] {status} {domain:7s} | acc={100*correct/(i+1):5.1f}% | hints={n_hints}", flush=True)
        await asyncio.sleep(0.2)

    return {"method": "hybrid", "correct": correct, "accuracy": round(100*correct/50, 1), "results": results}


async def main():
    print("="*70)
    print("HYBRID ACCUMULATION TEST")
    print("="*70)
    print("Approach: Fresh reasoning + accumulated hints (best of both)")
    print("="*70)

    async with httpx.AsyncClient() as client:
        llm1 = CerebrasLLM(client)
        fresh_result = await run_fresh(llm1, QUESTIONS)

        await asyncio.sleep(2)

        llm2 = CerebrasLLM(client)
        hybrid_result = await run_hybrid(llm2, QUESTIONS)

    def analyze(results):
        first, last = results[:25], results[25:]
        return {"first_25": round(100*sum(first)/25, 1), "last_25": round(100*sum(last)/25, 1),
                "delta": round(100*(sum(last)-sum(first))/25, 1)}

    fresh_a = analyze(fresh_result['results'])
    hybrid_a = analyze(hybrid_result['results'])

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Method':<15} {'Accuracy':>10} {'First 25':>10} {'Last 25':>10} {'Δ':>8}")
    print("-"*70)
    print(f"{'Fresh':<15} {fresh_result['accuracy']:>9.1f}% {fresh_a['first_25']:>9.1f}% {fresh_a['last_25']:>9.1f}% {fresh_a['delta']:>+7.1f}pp")
    print(f"{'Hybrid':<15} {hybrid_result['accuracy']:>9.1f}% {hybrid_a['first_25']:>9.1f}% {hybrid_a['last_25']:>9.1f}% {hybrid_a['delta']:>+7.1f}pp")
    print("-"*70)

    gate_passed = hybrid_result['accuracy'] >= fresh_result['accuracy']
    delta = hybrid_result['accuracy'] - fresh_result['accuracy']

    print(f"\nGATE: Hybrid ≥ Fresh → {'✅ PASSED' if gate_passed else '❌ FAILED'} ({delta:+.1f}pp)")
    print("="*70)

    with open("/workspace/data/hybrid_results.json", "w") as f:
        json.dump({"fresh": fresh_result, "hybrid": hybrid_result, "gate_passed": gate_passed,
                   "analysis": {"fresh": fresh_a, "hybrid": hybrid_a}}, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
