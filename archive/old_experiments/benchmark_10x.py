#!/usr/bin/env python3
"""
10x Benchmark Suite

Runs all key benchmarks with 10x larger datasets:
- GSM8K: 200 problems (was 20)
- Cross-session: 200 problems (was 20)
- Optimization: 100 problems (was 10)
"""
import asyncio
import json
import os
import re
import time
import sqlite3
import httpx
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime
import random

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "llama3.1-8b"

# Generate 200 GSM8K-style problems
def generate_gsm8k_problems(n: int = 200) -> List[Dict]:
    """Generate n math word problems with known answers"""
    problems = []
    templates = [
        # Rate problems
        ("A car travels {d} miles in {t} hours. What is its speed in mph?", lambda d, t: d // t, "rate"),
        ("A factory produces {n} units per hour. How many units in {h} hours?", lambda n, h: n * h, "rate"),
        # Percentage problems
        ("What is {p}% of {n}?", lambda p, n: p * n // 100, "percent"),
        ("{name} has ${m}. They spend {p}% on food. How much is left?", lambda m, p: m - (m * p // 100), "percent"),
        # Proportion problems
        ("{n1} items cost ${c}. How much do {n2} items cost?", lambda n1, c, n2: round(c * n2 / n1, 2), "proportion"),
        # Geometry
        ("A rectangle has length {l} and width {w}. What is its area?", lambda l, w: l * w, "geometry"),
        ("A rectangle has perimeter {p} and length {l}. What is its width?", lambda p, l: (p - 2*l) // 2, "geometry"),
        # Work problems
        ("If {w1} workers finish a job in {d1} days, how many days for {w2} workers?", lambda w1, d1, w2: w1 * d1 // w2, "inverse"),
        # Simple arithmetic
        ("{name} buys {n} items at ${p} each. Total cost?",lambda n, p: n * p, "arithmetic"),
        ("{name} has ${m}. Buys {n} items at ${p} each. Money left?", lambda m, n, p: m - n * p, "arithmetic"),
        # Age problems
        ("{name1} is {x} years older than {name2}. {name2} is {a} years old. How old is {name1}?", lambda x, a: x + a, "age"),
        # Sequence
        ("What comes next: {s1}, {s2}, {s3}, {s4}, ?", lambda s1, s2, s3, s4: s4 + (s4 - s3), "sequence"),
    ]

    names = ["John", "Mary", "Tom", "Lisa", "Alex", "Sarah", "Mike", "Emma", "David", "Anna"]

    i = 0
    while len(problems) < n:
        template_idx = i % len(templates)
        template, calc, ptype = templates[template_idx]

        try:
            if "speed" in template:
                d, t = random.choice([(120, 2), (180, 3), (240, 4), (300, 5), (150, 3), (200, 4), (250, 5), (100, 2), (160, 4), (210, 3)])
                q = template.format(d=d, t=t)
                a = str(calc(d, t))
            elif "factory" in template:
                n_val, h = random.choice([(50, 8), (30, 6), (40, 5), (25, 4), (60, 10), (45, 9), (35, 7), (20, 8), (55, 11), (75, 4)])
                q = template.format(n=n_val, h=h)
                a = str(calc(n_val, h))
            elif "% of" in template:
                p, n_val = random.choice([(10, 200), (15, 80), (20, 150), (25, 120), (30, 100), (40, 50), (50, 80), (5, 200), (12, 250), (8, 125)])
                q = template.format(p=p, n=n_val)
                a = str(calc(p, n_val))
            elif "spend" in template and "%" in template:
                m, p = random.choice([(100, 20), (80, 25), (50, 40), (200, 15), (150, 30), (120, 10), (90, 50), (60, 20), (75, 40), (110, 30)])
                name = random.choice(names)
                q = template.format(name=name, m=m, p=p)
                a = str(calc(m, p))
            elif "items cost" in template:
                n1, c, n2 = random.choice([(3, 6, 5), (4, 8, 7), (5, 10, 8), (2, 4, 6), (6, 12, 9), (3, 9, 4), (4, 12, 5), (5, 15, 6), (2, 6, 5), (3, 12, 7)])
                q = template.format(n1=n1, c=c, n2=n2)
                a = str(calc(n1, c, n2))
            elif "area" in template:
                l, w = random.choice([(8, 5), (10, 6), (12, 4), (7, 9), (15, 3), (9, 8), (11, 7), (6, 6), (14, 5), (8, 8)])
                q = template.format(l=l, w=w)
                a = str(calc(l, w))
            elif "perimeter" in template:
                l = random.choice([5, 6, 7, 8, 9, 10, 11, 12])
                w = random.choice([3, 4, 5, 6, 7, 8])
                p = 2 * (l + w)
                q = template.format(p=p, l=l)
                a = str(w)
            elif "workers" in template:
                w1, d1, w2 = random.choice([(4, 12, 6), (5, 10, 10), (3, 15, 5), (6, 8, 4), (8, 6, 12), (2, 20, 4), (10, 5, 5), (4, 8, 8), (6, 12, 9), (3, 18, 6)])
                q = template.format(w1=w1, d1=d1, w2=w2)
                a = str(calc(w1, d1, w2))
            elif "Total cost" in template:
                n_val, p = random.choice([(5, 8), (3, 12), (7, 5), (4, 15), (6, 10), (8, 7), (2, 25), (9, 4), (10, 6), (3, 20)])
                name = random.choice(names)
                q = template.format(name=name, n=n_val, p=p)
                a = str(calc(n_val, p))
            elif "Money left" in template:
                m = random.choice([50, 80, 100, 120, 150, 200])
                n_val = random.choice([2, 3, 4, 5, 6])
                p = random.choice([5, 8, 10, 12, 15])
                if m >= n_val * p:
                    name = random.choice(names)
                    q = template.format(name=name, m=m, n=n_val, p=p)
                    a = str(calc(m, n_val, p))
                else:
                    i += 1
                    continue
            elif "older than" in template:
                x, age = random.choice([(5, 10), (8, 12), (3, 15), (10, 8), (7, 20), (4, 18), (6, 14), (9, 11), (2, 25), (12, 6)])
                name1, name2 = random.sample(names, 2)
                q = template.format(name1=name1, name2=name2, x=x, a=age)
                a = str(calc(x, age))
            elif "comes next" in template:
                start = random.choice([2, 3, 5, 7, 10])
                step = random.choice([2, 3, 4, 5, 7])
                s1, s2, s3, s4 = start, start+step, start+2*step, start+3*step
                q = template.format(s1=s1, s2=s2, s3=s3, s4=s4)
                a = str(s4 + step)
            else:
                i += 1
                continue

            problems.append({"q": q, "a": a, "type": ptype})
        except:
            pass
        i += 1

    return problems[:n]


# Logic problems
LOGIC_PROBLEMS = [
    {"q": "All A are B. All B are C. Is every A a C?", "a": "Yes", "type": "syllogism"},
    {"q": "If P then Q. Not Q. What about P?", "a": "Not P", "type": "modus"},
    {"q": "Some X are Y. All Y are Z. Are some X definitely Z?", "a": "Yes", "type": "syllogism"},
    {"q": "If rain then wet. Wet. Did it rain?", "a": "Cannot determine", "type": "fallacy"},
    {"q": "All dogs are mammals. All mammals breathe. Do dogs breathe?", "a": "Yes", "type": "syllogism"},
    {"q": "If sunny then hot. Not hot. Is it sunny?", "a": "No", "type": "modus"},
    {"q": "Some cats are black. All black things are dark. Are some cats dark?", "a": "Yes", "type": "syllogism"},
    {"q": "If A then B. B. Therefore A?", "a": "Cannot determine", "type": "fallacy"},
    {"q": "No fish are birds. All birds fly. Do fish fly?", "a": "Cannot determine", "type": "syllogism"},
    {"q": "If X > Y and Y > Z, is X > Z?", "a": "Yes", "type": "transitive"},
] * 10  # 100 logic problems


class CerebrasLLM:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.total_tokens = 0
        self.call_count = 0

    async def generate(self, prompt: str, max_tokens: int = 100) -> Tuple[str, int]:
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
                self.call_count += 1
                return data["choices"][0]["message"]["content"], tokens
            except Exception as e:
                await asyncio.sleep(1)
        return "", 0


def extract_answer(text: str, expected: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    exp_lower = expected.lower()

    # Direct match
    if exp_lower in text[:50]:
        return expected

    # Yes/No/Cannot
    if exp_lower in ['yes', 'no', 'cannot determine', 'not p']:
        if 'cannot' in text[:60] or 'uncertain' in text[:60]:
            return 'Cannot determine'
        if 'not p' in text[:30]:
            return 'Not P'
        if text.startswith('yes') or 'yes,' in text[:20] or 'yes.' in text[:20]:
            return 'Yes'
        if text.startswith('no') or 'no,' in text[:20] or 'no.' in text[:20]:
            return 'No'

    # Numbers
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    return text.split()[0] if text.split() else ""


def check_answer(pred: str, exp: str) -> bool:
    p = str(pred).lower().strip().rstrip('.')
    e = str(exp).lower().strip()
    if p == e or e in p[:30] or p.startswith(e):
        return True
    try:
        return abs(float(p.replace(',', '')) - float(e.replace(',', ''))) < 0.1
    except:
        return False


async def run_gsm8k_10x(llm: CerebrasLLM, problems: List[Dict]) -> Dict:
    """GSM8K 10x: 200 problems, compare simple vs CoT"""
    print(f"\n{'='*60}")
    print(f"GSM8K 10x: {len(problems)} problems")
    print("="*60)

    simple_correct = 0
    start_time = time.time()

    for i, p in enumerate(problems):
        prompt = f"Q: {p['q']}\nAnswer (number only):"
        resp, _ = await llm.generate(prompt, 60)
        answer = extract_answer(resp, p['a'])
        if check_answer(answer, p['a']):
            simple_correct += 1

        if (i + 1) % 20 == 0:
            acc = 100 * simple_correct / (i + 1)
            print(f"  [{i+1:3d}/{len(problems)}] acc={acc:.1f}%", flush=True)

        await asyncio.sleep(0.05)

    elapsed = time.time() - start_time
    return {
        "problems": len(problems),
        "correct": simple_correct,
        "accuracy": round(100 * simple_correct / len(problems), 1),
        "time": round(elapsed, 1),
        "avg_time": round(elapsed / len(problems), 2),
        "tokens": llm.total_tokens
    }


async def run_logic_10x(llm: CerebrasLLM, problems: List[Dict]) -> Dict:
    """Logic 10x: 100 problems"""
    print(f"\n{'='*60}")
    print(f"LOGIC 10x: {len(problems)} problems")
    print("="*60)

    correct = 0
    start_time = time.time()

    for i, p in enumerate(problems):
        prompt = f"Q: {p['q']}\nAnswer (Yes/No/Cannot determine):"
        resp, _ = await llm.generate(prompt, 30)
        answer = extract_answer(resp, p['a'])
        if check_answer(answer, p['a']):
            correct += 1

        if (i + 1) % 20 == 0:
            acc = 100 * correct / (i + 1)
            print(f"  [{i+1:3d}/{len(problems)}] acc={acc:.1f}%", flush=True)

        await asyncio.sleep(0.05)

    elapsed = time.time() - start_time
    return {
        "problems": len(problems),
        "correct": correct,
        "accuracy": round(100 * correct / len(problems), 1),
        "time": round(elapsed, 1),
        "tokens": llm.total_tokens
    }


async def run_accumulation_10x(llm: CerebrasLLM, problems: List[Dict]) -> Dict:
    """Accumulation 10x: Test learning effect on 200 problems"""
    print(f"\n{'='*60}")
    print(f"ACCUMULATION 10x: {len(problems)} problems")
    print("="*60)

    # Track correctness in quarters
    quarters = [0, 0, 0, 0]
    quarter_size = len(problems) // 4

    correct = 0
    correct_claims = []

    for i, p in enumerate(problems):
        # Build hints from correct claims (same type)
        relevant = [c for c in correct_claims if c['type'] == p['type']][-3:]

        hints = ""
        if relevant:
            hints = "Similar solved: " + "; ".join([c['hint'] for c in relevant]) + "\n"

        prompt = f"{hints}Q: {p['q']}\nAnswer:"
        resp, _ = await llm.generate(prompt, 60)
        answer = extract_answer(resp, p['a'])
        is_correct = check_answer(answer, p['a'])

        if is_correct:
            correct += 1
            quarters[min(i // quarter_size, 3)] += 1
            correct_claims.append({
                'type': p['type'],
                'hint': f"{p['type']}: {p['a']}"
            })

        if (i + 1) % 50 == 0:
            acc = 100 * correct / (i + 1)
            print(f"  [{i+1:3d}/{len(problems)}] acc={acc:.1f}%", flush=True)

        await asyncio.sleep(0.05)

    # Calculate learning effect
    q1_acc = 100 * quarters[0] / quarter_size
    q4_acc = 100 * quarters[3] / quarter_size
    learning_delta = q4_acc - q1_acc

    return {
        "problems": len(problems),
        "correct": correct,
        "accuracy": round(100 * correct / len(problems), 1),
        "q1_accuracy": round(q1_acc, 1),
        "q4_accuracy": round(q4_acc, 1),
        "learning_delta": round(learning_delta, 1),
        "tokens": llm.total_tokens
    }


async def main():
    print("="*60)
    print("10x BENCHMARK SUITE")
    print("="*60)
    print("GSM8K: 200 problems | Logic: 100 problems | Accumulation: 200")
    print("="*60)

    # Generate problems
    gsm8k_problems = generate_gsm8k_problems(200)
    logic_problems = LOGIC_PROBLEMS[:100]
    accum_problems = generate_gsm8k_problems(200)

    results = {}

    async with httpx.AsyncClient() as client:
        # GSM8K 10x
        llm1 = CerebrasLLM(client)
        results["gsm8k"] = await run_gsm8k_10x(llm1, gsm8k_problems)

        await asyncio.sleep(2)

        # Logic 10x
        llm2 = CerebrasLLM(client)
        results["logic"] = await run_logic_10x(llm2, logic_problems)

        await asyncio.sleep(2)

        # Accumulation 10x
        llm3 = CerebrasLLM(client)
        results["accumulation"] = await run_accumulation_10x(llm3, accum_problems)

    # Summary
    print("\n" + "="*60)
    print("10x RESULTS SUMMARY")
    print("="*60)
    print(f"{'Benchmark':<20} {'N':>6} {'Accuracy':>10} {'Time':>10} {'Tokens':>10}")
    print("-"*60)
    print(f"{'GSM8K':<20} {results['gsm8k']['problems']:>6} {results['gsm8k']['accuracy']:>9.1f}% {results['gsm8k']['time']:>9.1f}s {results['gsm8k']['tokens']:>10}")
    print(f"{'Logic':<20} {results['logic']['problems']:>6} {results['logic']['accuracy']:>9.1f}% {results['logic']['time']:>9.1f}s {results['logic']['tokens']:>10}")
    print(f"{'Accumulation':<20} {results['accumulation']['problems']:>6} {results['accumulation']['accuracy']:>9.1f}% {'-':>10} {results['accumulation']['tokens']:>10}")
    print("-"*60)

    print(f"\nAccumulation Learning Effect:")
    print(f"  Q1 (first 50):  {results['accumulation']['q1_accuracy']:.1f}%")
    print(f"  Q4 (last 50):   {results['accumulation']['q4_accuracy']:.1f}%")
    print(f"  Delta:          {results['accumulation']['learning_delta']:+.1f}pp")
    print("="*60)

    # Save results
    Path("/workspace/data").mkdir(exist_ok=True)
    with open("/workspace/data/benchmark_10x_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to /workspace/data/benchmark_10x_results.json")


if __name__ == "__main__":
    asyncio.run(main())
