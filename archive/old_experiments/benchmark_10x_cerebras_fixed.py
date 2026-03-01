#!/usr/bin/env python3
"""
10x Benchmark Suite - FIXED EXTRACTION (Cerebras)

Uses proper answer extraction patterns from lm-evaluation-harness:
- GSM8K: #### (number) pattern
- Flexible fallback extraction
- Proper number normalization
"""
import asyncio
import json
import os
import re
import time
import httpx
from pathlib import Path
from typing import List, Dict, Tuple
import random

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "llama3.1-8b"


def generate_gsm8k_problems(n: int = 200) -> List[Dict]:
    """Generate n math word problems with known answers"""
    random.seed(42)  # Reproducible
    problems = []
    templates = [
        ("A car travels {d} miles in {t} hours. What is its speed in mph?", lambda d, t: d // t, "rate"),
        ("A factory produces {n} units per hour. How many units in {h} hours?", lambda n, h: n * h, "rate"),
        ("What is {p}% of {n}?", lambda p, n: p * n // 100, "percent"),
        ("{name} has ${m}. They spend {p}% on food. How much is left?", lambda m, p: m - (m * p // 100), "percent"),
        ("{n1} items cost ${c}. How much do {n2} items cost?", lambda n1, c, n2: round(c * n2 / n1, 2), "proportion"),
        ("A rectangle has length {l} and width {w}. What is its area?", lambda l, w: l * w, "geometry"),
        ("If {w1} workers finish a job in {d1} days, how many days for {w2} workers?", lambda w1, d1, w2: w1 * d1 // w2, "inverse"),
        ("{name} buys {n} items at ${p} each. Total cost?", lambda n, p: n * p, "arithmetic"),
    ]
    names = ["John", "Mary", "Tom", "Lisa", "Alex", "Sarah", "Mike", "Emma"]

    i = 0
    while len(problems) < n:
        template_idx = i % len(templates)
        template, calc, ptype = templates[template_idx]
        try:
            if "speed" in template:
                d, t = random.choice([(120, 2), (180, 3), (240, 4), (300, 5), (150, 3)])
                q = template.format(d=d, t=t)
                a = str(calc(d, t))
            elif "factory" in template:
                n_val, h = random.choice([(50, 8), (30, 6), (40, 5), (25, 4), (60, 10)])
                q = template.format(n=n_val, h=h)
                a = str(calc(n_val, h))
            elif "% of" in template:
                p, n_val = random.choice([(10, 200), (15, 80), (20, 150), (25, 120), (30, 100)])
                q = template.format(p=p, n=n_val)
                a = str(calc(p, n_val))
            elif "spend" in template:
                m, p = random.choice([(100, 20), (80, 25), (50, 40), (200, 15), (150, 30)])
                name = random.choice(names)
                q = template.format(name=name, m=m, p=p)
                a = str(calc(m, p))
            elif "items cost" in template:
                n1, c, n2 = random.choice([(3, 6, 5), (4, 8, 7), (5, 10, 8), (2, 4, 6)])
                q = template.format(n1=n1, c=c, n2=n2)
                a = str(calc(n1, c, n2))
            elif "area" in template:
                l, w = random.choice([(8, 5), (10, 6), (12, 4), (7, 9), (15, 3)])
                q = template.format(l=l, w=w)
                a = str(calc(l, w))
            elif "workers" in template:
                w1, d1, w2 = random.choice([(4, 12, 6), (5, 10, 10), (3, 15, 5), (6, 8, 4)])
                q = template.format(w1=w1, d1=d1, w2=w2)
                a = str(calc(w1, d1, w2))
            elif "Total cost" in template:
                n_val, p = random.choice([(5, 8), (3, 12), (7, 5), (4, 15)])
                name = random.choice(names)
                q = template.format(name=name, n=n_val, p=p)
                a = str(calc(n_val, p))
            else:
                i += 1
                continue
            problems.append({"q": q, "a": a, "type": ptype})
        except:
            pass
        i += 1
    return problems[:n]


LOGIC_PROBLEMS = [
    {"q": "All A are B. All B are C. Is every A a C?", "a": "Yes"},
    {"q": "If P then Q. Not Q. What about P?", "a": "Not P"},
    {"q": "Some X are Y. All Y are Z. Are some X definitely Z?", "a": "Yes"},
    {"q": "If rain then wet. Wet. Did it rain?", "a": "Cannot determine"},
    {"q": "All dogs are mammals. All mammals breathe. Do dogs breathe?", "a": "Yes"},
    {"q": "If sunny then hot. Not hot. Is it sunny?", "a": "No"},
    {"q": "Some cats are black. All black things are dark. Are some cats dark?", "a": "Yes"},
    {"q": "If A then B. B. Therefore A?", "a": "Cannot determine"},
    {"q": "No fish are birds. All birds fly. Do fish fly?", "a": "Cannot determine"},
    {"q": "If X > Y and Y > Z, is X > Z?", "a": "Yes"},
] * 10


class CerebrasLLM:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.total_tokens = 0

    async def generate(self, prompt: str, max_tokens: int = 150) -> str:
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
                    return ""
                data = resp.json()
                self.total_tokens += data.get("usage", {}).get("total_tokens", 0)
                return data["choices"][0]["message"]["content"]
            except:
                await asyncio.sleep(1)
        return ""


def extract_answer(text: str, expected: str) -> str:
    """Fixed extraction using lm-evaluation-harness patterns"""
    if not text:
        return ""

    text = text.strip()
    exp_lower = expected.lower()

    # Pattern 1: GSM8K #### pattern (highest priority)
    gsm8k = re.search(r'####\s*(\-?[\d,\.]+)', text)
    if gsm8k:
        return gsm8k.group(1).replace(',', '')

    # Pattern 2: "answer is X" variants
    for pattern in [
        r'answer\s*(?:is|:)\s*\$?(\-?[\d,\.]+)',
        r'(?:=|equals)\s*\$?(\-?[\d,\.]+)\s*$',
        r'total[:\s]+\$?(\-?[\d,\.]+)',
    ]:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).replace(',', '')

    # Pattern 3: Logic answers
    if exp_lower in ['yes', 'no', 'cannot determine', 'not p']:
        text_lower = text.lower()
        if any(p in text_lower for p in ['cannot determine', 'cannot be determined', 'insufficient']):
            return 'Cannot determine'
        if 'not p' in text_lower[:40]:
            return 'Not P'
        first = text_lower.split()[0] if text_lower.split() else ""
        if first.startswith('yes'):
            return 'Yes'
        if first.startswith('no'):
            return 'No'

    # Pattern 4: Last number fallback
    nums = re.findall(r'\-?[\d,]+\.?\d*', text)
    if nums:
        return nums[-1].replace(',', '')

    return text.split()[0] if text.split() else ""


def check(pred: str, exp: str) -> bool:
    """Check answer with tolerance"""
    p = str(pred).lower().strip().rstrip('.')
    e = str(exp).lower().strip()
    if p == e or e in p[:30]:
        return True
    try:
        pn, en = float(p.replace(',', '')), float(e.replace(',', ''))
        return abs(pn - en) < 0.01 if abs(en) < 100 else abs(pn - en) / abs(en) < 0.001
    except:
        return False


async def run_gsm8k(llm, problems):
    print(f"\n{'='*60}\nGSM8K 10x FIXED: {len(problems)} problems\n{'='*60}")
    correct = 0
    start = time.time()

    for i, p in enumerate(problems):
        prompt = f"Solve step by step. End with #### followed by your answer.\n\nProblem: {p['q']}\n\nSolution:"
        resp = await llm.generate(prompt, 200)
        ans = extract_answer(resp, p['a'])
        if check(ans, p['a']):
            correct += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):.1f}%")
        await asyncio.sleep(0.05)

    return {"problems": len(problems), "correct": correct, "accuracy": round(100*correct/len(problems), 1),
            "time": round(time.time()-start, 1), "tokens": llm.total_tokens}


async def run_logic(llm, problems):
    print(f"\n{'='*60}\nLOGIC 10x FIXED: {len(problems)} problems\n{'='*60}")
    correct = 0
    start = time.time()

    for i, p in enumerate(problems):
        prompt = f"Answer ONLY: Yes, No, or Cannot determine.\n\nQuestion: {p['q']}\n\nAnswer:"
        resp = await llm.generate(prompt, 30)
        if check(extract_answer(resp, p['a']), p['a']):
            correct += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):.1f}%")
        await asyncio.sleep(0.05)

    return {"problems": len(problems), "correct": correct, "accuracy": round(100*correct/len(problems), 1),
            "time": round(time.time()-start, 1), "tokens": llm.total_tokens}


async def run_accumulation(llm, problems):
    print(f"\n{'='*60}\nACCUMULATION 10x FIXED: {len(problems)} problems\n{'='*60}")
    quarters = [0, 0, 0, 0]
    qsize = len(problems) // 4
    correct = 0
    claims = []

    for i, p in enumerate(problems):
        rel = [c for c in claims if c['type'] == p['type']][-3:]
        hints = ("Similar: " + "; ".join([c['h'] for c in rel]) + "\n") if rel else ""

        prompt = f"{hints}Solve step by step. End with #### and answer.\n\nProblem: {p['q']}\n\nSolution:"
        resp = await llm.generate(prompt, 200)
        ans = extract_answer(resp, p['a'])
        ok = check(ans, p['a'])

        if ok:
            correct += 1
            quarters[min(i // qsize, 3)] += 1
            claims.append({'type': p['type'], 'h': f"{p['type']}:{p['a']}"})

        if (i + 1) % 50 == 0:
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):.1f}%")
        await asyncio.sleep(0.05)

    q1, q4 = 100*quarters[0]/qsize, 100*quarters[3]/qsize
    return {"problems": len(problems), "correct": correct, "accuracy": round(100*correct/len(problems), 1),
            "q1": round(q1, 1), "q4": round(q4, 1), "delta": round(q4-q1, 1), "tokens": llm.total_tokens}


async def main():
    print("="*60)
    print("10x BENCHMARK - FIXED EXTRACTION (Cerebras)")
    print("="*60)

    gsm = generate_gsm8k_problems(200)
    logic = LOGIC_PROBLEMS[:100]
    accum = generate_gsm8k_problems(200)
    results = {}

    async with httpx.AsyncClient() as c:
        results["gsm8k"] = await run_gsm8k(CerebrasLLM(c), gsm)
        await asyncio.sleep(2)
        results["logic"] = await run_logic(CerebrasLLM(c), logic)
        await asyncio.sleep(2)
        results["accumulation"] = await run_accumulation(CerebrasLLM(c), accum)

    print("\n" + "="*60)
    print("RESULTS - FIXED EXTRACTION")
    print("="*60)
    print(f"{'Benchmark':<15} {'N':>6} {'Acc':>8} {'Time':>8}")
    print("-"*40)
    print(f"{'GSM8K':<15} {results['gsm8k']['problems']:>6} {results['gsm8k']['accuracy']:>7.1f}% {results['gsm8k']['time']:>7.1f}s")
    print(f"{'Logic':<15} {results['logic']['problems']:>6} {results['logic']['accuracy']:>7.1f}% {results['logic']['time']:>7.1f}s")
    print(f"{'Accumulation':<15} {results['accumulation']['problems']:>6} {results['accumulation']['accuracy']:>7.1f}%")
    print("-"*40)
    print(f"\nLearning Effect: Q1={results['accumulation']['q1']:.1f}% → Q4={results['accumulation']['q4']:.1f}% (Δ={results['accumulation']['delta']:+.1f}pp)")
    print("="*60)

    Path("/workspace/data").mkdir(exist_ok=True)
    with open("/workspace/data/benchmark_10x_cerebras_fixed.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
