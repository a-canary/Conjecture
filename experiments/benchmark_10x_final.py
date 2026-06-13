#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
10x Final Benchmark - Simple prompts + Fixed extraction
"""
import asyncio
import json
import os
import re
import time
import httpx
from pathlib import Path
import random

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise EnvironmentError("CEREBRAS_API_KEY environment variable is not set")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "llama3.1-8b"


def generate_problems(n=200):
    random.seed(42)
    problems = []
    templates = [
        ("A car travels {d} miles in {t} hours. What is its speed in mph?", lambda d, t: d // t),
        ("A factory produces {n} units per hour. How many units in {h} hours?", lambda n, h: n * h),
        ("What is {p}% of {n}?", lambda p, n: p * n // 100),
        ("A rectangle has length {l} and width {w}. What is its area?", lambda l, w: l * w),
        ("If {w} workers finish in {d} days, how many days for {w2} workers?", lambda w, d, w2: w * d // w2),
        ("{name} buys {n} items at ${p} each. Total cost?", lambda n, p: n * p),
    ]
    names = ["John", "Mary", "Tom", "Lisa", "Alex"]
    vals = {
        0: [(120, 2), (180, 3), (240, 4), (300, 5), (150, 3), (200, 4)],
        1: [(50, 8), (30, 6), (40, 5), (25, 4), (60, 10), (45, 9)],
        2: [(10, 200), (15, 80), (20, 150), (25, 120), (30, 100), (40, 50)],
        3: [(8, 5), (10, 6), (12, 4), (7, 9), (15, 3), (9, 8)],
        4: [(4, 12, 6), (5, 10, 10), (3, 15, 5), (6, 8, 4), (8, 6, 12)],
        5: [(5, 8), (3, 12), (7, 5), (4, 15), (6, 10), (8, 7)],
    }

    while len(problems) < n:
        idx = len(problems) % len(templates)
        t, c = templates[idx]
        v = random.choice(vals[idx])
        if idx == 0:
            q, a = t.format(d=v[0], t=v[1]), str(c(v[0], v[1]))
        elif idx == 1:
            q, a = t.format(n=v[0], h=v[1]), str(c(v[0], v[1]))
        elif idx == 2:
            q, a = t.format(p=v[0], n=v[1]), str(c(v[0], v[1]))
        elif idx == 3:
            q, a = t.format(l=v[0], w=v[1]), str(c(v[0], v[1]))
        elif idx == 4:
            q, a = t.format(w=v[0], d=v[1], w2=v[2]), str(c(v[0], v[1], v[2]))
        elif idx == 5:
            q, a = t.format(name=random.choice(names), n=v[0], p=v[1]), str(c(v[0], v[1]))
        problems.append({"q": q, "a": a})
    return problems[:n]


LOGIC = [
    {"q": "All A are B. All B are C. Is every A a C?", "a": "Yes"},
    {"q": "If P then Q. Not Q. What about P?", "a": "Not P"},
    {"q": "Some X are Y. All Y are Z. Are some X definitely Z?", "a": "Yes"},
    {"q": "If rain then wet. Wet. Did it rain?", "a": "Cannot determine"},
    {"q": "All dogs are mammals. All mammals breathe. Do dogs breathe?", "a": "Yes"},
    {"q": "If sunny then hot. Not hot. Is it sunny?", "a": "No"},
    {"q": "Some cats are black. All black things are dark. Are some cats dark?", "a": "Yes"},
    {"q": "If A then B. B. Therefore A?", "a": "Cannot determine"},
    {"q": "If X > Y and Y > Z, is X > Z?", "a": "Yes"},
    {"q": "All squares are rectangles. Are all rectangles squares?", "a": "No"},
] * 10


class LLM:
    def __init__(self, client):
        self.client = client
        self.tokens = 0

    async def gen(self, prompt, max_tok=100):
        for _ in range(3):
            try:
                r = await self.client.post(CEREBRAS_URL,
                    headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
                    json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tok, "temperature": 0.1}, timeout=60.0)
                if r.status_code == 429:
                    await asyncio.sleep(5)
                    continue
                if r.status_code == 200:
                    d = r.json()
                    self.tokens += d.get("usage", {}).get("total_tokens", 0)
                    return d["choices"][0]["message"]["content"]
            except:
                await asyncio.sleep(1)
        return ""


def extract(text, exp):
    if not text:
        return ""
    t = text.strip().lower()
    e = exp.lower()

    # Logic answers
    if e in ['yes', 'no', 'cannot determine', 'not p']:
        if 'cannot' in t[:60]:
            return 'Cannot determine'
        if 'not p' in t[:40]:
            return 'Not P'
        w = t.split()[0] if t.split() else ""
        if w.startswith('yes'):
            return 'Yes'
        if w.startswith('no'):
            return 'No'

    # Numbers - multiple patterns
    for p in [r'answer[:\s]+\$?(\d+\.?\d*)', r'=\s*\$?(\d+\.?\d*)', r'(\d+\.?\d*)\s*(?:mph|units|dollars|\$|%)?$']:
        m = re.search(p, t)
        if m:
            return m.group(1)

    # Last number
    nums = re.findall(r'\d+\.?\d*', t)
    return nums[-1] if nums else ""


def check(p, e):
    p, e = str(p).lower().strip(), str(e).lower().strip()
    if p == e:
        return True
    try:
        return abs(float(p) - float(e)) < 0.1
    except:
        return False


async def run_math(llm, probs):
    print(f"\n{'='*50}\nMATH: {len(probs)} problems\n{'='*50}")
    correct = 0
    for i, p in enumerate(probs):
        r = await llm.gen(f"Q: {p['q']}\nAnswer (number only):", 60)
        if check(extract(r, p['a']), p['a']):
            correct += 1
        if (i+1) % 40 == 0:
            print(f"  [{i+1:3d}] acc={100*correct/(i+1):.1f}%")
        await asyncio.sleep(0.03)
    return {"n": len(probs), "correct": correct, "acc": round(100*correct/len(probs), 1)}


async def run_logic(llm, probs):
    print(f"\n{'='*50}\nLOGIC: {len(probs)} problems\n{'='*50}")
    correct = 0
    for i, p in enumerate(probs):
        r = await llm.gen(f"Q: {p['q']}\nAnswer (Yes/No/Cannot determine):", 30)
        if check(extract(r, p['a']), p['a']):
            correct += 1
        if (i+1) % 20 == 0:
            print(f"  [{i+1:3d}] acc={100*correct/(i+1):.1f}%")
        await asyncio.sleep(0.03)
    return {"n": len(probs), "correct": correct, "acc": round(100*correct/len(probs), 1)}


async def run_accum(llm, probs):
    print(f"\n{'='*50}\nACCUM: {len(probs)} problems\n{'='*50}")
    Q = [0, 0, 0, 0]
    qs = len(probs) // 4
    correct = 0
    hints = []

    for i, p in enumerate(probs):
        h = "; ".join(hints[-3:]) + "\n" if hints else ""
        r = await llm.gen(f"{h}Q: {p['q']}\nAnswer:", 60)
        ok = check(extract(r, p['a']), p['a'])
        if ok:
            correct += 1
            Q[min(i // qs, 3)] += 1
            hints.append(f"{p['a']}")
        if (i+1) % 50 == 0:
            print(f"  [{i+1:3d}] acc={100*correct/(i+1):.1f}%")
        await asyncio.sleep(0.03)

    q1, q4 = 100*Q[0]/qs, 100*Q[3]/qs
    return {"n": len(probs), "correct": correct, "acc": round(100*correct/len(probs), 1),
            "q1": round(q1, 1), "q4": round(q4, 1), "delta": round(q4-q1, 1)}


async def main():
    print("="*50)
    print("10x FINAL BENCHMARK")
    print("="*50)

    math_p = generate_problems(200)
    logic_p = LOGIC[:100]
    accum_p = generate_problems(200)

    async with httpx.AsyncClient() as c:
        math = await run_math(LLM(c), math_p)
        logic = await run_logic(LLM(c), logic_p)
        accum = await run_accum(LLM(c), accum_p)

    print("\n" + "="*50)
    print("FINAL RESULTS (10x)")
    print("="*50)
    print(f"Math:   {math['acc']:5.1f}% ({math['correct']}/{math['n']})")
    print(f"Logic:  {logic['acc']:5.1f}% ({logic['correct']}/{logic['n']})")
    print(f"Accum:  {accum['acc']:5.1f}% ({accum['correct']}/{accum['n']})")
    print(f"\nLearning: Q1={accum['q1']:.1f}% → Q4={accum['q4']:.1f}% (Δ={accum['delta']:+.1f}pp)")
    print("="*50)

    Path("/workspace/data").mkdir(exist_ok=True)
    with open("/workspace/data/benchmark_10x_final.json", "w") as f:
        json.dump({"math": math, "logic": logic, "accum": accum}, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
