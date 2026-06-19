#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Run benchmark with a specific prompt variant.

Usage:
    python experiments/run_variant_benchmark.py --variant v01_baseline --limit 50
"""
import asyncio
import json
import os
import re
import sys
import time
import httpx
from pathlib import Path
import random
import argparse

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.prompt_variants import get_variant, VARIANTS

# API config
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise EnvironmentError("CEREBRAS_API_KEY environment variable is not set")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "llama3.1-8b"


def generate_math_problems(n=100):
    """Generate math problems"""
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


def generate_logic_problems(n=50):
    """Generate logic problems"""
    base = [
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
    ]
    return (base * (n // len(base) + 1))[:n]


class LLM:
    def __init__(self, client):
        self.client = client
        self.tokens = 0
        self.calls = 0

    async def gen(self, prompt, max_tok=100):
        self.calls += 1
        for attempt in range(3):
            try:
                r = await self.client.post(
                    CEREBRAS_URL,
                    headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tok,
                        "temperature": 0.1
                    },
                    timeout=60.0
                )
                if r.status_code == 429:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                if r.status_code == 200:
                    d = r.json()
                    self.tokens += d.get("usage", {}).get("total_tokens", 0)
                    return d["choices"][0]["message"]["content"]
            except Exception as e:
                await asyncio.sleep(2)
        return ""


def extract_answer(text, expected):
    """Extract answer from response"""
    if not text:
        return ""
    t = text.strip().lower()
    e = expected.lower()

    # Logic answers
    if e in ['yes', 'no', 'cannot determine', 'not p']:
        if 'cannot' in t[:80]:
            return 'Cannot determine'
        if 'not p' in t[:60]:
            return 'Not P'
        w = t.split()[0] if t.split() else ""
        if w.startswith('yes'):
            return 'Yes'
        if w.startswith('no'):
            return 'No'
        # Check for yes/no anywhere in first line
        first_line = t.split('\n')[0][:100]
        if 'yes' in first_line and 'no' not in first_line:
            return 'Yes'
        if 'no' in first_line and 'yes' not in first_line:
            return 'No'

    # Numbers - multiple patterns
    patterns = [
        r'answer[:\s]*\$?(\d+\.?\d*)',
        r'=\s*\$?(\d+\.?\d*)',
        r'####\s*\$?(\d+\.?\d*)',
        r'"answer":\s*(\d+\.?\d*)',
        r'(\d+\.?\d*)\s*(?:mph|units|dollars|\$|%)?$',
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            return m.group(1)

    # Last number
    nums = re.findall(r'\d+\.?\d*', t)
    return nums[-1] if nums else ""


def check_answer(pred, expected):
    """Check if prediction matches expected"""
    p, e = str(pred).lower().strip(), str(expected).lower().strip()
    if p == e:
        return True
    try:
        return abs(float(p) - float(e)) < 0.1
    except:
        return False


async def run_benchmark(variant_id: str, math_limit: int = 100, logic_limit: int = 50):
    """Run benchmark with specific variant"""
    variant = get_variant(variant_id)
    results = {
        "variant_id": variant_id,
        "variant_name": variant["name"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL,
        "math": None,
        "logic": None,
    }

    print(f"\n{'='*60}")
    print(f"VARIANT: {variant_id} - {variant['name']}")
    print(f"{'='*60}")

    math_problems = generate_math_problems(math_limit)
    logic_problems = generate_logic_problems(logic_limit)

    async with httpx.AsyncClient() as client:
        llm = LLM(client)

        # Math benchmark
        print(f"\nMATH: {len(math_problems)} problems")
        math_correct = 0
        math_start = time.time()

        for i, p in enumerate(math_problems):
            prompt = variant["math"].format(q=p["q"])
            response = await llm.gen(prompt, max_tok=100)
            pred = extract_answer(response, p["a"])
            if check_answer(pred, p["a"]):
                math_correct += 1
            if (i + 1) % 25 == 0:
                print(f"  [{i+1:3d}] acc={100*math_correct/(i+1):.1f}%")
            await asyncio.sleep(0.05)

        results["math"] = {
            "correct": math_correct,
            "total": len(math_problems),
            "accuracy": round(100 * math_correct / len(math_problems), 2),
            "time": round(time.time() - math_start, 1),
        }
        print(f"  MATH FINAL: {results['math']['accuracy']:.1f}%")

        # Logic benchmark
        print(f"\nLOGIC: {len(logic_problems)} problems")
        logic_correct = 0
        logic_start = time.time()

        for i, p in enumerate(logic_problems):
            prompt = variant["logic"].format(q=p["q"])
            response = await llm.gen(prompt, max_tok=60)
            pred = extract_answer(response, p["a"])
            if check_answer(pred, p["a"]):
                logic_correct += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1:3d}] acc={100*logic_correct/(i+1):.1f}%")
            await asyncio.sleep(0.05)

        results["logic"] = {
            "correct": logic_correct,
            "total": len(logic_problems),
            "accuracy": round(100 * logic_correct / len(logic_problems), 2),
            "time": round(time.time() - logic_start, 1),
        }
        print(f"  LOGIC FINAL: {results['logic']['accuracy']:.1f}%")

        # Combined score
        results["combined_accuracy"] = round(
            (results["math"]["accuracy"] + results["logic"]["accuracy"]) / 2, 2
        )
        results["total_tokens"] = llm.tokens
        results["total_calls"] = llm.calls

    print(f"\n{'='*60}")
    print(f"COMBINED: {results['combined_accuracy']:.1f}%")
    print(f"Tokens: {results['total_tokens']} | Calls: {results['total_calls']}")
    print(f"{'='*60}")

    return results


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, help="Variant ID (e.g., v01_baseline)")
    parser.add_argument("--math-limit", type=int, default=100, help="Math problems")
    parser.add_argument("--logic-limit", type=int, default=50, help="Logic problems")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    if args.variant not in VARIANTS:
        print(f"Unknown variant: {args.variant}")
        print(f"Available: {list(VARIANTS.keys())}")
        sys.exit(1)

    results = await run_benchmark(args.variant, args.math_limit, args.logic_limit)

    # Save results
    output_dir = Path("/workspace/data/variant_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = args.output or output_dir / f"{args.variant}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
