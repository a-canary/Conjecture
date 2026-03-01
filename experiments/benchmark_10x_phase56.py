#!/usr/bin/env python3
"""
10x Benchmark: Phase 5 (Cross-Session) and Phase 6 (Optimization)
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
DB_PATH = "/workspace/data/phase5_10x_claims.db"


def generate_problems(n: int, seed: int = 42) -> List[Dict]:
    """Generate n diverse math problems"""
    random.seed(seed)
    problems = []
    types = ["rate", "percent", "geometry", "arithmetic", "inverse"]

    for i in range(n):
        ptype = types[i % len(types)]

        if ptype == "rate":
            d = random.choice([100, 120, 150, 180, 200, 240, 300])
            t = random.choice([2, 3, 4, 5])
            if d % t == 0:
                problems.append({
                    "q": f"A car travels {d} miles in {t} hours. What is its speed in mph?",
                    "a": str(d // t),
                    "type": ptype
                })
        elif ptype == "percent":
            p = random.choice([10, 15, 20, 25, 30, 40, 50])
            n_val = random.choice([80, 100, 120, 150, 200])
            problems.append({
                "q": f"What is {p}% of {n_val}?",
                "a": str(p * n_val // 100),
                "type": ptype
            })
        elif ptype == "geometry":
            l = random.choice([5, 6, 7, 8, 9, 10, 12])
            w = random.choice([3, 4, 5, 6, 7, 8])
            problems.append({
                "q": f"A rectangle has length {l} and width {w}. What is its area?",
                "a": str(l * w),
                "type": ptype
            })
        elif ptype == "arithmetic":
            a = random.choice([12, 15, 18, 20, 25, 30])
            b = random.choice([5, 7, 8, 10, 12, 15])
            problems.append({
                "q": f"What is {a} + {b}?",
                "a": str(a + b),
                "type": ptype
            })
        elif ptype == "inverse":
            w1 = random.choice([2, 3, 4, 5, 6])
            d1 = random.choice([6, 8, 10, 12, 15])
            w2 = w1 * 2
            problems.append({
                "q": f"If {w1} workers finish a job in {d1} days, how many days for {w2} workers?",
                "a": str(d1 // 2),
                "type": ptype
            })

    return problems[:n]


@dataclass
class Claim:
    id: str
    content: str
    problem_type: str
    confidence: float
    is_correct: bool


class ClaimDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                content TEXT,
                problem_type TEXT,
                confidence REAL,
                is_correct INTEGER
            )
        """)
        self.conn.commit()

    def clear(self):
        self.conn.execute("DELETE FROM claims")
        self.conn.commit()

    def save(self, claim: Claim):
        self.conn.execute(
            "INSERT OR REPLACE INTO claims VALUES (?, ?, ?, ?, ?)",
            (claim.id, claim.content, claim.problem_type, claim.confidence, int(claim.is_correct))
        )
        self.conn.commit()

    def get_relevant(self, problem_type: str, limit: int = 2) -> List[Claim]:
        cursor = self.conn.execute(
            "SELECT * FROM claims WHERE problem_type = ? AND is_correct = 1 LIMIT ?",
            (problem_type, limit)
        )
        return [Claim(r[0], r[1], r[2], r[3], bool(r[4])) for r in cursor.fetchall()]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]


class CerebrasLLM:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.total_tokens = 0

    async def generate(self, prompt: str, max_tokens: int = 80) -> str:
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


def extract_answer(text: str) -> str:
    if not text:
        return ""
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return numbers[-1] if numbers else ""


def check_answer(pred: str, exp: str) -> bool:
    try:
        return abs(float(pred) - float(exp)) < 0.1
    except:
        return False


async def run_phase5_10x(client: httpx.AsyncClient) -> Dict:
    """Phase 5 at 10x: 100 training + 100 test problems"""
    print("\n" + "="*60)
    print("PHASE 5 (10x): Cross-Session Learning")
    print("="*60)
    print("Session 1: 100 problems (training)")
    print("Session 2: 100 problems (test)")

    db = ClaimDB(DB_PATH)
    db.clear()

    train_problems = generate_problems(100, seed=42)
    test_problems = generate_problems(100, seed=123)

    # Session 1: Training
    print("\n--- Session 1: Training ---")
    llm1 = CerebrasLLM(client)
    s1_correct = 0

    for i, p in enumerate(train_problems):
        prompt = f"Q: {p['q']}\nAnswer:"
        resp = await llm1.generate(prompt, 50)
        answer = extract_answer(resp)
        is_correct = check_answer(answer, p['a'])

        if is_correct:
            s1_correct += 1

        db.save(Claim(
            id=f"s1_{i}",
            content=f"{p['type']}: {p['q'][:30]}... = {answer}",
            problem_type=p['type'],
            confidence=0.9 if is_correct else 0.3,
            is_correct=is_correct
        ))

        if (i + 1) % 25 == 0:
            print(f"  [{i+1:3d}/100] acc={100*s1_correct/(i+1):.1f}%", flush=True)

        await asyncio.sleep(0.05)

    # Session 2: Without claims
    print("\n--- Session 2: Without Claims ---")
    llm2 = CerebrasLLM(client)
    s2_base_correct = 0

    for i, p in enumerate(test_problems):
        prompt = f"Q: {p['q']}\nAnswer:"
        resp = await llm2.generate(prompt, 50)
        answer = extract_answer(resp)
        if check_answer(answer, p['a']):
            s2_base_correct += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i+1:3d}/100] acc={100*s2_base_correct/(i+1):.1f}%", flush=True)

        await asyncio.sleep(0.05)

    # Session 2: With claims
    print("\n--- Session 2: With Claims ---")
    llm3 = CerebrasLLM(client)
    s2_claims_correct = 0
    claims_used = 0

    for i, p in enumerate(test_problems):
        relevant = db.get_relevant(p['type'], limit=2)
        claims_used += len(relevant)

        hints = ""
        if relevant:
            hints = "Similar: " + "; ".join([c.content[:30] for c in relevant]) + "\n"

        prompt = f"{hints}Q: {p['q']}\nAnswer:"
        resp = await llm3.generate(prompt, 50)
        answer = extract_answer(resp)
        if check_answer(answer, p['a']):
            s2_claims_correct += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i+1:3d}/100] acc={100*s2_claims_correct/(i+1):.1f}%", flush=True)

        await asyncio.sleep(0.05)

    return {
        "s1_accuracy": round(100 * s1_correct / 100, 1),
        "s2_base_accuracy": round(100 * s2_base_correct / 100, 1),
        "s2_claims_accuracy": round(100 * s2_claims_correct / 100, 1),
        "improvement": round(100 * (s2_claims_correct - s2_base_correct) / 100, 1),
        "claims_saved": db.count(),
        "claims_used": claims_used
    }


async def run_phase6_10x(client: httpx.AsyncClient) -> Dict:
    """Phase 6 at 10x: 100 problems, compare approaches"""
    print("\n" + "="*60)
    print("PHASE 6 (10x): Production Optimization")
    print("="*60)

    problems = generate_problems(100, seed=456)

    # Multi-step (baseline)
    print("\n--- Multi-step (2 calls per problem) ---")
    llm1 = CerebrasLLM(client)
    multi_correct = 0
    multi_start = time.time()

    for i, p in enumerate(problems):
        resp1 = await llm1.generate(f"Problem: {p['q']}\nKey steps:", 80)
        resp2 = await llm1.generate(f"Steps: {resp1[:80]}\nAnswer:", 40)
        answer = extract_answer(resp2)
        if check_answer(answer, p['a']):
            multi_correct += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i+1:3d}/100] acc={100*multi_correct/(i+1):.1f}%", flush=True)

        await asyncio.sleep(0.05)

    multi_time = time.time() - multi_start
    multi_tokens = llm1.total_tokens

    # Single-step (optimized)
    print("\n--- Single-step (1 call per problem) ---")
    llm2 = CerebrasLLM(client)
    single_correct = 0
    single_start = time.time()

    for i, p in enumerate(problems):
        resp = await llm2.generate(f"Q: {p['q']}\nAnswer:", 40)
        answer = extract_answer(resp)
        if check_answer(answer, p['a']):
            single_correct += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i+1:3d}/100] acc={100*single_correct/(i+1):.1f}%", flush=True)

        await asyncio.sleep(0.05)

    single_time = time.time() - single_start
    single_tokens = llm2.total_tokens

    return {
        "multi_step": {
            "accuracy": round(100 * multi_correct / 100, 1),
            "time": round(multi_time, 1),
            "tokens": multi_tokens,
            "calls": 200
        },
        "single_step": {
            "accuracy": round(100 * single_correct / 100, 1),
            "time": round(single_time, 1),
            "tokens": single_tokens,
            "calls": 100
        },
        "time_reduction": round(100 * (multi_time - single_time) / multi_time, 1),
        "token_reduction": round(100 * (multi_tokens - single_tokens) / multi_tokens, 1)
    }


async def main():
    print("="*60)
    print("10x BENCHMARK: PHASE 5 & 6")
    print("="*60)

    results = {}

    async with httpx.AsyncClient() as client:
        results["phase5"] = await run_phase5_10x(client)
        await asyncio.sleep(2)
        results["phase6"] = await run_phase6_10x(client)

    # Summary
    print("\n" + "="*60)
    print("10x RESULTS: PHASE 5 & 6")
    print("="*60)

    p5 = results["phase5"]
    print(f"\nPHASE 5 (Cross-Session, 100+100 problems):")
    print(f"  Session 1 (training):  {p5['s1_accuracy']:.1f}%")
    print(f"  Session 2 (no claims): {p5['s2_base_accuracy']:.1f}%")
    print(f"  Session 2 (claims):    {p5['s2_claims_accuracy']:.1f}%")
    print(f"  Improvement:           {p5['improvement']:+.1f}pp")

    p6 = results["phase6"]
    print(f"\nPHASE 6 (Optimization, 100 problems):")
    print(f"  Multi-step: {p6['multi_step']['accuracy']:.1f}% | {p6['multi_step']['time']:.1f}s | {p6['multi_step']['tokens']} tokens")
    print(f"  Single-step: {p6['single_step']['accuracy']:.1f}% | {p6['single_step']['time']:.1f}s | {p6['single_step']['tokens']} tokens")
    print(f"  Time reduction: {p6['time_reduction']:.1f}%")
    print(f"  Token reduction: {p6['token_reduction']:.1f}%")
    print("="*60)

    # Save
    with open("/workspace/data/benchmark_10x_phase56.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
