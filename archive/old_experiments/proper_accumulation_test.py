#!/usr/bin/env python3
"""
Proper Accumulation Test - Train/Test Split

NO CONTAMINATION:
- Bootstrap claims from TRAINING set
- Evaluate on TEST set (all questions first-time-seen)
- Claims accumulate during test (Q1's claim helps Q2+)
- But no question ever sees claims derived from ITSELF

Uses GSM8K train/test split via HuggingFace datasets.
"""
import asyncio
import json
import time
import os
import re
import httpx
import sqlite3
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "llama3.1-8b"

# Paths
DB_PATH = Path("/workspace/data/claims.db")
RESULTS_PATH = Path("/workspace/data/proper_accumulation_results.json")


@dataclass
class Claim:
    id: str
    content: str
    domain: str
    confidence: float
    verified: bool = False
    source: str = ""  # "train" or "test"


class ClaimDB:
    """SQLite-backed claim storage"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                content TEXT,
                domain TEXT,
                confidence REAL,
                verified INTEGER,
                source TEXT,
                created_at REAL
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON claims(domain)")
        self.conn.commit()

    def save(self, claim: Claim):
        self.conn.execute("""
            INSERT OR REPLACE INTO claims (id, content, domain, confidence, verified, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (claim.id, claim.content, claim.domain, claim.confidence,
              int(claim.verified), claim.source, time.time()))
        self.conn.commit()

    def get_by_domain(self, domain: str, limit: int = 5) -> List[Claim]:
        cur = self.conn.execute("""
            SELECT id, content, domain, confidence, verified, source
            FROM claims WHERE domain = ?
            ORDER BY verified DESC, confidence DESC
            LIMIT ?
        """, (domain, limit))
        return [Claim(*row) for row in cur.fetchall()]

    def count(self) -> Dict[str, int]:
        cur = self.conn.execute("SELECT source, COUNT(*) FROM claims GROUP BY source")
        return dict(cur.fetchall())

    def clear(self):
        self.conn.execute("DELETE FROM claims")
        self.conn.commit()


class CerebrasLLM:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.total_tokens = 0

    async def generate(self, prompt: str, max_tokens: int = 200) -> Tuple[str, int]:
        for attempt in range(3):
            try:
                resp = await self.client.post(
                    CEREBRAS_URL,
                    headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.1
                    },
                    timeout=60.0
                )
                if resp.status_code == 429:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                if resp.status_code != 200:
                    return "", 0
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("total_tokens", 0)
                self.total_tokens += tokens
                return content, tokens
            except Exception as e:
                await asyncio.sleep(1)
        return "", 0


def extract_answer(response: str) -> str:
    """Extract numeric answer from GSM8K response"""
    if not response:
        return ""
    # Look for #### pattern (GSM8K format)
    match = re.search(r'####\s*([^\n]+)', response)
    if match:
        return match.group(1).strip().replace(',', '')
    # Look for final number
    numbers = re.findall(r'-?[\d,]+\.?\d*', response)
    if numbers:
        return numbers[-1].replace(',', '')
    return ""


def check_answer(predicted: str, expected: str) -> bool:
    pred = predicted.strip().replace(',', '')
    exp = expected.strip().replace(',', '')
    try:
        return abs(float(pred) - float(exp)) < 0.01
    except:
        return pred == exp


def load_gsm8k_split(split: str, limit: int = None) -> List[Dict]:
    """Load GSM8K from HuggingFace datasets"""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split=split, trust_remote_code=True)
        questions = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break
            # Extract answer from GSM8K format (#### answer)
            answer_match = re.search(r'####\s*([^\n]+)', item['answer'])
            answer = answer_match.group(1).strip().replace(',', '') if answer_match else ""
            questions.append({
                "q": item['question'],
                "a": answer,
                "full_answer": item['answer'],
                "domain": "math"
            })
        return questions
    except ImportError:
        print("datasets not installed, using fallback questions")
        return get_fallback_questions(split, limit)


def get_fallback_questions(split: str, limit: int = None) -> List[Dict]:
    """Fallback if datasets not available - use disjoint question sets"""
    # Training questions (for bootstrap)
    train_qs = [
        {"q": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 to bake. She sells the rest at $2 each. How much does she make daily?", "a": "18", "domain": "math"},
        {"q": "A robe takes 2 bolts of blue fiber and half that of white fiber. How many bolts total for 3 robes?", "a": "9", "domain": "math"},
        {"q": "Josh decides to try flipping a house. He buys for $80,000, puts $50,000 into repairs, and sells for 150% of total cost. How much profit?", "a": "70000", "domain": "math"},
        {"q": "James writes 10 pages per hour. He writes 5 hours a day for 2 days. How many pages total?", "a": "100", "domain": "math"},
        {"q": "Every day Wendi feeds each of her chickens 3 cups of feed. She has 6 chickens. How many cups per week?", "a": "126", "domain": "math"},
        {"q": "Kylar went to store with $100. He bought 5 apples at $2 each and 4 oranges at $3 each. How much left?", "a": "78", "domain": "math"},
        {"q": "Toulouse has twice as many sheep as Charleston. Charleston has 4x as many sheep as Seattle. Seattle has 20 sheep. How many total?", "a": "260", "domain": "math"},
        {"q": "Carla is downloading a 200 GB file at 2 GB/minute. Dan is downloading a 30 GB file at 2 GB/minute. How many minutes difference?", "a": "85", "domain": "math"},
        {"q": "John drives 1000 miles a month. He needs to get an oil change every 3000 miles. He gets 1 free. How much for oil changes in a year if each costs $50?", "a": "150", "domain": "math"},
        {"q": "Eliza buys 5 books at $10 each with a 20% discount. How much does she pay?", "a": "40", "domain": "math"},
        {"q": "A farmer has 52 cows. He sells 15 and buys 3 more. How many cows does he have?", "a": "40", "domain": "math"},
        {"q": "A store sells pencils in packs of 12. If Maria needs 100 pencils, how many packs must she buy?", "a": "9", "domain": "math"},
        {"q": "Tom reads 40 pages per day. His book has 280 pages. How many days to finish?", "a": "7", "domain": "math"},
        {"q": "A pizza has 8 slices. 3 friends share 2 pizzas equally. How many slices each?", "a": "5", "domain": "math"},
        {"q": "Sarah earns $15 per hour. She works 8 hours Mon-Fri. What's her weekly pay?", "a": "600", "domain": "math"},
        {"q": "A bus holds 45 passengers. If 12 buses are needed for a trip, minimum how many people?", "a": "496", "domain": "math"},
        {"q": "A recipe needs 3 cups flour for 24 cookies. How much flour for 40 cookies?", "a": "5", "domain": "math"},
        {"q": "Mark has 3x as many marbles as Mike. Mike has 15 marbles. They combine marbles and split equally. How many each?", "a": "30", "domain": "math"},
        {"q": "A train travels 60 mph for 2.5 hours, then 80 mph for 1.5 hours. Total distance?", "a": "270", "domain": "math"},
        {"q": "Lisa saves $25 weekly. After 8 weeks she spends $50. How much does she have?", "a": "150", "domain": "math"},
    ]

    # Test questions (for evaluation) - COMPLETELY DIFFERENT from training
    test_qs = [
        {"q": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents gave her $15 and grandparents gave her twice as much. How much more does Betty need?", "a": "5", "domain": "math"},
        {"q": "Julie is reading a 120-page book. She read 12 pages yesterday and twice that today. How many pages does she have left?", "a": "84", "domain": "math"},
        {"q": "James trains for track. He runs 60 miles per week. He runs 3 days a week with equal distance each day. He runs 5 miles on other days. How many miles on training days?", "a": "10", "domain": "math"},
        {"q": "Albert is 2x Mary's age. Mary is 10. In 5 years, how old will Albert be?", "a": "25", "domain": "math"},
        {"q": "Ken created a care package for his grandma. He placed a box of chocolates ($6.25), a tin of cookies ($4.25), and a bag of oranges ($4.45). He packed them in a $2 box. What's the total cost?", "a": "16.95", "domain": "math"},
        {"q": "Alexis is putting a 10-foot carpet in her room. She has 24 square feet of extra carpet. How long is her room?", "a": "34", "domain": "math"},
        {"q": "Tim is stuck in traffic for twice as long as he was driving. He drove 5 hours. How long was the total trip?", "a": "15", "domain": "math"},
        {"q": "There are 15 trees in the grove. 21% of trees are oak. How many are not oak (rounded)?", "a": "12", "domain": "math"},
        {"q": "A merchant wants to make a choice of 2 different items. He has 6 items. How many choices?", "a": "15", "domain": "math"},
        {"q": "Five friends ate at a restaurant. The bill was $50. They tipped 20%. What did each pay?", "a": "12", "domain": "math"},
        {"q": "A store marks up prices by 30%. An item costs $70 wholesale. What's retail?", "a": "91", "domain": "math"},
        {"q": "Sam runs 8 km/h. He runs for 45 minutes. How many km did he run?", "a": "6", "domain": "math"},
        {"q": "A tank is 2/5 full. 30 more liters fills it. What's the capacity?", "a": "50", "domain": "math"},
        {"q": "Jenny has 3x as many stickers as Mark. Mark has 28. They share equally. How many each?", "a": "56", "domain": "math"},
        {"q": "A theater has 250 seats in 10 rows. How many seats per row?", "a": "25", "domain": "math"},
    ]

    qs = train_qs if split == "train" else test_qs
    if limit:
        qs = qs[:limit]
    return qs


async def solve_with_claims(llm: CerebrasLLM, question: str, claims: List[Claim]) -> Tuple[str, str]:
    """Solve question using retrieved claims as context"""
    context = ""
    if claims:
        context = "Relevant knowledge:\n"
        for c in claims[:3]:
            status = "[verified]" if c.verified else ""
            context += f"- {c.content[:100]} {status}\n"
        context += "\n"

    prompt = f"""{context}Question: {question}

Solve step by step, then give final answer after ####."""

    response, _ = await llm.generate(prompt, 300)
    answer = extract_answer(response)
    return answer, response


async def create_claim_from_solution(llm: CerebrasLLM, question: str, solution: str,
                                      is_correct: bool, domain: str, claim_id: str,
                                      source: str) -> Claim:
    """Create a claim from the solution - can be specific or pattern"""
    # Extract key insight from solution
    prompt = f"""From this math solution, extract the key method or fact as a single sentence.

Question: {question[:100]}
Solution: {solution[:200]}

Key insight (one sentence):"""

    insight, _ = await llm.generate(prompt, 60)
    content = insight.strip().split('\n')[0][:150] if insight else f"Math problem solving approach"

    return Claim(
        id=claim_id,
        content=content,
        domain=domain,
        confidence=0.9 if is_correct else 0.4,
        verified=is_correct,
        source=source
    )


async def bootstrap_phase(llm: CerebrasLLM, db: ClaimDB, train_questions: List[Dict]) -> Dict:
    """Bootstrap claims from training set"""
    print("\n" + "="*70)
    print("BOOTSTRAP PHASE (training set)")
    print(f"Questions: {len(train_questions)}")
    print("="*70)

    correct = 0
    for i, q in enumerate(train_questions):
        answer, solution = await solve_with_claims(llm, q['q'], [])  # No claims during bootstrap
        is_correct = check_answer(answer, q['a'])
        if is_correct:
            correct += 1

        # Create and save claim
        claim = await create_claim_from_solution(
            llm, q['q'], solution, is_correct, q['domain'],
            f"train_{i}", "train"
        )
        db.save(claim)

        acc = 100 * correct / (i + 1)
        status = "ok" if is_correct else "x"
        print(f"  [{i+1:2d}/{len(train_questions)}] {status} | acc={acc:5.1f}% | claims={db.count()}", flush=True)
        await asyncio.sleep(0.3)

    return {
        "phase": "bootstrap",
        "questions": len(train_questions),
        "correct": correct,
        "accuracy": round(100 * correct / len(train_questions), 1),
        "claims_created": db.count()
    }


async def evaluation_phase(llm: CerebrasLLM, db: ClaimDB, test_questions: List[Dict]) -> Dict:
    """Evaluate on test set - all questions first-time-seen"""
    print("\n" + "="*70)
    print("EVALUATION PHASE (test set - all first-time-seen)")
    print(f"Questions: {len(test_questions)}")
    print(f"Starting claims: {db.count()}")
    print("="*70)

    correct = 0
    results = []

    for i, q in enumerate(test_questions):
        # Retrieve relevant claims (from bootstrap + earlier test questions)
        claims = db.get_by_domain(q['domain'], limit=5)

        # Solve with claims
        answer, solution = await solve_with_claims(llm, q['q'], claims)
        is_correct = check_answer(answer, q['a'])
        if is_correct:
            correct += 1
        results.append(is_correct)

        # Create and save NEW claim (accumulates during test)
        claim = await create_claim_from_solution(
            llm, q['q'], solution, is_correct, q['domain'],
            f"test_{i}", "test"
        )
        db.save(claim)

        acc = 100 * correct / (i + 1)
        status = "ok" if is_correct else "x"
        n_claims = len(claims)
        print(f"  [{i+1:2d}/{len(test_questions)}] {status} | acc={acc:5.1f}% | used={n_claims} claims | total={sum(db.count().values())}", flush=True)
        await asyncio.sleep(0.3)

    # Learning analysis
    mid = len(results) // 2
    first_half = sum(results[:mid]) / mid if mid > 0 else 0
    second_half = sum(results[mid:]) / (len(results) - mid) if len(results) > mid else 0

    return {
        "phase": "evaluation",
        "questions": len(test_questions),
        "correct": correct,
        "accuracy": round(100 * correct / len(test_questions), 1),
        "first_half_acc": round(100 * first_half, 1),
        "second_half_acc": round(100 * second_half, 1),
        "learning_delta": round(100 * (second_half - first_half), 1),
        "final_claims": db.count(),
        "results": results
    }


async def run_bare_baseline(llm: CerebrasLLM, test_questions: List[Dict]) -> Dict:
    """Bare LLM baseline - no claims"""
    print("\n" + "="*70)
    print("BARE BASELINE (no claims)")
    print("="*70)

    correct = 0
    results = []

    for i, q in enumerate(test_questions):
        prompt = f"Question: {q['q']}\n\nSolve step by step, then give final answer after ####."
        response, _ = await llm.generate(prompt, 300)
        answer = extract_answer(response)
        is_correct = check_answer(answer, q['a'])
        if is_correct:
            correct += 1
        results.append(is_correct)

        acc = 100 * correct / (i + 1)
        status = "ok" if is_correct else "x"
        print(f"  [{i+1:2d}/{len(test_questions)}] {status} | acc={acc:5.1f}%", flush=True)
        await asyncio.sleep(0.3)

    return {
        "method": "bare",
        "accuracy": round(100 * correct / len(test_questions), 1),
        "results": results
    }


async def main():
    print("="*70)
    print("PROPER ACCUMULATION TEST")
    print("="*70)
    print("NO CONTAMINATION:")
    print("  - Bootstrap claims from TRAINING set")
    print("  - Evaluate on TEST set (first-time-seen)")
    print("  - Claims accumulate during test")
    print("  - No question sees claims from ITSELF")
    print("="*70)

    # Load train/test split
    print("\nLoading GSM8K train/test split...")
    train_qs = load_gsm8k_split("train", limit=20)  # Bootstrap from 20 train questions
    test_qs = load_gsm8k_split("test", limit=15)    # Evaluate on 15 test questions

    print(f"Train: {len(train_qs)} questions")
    print(f"Test: {len(test_qs)} questions (all first-time-seen)")

    # Fresh DB for this run
    db = ClaimDB(DB_PATH)
    db.clear()

    async with httpx.AsyncClient() as client:
        llm = CerebrasLLM(client)

        # 1. Bare baseline (same test questions, no claims)
        bare_result = await run_bare_baseline(llm, test_qs)

        await asyncio.sleep(2)

        # 2. Bootstrap claims from training set
        bootstrap_result = await bootstrap_phase(llm, db, train_qs)

        await asyncio.sleep(2)

        # 3. Evaluate on test set with accumulated claims
        eval_result = await evaluation_phase(llm, db, test_qs)

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Method':<30} {'Accuracy':>10} {'1st Half':>10} {'2nd Half':>10} {'Delta':>8}")
    print("-"*70)
    print(f"{'Bare (no claims)':<30} {bare_result['accuracy']:>9.1f}%")
    print(f"{'Accumulated (train+test)':<30} {eval_result['accuracy']:>9.1f}% {eval_result['first_half_acc']:>9.1f}% {eval_result['second_half_acc']:>9.1f}% {eval_result['learning_delta']:>+7.1f}pp")
    print("-"*70)

    improvement = eval_result['accuracy'] - bare_result['accuracy']
    print(f"\nImprovement over bare: {improvement:+.1f}pp")
    print(f"Learning effect: {eval_result['learning_delta']:+.1f}pp (2nd half vs 1st half)")
    print(f"Final claims: {eval_result['final_claims']}")

    gate_passed = eval_result['accuracy'] > bare_result['accuracy']
    print(f"\nGATE: Accumulated > Bare: {'PASSED' if gate_passed else 'FAILED'}")
    print("="*70)

    # Save results
    results = {
        "bare": bare_result,
        "bootstrap": bootstrap_result,
        "evaluation": eval_result,
        "improvement_pp": round(improvement, 1),
        "gate_passed": gate_passed,
        "contamination_free": True,
        "methodology": "train/test split - all test questions first-time-seen",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
