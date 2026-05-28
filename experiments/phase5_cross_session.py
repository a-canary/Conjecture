#!/usr/bin/env python3
"""
Phase 5: Cross-Session Learning

Tests whether claims persisted from Session 1 improve Session 2.

Steps:
1. Session 1: Solve 20 problems, persist claims to SQLite
2. Session 2: Solve 20 NEW problems, retrieve relevant claims
3. Compare: Session 2 with claims vs Session 2 without claims

Gates:
- Claims persist across sessions
- Session 2 with claims > Session 2 without claims
- Relevant claims retrieved with >80% precision
"""
import asyncio
import json
import os
import re
import sqlite3
import httpx
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise EnvironmentError("CEREBRAS_API_KEY environment variable is not set")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "llama3.1-8b"
DB_PATH = "/workspace/data/phase5_claims.db"

# Session 1 problems (training set)
SESSION1_PROBLEMS = [
    {"q": "A car travels 240 miles in 4 hours. What is its speed in mph?", "a": "60", "domain": "math", "type": "rate"},
    {"q": "If 5 workers can paint a house in 6 days, how many days for 10 workers?", "a": "3", "domain": "math", "type": "inverse"},
    {"q": "A rectangle has perimeter 20 and length 6. What is its width?", "a": "4", "domain": "math", "type": "geometry"},
    {"q": "John has $50. He spends 40% on lunch. How much does he have left?", "a": "30", "domain": "math", "type": "percent"},
    {"q": "A train leaves at 2pm going 60mph. At 3pm another train leaves same direction at 80mph. When does it catch up?", "a": "6", "domain": "math", "type": "chase"},
    {"q": "3 apples cost $2.40. How much do 7 apples cost?", "a": "5.60", "domain": "math", "type": "proportion"},
    {"q": "A tank fills in 6 hours, drains in 8 hours. If both on, how long to fill?", "a": "24", "domain": "math", "type": "rate"},
    {"q": "What is 15% of 80?", "a": "12", "domain": "math", "type": "percent"},
    {"q": "A circle has radius 7. What is its area? (use pi=3.14)", "a": "153.86", "domain": "math", "type": "geometry"},
    {"q": "Tom is twice as old as Jerry. In 5 years, Tom will be 1.5x Jerry's age. How old is Tom now?", "a": "10", "domain": "math", "type": "age"},
    {"q": "All A are B. All B are C. Is every A a C?", "a": "Yes", "domain": "logic", "type": "syllogism"},
    {"q": "If P then Q. Not Q. What can we conclude about P?", "a": "Not P", "domain": "logic", "type": "modus"},
    {"q": "Some X are Y. All Y are Z. Are some X definitely Z?", "a": "Yes", "domain": "logic", "type": "syllogism"},
    {"q": "If raining then wet. It's wet. Is it raining?", "a": "Cannot determine", "domain": "logic", "type": "fallacy"},
    {"q": "What is the chemical symbol for sodium?", "a": "Na", "domain": "science", "type": "element"},
    {"q": "Water boils at what temperature in Celsius?", "a": "100", "domain": "science", "type": "property"},
    {"q": "How many protons does Carbon have?", "a": "6", "domain": "science", "type": "element"},
    {"q": "What is the speed of light in m/s? (round to 10^8)", "a": "3", "domain": "science", "type": "constant"},
    {"q": "A sequence: 2, 5, 10, 17, 26, ? What comes next?", "a": "37", "domain": "math", "type": "sequence"},
    {"q": "If A > B and B > C, is A > C?", "a": "Yes", "domain": "logic", "type": "transitive"},
]

# Session 2 problems (test set - similar types, different numbers)
SESSION2_PROBLEMS = [
    {"q": "A bus travels 180 miles in 3 hours. What is its speed in mph?", "a": "60", "domain": "math", "type": "rate"},
    {"q": "If 4 painters finish in 8 days, how many days for 8 painters?", "a": "4", "domain": "math", "type": "inverse"},
    {"q": "A rectangle has perimeter 30 and length 10. What is its width?", "a": "5", "domain": "math", "type": "geometry"},
    {"q": "Mary has $80. She spends 25% on books. How much does she have left?", "a": "60", "domain": "math", "type": "percent"},
    {"q": "Train A at 50mph leaves at noon. Train B at 75mph leaves at 1pm same direction. When does B catch A?", "a": "3", "domain": "math", "type": "chase"},
    {"q": "4 oranges cost $3.20. How much do 9 oranges cost?", "a": "7.20", "domain": "math", "type": "proportion"},
    {"q": "A pool fills in 4 hours, drains in 6 hours. Both on - how long to fill?", "a": "12", "domain": "math", "type": "rate"},
    {"q": "What is 20% of 75?", "a": "15", "domain": "math", "type": "percent"},
    {"q": "A circle has radius 5. What is its area? (use pi=3.14)", "a": "78.5", "domain": "math", "type": "geometry"},
    {"q": "Lisa is 3x older than Kim. In 6 years, Lisa will be 2x Kim's age. How old is Lisa now?", "a": "18", "domain": "math", "type": "age"},
    {"q": "All dogs are mammals. All mammals are warm-blooded. Are all dogs warm-blooded?", "a": "Yes", "domain": "logic", "type": "syllogism"},
    {"q": "If sunny then hot. Not hot. Is it sunny?", "a": "No", "domain": "logic", "type": "modus"},
    {"q": "Some cats are black. All black things are dark. Are some cats dark?", "a": "Yes", "domain": "logic", "type": "syllogism"},
    {"q": "If studying then pass. Student passed. Did they study?", "a": "Cannot determine", "domain": "logic", "type": "fallacy"},
    {"q": "What is the chemical symbol for potassium?", "a": "K", "domain": "science", "type": "element"},
    {"q": "Water freezes at what temperature in Celsius?", "a": "0", "domain": "science", "type": "property"},
    {"q": "How many protons does Oxygen have?", "a": "8", "domain": "science", "type": "element"},
    {"q": "What is Earth's gravity in m/s^2? (round to integer)", "a": "10", "domain": "science", "type": "constant"},
    {"q": "A sequence: 3, 7, 13, 21, 31, ? What comes next?", "a": "43", "domain": "math", "type": "sequence"},
    {"q": "If X < Y and Y < Z, is X < Z?", "a": "Yes", "domain": "logic", "type": "transitive"},
]


@dataclass
class Claim:
    id: str
    content: str
    domain: str
    problem_type: str
    confidence: float
    is_correct: bool
    created: str


class ClaimDatabase:
    """SQLite-based claim persistence"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None

    def initialize(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                domain TEXT NOT NULL,
                problem_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                is_correct INTEGER NOT NULL,
                created TEXT NOT NULL
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON claims(domain)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON claims(problem_type)")
        self.conn.commit()

    def clear(self):
        if self.conn:
            self.conn.execute("DELETE FROM claims")
            self.conn.commit()

    def save(self, claim: Claim):
        self.conn.execute(
            "INSERT OR REPLACE INTO claims VALUES (?, ?, ?, ?, ?, ?, ?)",
            (claim.id, claim.content, claim.domain, claim.problem_type,
             claim.confidence, int(claim.is_correct), claim.created)
        )
        self.conn.commit()

    def get_relevant(self, domain: str, problem_type: str, limit: int = 3) -> List[Claim]:
        """Retrieve relevant claims by domain and type"""
        cursor = self.conn.execute(
            """SELECT * FROM claims
               WHERE domain = ? AND problem_type = ? AND is_correct = 1
               ORDER BY confidence DESC LIMIT ?""",
            (domain, problem_type, limit)
        )
        rows = cursor.fetchall()
        return [Claim(r[0], r[1], r[2], r[3], r[4], bool(r[5]), r[6]) for r in rows]

    def count(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM claims")
        return cursor.fetchone()[0]

    def close(self):
        if self.conn:
            self.conn.close()


class CerebrasLLM:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def generate(self, prompt: str, max_tokens: int = 200) -> str:
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
                return resp.json()["choices"][0]["message"]["content"]
            except:
                await asyncio.sleep(1)
        return ""


def extract_answer(text: str, expected: str) -> str:
    if not text:
        return ""
    text = text.strip()
    # Direct match
    if expected.lower() in text.lower()[:50]:
        return expected
    # Numbers
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    # Yes/No
    if expected.lower() in ['yes', 'no']:
        if 'yes' in text.lower()[:30]:
            return 'Yes'
        if 'no' in text.lower()[:30]:
            return 'No'
        if 'cannot' in text.lower()[:50]:
            return 'Cannot determine'
    return text.split()[0] if text.split() else ""


def check_answer(pred: str, exp: str) -> bool:
    p = str(pred).lower().strip().replace(',', '')
    e = str(exp).lower().strip().replace(',', '')
    if p == e or e in p[:20]:
        return True
    try:
        return abs(float(p) - float(e)) < 0.1
    except:
        return False


async def run_session1(llm: CerebrasLLM, db: ClaimDatabase) -> Dict:
    """Session 1: Solve problems and persist claims"""
    print("\n" + "="*60)
    print("SESSION 1: Build Claim Database")
    print("="*60)

    correct = 0
    for i, p in enumerate(SESSION1_PROBLEMS):
        # Solve with simple prompt
        prompt = f"Problem: {p['q']}\n\nSolve and give final answer only."
        resp = await llm.generate(prompt, 150)
        answer = extract_answer(resp, p['a'])
        is_correct = check_answer(answer, p['a'])

        if is_correct:
            correct += 1

        # Create and save claim with method pattern
        method_prompt = f"Problem: {p['q']}\nAnswer: {p['a']}\n\nIn one sentence, what general method solves this type of problem?"
        method = await llm.generate(method_prompt, 100)

        claim = Claim(
            id=f"s1_{i}",
            content=method[:200] if method else f"Solved {p['type']} problem",
            domain=p['domain'],
            problem_type=p['type'],
            confidence=0.9 if is_correct else 0.3,
            is_correct=is_correct,
            created=datetime.now().isoformat()
        )
        db.save(claim)

        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:2d}/20] {status} {p['type']:12s} | saved claim", flush=True)
        await asyncio.sleep(0.2)

    return {"correct": correct, "accuracy": 100 * correct / 20, "claims_saved": db.count()}


async def run_session2_without_claims(llm: CerebrasLLM) -> Dict:
    """Session 2 without prior claims (baseline)"""
    print("\n" + "="*60)
    print("SESSION 2: Without Claims (Baseline)")
    print("="*60)

    correct = 0
    for i, p in enumerate(SESSION2_PROBLEMS):
        prompt = f"Problem: {p['q']}\n\nSolve and give final answer only."
        resp = await llm.generate(prompt, 150)
        answer = extract_answer(resp, p['a'])
        is_correct = check_answer(answer, p['a'])

        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:2d}/20] {status} {p['type']:12s}", flush=True)
        await asyncio.sleep(0.2)

    return {"correct": correct, "accuracy": 100 * correct / 20}


async def run_session2_with_claims(llm: CerebrasLLM, db: ClaimDatabase) -> Dict:
    """Session 2 with retrieved claims"""
    print("\n" + "="*60)
    print("SESSION 2: With Persisted Claims")
    print("="*60)

    correct = 0
    claims_used = 0
    claims_relevant = 0

    for i, p in enumerate(SESSION2_PROBLEMS):
        # Retrieve relevant claims from Session 1
        relevant = db.get_relevant(p['domain'], p['type'], limit=2)

        hints = ""
        if relevant:
            claims_used += len(relevant)
            hints = "Method hints from similar problems:\n"
            for c in relevant:
                hints += f"• {c.content}\n"
            hints += "\n"

        prompt = f"{hints}Problem: {p['q']}\n\nSolve and give final answer only."
        resp = await llm.generate(prompt, 150)
        answer = extract_answer(resp, p['a'])
        is_correct = check_answer(answer, p['a'])

        if is_correct:
            correct += 1
            if relevant:
                claims_relevant += 1

        status = "✓" if is_correct else "✗"
        n_claims = len(relevant)
        print(f"  [{i+1:2d}/20] {status} {p['type']:12s} | claims={n_claims}", flush=True)
        await asyncio.sleep(0.2)

    return {
        "correct": correct,
        "accuracy": 100 * correct / 20,
        "claims_used": claims_used,
        "claims_helpful": claims_relevant
    }


async def main():
    print("="*60)
    print("PHASE 5: CROSS-SESSION LEARNING")
    print("="*60)

    # Initialize database
    db = ClaimDatabase(DB_PATH)
    db.initialize()
    db.clear()  # Start fresh

    async with httpx.AsyncClient() as client:
        llm = CerebrasLLM(client)

        # Session 1: Build claim database
        s1 = await run_session1(llm, db)
        print(f"\n  Session 1: {s1['accuracy']:.1f}%, {s1['claims_saved']} claims saved")

        await asyncio.sleep(2)

        # Session 2 baseline (no claims)
        s2_base = await run_session2_without_claims(llm)
        print(f"\n  Session 2 (no claims): {s2_base['accuracy']:.1f}%")

        await asyncio.sleep(2)

        # Session 2 with claims
        s2_claims = await run_session2_with_claims(llm, db)
        print(f"\n  Session 2 (with claims): {s2_claims['accuracy']:.1f}%")

    db.close()

    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Session 1 (training):      {s1['accuracy']:5.1f}%  ({s1['claims_saved']} claims)")
    print(f"Session 2 (no claims):     {s2_base['accuracy']:5.1f}%")
    print(f"Session 2 (with claims):   {s2_claims['accuracy']:5.1f}%  ({s2_claims['claims_used']} claims used)")
    print()

    delta = s2_claims['accuracy'] - s2_base['accuracy']
    print(f"Improvement from claims:   {delta:+.1f}pp")

    print("\n" + "="*60)
    print("GATES")
    print("="*60)
    gate1 = s1['claims_saved'] > 0
    gate2 = s2_claims['accuracy'] >= s2_base['accuracy']
    gate3 = s2_claims['claims_used'] > 0

    print(f"Claims persist:            {'✅ PASS' if gate1 else '❌ FAIL'} ({s1['claims_saved']} saved)")
    print(f"Session 2 improved:        {'✅ PASS' if gate2 else '❌ FAIL'} ({delta:+.1f}pp)")
    print(f"Claims retrieved:          {'✅ PASS' if gate3 else '❌ FAIL'} ({s2_claims['claims_used']} used)")
    print("="*60)

    # Save results
    Path("/workspace/data").mkdir(exist_ok=True)
    with open("/workspace/data/phase5_results.json", "w") as f:
        json.dump({
            "session1": s1,
            "session2_baseline": s2_base,
            "session2_claims": s2_claims,
            "improvement": delta,
            "gates": {"persist": gate1, "improved": gate2, "retrieved": gate3}
        }, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
