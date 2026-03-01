#!/usr/bin/env python3
"""
Smart Accumulation Test - Phase 3 Implementation

Improvements over naive accumulation:
1. Domain-tagged claims (math, logic, science, etc.)
2. Confidence gating (exclude <50% confidence)
3. Correctness tracking (prioritize verified claims)
4. Relevance filtering (same-domain + high-value cross-domain)

Gate: Accumulated accuracy must reach or exceed Fresh (72%)
"""
import asyncio
import json
import time
import os
import re
import httpx
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "llama3.1-8b"

# Same 50 questions as original test
QUESTIONS = [
    # Math - 15 questions
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
    # Logic - 15 questions
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
    # Science - 10 questions
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
    # Mixed - 10 questions
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
    """Enhanced claim with domain, confidence, and correctness tracking"""
    id: str
    content: str
    domain: str
    confidence: float
    question_idx: int
    is_correct: bool = False
    reasoning_type: str = ""  # "analysis", "solution", "pattern"

    def quality_score(self) -> float:
        """Compute quality score for ranking"""
        # Correct claims get big boost
        correctness_bonus = 0.5 if self.is_correct else 0.0
        return self.confidence + correctness_bonus


@dataclass
class SmartClaimMemory:
    """
    Smart claim storage with domain separation and quality filtering

    Key improvements:
    1. Domain-specific pools
    2. Confidence threshold
    3. Correctness weighting
    4. Limited context window
    """
    claims: List[Claim] = field(default_factory=list)
    domain_pools: Dict[str, List[Claim]] = field(default_factory=lambda: defaultdict(list))

    # Configuration
    confidence_threshold: float = 0.5  # Gate: exclude low-confidence claims
    max_claims_per_query: int = 5      # Limit context size
    cross_domain_limit: int = 1        # Max claims from other domains

    def add(self, claim: Claim):
        """Add claim if it meets quality threshold"""
        # Gate 1: Confidence threshold
        if claim.confidence < self.confidence_threshold:
            return  # Reject low-confidence claims

        self.claims.append(claim)
        self.domain_pools[claim.domain].append(claim)

    def get_relevant(self, domain: str) -> List[Claim]:
        """
        Get relevant claims with smart filtering:
        1. Same-domain claims (sorted by quality)
        2. High-quality cross-domain claims (correct + high confidence)
        """
        relevant = []

        # Same-domain claims (prioritize correct ones)
        domain_claims = sorted(
            self.domain_pools[domain],
            key=lambda c: c.quality_score(),
            reverse=True
        )
        relevant.extend(domain_claims[:self.max_claims_per_query - self.cross_domain_limit])

        # Cross-domain: only correct, high-confidence claims
        cross_domain = []
        for other_domain, pool in self.domain_pools.items():
            if other_domain != domain:
                for claim in pool:
                    if claim.is_correct and claim.confidence >= 0.7:
                        cross_domain.append(claim)

        # Sort cross-domain by quality and take top N
        cross_domain.sort(key=lambda c: c.quality_score(), reverse=True)
        relevant.extend(cross_domain[:self.cross_domain_limit])

        return relevant[:self.max_claims_per_query]

    def stats(self) -> Dict:
        """Get memory statistics"""
        total = len(self.claims)
        correct = sum(1 for c in self.claims if c.is_correct)
        by_domain = {d: len(pool) for d, pool in self.domain_pools.items()}
        avg_conf = sum(c.confidence for c in self.claims) / total if total > 0 else 0

        return {
            "total": total,
            "correct": correct,
            "accuracy": round(100 * correct / total, 1) if total > 0 else 0,
            "avg_confidence": round(avg_conf, 2),
            "by_domain": by_domain
        }


class CerebrasLLM:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.total_tokens = 0
        self.total_calls = 0

    async def generate(self, prompt: str, max_tokens: int = 200) -> tuple[str, int, float]:
        start = time.time()
        self.total_calls += 1

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
                elapsed = time.time() - start

                if resp.status_code == 429:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue

                if resp.status_code != 200:
                    return "", 0, elapsed

                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                tokens = data.get("usage", {}).get("total_tokens", 0)
                self.total_tokens += tokens
                return content, tokens, elapsed

            except Exception:
                await asyncio.sleep(1)
        return "", 0, time.time() - start


def extract_answer(response: str, expected: str) -> str:
    if not response:
        return ""
    response = response.strip()

    # Direct match
    if expected.lower() in response.lower()[:50]:
        return expected

    # Number extraction
    numbers = re.findall(r'-?\d+\.?\d*', response)
    if numbers:
        for num in numbers:
            try:
                if abs(float(num) - float(expected)) < 0.1:
                    return num
            except:
                pass
        return numbers[0]

    # Yes/No
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
        pred_num = float(pred.replace(',', ''))
        exp_num = float(exp.replace(',', ''))
        return abs(pred_num - exp_num) < 0.1
    except:
        pass

    return False


async def run_smart_accumulated(llm: CerebrasLLM, questions: List[Dict]) -> Dict:
    """Run with SMART claim accumulation"""
    print("\n" + "="*70)
    print("SMART ACCUMULATED CONJECTURE")
    print("Features: domain pools, confidence gating, correctness weighting")
    print("="*70)

    memory = SmartClaimMemory()
    correct = 0
    results = []

    for i, q in enumerate(questions):
        domain = q['domain']

        # Get RELEVANT prior claims (smart filtering)
        prior_claims = memory.get_relevant(domain)

        # Build context from relevant claims only
        prior_context = ""
        if prior_claims:
            prior_context = f"Relevant prior knowledge ({domain}):\n"
            for c in prior_claims:
                status = "✓" if c.is_correct else "?"
                prior_context += f"- [{status}] {c.content[:60]}...\n"
            prior_context += "\n"

        # Step 1: Domain-aware analysis
        prompt1 = f"""{prior_context}Question ({domain}): {q['q']}

Analyze: What key facts or calculation steps are needed? Be specific to {domain}."""

        resp1, _, _ = await llm.generate(prompt1, 120)

        # Step 2: Solve with context
        prompt2 = f"""Analysis: {resp1[:150] if resp1 else 'solve step by step'}

Question: {q['q']}

Answer with just the value (number, Yes/No, or short answer)."""

        resp2, _, _ = await llm.generate(prompt2, 50)

        answer = extract_answer(resp2, q['a'])
        is_correct = check_answer(answer, q['a'])

        if is_correct:
            correct += 1
        results.append(is_correct)

        # Create claims with proper metadata
        # Analysis claim
        analysis_claim = Claim(
            id=f"c{i}_analysis",
            content=f"[{domain}] {resp1[:100]}" if resp1 else f"[{domain}] analysis",
            domain=domain,
            confidence=0.7 if resp1 else 0.3,
            question_idx=i,
            is_correct=is_correct,  # Tag with correctness
            reasoning_type="analysis"
        )
        memory.add(analysis_claim)

        # Solution claim (higher confidence if correct)
        solution_claim = Claim(
            id=f"c{i}_solution",
            content=f"[{domain}] Q: {q['q'][:40]}... → {answer}",
            domain=domain,
            confidence=0.9 if is_correct else 0.3,  # High confidence only if correct
            question_idx=i,
            is_correct=is_correct,
            reasoning_type="solution"
        )
        memory.add(solution_claim)

        # Progress
        running_acc = 100 * correct / (i + 1)
        n_relevant = len(prior_claims)
        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:2d}/50] {status} {domain:7s} | acc={running_acc:5.1f}% | relevant={n_relevant} | pool={len(memory.claims)}", flush=True)

        await asyncio.sleep(0.2)

    return {
        "method": "smart_accumulated",
        "correct": correct,
        "accuracy": round(100 * correct / len(questions), 1),
        "memory_stats": memory.stats(),
        "results": results
    }


async def run_fresh(llm: CerebrasLLM, questions: List[Dict]) -> Dict:
    """Fresh Conjecture baseline (for comparison)"""
    print("\n" + "="*70)
    print("FRESH CONJECTURE (baseline)")
    print("="*70)

    correct = 0
    results = []

    for i, q in enumerate(questions):
        # Step 1: Analyze
        prompt1 = f"Question: {q['q']}\n\nWhat key facts or steps are needed? Be brief."
        resp1, _, _ = await llm.generate(prompt1, 100)

        # Step 2: Solve
        prompt2 = f"Analysis: {resp1[:150] if resp1 else 'solve carefully'}\n\nQuestion: {q['q']}\n\nAnswer with just the value."
        resp2, _, _ = await llm.generate(prompt2, 50)

        answer = extract_answer(resp2, q['a'])
        is_correct = check_answer(answer, q['a'])

        if is_correct:
            correct += 1
        results.append(is_correct)

        running_acc = 100 * correct / (i + 1)
        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:2d}/50] {status} {q['domain']:7s} | acc={running_acc:5.1f}%", flush=True)

        await asyncio.sleep(0.2)

    return {
        "method": "fresh",
        "correct": correct,
        "accuracy": round(100 * correct / len(questions), 1),
        "results": results
    }


def analyze_learning(results: List[bool], window: int = 10) -> Dict:
    """Analyze learning progression"""
    first_half = results[:25]
    second_half = results[25:]

    return {
        "first_25": round(100 * sum(first_half) / 25, 1),
        "last_25": round(100 * sum(second_half) / 25, 1),
        "delta": round(100 * (sum(second_half) - sum(first_half)) / 25, 1)
    }


async def main():
    print("="*70)
    print("SMART ACCUMULATION TEST - Phase 3")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Questions: {len(QUESTIONS)}")
    print(f"\nImprovements over naive accumulation:")
    print("  1. Domain-specific claim pools")
    print("  2. Confidence gating (>50%)")
    print("  3. Correctness weighting")
    print("  4. Relevance filtering (max 5 claims)")
    print()
    print(f"GATE: Smart Accumulated must reach Fresh baseline (72%)")
    print("="*70)

    async with httpx.AsyncClient() as client:
        # Fresh baseline
        llm1 = CerebrasLLM(client)
        fresh_result = await run_fresh(llm1, QUESTIONS)

        await asyncio.sleep(2)

        # Smart accumulated
        llm2 = CerebrasLLM(client)
        smart_result = await run_smart_accumulated(llm2, QUESTIONS)

    # Analysis
    fresh_learning = analyze_learning(fresh_result['results'])
    smart_learning = analyze_learning(smart_result['results'])

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Method':<25} {'Accuracy':>10} {'First 25':>10} {'Last 25':>10} {'Δ':>8}")
    print("-"*70)
    print(f"{'Fresh Conjecture':<25} {fresh_result['accuracy']:>9.1f}% {fresh_learning['first_25']:>9.1f}% {fresh_learning['last_25']:>9.1f}% {fresh_learning['delta']:>+7.1f}pp")
    print(f"{'Smart Accumulated':<25} {smart_result['accuracy']:>9.1f}% {smart_learning['first_25']:>9.1f}% {smart_learning['last_25']:>9.1f}% {smart_learning['delta']:>+7.1f}pp")
    print("-"*70)

    # Gate check
    gate_passed = smart_result['accuracy'] >= fresh_result['accuracy']
    improvement = smart_result['accuracy'] - fresh_result['accuracy']

    print(f"\nGATE CHECK: Smart Accumulated ≥ Fresh")
    print(f"  Fresh:  {fresh_result['accuracy']}%")
    print(f"  Smart:  {smart_result['accuracy']}%")
    print(f"  Delta:  {improvement:+.1f}pp")
    print(f"  Result: {'✅ PASSED' if gate_passed else '❌ FAILED'}")

    # Memory stats
    print(f"\nMemory Stats: {smart_result['memory_stats']}")
    print("="*70)

    # Save results
    results = {
        "fresh": fresh_result,
        "smart_accumulated": smart_result,
        "learning": {
            "fresh": fresh_learning,
            "smart": smart_learning
        },
        "gate_passed": gate_passed,
        "improvement_pp": round(improvement, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open("/workspace/data/smart_accumulation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to /workspace/data/smart_accumulation_results.json")


if __name__ == "__main__":
    asyncio.run(main())
