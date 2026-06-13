#!/usr/bin/env python3
"""
Phase 4: Claim Quality Improvement

Implements:
1. Self-verification: Check answer before submitting
2. Claim chaining: Multi-step with explicit dependencies
3. Error detection measurement: Track errors caught by verification

Target: GSM8K 50% → 60%+
"""
import asyncio
import json
import os
import re
import httpx
from dataclasses import dataclass
from typing import List, Tuple

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise EnvironmentError("CEREBRAS_API_KEY environment variable is not set")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "llama3.1-8b"

# GSM8K-style math problems (multi-step required)
GSM8K_PROBLEMS = [
    {"q": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. She sells the rest at $2 each. How much does she make daily?", "a": "18"},
    {"q": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total for 3 robes?", "a": "9"},
    {"q": "Josh decides to try flipping a house. He buys a house for $80,000 and spends $50,000 on repairs. This increased the value to $150,000. How much profit did he make?", "a": "20000"},
    {"q": "James writes 10 pages per hour. If he writes 5 hours a day for 2 days, how many pages does he write?", "a": "100"},
    {"q": "Every day, Wendi feeds each of her chickens 3 cups of feed. She has 20 chickens. How many cups of feed does she need for 5 days?", "a": "300"},
    {"q": "Kylar went to the store. He bought 20 pens at $1.50 each and 10 notebooks at $3 each. How much did he spend?", "a": "60"},
    {"q": "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. Seattle has 20 sheep. How many sheep do they have total?", "a": "260"},
    {"q": "A merchant wants to make a mixture of nuts worth $6/kg. He mixes 10kg at $4/kg with some amount at $8/kg. How many kg at $8/kg?", "a": "10"},
    {"q": "Carla is downloading a 200GB file. The download speed is 2GB/minute. She already downloaded 80GB. How many more minutes to finish?", "a": "60"},
    {"q": "John takes a 10 minute shower every other day for 4 weeks. How many minutes does he spend showering?", "a": "140"},
    {"q": "A farmer has 52 chickens. Each chicken produces 2 eggs per week. He sells eggs at $3/dozen. How much money per week?", "a": "26"},
    {"q": "Tim rides his bike to work 5 days a week. It takes 30 minutes each way. How many hours does he spend biking per week?", "a": "5"},
    {"q": "Maria has $50. She buys 3 books at $8 each and 2 pens at $2 each. How much money does she have left?", "a": "22"},
    {"q": "A train travels at 60 mph. It needs to cover 240 miles. If it leaves at 2pm, what time does it arrive?", "a": "6"},
    {"q": "Tom has 3 times as many marbles as Jerry. Jerry has 5 more marbles than Kim. Kim has 10 marbles. How many marbles does Tom have?", "a": "45"},
    {"q": "A rectangle's length is 3 times its width. If the perimeter is 48cm, what is the area?", "a": "108"},
    {"q": "Lisa saves $20/week. After 8 weeks, she spends half on a gift. How much does she have left?", "a": "80"},
    {"q": "A baker makes 60 cookies. He puts them in boxes of 12. Each box sells for $5. How much does he make if he sells all boxes?", "a": "25"},
    {"q": "Mike runs 3 miles in 24 minutes. At this pace, how long to run 5 miles?", "a": "40"},
    {"q": "A store sells shirts at $15 each. They offer a 20% discount on orders over $50. How much for 5 shirts?", "a": "60"},
]


@dataclass
class ClaimChain:
    """Multi-step claim with explicit dependencies"""
    steps: List[str]
    final_answer: str
    verified: bool = False
    verification_changed: bool = False


class CerebrasLLM:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.total_tokens = 0

    async def generate(self, prompt: str, max_tokens: int = 300) -> str:
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
            except Exception as e:
                await asyncio.sleep(1)
        return ""


def extract_number(text: str) -> str:
    """Extract final numeric answer"""
    if not text:
        return ""
    # Look for boxed answer
    boxed = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed.group(1).replace(",", "").replace("$", "")
    # Look for "answer is X" pattern
    answer_match = re.search(r'(?:answer|total|result|makes?|profit|spend|cost|have|left|arrive)[^\d]*(\d+(?:\.\d+)?)', text.lower())
    if answer_match:
        return answer_match.group(1)
    # Last number in response
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else ""


def check_answer(predicted: str, expected: str) -> bool:
    try:
        pred = float(predicted.replace(",", "").replace("$", ""))
        exp = float(expected.replace(",", "").replace("$", ""))
        return abs(pred - exp) < 0.01
    except:
        return predicted.strip() == expected.strip()


async def solve_baseline(llm: CerebrasLLM, question: str) -> Tuple[str, int]:
    """Baseline: Single-shot solution"""
    prompt = f"""Solve this math problem step by step. Show your work, then give the final numeric answer.

Problem: {question}

Work through it carefully:"""

    response = await llm.generate(prompt, 400)
    answer = extract_number(response)
    return answer, 1


async def solve_with_verification(llm: CerebrasLLM, question: str) -> Tuple[str, int, bool]:
    """Phase 4: Multi-step with self-verification"""

    # Step 1: Identify what we need to find
    prompt1 = f"""Problem: {question}

What is the question asking for? What information is given? List the key facts."""

    step1 = await llm.generate(prompt1, 150)

    # Step 2: Set up the calculation
    prompt2 = f"""Problem: {question}

Given information:
{step1[:300]}

Set up the mathematical steps needed to solve this. What calculations are required?"""

    step2 = await llm.generate(prompt2, 200)

    # Step 3: Execute calculation
    prompt3 = f"""Problem: {question}

Setup:
{step2[:300]}

Now perform the calculations and find the final answer. Show each step."""

    step3 = await llm.generate(prompt3, 250)
    initial_answer = extract_number(step3)

    # Step 4: Self-verification
    prompt4 = f"""Problem: {question}

My solution:
{step3[:400]}

My answer: {initial_answer}

CHECK YOUR WORK:
1. Did I use all the given information correctly?
2. Are my calculations accurate?
3. Does this answer make sense for the problem?

If there's an error, provide the CORRECT answer. Otherwise confirm the answer.

Final verified answer:"""

    verification = await llm.generate(prompt4, 200)
    verified_answer = extract_number(verification)

    # Check if verification changed the answer
    answer_changed = initial_answer != verified_answer and verified_answer
    final_answer = verified_answer if verified_answer else initial_answer

    return final_answer, 4, answer_changed


async def run_benchmark():
    print("="*70)
    print("PHASE 4: CLAIM QUALITY IMPROVEMENT")
    print("="*70)
    print("Testing on 20 GSM8K-style problems")
    print("Comparing: Baseline (1-shot) vs Verified (4-step chain)")
    print("="*70)

    async with httpx.AsyncClient() as client:
        llm = CerebrasLLM(client)

        baseline_correct = 0
        verified_correct = 0
        errors_caught = 0  # Verification changed wrong → right
        errors_introduced = 0  # Verification changed right → wrong
        verification_changes = 0

        print("\n" + "-"*70)
        print(f"{'#':>2} {'Expected':>8} {'Base':>8} {'Verif':>8} {'B':>3} {'V':>3} {'Changed':>8}")
        print("-"*70)

        for i, prob in enumerate(GSM8K_PROBLEMS):
            # Run baseline
            base_ans, _ = await solve_baseline(llm, prob["q"])
            base_correct = check_answer(base_ans, prob["a"])

            await asyncio.sleep(0.3)

            # Run verified
            verif_ans, _, changed = await solve_with_verification(llm, prob["q"])
            verif_correct = check_answer(verif_ans, prob["a"])

            if base_correct:
                baseline_correct += 1
            if verif_correct:
                verified_correct += 1
            if changed:
                verification_changes += 1
                if not base_correct and verif_correct:
                    errors_caught += 1
                elif base_correct and not verif_correct:
                    errors_introduced += 1

            b_mark = "✓" if base_correct else "✗"
            v_mark = "✓" if verif_correct else "✗"
            chg = "YES" if changed else ""

            print(f"{i+1:2d} {prob['a']:>8} {base_ans[:8]:>8} {verif_ans[:8]:>8} {b_mark:>3} {v_mark:>3} {chg:>8}")

            await asyncio.sleep(0.3)

        # Results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)

        base_acc = 100 * baseline_correct / len(GSM8K_PROBLEMS)
        verif_acc = 100 * verified_correct / len(GSM8K_PROBLEMS)
        delta = verif_acc - base_acc

        print(f"Baseline (1-shot):     {baseline_correct}/{len(GSM8K_PROBLEMS)} = {base_acc:.1f}%")
        print(f"Verified (4-step):     {verified_correct}/{len(GSM8K_PROBLEMS)} = {verif_acc:.1f}%")
        print(f"Improvement:           {delta:+.1f}pp")
        print()
        print(f"Verification changes:  {verification_changes}")
        print(f"Errors caught:         {errors_caught} (wrong→right)")
        print(f"Errors introduced:     {errors_introduced} (right→wrong)")

        if verification_changes > 0:
            catch_rate = 100 * errors_caught / verification_changes
            print(f"Net benefit rate:      {catch_rate:.1f}%")

        print()
        print("GATES:")
        gate1 = errors_caught >= 0.2 * len(GSM8K_PROBLEMS)
        gate2 = verif_acc >= 60
        print(f"  Self-verification catches 20%+ errors: {'✅ PASS' if gate1 else '❌ FAIL'} ({errors_caught}/{len(GSM8K_PROBLEMS)})")
        print(f"  GSM8K accuracy ≥ 60%: {'✅ PASS' if gate2 else '❌ FAIL'} ({verif_acc:.1f}%)")
        print("="*70)

        # Save results
        results = {
            "baseline_correct": baseline_correct,
            "verified_correct": verified_correct,
            "baseline_accuracy": base_acc,
            "verified_accuracy": verif_acc,
            "improvement": delta,
            "verification_changes": verification_changes,
            "errors_caught": errors_caught,
            "errors_introduced": errors_introduced,
            "gates": {"error_catch_20pct": gate1, "gsm8k_60pct": gate2}
        }

        with open("/workspace/data/phase4_results.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_benchmark())
