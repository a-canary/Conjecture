#!/usr/bin/env python3
"""
Phase 4: Chain-of-Thought Single Prompt

Previous approach (4 separate prompts) failed - context lost between calls.
New approach: Single prompt with structured reasoning + self-check.
"""
import asyncio
import json
import os
import re
import httpx

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "llama3.1-8b"

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


class CerebrasLLM:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def generate(self, prompt: str, max_tokens: int = 500) -> str:
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


def extract_final_answer(text: str) -> str:
    """Extract answer from FINAL ANSWER: X format"""
    if not text:
        return ""

    # Look for explicit final answer
    final_match = re.search(r'FINAL ANSWER[:\s]+\$?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if final_match:
        return final_match.group(1)

    # Look for boxed
    boxed = re.search(r'\\boxed\{(\d+(?:\.\d+)?)\}', text)
    if boxed:
        return boxed.group(1)

    # Look for "= X" at end of line
    equals = re.findall(r'=\s*\$?(\d+(?:\.\d+)?)\s*$', text, re.MULTILINE)
    if equals:
        return equals[-1]

    # Last number
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else ""


def check_answer(predicted: str, expected: str) -> bool:
    try:
        pred = float(predicted.replace(",", ""))
        exp = float(expected.replace(",", ""))
        return abs(pred - exp) < 0.01
    except:
        return False


async def solve_baseline(llm: CerebrasLLM, question: str) -> str:
    """Baseline: Direct question"""
    prompt = f"Problem: {question}\n\nSolve this and give the final numeric answer."
    return extract_final_answer(await llm.generate(prompt, 300))


async def solve_cot(llm: CerebrasLLM, question: str) -> str:
    """Chain-of-Thought: Step-by-step in single prompt"""
    prompt = f"""Problem: {question}

Solve step-by-step:
1. List the key information given
2. Identify what we need to find
3. Set up the calculation
4. Compute the answer
5. Verify: Does this make sense?

FINAL ANSWER: [number only]"""

    return extract_final_answer(await llm.generate(prompt, 500))


async def solve_cot_verify(llm: CerebrasLLM, question: str) -> str:
    """CoT + Explicit verification step"""
    prompt = f"""Problem: {question}

STEP 1 - UNDERSTAND: What information is given? What are we solving for?

STEP 2 - PLAN: What calculations do we need?

STEP 3 - SOLVE: Do the math step by step.

STEP 4 - CHECK: Plug the answer back in. Does it work?

FINAL ANSWER: [just the number]"""

    return extract_final_answer(await llm.generate(prompt, 600))


async def main():
    print("="*70)
    print("PHASE 4: CHAIN-OF-THOUGHT COMPARISON")
    print("="*70)
    print("Comparing three approaches:")
    print("  1. Baseline: Direct question")
    print("  2. CoT: Step-by-step reasoning")
    print("  3. CoT+Verify: Steps + explicit verification")
    print("="*70)

    async with httpx.AsyncClient() as client:
        llm = CerebrasLLM(client)

        results = {"baseline": [], "cot": [], "cot_verify": []}

        print(f"\n{'#':>2} {'Exp':>8} {'Base':>8} {'CoT':>8} {'CoT+V':>8} | B C V")
        print("-"*70)

        for i, prob in enumerate(GSM8K_PROBLEMS):
            base = await solve_baseline(llm, prob["q"])
            await asyncio.sleep(0.2)

            cot = await solve_cot(llm, prob["q"])
            await asyncio.sleep(0.2)

            cot_v = await solve_cot_verify(llm, prob["q"])
            await asyncio.sleep(0.2)

            b_ok = check_answer(base, prob["a"])
            c_ok = check_answer(cot, prob["a"])
            v_ok = check_answer(cot_v, prob["a"])

            results["baseline"].append(b_ok)
            results["cot"].append(c_ok)
            results["cot_verify"].append(v_ok)

            print(f"{i+1:2d} {prob['a']:>8} {base[:8]:>8} {cot[:8]:>8} {cot_v[:8]:>8} | {'✓' if b_ok else '✗'} {'✓' if c_ok else '✗'} {'✓' if v_ok else '✗'}")

        # Summary
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)

        n = len(GSM8K_PROBLEMS)
        base_acc = 100 * sum(results["baseline"]) / n
        cot_acc = 100 * sum(results["cot"]) / n
        cot_v_acc = 100 * sum(results["cot_verify"]) / n

        print(f"Baseline:        {sum(results['baseline']):2d}/{n} = {base_acc:5.1f}%")
        print(f"CoT:             {sum(results['cot']):2d}/{n} = {cot_acc:5.1f}%")
        print(f"CoT+Verify:      {sum(results['cot_verify']):2d}/{n} = {cot_v_acc:5.1f}%")
        print()

        best = max(base_acc, cot_acc, cot_v_acc)
        print(f"Best approach:   {best:.1f}%")
        print(f"GSM8K target:    60.0%")
        print(f"Gate status:     {'✅ PASS' if best >= 60 else '❌ FAIL'}")
        print("="*70)

        with open("/workspace/data/phase4_cot_results.json", "w") as f:
            json.dump({
                "baseline": {"correct": sum(results["baseline"]), "accuracy": base_acc},
                "cot": {"correct": sum(results["cot"]), "accuracy": cot_acc},
                "cot_verify": {"correct": sum(results["cot_verify"]), "accuracy": cot_v_acc},
                "best": best,
                "gate_passed": best >= 60
            }, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
