#!/usr/bin/env python3
"""Quick test of position primacy hypothesis."""

import asyncio
import os
import re
import httpx
from dotenv import load_dotenv

load_dotenv("/workspace/.env")


def extract_answer(response: str) -> str:
    """Extract numerical answer from response."""
    if not response:
        return ""

    # \boxed{} pattern
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        return match.group(1).replace(",", "")

    # #### pattern
    match = re.search(r'####\s*(\-?[\d,\.]+)', response)
    if match:
        return match.group(1).replace(",", "")

    # "answer is X" pattern
    match = re.search(r'answer\s*(?:is|:)\s*\$?(\-?[\d,\.]+)', response, re.I)
    if match:
        return match.group(1).replace(",", "")

    # Last number in response
    numbers = re.findall(r'\-?[\d,]+\.?\d*', response)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


async def generate(prompt: str) -> str:
    """Generate response."""
    url = "https://llm.chutes.ai/v1/chat/completions"
    api_key = os.getenv("CHUTES_API_KEY")

    async with httpx.AsyncClient() as client:
        for attempt in range(3):
            try:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "deepseek-ai/DeepSeek-V3",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 150,
                        "temperature": 0.1
                    },
                    timeout=120.0
                )
                if resp.status_code == 429:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"  Error: {e}")
                await asyncio.sleep(2)
    return ""


def generate_problems(n: int = 20):
    """Generate simple math problems."""
    import random
    random.seed(42)
    problems = []
    for i in range(n):
        if i % 4 == 0:
            a, b = random.randint(10, 50), random.randint(10, 50)
            problems.append((f"Store sells {a} at $2, {b} at $3. Total?", str(a*2 + b*3)))
        elif i % 4 == 1:
            s, t = random.randint(40, 80), random.randint(2, 6)
            problems.append((f"Car at {s} mph for {t} hours. Miles?", str(s*t)))
        elif i % 4 == 2:
            total, pct = random.randint(100, 500), random.randint(20, 80)
            problems.append((f"{pct}% of {total}?", str(int(total*pct/100))))
        else:
            items, groups = random.randint(24, 120), random.randint(3, 8)
            problems.append((f"{items} divided by {groups}?", str(items//groups)))
    return problems


async def test_position(position: str, problems: list) -> float:
    """Test a position strategy."""
    print(f"\n--- {position.upper()} ---", flush=True)

    memory = []
    correct = 0

    for i, (q, ans) in enumerate(problems):
        # Build prompt based on position
        if memory:
            hints_list = memory[-3:]
            hints_text = "\n".join(f"- {c}" for c in hints_list)
        else:
            hints_text = ""

        if position == "start" and hints_text:
            prompt = f"KEY PATTERNS:\n{hints_text}\n\nProblem: {q}\n\nSolve and give the answer."
        elif position == "middle" and hints_text:
            prompt = f"Problem: {q}\n\nHints:\n{hints_text}\n\nSolve and give the answer."
        elif position == "end" and hints_text:
            prompt = f"Problem: {q}\n\nSolve and give the answer.\n\nReference patterns:\n{hints_text}"
        else:
            prompt = f"Problem: {q}\n\nSolve and give the answer."

        response = await generate(prompt)
        pred = extract_answer(response)
        is_correct = str(pred) == str(ans)

        if is_correct:
            correct += 1

        memory.append(f"{q[:25]}... = {pred}")

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(problems)}] acc={100*correct/(i+1):.1f}%", flush=True)

        await asyncio.sleep(0.1)

    acc = 100 * correct / len(problems)
    print(f"  Final: {acc:.1f}%", flush=True)
    return acc


async def main():
    print("=" * 60, flush=True)
    print("QUICK POSITION PRIMACY TEST (20 problems)", flush=True)
    print("=" * 60, flush=True)

    problems = generate_problems(20)
    print(f"Generated {len(problems)} problems", flush=True)

    results = {}

    for pos in ["middle", "start", "end"]:
        results[pos] = await test_position(pos, problems)
        await asyncio.sleep(2)

    print("\n" + "=" * 60, flush=True)
    print("RESULTS:", flush=True)
    print("-" * 40, flush=True)
    for pos, acc in results.items():
        print(f"  {pos.upper():8s}: {acc:.1f}%", flush=True)

    best = max(results.items(), key=lambda x: x[1])
    print(f"\nBest position: {best[0]} ({best[1]:.1f}%)", flush=True)

    if results.get("start", 0) > results.get("middle", 0):
        print("\n✓ PRIMACY CONFIRMED: START beats MIDDLE", flush=True)
    else:
        print("\n✗ Primacy not confirmed", flush=True)

    print("=" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
