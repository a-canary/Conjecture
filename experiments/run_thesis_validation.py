#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Run Thesis Validation Experiment

Compares direct prompting vs reasoning loop with decomposition.
Uses Chutes.ai API with existing infrastructure.

Usage:
    python experiments/run_thesis_validation.py --quick     # 20 problems
    python experiments/run_thesis_validation.py --full      # 200 problems
"""

import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


class LLMClient:
    """Minimal LLM client for validation experiment."""

    def __init__(self, model: str = "deepseek/deepseek-chat-v3-0324"):
        # Use OpenRouter for better availability
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = model
        self._client = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "content": response.choices[0].message.content,
            "model": response.model
        }


@dataclass
class Problem:
    id: str
    question: str
    answer: float
    category: str


def generate_store_problem() -> Problem:
    """Generate a store discount problem."""
    item = random.choice(['widgets', 'gadgets', 'items', 'products'])
    price = random.randint(7, 97)
    qty1 = random.randint(2, 25)
    qty2 = random.randint(2, 25)
    discount = random.randint(5, 35)

    total_qty = qty1 + qty2
    subtotal = total_qty * price
    discount_amt = subtotal * discount / 100
    answer = round(subtotal - discount_amt, 2)

    question = (
        f"A store sells {item} for ${price} each. A customer buys {qty1} on Monday "
        f"and {qty2} on Tuesday, then gets a {discount}% discount on the total. "
        f"How much do they pay? Give only the number."
    )

    return Problem(
        id=f"store_{random.randint(1000, 9999)}",
        question=question,
        answer=answer,
        category="math"
    )


def generate_handshake_problem() -> Problem:
    """Generate a handshake counting problem."""
    n = random.randint(5, 20)
    answer = n * (n - 1) // 2

    question = (
        f"At a party, {n} people each shake hands exactly once with every other person. "
        f"How many handshakes occur in total? Give only the number."
    )

    return Problem(
        id=f"handshake_{random.randint(1000, 9999)}",
        question=question,
        answer=float(answer),
        category="logic"
    )


def generate_work_problem() -> Problem:
    """Generate a work rate problem."""
    rate1 = random.randint(2, 8)
    rate2 = random.randint(2, 8)
    # Combined rate = 1/rate1 + 1/rate2
    # Combined days = 1 / combined_rate
    combined = 1 / (1/rate1 + 1/rate2)
    answer = round(combined, 2)

    question = (
        f"Worker A can complete a job in {rate1} days. Worker B can complete it in {rate2} days. "
        f"If they work together, how many days to complete the job? Round to 2 decimals. Give only the number."
    )

    return Problem(
        id=f"work_{random.randint(1000, 9999)}",
        question=question,
        answer=answer,
        category="math"
    )


def generate_reverse_problem() -> Problem:
    """Generate a reverse engineering problem."""
    original = random.randint(5, 50)
    multiplier = random.randint(2, 5)
    addend = random.randint(5, 30)

    doubled = original * multiplier
    final = doubled + addend

    question = (
        f"A number is multiplied by {multiplier}, then {addend} is added, giving {final}. "
        f"What was the original number? Give only the number."
    )

    return Problem(
        id=f"reverse_{random.randint(1000, 9999)}",
        question=question,
        answer=float(original),
        category="reasoning"
    )


def generate_problems(n: int) -> List[Problem]:
    """Generate n mixed problems."""
    generators = [
        generate_store_problem,
        generate_handshake_problem,
        generate_work_problem,
        generate_reverse_problem,
    ]

    problems = []
    for i in range(n):
        gen = generators[i % len(generators)]
        problems.append(gen())

    random.shuffle(problems)
    return problems


def extract_number(text: str) -> Optional[float]:
    """Extract a number from LLM response."""
    if not text:
        return None

    # Clean up
    text = text.strip()

    # Look for common patterns
    patterns = [
        r'\$?([\d,]+\.?\d*)',  # $123.45 or 123.45
        r'([\d,]+\.?\d*)\s*(?:dollars|items|handshakes|days|people)',
        r'answer[:\s]+\$?([\d,]+\.?\d*)',
        r'result[:\s]+\$?([\d,]+\.?\d*)',
        r'total[:\s]+\$?([\d,]+\.?\d*)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                num_str = match.group(1).replace(',', '')
                return float(num_str)
            except:
                continue

    # Try to find any number
    numbers = re.findall(r'[\d,]+\.?\d*', text)
    if numbers:
        try:
            # Take the last number (usually the final answer)
            return float(numbers[-1].replace(',', ''))
        except:
            pass

    return None


def check_answer(predicted: Optional[float], expected: float) -> bool:
    """Check if prediction matches expected within tolerance."""
    if predicted is None:
        return False
    return abs(predicted - expected) < 0.1


DIRECT_SYSTEM = """You are a math problem solver.
Give only the numerical answer, nothing else.
No explanation, no units, just the number."""

REASONING_SYSTEM = """You are a careful reasoning assistant.
Break down the problem step by step:
1. Identify the key information
2. State your assumptions
3. Work through the calculation
4. Verify your answer makes sense
5. Give the final numerical answer

Always show your reasoning, then give just the number as your final answer."""


async def run_direct(problem: Problem, client: LLMClient) -> Tuple[Optional[float], float]:
    """Run direct prompting."""
    start = time.time()
    try:
        response = await client.generate(
            prompt=problem.question,
            system_prompt=DIRECT_SYSTEM,
            temperature=0.1,
            max_tokens=100
        )
        elapsed = time.time() - start
        answer = extract_number(response.get("content", ""))
        return answer, elapsed
    except Exception as e:
        print(f"  Direct error: {e}")
        return None, time.time() - start


async def run_reasoning(problem: Problem, client: LLMClient) -> Tuple[Optional[float], float]:
    """Run reasoning with decomposition prompt."""
    start = time.time()

    reasoning_prompt = f"""Problem: {problem.question}

Let me break this down step by step:

Step 1 - Identify key information:
What are the given values?

Step 2 - State assumptions:
What do I need to assume or validate?

Step 3 - Work through calculation:
Show the math step by step.

Step 4 - Verify:
Does this answer make sense?

Step 5 - Final answer:
[Just the number]"""

    try:
        response = await client.generate(
            prompt=reasoning_prompt,
            system_prompt=REASONING_SYSTEM,
            temperature=0.1,
            max_tokens=500
        )
        elapsed = time.time() - start
        content = response.get("content", "")

        # Better extraction for reasoning responses - look for final answer patterns
        answer = None

        # Look for explicit final answer marker
        final_patterns = [
            r'(?:final answer|answer)[:\s]*\$?([\d,]+\.?\d*)',
            r'(?:Step 5|Final)[^:]*:[^\d]*([\d,]+\.?\d*)',
            r'\*\*([\d,]+\.?\d*)\*\*',  # Bold numbers
            r'= \$?([\d,]+\.?\d*)\s*$',  # Last equation result
        ]

        for pattern in final_patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    answer = float(match.group(1).replace(',', ''))
                    break
                except:
                    continue

        # Fallback: take last number in content
        if answer is None:
            numbers = re.findall(r'[\d,]+\.?\d*', content)
            if numbers:
                # Filter out step numbers (1, 2, 3, 4, 5)
                valid_numbers = [n for n in numbers if float(n.replace(',', '')) > 10]
                if valid_numbers:
                    try:
                        answer = float(valid_numbers[-1].replace(',', ''))
                    except:
                        pass

        return answer, elapsed
    except Exception as e:
        print(f"  Reasoning error: {e}")
        return None, time.time() - start


async def run_experiment(n_problems: int = 20):
    """Run the validation experiment."""
    print(f"\n{'='*60}")
    print("THESIS VALIDATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Problems: {n_problems}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Initialize client with OpenRouter model ID
    # Can override via --model flag
    model = os.environ.get("TEST_MODEL", "deepseek/deepseek-chat-v3-0324")
    client = LLMClient(model=model)

    # Generate problems
    print("Generating problems...")
    problems = generate_problems(n_problems)
    print(f"Generated {len(problems)} problems\n")

    # Run direct baseline
    print("--- BASELINE (Direct Prompting) ---")
    direct_correct = 0
    direct_times = []

    for i, prob in enumerate(problems):
        answer, elapsed = await run_direct(prob, client)
        correct = check_answer(answer, prob.answer)
        direct_correct += correct
        direct_times.append(elapsed)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n_problems}] Expected: {prob.answer}, Got: {answer}, Correct: {correct}")

        await asyncio.sleep(0.5)  # Rate limiting

    direct_acc = direct_correct / n_problems
    print(f"\nBaseline: {direct_acc:.1%} ({direct_correct}/{n_problems})")
    print(f"Avg time: {sum(direct_times)/len(direct_times):.2f}s")

    # Run reasoning
    print("\n--- REASONING (Decomposition + Validation) ---")
    reasoning_correct = 0
    reasoning_times = []

    for i, prob in enumerate(problems):
        answer, elapsed = await run_reasoning(prob, client)
        correct = check_answer(answer, prob.answer)
        reasoning_correct += correct
        reasoning_times.append(elapsed)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n_problems}] Expected: {prob.answer}, Got: {answer}, Correct: {correct}")

        await asyncio.sleep(0.5)  # Rate limiting

    reasoning_acc = reasoning_correct / n_problems
    print(f"\nReasoning: {reasoning_acc:.1%} ({reasoning_correct}/{n_problems})")
    print(f"Avg time: {sum(reasoning_times)/len(reasoning_times):.2f}s")

    # Results summary
    improvement = (reasoning_acc - direct_acc) * 100

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Direct Baseline:  {direct_acc:.1%}")
    print(f"With Reasoning:   {reasoning_acc:.1%}")
    print(f"Improvement:      {improvement:+.1f}pp")
    print()

    if improvement > 5:
        conclusion = "THESIS VALIDATED: Decomposition improves accuracy"
    elif improvement > 0:
        conclusion = "THESIS PARTIALLY SUPPORTED: Modest improvement"
    elif improvement > -5:
        conclusion = "THESIS INCONCLUSIVE: No significant difference"
    else:
        conclusion = "THESIS CHALLENGED: Decomposition decreases accuracy"

    print(f"CONCLUSION: {conclusion}")
    print(f"{'='*60}\n")

    # Save results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_problems": n_problems,
        "model": client.model,
        "direct_accuracy": direct_acc,
        "reasoning_accuracy": reasoning_acc,
        "improvement_pp": improvement,
        "direct_avg_time": sum(direct_times) / len(direct_times),
        "reasoning_avg_time": sum(reasoning_times) / len(reasoning_times),
        "conclusion": conclusion
    }

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"thesis_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick test (20 problems)")
    parser.add_argument("--full", action="store_true", help="Full test (200 problems)")
    parser.add_argument("-n", type=int, default=20, help="Number of problems")
    args = parser.parse_args()

    if args.full:
        n = 200
    elif args.quick:
        n = 20
    else:
        n = args.n

    asyncio.run(run_experiment(n))
