#!/usr/bin/env python3
"""
E9: Thesis Replication with MiniMax-M2.7

Replicate the core thesis: decomposition improves accuracy +18pp,
using MiniMax-M2.7 instead of DeepSeek-V3.

Generates 50 novel math/logic problems and compares DIRECT vs DECOMPOSITION prompting.
"""

import json
import random
import re
import time
import signal
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("API call timed out after 30 seconds")

signal.signal(signal.SIGALRM, timeout_handler)

from openai import OpenAI


# Configuration
MODEL = "MiniMax-M2.7"
BASE_URL = "https://api.minimax.io/v1"
API_KEY = "sk-cp-bhJs4p1RP_THEJodFqXSCbmSUmxEDPPeWWAhnDztOTw7sX-eJjBbMnp2feo18Y60vHoXTneNOJDmkX_R7H2Q0Dlnp3phlY7MEEWMEdJh2TxDZMFxy4b0lzA"
N_PROBLEMS = 30

OUTPUT_DIR = Path("/home/aaron/projects/conjecture/research/rnd-sprint-2026-05-04")


@dataclass
class Problem:
    id: str
    question: str
    answer: float
    category: str


def generate_store_problem() -> Problem:
    """Generate a store discount problem (multi-step arithmetic)."""
    item = random.choice(['widgets', 'gadgets', 'items', 'products', 'books', 'tools'])
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
        f"How much do they pay?"
    )

    return Problem(
        id=f"store_{random.randint(10000, 99999)}",
        question=question,
        answer=answer,
        category="store_discount"
    )


def generate_handshake_problem() -> Problem:
    """Generate a handshake counting problem (combinatorics)."""
    n = random.randint(5, 20)
    answer = n * (n - 1) // 2

    question = (
        f"At a party, {n} people each shake hands exactly once with every other person. "
        f"How many handshakes occur in total?"
    )

    return Problem(
        id=f"handshake_{random.randint(10000, 99999)}",
        question=question,
        answer=float(answer),
        category="handshake"
    )


def generate_work_problem() -> Problem:
    """Generate a work rate problem (reciprocals)."""
    rate1 = random.randint(2, 8)
    rate2 = random.randint(2, 8)
    combined = 1 / (1/rate1 + 1/rate2)
    answer = round(combined, 2)

    question = (
        f"Worker A can complete a job in {rate1} days. Worker B can complete it in {rate2} days. "
        f"If they work together, how many days to complete the job? Round to 2 decimals."
    )

    return Problem(
        id=f"work_{random.randint(10000, 99999)}",
        question=question,
        answer=answer,
        category="work_rate"
    )


def generate_reverse_problem() -> Problem:
    """Generate a reverse engineering problem (algebraic reasoning)."""
    original = random.randint(5, 50)
    multiplier = random.randint(2, 5)
    addend = random.randint(5, 30)

    doubled = original * multiplier
    final = doubled + addend

    question = (
        f"A number is multiplied by {multiplier}, then {addend} is added, giving {final}. "
        f"What was the original number?"
    )

    return Problem(
        id=f"reverse_{random.randint(10000, 99999)}",
        question=question,
        answer=float(original),
        category="reverse"
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

    text = text.strip()

    # Look for common patterns
    patterns = [
        r'\$?([\d,]+\.?\d*)',
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
            return float(numbers[-1].replace(',', ''))
        except:
            pass

    return None


def check_answer(predicted: Optional[float], expected: float) -> bool:
    """Check if prediction matches expected within tolerance."""
    if predicted is None:
        return False
    return abs(predicted - expected) < 0.1


class MiniMaxClient:
    """Client for MiniMax API."""

    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.1, max_tokens: int = 500) -> Dict[str, Any]:
        """Generate a response."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": response.usage.total_tokens if response.usage else 0
        }


def run_direct(client: MiniMaxClient, problem: Problem) -> Tuple[Optional[float], float, int]:
    """Run direct prompting - 'Give only the numerical answer.'"""
    start = time.time()

    system_prompt = "You are a math problem solver. Give only the numerical answer, nothing else. No explanation, no units, just the number."
    prompt = f"Give only the numerical answer. {problem.question}"

    try:
        signal.alarm(30)  # 30 second timeout
        response = client.generate(prompt, system_prompt=system_prompt, temperature=0.1, max_tokens=100)
        signal.alarm(0)  # Cancel alarm
        elapsed = time.time() - start
        answer = extract_number(response.get("content", ""))
        tokens = response.get("usage", 0)
        return answer, elapsed, tokens
    except TimeoutError:
        print(f"  [TIMEOUT] Direct API call timed out after 30s")
        return None, time.time() - start, 0
    except Exception as e:
        signal.alarm(0)
        print(f"  Direct error: {e}")
        return None, time.time() - start, 0


def run_decomposition(client: MiniMaxClient, problem: Problem) -> Tuple[Optional[float], float, int]:
    """Run decomposition prompting - 'Break this problem down step by step.'"""
    start = time.time()

    system_prompt = """You are a careful reasoning assistant.
Break down the problem step by step:
1. Identify the key information
2. State your assumptions
3. Work through the calculation
4. Verify your answer makes sense
5. Give the final numerical answer

Always show your reasoning, then give just the number as your final answer."""

    prompt = f"Break this problem down step by step. Show your reasoning. Then give the final numerical answer. {problem.question}"

    try:
        signal.alarm(30)  # 30 second timeout
        response = client.generate(prompt, system_prompt=system_prompt, temperature=0.1, max_tokens=500)
        signal.alarm(0)  # Cancel alarm
        elapsed = time.time() - start
        content = response.get("content", "")
        tokens = response.get("usage", 0)

        # Extract answer from reasoning response
        answer = None

        # Look for explicit final answer marker
        final_patterns = [
            r'(?:final answer|answer)[:\s]*\$?([\d,]+\.?\d*)',
            r'(?:Step 5|Final)[^:]*:[^\d]*([\d,]+\.?\d*)',
            r'\*\*([\d,]+\.?\d*)\*\*',
            r'= \$?([\d,]+\.?\d*)\s*$',
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

        return answer, elapsed, tokens
    except TimeoutError:
        print(f"  [TIMEOUT] Decomposition API call timed out after 30s")
        return None, time.time() - start, 0
    except Exception as e:
        signal.alarm(0)
        print(f"  Decomposition error: {e}")
        return None, time.time() - start, 0


def main():
    print(f"\n{'='*70}")
    print("E9: THESIS REPLICATION WITH MINIMAX-M2.7")
    print(f"{'='*70}")
    print(f"Model: {MODEL}")
    print(f"Problems: {N_PROBLEMS}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Initialize client
    client = MiniMaxClient()

    # Generate problems
    print("Generating problems...")
    problems = generate_problems(N_PROBLEMS)
    print(f"Generated {len(problems)} problems")
    print(f"  - Store discount: {sum(1 for p in problems if p.category == 'store_discount')}")
    print(f"  - Handshake: {sum(1 for p in problems if p.category == 'handshake')}")
    print(f"  - Work rate: {sum(1 for p in problems if p.category == 'work_rate')}")
    print(f"  - Reverse engineering: {sum(1 for p in problems if p.category == 'reverse')}")
    print()

    # Track results
    direct_correct = 0
    decomposition_correct = 0
    direct_times = []
    decomposition_times = []
    direct_tokens = 0
    decomposition_tokens = 0
    detailed_results = []

    # Run DIRECT prompting
    print("--- DIRECT PROMPTING (Baseline) ---")
    for i, prob in enumerate(problems, 1):
        answer, elapsed, tokens = run_direct(client, prob)
        correct = check_answer(answer, prob.answer)
        direct_correct += correct
        direct_times.append(elapsed)
        direct_tokens += tokens

        status = "✓" if correct else "✗"
        print(f"  [{i:2d}/{N_PROBLEMS}] {status} Expected: {prob.answer}, Got: {answer}")

        detailed_results.append({
            "problem_id": prob.id,
            "category": prob.category,
            "question": prob.question,
            "expected": prob.answer,
            "direct_answer": answer,
            "direct_correct": correct,
        })

        time.sleep(0.5)  # Rate limiting

    direct_acc = direct_correct / N_PROBLEMS
    print(f"\nDirect Accuracy: {direct_acc:.1%} ({direct_correct}/{N_PROBLEMS})")
    print(f"Avg time: {sum(direct_times)/len(direct_times):.2f}s")
    print(f"Total tokens: {direct_tokens:,}")
    print()

    # Run DECOMPOSITION prompting
    print("--- DECOMPOSITION PROMPTING (Reasoning) ---")
    for i, prob in enumerate(problems, 1):
        answer, elapsed, tokens = run_decomposition(client, prob)
        correct = check_answer(answer, prob.answer)
        decomposition_correct += correct
        decomposition_times.append(elapsed)
        decomposition_tokens += tokens

        status = "✓" if correct else "✗"
        print(f"  [{i:2d}/{N_PROBLEMS}] {status} Expected: {prob.answer}, Got: {answer}")

        # Update detailed results
        for r in detailed_results:
            if r["problem_id"] == prob.id:
                r["decomposition_answer"] = answer
                r["decomposition_correct"] = correct
                break

        time.sleep(0.5)  # Rate limiting

    decomposition_acc = decomposition_correct / N_PROBLEMS
    print(f"\nDecomposition Accuracy: {decomposition_acc:.1%} ({decomposition_correct}/{N_PROBLEMS})")
    print(f"Avg time: {sum(decomposition_times)/len(decomposition_times):.2f}s")
    print(f"Total tokens: {decomposition_tokens:,}")
    print()

    # Calculate improvement
    improvement_pp = (decomposition_acc - direct_acc) * 100

    # Simple p-value approximation using McNemar-like comparison
    # Count cases where decomposition helped and where it hurt
    decomp_helped = 0
    decomp_hurt = 0
    for r in detailed_results:
        if r.get("decomposition_correct") and not r["direct_correct"]:
            decomp_helped += 1
        elif not r.get("decomposition_correct") and r["direct_correct"]:
            decomp_hurt += 1

    # Simple sign test p-value approximation
    total_diff_cases = decomp_helped + decomp_hurt
    if total_diff_cases > 0:
        p_value = 2 ** (-total_diff_cases)  # Approximate sign test
    else:
        p_value = 1.0

    # Results summary
    print(f"{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Direct (Baseline):        {direct_acc:.1%} ({direct_correct}/{N_PROBLEMS})")
    print(f"Decomposition (Reasoning): {decomposition_acc:.1%} ({decomposition_correct}/{N_PROBLEMS})")
    print(f"Improvement:             {improvement_pp:+.1f} percentage points")
    print(f"Decomposition helped:   {decomp_helped} cases")
    print(f"Decomposition hurt:     {decomp_hurt} cases")
    print(f"Approximate p-value:    {p_value:.6f}")
    print()

    # Pass/fail criteria: improvement > 10pp
    e9_pass = improvement_pp > 10.0
    pass_fail = "PASS" if e9_pass else "FAIL"

    if e9_pass:
        conclusion = "THESIS REPLICATED: Decomposition improves accuracy by >10pp with MiniMax-M2.7"
    else:
        conclusion = "THESIS NOT REPLICATED: Improvement <= 10pp with MiniMax-M2.7"

    print(f"E9 Status: {pass_fail}")
    print(f"Conclusion: {conclusion}")
    print(f"{'='*70}")
    print()

    # Prepare results
    results = {
        "experiment": "E9_thesis_replication_minimax",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "n_problems": N_PROBLEMS,
        "direct_accuracy": direct_acc,
        "decomposition_accuracy": decomposition_acc,
        "improvement_pp": improvement_pp,
        "p_value": p_value,
        "E9_pass": e9_pass,
        "direct_correct": direct_correct,
        "decomposition_correct": decomposition_correct,
        "direct_avg_time": sum(direct_times) / len(direct_times) if direct_times else 0,
        "decomposition_avg_time": sum(decomposition_times) / len(decomposition_times) if decomposition_times else 0,
        "direct_total_tokens": direct_tokens,
        "decomposition_total_tokens": decomposition_tokens,
        "decomp_helped": decomp_helped,
        "decomp_hurt": decomp_hurt,
        "conclusion": conclusion,
        "pass_fail": pass_fail,
        "detailed_results": detailed_results,
    }

    # Save results
    results_file = OUTPUT_DIR / "E9-results.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved to: {results_file}")

    return results


if __name__ == "__main__":
    results = main()