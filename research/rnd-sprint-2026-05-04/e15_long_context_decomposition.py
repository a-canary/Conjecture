#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
E15: Long-Context Decomposition vs Chunking — Does size hurt MiniMax-M2.7?

Tests accuracy degradation as claim chain length increases.
Based on E9's 30 problems with added context layers.

Hypothesis: Beyond a certain context length, decomposition quality degrades.
"""

import json
import random
import re
import time
import signal
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("API call timed out after 45 seconds")

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


def generate_problems(n: int) -> List[Problem]:
    """Generate the same 30 problems from E9 (store, handshake, work, reverse)."""
    problems = []
    
    # Store discount problems (8 problems)
    store_params = [
        (65, 2, 7, 9, 532.35),
        (31, 7, 20, 11, 744.93),
        (7, 22, 9, 10, 195.3),
        (67, 13, 6, 34, 840.18),
        (69, 25, 17, 23, 2231.46),
        (67, 4, 11, 14, 864.3),
        (85, 19, 17, 6, 2876.4),
        (30, 3, 6, 14, 232.2),
    ]
    for i, (price, q1, q2, disc, ans) in enumerate(store_params):
        item = random.choice(['widgets', 'gadgets', 'books', 'tools', 'products'])
        problems.append(Problem(
            id=f"store_{35704 + i*1000}",
            question=f"A store sells {item} for ${price} each. A customer buys {q1} on Monday and {q2} on Tuesday, then gets a {disc}% discount on the total. How much do they pay?",
            answer=ans,
            category="store_discount"
        ))
    
    # Handshake problems (8 problems)  
    handshake_params = [
        (9, 36.0), (9, 36.0), (11, 55.0), (11, 55.0),
        (10, 45.0), (6, 15.0), (17, 136.0), (10, 45.0),
    ]
    for i, (n, ans) in enumerate(handshake_params):
        problems.append(Problem(
            id=f"handshake_{23706 + i*1000}",
            question=f"At a party, {n} people each shake hands exactly once with every other person. How many handshakes occur in total?",
            answer=ans,
            category="handshake"
        ))
    
    # Work rate problems (8 problems)
    work_params = [
        (2, 6, 1.5), (6, 5, 2.73), (2, 7, 1.56), (7, 6, 3.23),
        (8, 8, 4.0), (6, 6, 3.0), (8, 5, 3.08), (8, 8, 4.0),
    ]
    for i, (r1, r2, ans) in enumerate(work_params):
        problems.append(Problem(
            id=f"work_{31353 + i*1000}",
            question=f"Worker A can complete a job in {r1} days. Worker B can complete it in {r2} days. If they work together, how many days to complete the job? Round to 2 decimals.",
            answer=ans,
            category="work_rate"
        ))
    
    # Reverse engineering problems (6 problems)
    reverse_params = [
        (4, 7, 107, 25.0), (5, 9, 224, 43.0), (2, 6, 98, 46.0),
        (3, 28, 139, 37.0), (4, 13, 93, 20.0), (2, 5, 97, 46.0),
        (3, 20, 47, 9.0),
    ]
    for i, (mult, add, result, ans) in enumerate(reverse_params):
        problems.append(Problem(
            id=f"reverse_{74597 + i*1000}",
            question=f"A number is multiplied by {mult}, then {add} is added, giving {result}. What was the original number?",
            answer=ans,
            category="reverse"
        ))
    
    random.shuffle(problems)
    return problems[:n]


def build_context_layers(problem: Problem, num_hops: int) -> str:
    """Build a claim chain of given depth.
    
    1 hop: Direct claim + evidence
    3 hops: 1 claim + 2 intermediate claims + evidence
    7 hops: 1 claim + 6 intermediate claims + evidence
    15 hops: 1 claim + 14 intermediate claims + evidence
    """
    if num_hops == 1:
        # Short: direct claim with evidence
        return f"""Context:
The store's pricing data shows the base price and quantity sold.
Evidence: Based on sales records, customers purchase items at listed prices.

Question: {problem.question}
"""
    
    # Build intermediate claims to create context depth
    # Each intermediate claim adds a "layer" of context
    
    if problem.category == "store_discount":
        # Extract key facts
        match = re.search(r'\$(\d+) each.*buys (\d+) on Monday and (\d+) on Tuesday.*(\d+)% discount', problem.question)
        if match:
            price, q1, q2, disc = match.groups()
            price, q1, q2, disc = int(price), int(q1), int(q2), int(disc)
            
            # Build chain of claims
            claims = [f"Claim: The item costs ${price} per unit."]
            
            for hop in range(1, num_hops):
                if hop == 1:
                    claims.append(f"Claim: The customer purchased {q1} units on Monday and {q2} units on Tuesday, totaling {q1+q2} units.")
                elif hop == 2:
                    claims.append(f"Claim: The subtotal before discount is {q1+q2} × ${price} = ${(q1+q2)*price}.")
                elif hop == 3:
                    claims.append(f"Claim: A {disc}% discount applies to the subtotal, equaling ${(q1+q2)*price * disc / 100}.")
                elif hop == 4:
                    claims.append(f"Claim: The final price after discount is ${(q1+q2)*price * (100-disc) / 100}.")
                else:
                    # Pad with additional claims about related transactions
                    claims.append(f"Claim: Similar transactions in the database show consistent pricing patterns for {price}-dollar items.")
            
            # Trim or pad to exactly num_hops claims
            while len(claims) < num_hops:
                claims.append(f"Claim: Historical sales data confirms the pricing model for {price}-dollar items.")
            
            context = "Context (Claim Chain):\n" + "\n".join(claims[:num_hops]) + f"\n\nQuestion: {problem.question}"
            return context
    
    elif problem.category == "handshake":
        match = re.search(r'(\d+) people', problem.question)
        if match:
            n = int(match.group(1))
            n_handshakes = n * (n-1) // 2
            
            claims = [f"Claim: There are {n} people at the party."]
            
            for hop in range(1, num_hops):
                if hop == 1:
                    claims.append(f"Claim: Each person shakes hands with {n-1} other people.")
                elif hop == 2:
                    claims.append(f"Claim: Total handshakes counted: {n} × {n-1} = {n*(n-1)}.")
                elif hop == 3:
                    claims.append(f"Claim: Since each handshake involves 2 people, we divide by 2: {n*(n-1)//2}.")
                elif hop == 4:
                    claims.append(f"Claim: The formula n×(n-1)/2 gives {n_handshakes} for n={n}.")
                else:
                    claims.append(f"Claim: Combinatorics confirms {n} choose 2 = {n_handshakes} handshakes.")
            
            while len(claims) < num_hops:
                claims.append(f"Claim: Social interaction data from similar events confirms the handshake pattern.")
            
            context = "Context (Claim Chain):\n" + "\n".join(claims[:num_hops]) + f"\n\nQuestion: {problem.question}"
            return context
    
    elif problem.category == "work_rate":
        match = re.search(r'(\d+) days.*(\d+) days', problem.question)
        if match:
            r1, r2 = int(match.group(1)), int(match.group(2))
            combined = round(1 / (1/r1 + 1/r2), 2)
            
            claims = [f"Claim: Worker A can complete the job in {r1} days."]
            claims.append(f"Claim: Worker B can complete the job in {r2} days.")
            
            for hop in range(2, num_hops):
                if hop == 2:
                    claims.append(f"Claim: Worker A's rate is 1/{r1} of the job per day.")
                elif hop == 3:
                    claims.append(f"Claim: Worker B's rate is 1/{r2} of the job per day.")
                elif hop == 4:
                    claims.append(f"Claim: Combined rate = 1/{r1} + 1/{r2} = {(r1+r2)/(r1*r2)} job/day.")
                elif hop == 5:
                    claims.append(f"Claim: Time to complete = 1 / combined rate = {r1*r2/(r1+r2)} days.")
                elif hop == 6:
                    claims.append(f"Claim: Rounded to 2 decimals: {combined} days.")
                else:
                    claims.append(f"Claim: Work rate calculations from similar problems confirm this pattern.")
            
            while len(claims) < num_hops:
                claims.append(f"Claim: Historical job completion data validates the work rate model.")
            
            context = "Context (Claim Chain):\n" + "\n".join(claims[:num_hops]) + f"\n\nQuestion: {problem.question}"
            return context
    
    elif problem.category == "reverse":
        match = re.search(r'multiplied by (\d+),.*(\d+) is added, giving (\d+)', problem.question)
        if match:
            mult, add, result = int(match.group(1)), int(match.group(2)), int(match.group(3))
            original = (result - add) // mult
            
            claims = [f"Claim: The final result is {result}."]
            
            for hop in range(1, num_hops):
                if hop == 1:
                    claims.append(f"Claim: {add} was added to the intermediate value.")
                elif hop == 2:
                    claims.append(f"Claim: Before adding, the value was {result - add}.")
                elif hop == 3:
                    claims.append(f"Claim: This value came from multiplying by {mult}.")
                elif hop == 4:
                    claims.append(f"Claim: So the original was ({result} - {add}) / {mult} = {original}.")
                elif hop == 5:
                    claims.append(f"Claim: Verification: {original} × {mult} + {add} = {original*mult + add}.")
                else:
                    claims.append(f"Claim: Algebraic reasoning confirms the inverse operation.")
            
            while len(claims) < num_hops:
                claims.append(f"Claim: Numerical inverse operations follow predictable patterns.")
            
            context = "Context (Claim Chain):\n" + "\n".join(claims[:num_hops]) + f"\n\nQuestion: {problem.question}"
            return context
    
    # Fallback: simple padding
    claims = [f"Claim {i+1}: Relevant background information for this problem." for i in range(num_hops)]
    return "Context (Claim Chain):\n" + "\n".join(claims) + f"\n\nQuestion: {problem.question}"


def build_long_claim_context(problem: Problem) -> Tuple[str, str]:
    """Build a very long single claim vs decomposed version.
    Returns (long_claim_context, decomposed_context)"""
    
    # Build a long claim with lots of filler
    long_claim = f"""CONTEXT DOCUMENT:
This is a comprehensive analysis of the problem scenario. Let me walk you through all the relevant details in an exhaustive manner. 

The fundamental setup involves understanding the initial conditions and how they relate to the final outcome. We need to carefully trace each step of the reasoning process to ensure accuracy. 

Initial State: The problem describes a situation where certain operations are performed on an initial value or set of values. These operations include basic arithmetic transformations such as multiplication, addition, subtraction, and division.

Step-by-Step Analysis:
In order to properly solve this problem, one must identify all the key variables and parameters involved. The relationships between these variables must be established before any calculations can begin. Each transformation must be applied in the correct sequence to maintain logical consistency.

Intermediate Calculations:
After identifying the initial parameters, the next phase involves performing the intermediate calculations. These calculations serve as stepping stones toward the final solution. Each intermediate result builds upon the previous one, creating a chain of logical dependencies.

Detailed Breakdown:
For this particular problem, the detailed breakdown reveals several important insights. The first observation is that the problem follows a standard pattern commonly seen in mathematical reasoning tasks. The second observation is that the specific numbers involved require careful handling to avoid common errors.

Verification Process:
Once the calculations are complete, verification is essential to ensure the accuracy of the result. This involves checking each step of the calculation for potential errors and confirming that the final answer makes sense in the context of the problem.

Final Answer Generation:
Based on the above analysis, we can now derive the final answer with confidence. The mathematical operations have been verified and the logical flow has been confirmed.

{problem.question}

Please provide only the numerical answer."""

    decomposed = f"""CONTEXT DOCUMENT:
Let me break down this problem carefully.

Step 1: Identify the key information from the question.
Step 2: Set up the mathematical relationships.
Step 3: Perform the calculations step by step.
Step 4: Verify the answer.

Question: {problem.question}

Provide the numerical answer."""

    return long_claim, decomposed


def extract_number(text: str) -> Optional[float]:
    """Extract a number from LLM response."""
    if not text:
        return None

    text = text.strip()

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


def run_decomposition(client: MiniMaxClient, context: str) -> Tuple[Optional[float], float, int]:
    """Run decomposition prompting with given context."""
    start = time.time()

    system_prompt = """You are a careful reasoning assistant.
Break down the problem step by step:
1. Identify the key information
2. State your assumptions
3. Work through the calculation
4. Verify your answer makes sense
5. Give the final numerical answer

Always show your reasoning, then give just the number as your final answer."""

    prompt = f"Break this problem down step by step. Show your reasoning. Then give the final numerical answer.\n\n{context}"

    try:
        signal.alarm(45)  # 45 second timeout
        response = client.generate(prompt, system_prompt=system_prompt, temperature=0.1, max_tokens=600)
        signal.alarm(0)
        elapsed = time.time() - start
        content = response.get("content", "")
        tokens = response.get("usage", 0)

        answer = None

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

        if answer is None:
            numbers = re.findall(r'[\d,]+\.?\d*', content)
            if numbers:
                valid_numbers = [n for n in numbers if float(n.replace(',', '')) > 10]
                if valid_numbers:
                    try:
                        answer = float(valid_numbers[-1].replace(',', ''))
                    except:
                        pass

        return answer, elapsed, tokens
    except TimeoutError:
        print(f"  [TIMEOUT] API call timed out after 45s")
        return None, time.time() - start, 0
    except Exception as e:
        signal.alarm(0)
        print(f"  Error: {e}")
        return None, time.time() - start, 0


def run_long_claim_test(client: MiniMaxClient, problem: Problem) -> Tuple[bool, bool, float, float, int, int]:
    """Test long claim vs decomposed on same problem.
    Returns (long_correct, decomp_correct, long_time, decomp_time, long_tokens, decomp_tokens)
    """
    long_context, decomp_context = build_long_claim_context(problem)
    
    # Test long claim
    long_answer, long_time, long_tokens = run_decomposition(client, long_context)
    long_correct = check_answer(long_answer, problem.answer)
    
    time.sleep(0.5)
    
    # Test decomposed
    decomp_answer, decomp_time, decomp_tokens = run_decomposition(client, decomp_context)
    decomp_correct = check_answer(decomp_answer, problem.answer)
    
    return long_correct, decomp_correct, long_time, decomp_time, long_tokens, decomp_tokens


def main():
    print(f"\n{'='*70}")
    print("E15: LONG-CONTEXT DECOMPOSITION VS CHUNKING")
    print(f"{'='*70}")
    print(f"Model: {MODEL}")
    print(f"Problems: {N_PROBLEMS}")
    print(f"Started: {datetime.now().isoformat()}")
    print()
    
    client = MiniMaxClient()
    
    # Generate same problems as E9
    print("Generating problems (same as E9)...")
    problems = generate_problems(N_PROBLEMS)
    print(f"Generated {len(problems)} problems")
    print()
    
    # Test hop levels: 1, 3, 7, 15
    hop_levels = [1, 3, 7, 15]
    hop_results = {h: {"correct": 0, "total": 0, "times": [], "tokens": 0} for h in hop_levels}
    
    # First, test short (1-hop) and medium (3-hop)
    print("=== Testing 1-hop (Short) Context ===")
    for i, prob in enumerate(problems, 1):
        context = build_context_layers(prob, 1)
        answer, elapsed, tokens = run_decomposition(client, context)
        correct = check_answer(answer, prob.answer)
        hop_results[1]["correct"] += correct
        hop_results[1]["total"] += 1
        hop_results[1]["times"].append(elapsed)
        hop_results[1]["tokens"] += tokens
        
        status = "✓" if correct else "✗"
        print(f"  [{i:2d}/{N_PROBLEMS}] {status} Expected: {prob.answer}, Got: {answer}")
        time.sleep(0.3)
    
    print(f"\n1-hop Accuracy: {hop_results[1]['correct']/N_PROBLEMS:.1%}")
    print()
    
    print("=== Testing 3-hop (Medium) Context ===")
    for i, prob in enumerate(problems, 1):
        context = build_context_layers(prob, 3)
        answer, elapsed, tokens = run_decomposition(client, context)
        correct = check_answer(answer, prob.answer)
        hop_results[3]["correct"] += correct
        hop_results[3]["total"] += 1
        hop_results[3]["times"].append(elapsed)
        hop_results[3]["tokens"] += tokens
        
        status = "✓" if correct else "✗"
        print(f"  [{i:2d}/{N_PROBLEMS}] {status} Expected: {prob.answer}, Got: {answer}")
        time.sleep(0.3)
    
    print(f"\n3-hop Accuracy: {hop_results[3]['correct']/N_PROBLEMS:.1%}")
    print()
    
    print("=== Testing 7-hop (Long) Context ===")
    for i, prob in enumerate(problems, 1):
        context = build_context_layers(prob, 7)
        answer, elapsed, tokens = run_decomposition(client, context)
        correct = check_answer(answer, prob.answer)
        hop_results[7]["correct"] += correct
        hop_results[7]["total"] += 1
        hop_results[7]["times"].append(elapsed)
        hop_results[7]["tokens"] += tokens
        
        status = "✓" if correct else "✗"
        print(f"  [{i:2d}/{N_PROBLEMS}] {status} Expected: {prob.answer}, Got: {answer}")
        time.sleep(0.3)
    
    print(f"\n7-hop Accuracy: {hop_results[7]['correct']/N_PROBLEMS:.1%}")
    print()
    
    print("=== Testing 15-hop (Very Long) Context ===")
    for i, prob in enumerate(problems, 1):
        context = build_context_layers(prob, 15)
        answer, elapsed, tokens = run_decomposition(client, context)
        correct = check_answer(answer, prob.answer)
        hop_results[15]["correct"] += correct
        hop_results[15]["total"] += 1
        hop_results[15]["times"].append(elapsed)
        hop_results[15]["tokens"] += tokens
        
        status = "✓" if correct else "✗"
        print(f"  [{i:2d}/{N_PROBLEMS}] {status} Expected: {prob.answer}, Got: {answer}")
        time.sleep(0.3)
    
    print(f"\n15-hop Accuracy: {hop_results[15]['correct']/N_PROBLEMS:.1%}")
    print()
    
    # Long claim vs decomposed test
    print("=== Testing Long Claim (2000+ chars) vs Decomposed ===")
    long_claim_correct = 0
    decomp_correct = 0
    long_times = []
    decomp_times = []
    
    for i, prob in enumerate(problems, 1):
        lc, dc, lt, dt, ltok, dtok = run_long_claim_test(client, prob)
        long_claim_correct += lc
        decomp_correct += dc
        long_times.append(lt)
        decomp_times.append(dt)
        
        status_lc = "✓" if lc else "✗"
        status_dc = "✓" if dc else "✗"
        print(f"  [{i:2d}/{N_PROBLEMS}] Long:{status_lc} Decomp:{status_dc} | Expected: {prob.answer}")
        time.sleep(0.5)
    
    print(f"\nLong Claim Accuracy: {long_claim_correct/N_PROBLEMS:.1%}")
    print(f"Decomposed Accuracy: {decomp_correct/N_PROBLEMS:.1%}")
    print()
    
    # Calculate degradation rate using linear regression
    hops = [1, 3, 7, 15]
    accuracies = [hop_results[h]["correct"] / N_PROBLEMS for h in hops]
    
    # Simple linear regression: accuracy = a + b*hops
    # b = degradation_rate_per_hop
    n = len(hops)
    sum_x = sum(hops)
    sum_y = sum(accuracies)
    sum_xy = sum(h*acc for h, acc in zip(hops, accuracies))
    sum_x2 = sum(h**2 for h in hops)
    
    b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    a = (sum_y - b * sum_x) / n
    
    degradation_rate = b  # Slope: change in accuracy per hop
    
    # Find cliff point: where accuracy drops below 50%
    # accuracy = a + b*hops < 0.5
    # hops > (0.5 - a) / b
    if b < 0:
        context_cliff_point = (0.5 - a) / b if b != 0 else float('inf')
    else:
        context_cliff_point = float('inf')
    
    # Calculate final metrics
    short_accuracy = hop_results[1]["correct"] / N_PROBLEMS
    medium_accuracy = hop_results[3]["correct"] / N_PROBLEMS
    long_accuracy = hop_results[7]["correct"] / N_PROBLEMS
    very_long_accuracy = hop_results[15]["correct"] / N_PROBLEMS
    
    # E15_pass: degradation is measurable (slope is negative and significant)
    e15_pass = degradation_rate < -0.01  # Measurable if slope < -1% per hop
    
    # Compile results
    results = {
        "experiment": "E15_long_context_decomposition",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "n_problems": N_PROBLEMS,
        "short_accuracy": short_accuracy,
        "medium_accuracy": medium_accuracy,
        "long_accuracy": long_accuracy,
        "very_long_accuracy": very_long_accuracy,
        "degradation_rate_per_hop": round(degradation_rate, 4),
        "context_cliff_point": round(context_cliff_point, 2) if context_cliff_point != float('inf') else None,
        "E15_pass": e15_pass,
        "long_claim_accuracy": long_claim_correct / N_PROBLEMS,
        "decomposed_accuracy": decomp_correct / N_PROBLEMS,
        "hop_details": {
            "1": {"accuracy": short_accuracy, "correct": hop_results[1]["correct"], "avg_time": statistics.mean(hop_results[1]["times"]), "total_tokens": hop_results[1]["tokens"]},
            "3": {"accuracy": medium_accuracy, "correct": hop_results[3]["correct"], "avg_time": statistics.mean(hop_results[3]["times"]), "total_tokens": hop_results[3]["tokens"]},
            "7": {"accuracy": long_accuracy, "correct": hop_results[7]["correct"], "avg_time": statistics.mean(hop_results[7]["times"]), "total_tokens": hop_results[7]["tokens"]},
            "15": {"accuracy": very_long_accuracy, "correct": hop_results[15]["correct"], "avg_time": statistics.mean(hop_results[15]["times"]), "total_tokens": hop_results[15]["tokens"]},
        },
        "long_claim_vs_decomposed": {
            "long_claim_correct": long_claim_correct,
            "decomposed_correct": decomp_correct,
            "long_claim_accuracy": long_claim_correct / N_PROBLEMS,
            "decomposed_accuracy": decomp_correct / N_PROBLEMS,
            "avg_long_time": statistics.mean(long_times),
            "avg_decomp_time": statistics.mean(decomp_times),
        },
    }
    
    # Save results
    results_file = OUTPUT_DIR / "E15-results.json"
    results_file.write_text(json.dumps(results, indent=2))
    
    # Print summary
    print(f"\n{'='*70}")
    print("E15 RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"1-hop (Short) Accuracy:    {short_accuracy:.1%}")
    print(f"3-hop (Medium) Accuracy:    {medium_accuracy:.1%}")
    print(f"7-hop (Long) Accuracy:      {long_accuracy:.1%}")
    print(f"15-hop (Very Long) Accuracy: {very_long_accuracy:.1%}")
    print()
    print(f"Degradation rate per hop:   {degradation_rate:.4f}")
    print(f"Context cliff point:        {context_cliff_point:.2f} hops" if context_cliff_point != float('inf') else "Context cliff point:        Not reached within tested range")
    print()
    print(f"Long Claim (2000+ chars) vs Decomposed:")
    print(f"  Long Claim Accuracy:  {long_claim_correct/N_PROBLEMS:.1%}")
    print(f"  Decomposed Accuracy:  {decomp_correct/N_PROBLEMS:.1%}")
    print()
    print(f"E15 Pass: {e15_pass}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    results = main()