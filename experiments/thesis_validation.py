#!/usr/bin/env python3
"""
Thesis Validation Experiment

Core Thesis: Decomposition, assumption-validation, and exploration improves
LLM accuracy and reasoning on unseen problems.

Methodology:
1. Generate fresh, novel problems unlikely to be in training data
2. Compare direct prompting vs reasoning loop with claims
3. Use multiple problem types (math, logic, reasoning)
4. 200+ samples per condition for statistical validity
5. Clear success criteria: >5% improvement with p<0.05

Author: Director
Date: 2026-03-06
"""

import asyncio
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Problem generation templates for truly novel problems
NOVEL_MATH_TEMPLATES = [
    # Multi-step word problems with random numbers
    "A store sells {item} for ${price} each. If a customer buys {qty1} on Monday and {qty2} on Tuesday, and gets a {discount}% discount on the total, how much do they pay?",
    "A tank fills at {rate1} gallons per minute but leaks at {rate2} gallons per minute. If it starts with {initial} gallons and needs {target} gallons, how many minutes until full?",
    "Three workers can complete a job in {days1}, {days2}, and {days3} days respectively. If they all work together, how many days to complete the job?",
    "A train travels {dist1} miles at {speed1} mph, then {dist2} miles at {speed2} mph. What is the average speed for the entire journey?",
    "If {people1} people can paint {rooms1} rooms in {hours1} hours, how many hours would {people2} people need to paint {rooms2} rooms?",
]

NOVEL_LOGIC_TEMPLATES = [
    # Novel deduction problems
    "In a group of {n} people, each person shakes hands exactly once with every other person. How many handshakes occur in total?",
    "A password has {digits} digits, each from 0-9. If no digit can repeat and it must start with an odd number, how many valid passwords exist?",
    "You have {red} red balls and {blue} blue balls in a bag. What is the minimum number of balls you must draw to guarantee at least {min_same} balls of the same color?",
    "A sequence follows the rule: each term is {mult}x the previous term plus {add}. If the first term is {start}, what is the {nth} term?",
]

NOVEL_REASONING_TEMPLATES = [
    # Counterfactual reasoning
    "If doubling a number gives {double_result}, and adding {add_val} to that gives {final_result}, what was the original number?",
    "Alice is {years_diff} years older than Bob. In {future_years} years, Alice will be {mult}x as old as Bob is now. How old is Bob now?",
    "A rope is cut into {num_pieces} pieces. If the pieces have lengths in ratio {ratio}, and the longest piece is {longest} meters, what was the original length?",
]


@dataclass
class Problem:
    """A generated problem for validation."""
    id: str
    category: str
    question: str
    answer: Any
    reasoning_steps: List[str]
    difficulty: int  # 1-5


@dataclass
class ExperimentResult:
    """Result of running one condition."""
    condition: str
    problems_tested: int
    correct: int
    accuracy: float
    avg_response_time: float
    avg_tokens_used: int
    reasoning_depth: float  # avg iterations for reasoning loop


@dataclass
class ValidationReport:
    """Full validation report."""
    timestamp: str
    thesis: str
    conditions: List[ExperimentResult]
    baseline_accuracy: float
    reasoning_accuracy: float
    improvement_pp: float
    p_value: float
    conclusion: str


def generate_novel_problem(template: str, category: str, difficulty: int) -> Problem:
    """Generate a problem with random parameters that's unlikely to be in training."""
    # Use truly random values
    params = {
        'item': random.choice(['widgets', 'gadgets', 'sprockets', 'gizmos', 'thingamajigs']),
        'price': random.randint(7, 97),  # Odd prices less common
        'qty1': random.randint(3, 47),
        'qty2': random.randint(2, 38),
        'discount': random.randint(5, 35),
        'rate1': random.randint(3, 19),
        'rate2': random.randint(1, 7),
        'initial': random.randint(10, 90),
        'target': random.randint(100, 500),
        'days1': random.randint(4, 12),
        'days2': random.randint(5, 15),
        'days3': random.randint(6, 18),
        'dist1': random.randint(50, 200),
        'dist2': random.randint(30, 150),
        'speed1': random.randint(30, 80),
        'speed2': random.randint(20, 60),
        'people1': random.randint(2, 8),
        'rooms1': random.randint(3, 12),
        'hours1': random.randint(4, 16),
        'people2': random.randint(3, 10),
        'rooms2': random.randint(4, 15),
        'n': random.randint(5, 25),
        'digits': random.randint(3, 5),
        'red': random.randint(3, 10),
        'blue': random.randint(3, 10),
        'min_same': random.randint(2, 4),
        'mult': random.randint(2, 5),
        'add': random.randint(1, 10),
        'start': random.randint(1, 10),
        'nth': random.randint(4, 8),
        'double_result': random.randint(20, 100),
        'add_val': random.randint(5, 30),
        'final_result': random.randint(30, 150),
        'years_diff': random.randint(3, 15),
        'future_years': random.randint(5, 20),
        'longest': random.randint(5, 30),
        'num_pieces': random.randint(3, 6),
        'ratio': f"{random.randint(1,3)}:{random.randint(2,4)}:{random.randint(3,5)}",
    }

    # Format question
    try:
        question = template.format(**params)
    except KeyError:
        question = template

    # Generate unique ID
    prob_id = f"{category}_{random.randint(10000, 99999)}"

    # Compute answer (simplified - real impl would solve each type)
    answer = None
    reasoning = []

    if 'store sells' in template:
        total_qty = params['qty1'] + params['qty2']
        subtotal = total_qty * params['price']
        discount_amt = subtotal * params['discount'] / 100
        answer = round(subtotal - discount_amt, 2)
        reasoning = [
            f"Total items: {params['qty1']} + {params['qty2']} = {total_qty}",
            f"Subtotal: {total_qty} × ${params['price']} = ${subtotal}",
            f"Discount: ${subtotal} × {params['discount']}% = ${discount_amt}",
            f"Final: ${subtotal} - ${discount_amt} = ${answer}"
        ]
    elif 'handshakes' in template:
        n = params['n']
        answer = n * (n - 1) // 2
        reasoning = [
            f"Each person shakes hands with {n-1} others",
            f"Total: {n} × {n-1} = {n*(n-1)} (but each handshake counted twice)",
            f"Answer: {n*(n-1)} ÷ 2 = {answer}"
        ]
    elif 'doubling' in template.lower():
        # Work backwards
        original = (params['final_result'] - params['add_val']) // 2
        answer = original
        reasoning = [
            f"final_result = doubled + add_val",
            f"{params['final_result']} = doubled + {params['add_val']}",
            f"doubled = {params['final_result'] - params['add_val']}",
            f"original = {params['final_result'] - params['add_val']} ÷ 2 = {answer}"
        ]
    else:
        answer = "COMPUTE"  # Placeholder
        reasoning = ["Step-by-step solution needed"]

    return Problem(
        id=prob_id,
        category=category,
        question=question,
        answer=answer,
        reasoning_steps=reasoning,
        difficulty=difficulty
    )


def generate_problem_set(n_problems: int = 200) -> List[Problem]:
    """Generate a balanced set of novel problems."""
    problems = []
    categories = [
        ('math', NOVEL_MATH_TEMPLATES),
        ('logic', NOVEL_LOGIC_TEMPLATES),
        ('reasoning', NOVEL_REASONING_TEMPLATES),
    ]

    per_category = n_problems // len(categories)

    for category, templates in categories:
        for i in range(per_category):
            template = random.choice(templates)
            difficulty = random.randint(1, 5)
            prob = generate_novel_problem(template, category, difficulty)
            problems.append(prob)

    random.shuffle(problems)
    return problems


async def run_direct_baseline(problem: Problem, llm_client: Any) -> Tuple[str, float, int]:
    """Run direct prompting (no decomposition)."""
    prompt = f"""Solve this problem directly. Give only the final numerical answer.

Problem: {problem.question}

Answer:"""

    start = time.time()
    response = await llm_client.generate(prompt)
    elapsed = time.time() - start

    # Extract answer
    answer = response.strip()
    tokens = len(response.split())  # Rough estimate

    return answer, elapsed, tokens


async def run_reasoning_loop(problem: Problem, reasoning_loop: Any) -> Tuple[str, float, int, int]:
    """Run the full reasoning loop with decomposition."""
    start = time.time()
    result = await reasoning_loop.run(problem.question)
    elapsed = time.time() - start

    answer = result.response
    tokens = sum(len(str(tc)) for tc in result.tool_calls)  # Rough estimate
    iterations = result.iterations

    return answer, elapsed, tokens, iterations


def check_answer(predicted: str, expected: Any) -> bool:
    """Check if predicted answer matches expected."""
    if expected is None or expected == "COMPUTE":
        return False

    # Try numeric comparison
    try:
        pred_num = float(predicted.replace('$', '').replace(',', '').strip())
        exp_num = float(str(expected).replace('$', '').replace(',', '').strip())
        return abs(pred_num - exp_num) < 0.01
    except:
        pass

    # String comparison
    return str(predicted).strip().lower() == str(expected).strip().lower()


def calculate_p_value(baseline_correct: List[bool], reasoning_correct: List[bool]) -> float:
    """Calculate statistical significance using McNemar's test (simplified)."""
    # Count discordant pairs
    b = sum(1 for bl, rs in zip(baseline_correct, reasoning_correct) if bl and not rs)
    c = sum(1 for bl, rs in zip(baseline_correct, reasoning_correct) if not bl and rs)

    if b + c == 0:
        return 1.0

    # McNemar's chi-squared (simplified)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # Rough p-value approximation
    if chi2 > 10.83:
        return 0.001
    elif chi2 > 6.63:
        return 0.01
    elif chi2 > 3.84:
        return 0.05
    else:
        return 0.1


async def run_validation_experiment(
    llm_client: Any,
    reasoning_loop: Any,
    n_problems: int = 200
) -> ValidationReport:
    """Run the full validation experiment."""

    print(f"\n{'='*60}")
    print("THESIS VALIDATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Problems: {n_problems}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print()

    # Generate problems
    print("Generating novel problems...")
    problems = generate_problem_set(n_problems)
    print(f"Generated {len(problems)} problems")

    # Run baseline
    print("\n--- Running BASELINE (direct prompting) ---")
    baseline_results = []
    baseline_correct = []
    baseline_times = []
    baseline_tokens = []

    for i, prob in enumerate(problems):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(problems)}")
        try:
            answer, elapsed, tokens = await run_direct_baseline(prob, llm_client)
            correct = check_answer(answer, prob.answer)
            baseline_results.append(answer)
            baseline_correct.append(correct)
            baseline_times.append(elapsed)
            baseline_tokens.append(tokens)
        except Exception as e:
            print(f"  Error on problem {prob.id}: {e}")
            baseline_results.append("")
            baseline_correct.append(False)
            baseline_times.append(0)
            baseline_tokens.append(0)

    baseline_acc = sum(baseline_correct) / len(baseline_correct)
    print(f"\nBaseline accuracy: {baseline_acc:.1%} ({sum(baseline_correct)}/{len(baseline_correct)})")

    # Run reasoning loop
    print("\n--- Running REASONING LOOP (decomposition + exploration) ---")
    reasoning_results = []
    reasoning_correct = []
    reasoning_times = []
    reasoning_tokens = []
    reasoning_iterations = []

    for i, prob in enumerate(problems):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(problems)}")
        try:
            answer, elapsed, tokens, iters = await run_reasoning_loop(prob, reasoning_loop)
            correct = check_answer(answer, prob.answer)
            reasoning_results.append(answer)
            reasoning_correct.append(correct)
            reasoning_times.append(elapsed)
            reasoning_tokens.append(tokens)
            reasoning_iterations.append(iters)
        except Exception as e:
            print(f"  Error on problem {prob.id}: {e}")
            reasoning_results.append("")
            reasoning_correct.append(False)
            reasoning_times.append(0)
            reasoning_tokens.append(0)
            reasoning_iterations.append(0)

    reasoning_acc = sum(reasoning_correct) / len(reasoning_correct)
    print(f"\nReasoning accuracy: {reasoning_acc:.1%} ({sum(reasoning_correct)}/{len(reasoning_correct)})")

    # Calculate statistics
    improvement_pp = (reasoning_acc - baseline_acc) * 100
    p_value = calculate_p_value(baseline_correct, reasoning_correct)

    # Determine conclusion
    if improvement_pp > 5 and p_value < 0.05:
        conclusion = "THESIS VALIDATED: Statistically significant improvement (>5pp, p<0.05)"
    elif improvement_pp > 0 and p_value < 0.05:
        conclusion = "THESIS PARTIALLY VALIDATED: Significant but modest improvement"
    elif improvement_pp > 0:
        conclusion = "THESIS INCONCLUSIVE: Improvement observed but not statistically significant"
    elif improvement_pp < -5:
        conclusion = "THESIS REFUTED: Decomposition significantly DECREASES accuracy"
    else:
        conclusion = "THESIS NOT SUPPORTED: No meaningful improvement observed"

    # Build report
    report = ValidationReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        thesis="Decomposition, assumption-validation, and exploration improves LLM accuracy on unseen problems",
        conditions=[
            ExperimentResult(
                condition="baseline_direct",
                problems_tested=len(problems),
                correct=sum(baseline_correct),
                accuracy=baseline_acc,
                avg_response_time=statistics.mean(baseline_times) if baseline_times else 0,
                avg_tokens_used=int(statistics.mean(baseline_tokens)) if baseline_tokens else 0,
                reasoning_depth=1.0
            ),
            ExperimentResult(
                condition="reasoning_loop",
                problems_tested=len(problems),
                correct=sum(reasoning_correct),
                accuracy=reasoning_acc,
                avg_response_time=statistics.mean(reasoning_times) if reasoning_times else 0,
                avg_tokens_used=int(statistics.mean(reasoning_tokens)) if reasoning_tokens else 0,
                reasoning_depth=statistics.mean(reasoning_iterations) if reasoning_iterations else 0
            )
        ],
        baseline_accuracy=baseline_acc,
        reasoning_accuracy=reasoning_acc,
        improvement_pp=improvement_pp,
        p_value=p_value,
        conclusion=conclusion
    )

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline:  {baseline_acc:.1%}")
    print(f"Reasoning: {reasoning_acc:.1%}")
    print(f"Improvement: {improvement_pp:+.1f}pp")
    print(f"P-value: {p_value:.3f}")
    print()
    print(f"CONCLUSION: {conclusion}")
    print(f"{'='*60}\n")

    return report


def save_report(report: ValidationReport, path: Path):
    """Save the validation report to JSON."""
    data = {
        'timestamp': report.timestamp,
        'thesis': report.thesis,
        'conditions': [
            {
                'condition': c.condition,
                'problems_tested': c.problems_tested,
                'correct': c.correct,
                'accuracy': c.accuracy,
                'avg_response_time': c.avg_response_time,
                'avg_tokens_used': c.avg_tokens_used,
                'reasoning_depth': c.reasoning_depth
            }
            for c in report.conditions
        ],
        'baseline_accuracy': report.baseline_accuracy,
        'reasoning_accuracy': report.reasoning_accuracy,
        'improvement_pp': report.improvement_pp,
        'p_value': report.p_value,
        'conclusion': report.conclusion
    }
    path.write_text(json.dumps(data, indent=2))
    print(f"Report saved to: {path}")


if __name__ == "__main__":
    # Demo: show problem generation
    print("Generating sample problems...")
    problems = generate_problem_set(10)
    for p in problems[:5]:
        print(f"\n[{p.category}] {p.question}")
        print(f"  Answer: {p.answer}")
        print(f"  Steps: {p.reasoning_steps}")
