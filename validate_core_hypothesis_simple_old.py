#!/usr/bin/env python3
"""
Simplified Core Hypothesis Validation
Uses direct CLI subprocess calls for maximum compatibility
"""

import subprocess
import json
import time
import re
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics


@dataclass
class MathProblem:
    """A math problem with expected answer"""

    id: str
    question: str
    expected_answer: float
    category: str
    difficulty: str


@dataclass
class Result:
    """Benchmark result"""

    problem_id: str
    system: str
    correct: bool
    time: float
    error: str = None


def extract_number(text: str) -> float:
    """Extract numeric answer from text"""
    # Look for common patterns
    patterns = [
        r"answer is (\d+(?:\.\d+)?)",
        r"= (\d+(?:\.\d+)?)",
        r"\$(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars|meters|cubic meters|eggs|pages|passengers)",
        r"total[:\s]+(\d+(?:,\d{3})*(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return float(match.group(1).replace(",", ""))

    # Fallback: last number in text
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if numbers:
        return float(numbers[-1])

    return 0.0


def get_problems() -> List[MathProblem]:
    """Get test problems"""
    return [
        MathProblem(
            id="gsm8k_janet_eggs",
            question="Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the remainder at $2 each. How much does she make?",
            expected_answer=18.0,
            category="arithmetic",
            difficulty="easy",
        ),
        MathProblem(
            id="gsm8k_swimming_pool",
            question="A swimming pool is 50 meters long and 25 meters wide. If the water depth is 2 meters, how many cubic meters of water does it contain?",
            expected_answer=2500.0,
            category="geometry",
            difficulty="easy",
        ),
        MathProblem(
            id="gsm8k_fruit_stand",
            question="Apples cost $3 per pound and oranges cost $2 per pound. If John buys 5 pounds of apples and 8 pounds of oranges, how much does he spend?",
            expected_answer=31.0,
            category="arithmetic",
            difficulty="easy",
        ),
        MathProblem(
            id="gsm8k_bus_problem",
            question="A bus starts with 45 passengers. At the first stop, 12 get off and 8 get on. At the second stop, 15 get off and 20 get on. How many are on the bus?",
            expected_answer=46.0,
            category="arithmetic",
            difficulty="medium",
        ),
        MathProblem(
            id="gsm8k_reading_challenge",
            question="Sarah wants to read a 480-page book in 8 days reading the same number each day. How many pages per day?",
            expected_answer=60.0,
            category="arithmetic",
            difficulty="easy",
        ),
    ]


def run_baseline(problem: MathProblem) -> Result:
    """Run baseline: simple prompt through CLI"""
    start = time.time()

    try:
        # Use 'conjecture prompt' for baseline
        prompt = f"Solve: {problem.question} Provide just the numeric answer."

        # Run through CLI
        result = subprocess.run(
            ["python", "conjecture", "prompt", prompt],
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
        )

        elapsed = time.time() - start

        if result.returncode != 0:
            return Result(
                problem_id=problem.id,
                system="baseline",
                correct=False,
                time=elapsed,
                error=f"CLI error: {result.stderr[:200]}",
            )

        # Extract answer
        answer = extract_number(result.stdout)
        correct = abs(answer - problem.expected_answer) < 0.01

        return Result(
            problem_id=problem.id, system="baseline", correct=correct, time=elapsed
        )

    except Exception as e:
        return Result(
            problem_id=problem.id,
            system="baseline",
            correct=False,
            time=time.time() - start,
            error=str(e),
        )


def run_conjecture(problem: MathProblem) -> Result:
    """Run Conjecture: claim-based reasoning"""
    start = time.time()

    try:
        # First create claim
        create_result = subprocess.run(
            ["python", "conjecture", "create", problem.question, "--confidence", "0.5"],
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
        )

        elapsed = time.time() - start

        if create_result.returncode != 0:
            return Result(
                problem_id=problem.id,
                system="conjecture",
                correct=False,
                time=elapsed,
                error=f"Create claim failed: {create_result.stderr[:200]}",
            )

        # Extract claim ID
        claim_id_match = re.search(r"(c\d{7})", create_result.stdout)
        if not claim_id_match:
            return Result(
                problem_id=problem.id,
                system="conjecture",
                correct=False,
                time=elapsed,
                error="Could not extract claim ID",
            )

        claim_id = claim_id_match.group(1)

        # Now analyze/evaluate the claim
        analyze_result = subprocess.run(
            ["python", "conjecture", "analyze", claim_id],
            capture_output=True,
            text=True,
            timeout=60,
            encoding="utf-8",
            errors="replace",
        )

        elapsed = time.time() - start

        if analyze_result.returncode != 0:
            return Result(
                problem_id=problem.id,
                system="conjecture",
                correct=False,
                time=elapsed,
                error=f"Analyze failed: {analyze_result.stderr[:200]}",
            )

        # Extract answer from analysis
        answer = extract_number(analyze_result.stdout)
        correct = abs(answer - problem.expected_answer) < 0.01

        return Result(
            problem_id=problem.id, system="conjecture", correct=correct, time=elapsed
        )

    except Exception as e:
        return Result(
            problem_id=problem.id,
            system="conjecture",
            correct=False,
            time=time.time() - start,
            error=str(e),
        )


def main():
    """Run benchmark"""
    print("=" * 60)
    print("CORE HYPOTHESIS VALIDATION - SIMPLIFIED")
    print("=" * 60)
    print("Hypothesis: Conjecture improves intelligence & truthfulness")
    print("Benchmark: GSM8K-style math (simplified)")
    print("=" * 60)
    print()

    problems = get_problems()
    print(f"Running {len(problems)} problems...")
    print()

    baseline_results = []
    conjecture_results = []

    for i, problem in enumerate(problems, 1):
        print(f"[{i}/{len(problems)}] {problem.id}")
        print(f"  Q: {problem.question[:60]}...")

        # Baseline
        print(f"  Baseline...", end="", flush=True)
        baseline_result = run_baseline(problem)
        baseline_results.append(baseline_result)
        status = "PASS" if baseline_result.correct else "FAIL"
        print(f" {status} ({baseline_result.time:.1f}s)")
        if baseline_result.error:
            print(f"    Error: {baseline_result.error[:80]}")

        # Conjecture
        print(f"  Conjecture...", end="", flush=True)
        conjecture_result = run_conjecture(problem)
        conjecture_results.append(conjecture_result)
        status = "PASS" if conjecture_result.correct else "FAIL"
        print(f" {status} ({conjecture_result.time:.1f}s)")
        if conjecture_result.error:
            print(f"    Error: {conjecture_result.error[:80]}")

        print()

    # Calculate metrics
    baseline_correct = sum(r.correct for r in baseline_results)
    conjecture_correct = sum(r.correct for r in conjecture_results)

    baseline_accuracy = baseline_correct / len(baseline_results)
    conjecture_accuracy = conjecture_correct / len(conjecture_results)

    baseline_times = [r.time for r in baseline_results if r.time > 0]
    conjecture_times = [r.time for r in conjecture_results if r.time > 0]

    baseline_avg_time = statistics.mean(baseline_times) if baseline_times else 0
    conjecture_avg_time = statistics.mean(conjecture_times) if conjecture_times else 0

    improvement = conjecture_accuracy - baseline_accuracy
    improvement_pct = (
        (improvement / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
    )

    # Print results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(
        f"Baseline:    {baseline_accuracy:.1%} ({baseline_correct}/{len(baseline_results)}) @ {baseline_avg_time:.1f}s avg"
    )
    print(
        f"Conjecture:  {conjecture_accuracy:.1%} ({conjecture_correct}/{len(conjecture_results)}) @ {conjecture_avg_time:.1f}s avg"
    )
    print()
    print(f"Improvement: {improvement:+.1%} ({improvement_pct:+.1f}%)")
    print()

    if improvement > 0.05:
        print("HYPOTHESIS SUPPORTED")
        print("  Conjecture shows meaningful accuracy improvement (>5%)")
    else:
        print("HYPOTHESIS NOT SUPPORTED")
        print("  Conjecture does not show meaningful improvement")
    print("=" * 60)

    # Save results
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_problems": len(problems),
        "baseline": {
            "accuracy": baseline_accuracy,
            "correct": baseline_correct,
            "total": len(baseline_results),
            "avg_time": baseline_avg_time,
            "results": [asdict(r) for r in baseline_results],
        },
        "conjecture": {
            "accuracy": conjecture_accuracy,
            "correct": conjecture_correct,
            "total": len(conjecture_results),
            "avg_time": conjecture_avg_time,
            "results": [asdict(r) for r in conjecture_results],
        },
        "comparison": {
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "hypothesis_supported": improvement > 0.05,
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hypothesis_validation_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Results saved to: {filename}")


if __name__ == "__main__":
    main()
