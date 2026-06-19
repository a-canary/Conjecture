#!/usr/bin/env python3
"""
MATH Benchmark — O-0008 Validation

Tests decomposition-based reasoning on competition mathematics (MATH dataset).
MATH covers 12 subject areas at 5 difficulty levels — much harder than GSM8K.

Dataset: HuggingFaceH4/MATH (test split)
n=100 problems from test set

Expected pattern: MATH is a HARD REASONING task.
- Baseline likely 40-60% for competition-level problems
- Decomposition should help: +5-15pp (similar to BBH +9pp)
- The O-0008 report predicted: "Potential benefit if baseline <90% (reasoning task)"

Key difference from GSM8K: GSM8K (92% baseline) showed +1pp (saturated).
MATH is harder → should show stronger decomposition benefit.

References:
- O-0008_VALIDATION_REPORT.md — hypothesis framing
- experiments/gsm8k_three_prompt_benchmark.py — related math benchmark
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import httpx
from datasets import load_dataset


# =============================================================================
# CONFIGURATION
# =============================================================================

N_PROBLEMS = int(os.environ.get("BENCHMARK_N", "100"))
API_URL = os.environ.get("CHUTES_URL", "https://llm.chutes.ai/v1/chat/completions")
API_KEY = os.environ.get("CHUTES_API_KEY", "")
MODEL = os.environ.get("BENCHMARK_MODEL", "deepseek-ai/DeepSeek-V3.2-TEE")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "800"))


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

DIRECT_SYSTEM = """You are an expert mathematics problem solver.
Read the problem carefully, solve it step by step, and give your final answer.
Show your work briefly, then state the answer clearly."""

DECOMP_SYSTEM = """You are an expert mathematics problem solver using decomposition.

For complex competition math problems, break down the solution into clear steps:
1. Understand what the problem asks (identify the goal)
2. Identify relevant mathematical concepts and techniques
3. Break the problem into sub-problems if needed
4. Solve each sub-problem systematically
5. Combine results to get the final answer

Be rigorous. Show enough work to verify your reasoning."""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BenchmarkResult:
    task_id: str
    model_name: str
    using_conjecture: bool
    problem: str
    level: str
    problem_type: str
    response: str
    expected_answer: str
    correct: bool
    score: float
    execution_time: float
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    benchmark_name: str
    model_name: str
    using_conjecture: bool
    total_tasks: int
    correct_answers: int
    accuracy: float
    average_time: float
    total_time: float


# =============================================================================
# LLM CLIENT
# =============================================================================

async def call_llm(prompt: str, system: str = "", max_tokens: int = MAX_TOKENS) -> tuple[str, float]:
    """Call DeepSeek-V3 via Chutes API. Returns (response, time_seconds)."""
    if not API_KEY:
        raise RuntimeError("CHUTES_API_KEY not set")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    async with httpx.AsyncClient() as client:
        start = asyncio.get_event_loop().time()
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            timeout=120.0,
        )
        elapsed = asyncio.get_event_loop().time() - start
        if resp.status_code != 200:
            raise RuntimeError(f"API error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content, elapsed


# =============================================================================
# ANSWER EXTRACTION AND CHECKING
# =============================================================================

def extract_answer_from_solution(solution: str) -> str:
    """Extract the final boxed answer from a LaTeX-formatted solution."""
    # Match \boxed{answer} or the final numeric/text answer
    boxed = re.search(r'\\boxed\{([^{}]+)\}', solution)
    if boxed:
        return boxed.group(1).strip()
    # Fallback: last line of solution
    lines = [l.strip() for l in solution.split('\n') if l.strip()]
    if lines:
        return lines[-1].strip()
    return solution.strip()


def normalize_math_answer(text: str) -> str:
    """Normalize a math answer for comparison."""
    text = text.lower().strip()
    # Remove common LaTeX commands
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', text)
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = re.sub(r'[^\w\s./-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def check_math_answer(response: str, expected: str) -> bool:
    """Check if the model's answer matches the expected answer."""
    resp_norm = normalize_math_answer(response)
    exp_norm = normalize_math_answer(expected)

    # Direct substring match
    if exp_norm in resp_norm:
        return True

    # Numeric comparison
    resp_nums = set(re.findall(r'-?\d+\.?\d*', response))
    exp_nums = set(re.findall(r'-?\d+\.?\d*', expected))
    if exp_nums and resp_nums:
        # Allow floating point tolerance
        try:
            exp_num = float(list(exp_nums)[0])
            for r in resp_nums:
                try:
                    if abs(float(r) - exp_num) < 0.01:
                        return True
                except ValueError:
                    pass
        except ValueError:
            pass

    return False


# =============================================================================
# PROBLEM LOADING
# =============================================================================

def load_math_problems(n: int = N_PROBLEMS) -> List[dict]:
    """Load MATH problems from test set."""
    ds = load_dataset("HuggingFaceH4/MATH", split="test")
    problems = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        expected = extract_answer_from_solution(item["solution"])
        problems.append({
            "id": f"math_{i}",
            "problem": item["problem"],
            "level": item["level"],
            "type": item["type"],
            "expected": expected,
        })
    return problems


# =============================================================================
# EVALUATION
# =============================================================================

async def evaluate_problem(
    problem: dict,
    use_conjecture: bool,
    model: str,
) -> BenchmarkResult:
    """Evaluate a single MATH problem."""
    prompt = problem["problem"]
    system = DECOMP_SYSTEM if use_conjecture else DIRECT_SYSTEM

    try:
        response, elapsed = await call_llm(prompt, system)
        correct = check_math_answer(response, problem["expected"])
    except Exception as e:
        response = ""
        elapsed = 0.0
        correct = False

    return BenchmarkResult(
        task_id=problem["id"],
        model_name=model,
        using_conjecture=use_conjecture,
        problem=problem["problem"],
        level=problem["level"],
        problem_type=problem["type"],
        response=response[:500],
        expected_answer=problem["expected"],
        correct=correct,
        score=1.0 if correct else 0.0,
        execution_time=elapsed,
    )


async def run_evaluation(problems: List[dict], use_conjecture: bool) -> tuple[List[BenchmarkResult], BenchmarkSummary]:
    """Run evaluation on all problems."""
    results: List[BenchmarkResult] = []
    total_time = 0.0

    for i, problem in enumerate(problems):
        result = await evaluate_problem(problem, use_conjecture, MODEL)
        results.append(result)
        total_time += result.execution_time
        if (i + 1) % 10 == 0:
            acc = sum(r.correct for r in results) / len(results) * 100
            print(f"  [{i+1}/{len(problems)}] accuracy: {acc:.1f}%")

    correct = sum(r.correct for r in results)
    accuracy = correct / len(results) * 100 if results else 0

    summary = BenchmarkSummary(
        benchmark_name="MATH",
        model_name=MODEL,
        using_conjecture=use_conjecture,
        total_tasks=len(results),
        correct_answers=correct,
        accuracy=accuracy,
        average_time=total_time / len(results) if results else 0,
        total_time=total_time,
    )
    return results, summary


# =============================================================================
# MAIN
# =============================================================================

async def run_benchmark() -> dict:
    print(f"MATH Benchmark — O-0008 Validation")
    print(f"=" * 70)
    print(f"Model: {MODEL}")
    print(f"Problems: {N_PROBLEMS}")
    print(f"API: {API_URL}")

    problems = load_math_problems(N_PROBLEMS)
    print(f"\nLoaded {len(problems)} MATH problems")

    print("\n--- Direct Prompting ---")
    direct_results, direct_summary = await run_evaluation(problems, use_conjecture=False)
    print(f"Direct accuracy: {direct_summary.accuracy:.1f}% ({direct_summary.correct_answers}/{direct_summary.total_tasks})")

    print("\n--- Decomposition (Conjecture) ---")
    decomp_results, decomp_summary = await run_evaluation(problems, use_conjecture=True)
    print(f"Conjecture accuracy: {decomp_summary.accuracy:.1f}% ({decomp_summary.correct_answers}/{decomp_summary.total_tasks})")

    improvement = decomp_summary.accuracy - direct_summary.accuracy
    print(f"\n--- Result ---")
    print(f"Improvement: {improvement:+.1f}pp")

    if improvement > 5:
        conclusion = "SUCCESS: Decomposition significantly improves MATH accuracy"
    elif improvement > 2:
        conclusion = "PROMISING: Decomposition shows mild improvement on MATH"
    elif improvement > -2:
        conclusion = "NEUTRAL: No significant difference on MATH"
    else:
        conclusion = "CHALLENGE: Decomposition hurts MATH accuracy"

    print(f"CONCLUSION: {conclusion}")

    results = {
        "benchmark": "MATH",
        "model": MODEL,
        "n_problems": len(problems),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "direct": asdict(direct_summary),
        "conjecture": asdict(decomp_summary),
        "improvement_pp": improvement,
        "conclusion": conclusion,
        "individual_results": [asdict(r) for r in direct_results + decomp_results],
    }

    results_dir = Path("experiments/results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"math_o0008_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {results_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100, help="Number of problems")
    parser.add_argument("--model", type=str, default=MODEL)
    args = parser.parse_args()
    os.environ["BENCHMARK_N"] = str(args.n)
    os.environ["BENCHMARK_MODEL"] = args.model
    asyncio.run(run_benchmark())
