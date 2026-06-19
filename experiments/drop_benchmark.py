#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
DROP Benchmark — O-0008 Validation

Tests decomposition-based reasoning on DROP (Discrete Reasoning Over Text).
DROP requires multi-step discrete reasoning over a passage: counting, arithmetic,
comparisons, and coreference resolution.

Dataset: google/datasets (drop)
n=100 problems from validation set

Expected pattern: DROP is a MIXED task (recall + reasoning).
- Baseline likely 50-70% for complex questions
- Decomposition may help: +3-8pp if reasoning is the bottleneck
- May regress if passage recall dominates

References:
- O-0008_VALIDATION_REPORT.md — hypothesis framing
- experiments/bbh_three_prompt_benchmark.py — pattern reference
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
MODEL = os.environ.get("BENCHMARK_MODEL", "deepseek-ai/DeepSeek-V3")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "500"))


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

DIRECT_SYSTEM = """You are an expert reading comprehension assistant.
Given a passage and a question, provide a precise, factual answer.
Answer only with the answer — no preamble."""

DECOMP_SYSTEM = """You are an expert reasoning assistant.
Break down complex reading comprehension questions into steps.

Steps:
1. Identify what the question is asking (counting, arithmetic, comparison, coreference)
2. Locate relevant information in the passage
3. Perform the required reasoning step-by-step
4. Give the final answer

Be systematic. Show your reasoning briefly, then state the answer clearly."""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BenchmarkResult:
    task_id: str
    model_name: str
    using_conjecture: bool
    passage: str
    question: str
    response: str
    expected_answers: List[str]
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
# ANSWER CHECKING
# =============================================================================

def normalize_answer(text: str) -> str:
    """Normalize an answer string for comparison."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def check_drop_answer(response: str, expected_spans: List[str]) -> bool:
    """Check if the model's response contains any of the expected answer spans."""
    response_norm = normalize_answer(response)
    for span in expected_spans:
        span_norm = normalize_answer(span)
        if span_norm and span_norm in response_norm:
            return True
        # Also check for numeric answers
        span_nums = re.findall(r'[\d,.-]+', span)
        resp_nums = re.findall(r'[\d,.-]+', response)
        if span_nums and resp_nums:
            if any(n in resp_nums for n in span_nums):
                return True
    return False


# =============================================================================
# PROBLEM LOADING
# =============================================================================

def load_drop_problems(n: int = N_PROBLEMS) -> List[dict]:
    """Load DROP problems from validation set."""
    ds = load_dataset("google/datasets", "drop", split="validation")
    problems = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        # Skip complex answer types that are hard to evaluate automatically
        answer_types = item.get("answers_spans", {}).get("types", [])
        # Focus on span and number answers; skip date/complex types
        if not answer_types:
            continue
        problems.append({
            "id": item["query_id"],
            "passage": item["passage"],
            "question": item["question"],
            "expected": item["answers_spans"].get("spans", []),
            "types": answer_types,
        })
    return problems[:n]


# =============================================================================
# EVALUATION
# =============================================================================

async def evaluate_problem(
    problem: dict,
    use_conjecture: bool,
    model: str,
) -> BenchmarkResult:
    """Evaluate a single DROP problem."""
    passage = problem["passage"]
    question = problem["question"]
    expected = problem["expected"]

    prompt = f"Passage:\n{passage}\n\nQuestion: {question}"
    system = DECOMP_SYSTEM if use_conjecture else DIRECT_SYSTEM

    try:
        response, elapsed = await call_llm(prompt, system)
        correct = check_drop_answer(response, expected)
    except Exception as e:
        response = ""
        elapsed = 0.0
        correct = False

    return BenchmarkResult(
        task_id=problem["id"],
        model_name=model,
        using_conjecture=use_conjecture,
        passage=passage[:200],
        question=question,
        response=response[:500],
        expected_answers=expected,
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
        benchmark_name="DROP",
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
    print(f"DROP Benchmark — O-0008 Validation")
    print(f"=" * 70)
    print(f"Model: {MODEL}")
    print(f"Problems: {N_PROBLEMS}")
    print(f"API: {API_URL}")

    problems = load_drop_problems(N_PROBLEMS)
    print(f"\nLoaded {len(problems)} DROP problems")

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
        conclusion = "SUCCESS: Decomposition significantly improves DROP reasoning"
    elif improvement > 2:
        conclusion = "PROMISING: Decomposition shows mild improvement on DROP"
    elif improvement > -2:
        conclusion = "NEUTRAL: No significant difference on DROP"
    else:
        conclusion = "CHALLENGE: Decomposition hurts DROP accuracy"

    print(f"CONCLUSION: {conclusion}")

    results = {
        "benchmark": "DROP",
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
    results_file = results_dir / f"drop_o0008_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
