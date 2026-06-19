#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
BBH (Big-Bench Hard) Benchmark: Direct vs Decomposition

BBH contains 23 challenging tasks from BIG-Bench that are difficult for language models.
Tests complex reasoning, symbolic manipulation, and multi-step inference.

Hypothesis: Decomposition should help significantly (hardest reasoning tasks)
Expected: Strong positive improvement
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from datasets import load_dataset
from openai import AsyncOpenAI


MODEL = os.environ.get("BENCHMARK_MODEL", "deepseek/deepseek-chat-v3-0324")
N_PROBLEMS = int(os.environ.get("BENCHMARK_N", "100"))
BBH_TASK = os.environ.get("BBH_TASK", "logical_deduction_three_objects")  # Default task
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")


# =============================================================================
# PROMPTS
# =============================================================================

DIRECT_SYSTEM = """You are a reasoning assistant for challenging logical problems.
Read the problem carefully and give your answer.
For multiple choice, give just the letter or option."""

DECOMPOSITION_SYSTEM = """You are a careful reasoning assistant for hard problems.
For each problem:

1. PARSE: Break down what the problem is asking
2. IDENTIFY: What type of reasoning is needed? (logical, mathematical, symbolic)
3. DECOMPOSE: Split into smaller sub-problems if needed
4. SOLVE: Work through step by step
5. VERIFY: Check your answer against the constraints
6. ANSWER: Give your final answer

Show your reasoning, then state the final answer clearly."""


# =============================================================================
# CLIENT
# =============================================================================

class BenchmarkClient:
    def __init__(self, model: str):
        self.model = model
        self._client = AsyncOpenAI(
            api_key=OPENROUTER_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        self.total_tokens = 0

    async def generate(
        self,
        prompt: str,
        system: str,
        max_tokens: int = 500
    ) -> Tuple[str, int]:
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0
            self.total_tokens += tokens
            return content, tokens
        except Exception as e:
            print(f"  API error: {e}")
            return "", 0


# =============================================================================
# DATA
# =============================================================================

def load_bbh_problems(task: str, limit: int) -> List[dict]:
    """Load BBH task."""
    print(f"Loading BBH task: {task} (limit={limit})...")

    try:
        # BBH tasks are in the lukaemon/bbh dataset
        ds = load_dataset("lukaemon/bbh", task)
        split = "test" if "test" in ds else list(ds.keys())[0]
        data = ds[split]
    except Exception as e:
        print(f"Error loading task {task}: {e}")
        print("Using default task: logical_deduction_three_objects")
        ds = load_dataset("lukaemon/bbh", "logical_deduction_three_objects")
        split = "test" if "test" in ds else list(ds.keys())[0]
        data = ds[split]

    problems = []
    for i, item in enumerate(data):
        if i >= limit:
            break

        problems.append({
            "id": f"bbh_{task}_{i}",
            "input": item["input"],
            "target": item["target"]
        })

    print(f"Loaded {len(problems)} problems from task '{task}'")
    return problems


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_answer(text: str, target: str) -> Optional[str]:
    """Extract answer from response."""
    if not text:
        return None

    text = text.strip()

    # Try to find exact target match
    if target.lower() in text.lower():
        return target

    # Look for common answer patterns
    patterns = [
        r'(?:answer|solution|result)[:\s]+([^\n.]+)',
        r'(?:final answer)[:\s]+([^\n.]+)',
        r'therefore[,\s]+([^\n.]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Check if it matches target (case-insensitive, flexible)
            if answer.lower() == target.lower():
                return target
            # Return extracted answer anyway for comparison
            return answer

    # Look for letter answers (A, B, C...)
    letter_match = re.search(r'\b([A-E])\b', text.upper())
    if letter_match:
        return letter_match.group(1)

    # Return last line as fallback
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        return lines[-1]

    return None


def check_answer(predicted: Optional[str], expected: str) -> bool:
    """Check if predicted matches expected."""
    if predicted is None:
        return False

    # Normalize both
    pred_norm = predicted.lower().strip().strip('.')
    exp_norm = expected.lower().strip().strip('.')

    # Exact match
    if pred_norm == exp_norm:
        return True

    # Check if expected is contained in predicted
    if exp_norm in pred_norm:
        return True

    # Check if predicted is contained in expected (for verbose answers)
    if pred_norm in exp_norm:
        return True

    return False


# =============================================================================
# RUNNER
# =============================================================================

@dataclass
class MethodResult:
    method: str
    correct: int
    total: int
    accuracy: float
    avg_time: float
    total_tokens: int
    extraction_failures: int


async def run_method(
    client: BenchmarkClient,
    problems: List[dict],
    system_prompt: str,
    method_name: str
) -> MethodResult:
    """Run benchmark with specific method."""
    print(f"\n{'='*60}")
    print(f"METHOD: {method_name}")
    print(f"{'='*60}")

    correct = 0
    extraction_failures = 0
    times = []

    for i, prob in enumerate(problems):
        start = time.time()

        response, tokens = await client.generate(
            prompt=prob["input"],
            system=system_prompt,
            max_tokens=600 if method_name == "DECOMPOSITION" else 200
        )

        elapsed = time.time() - start
        times.append(elapsed)

        predicted = extract_answer(response, prob["target"])
        expected = prob["target"]

        if predicted is None:
            extraction_failures += 1
            is_correct = False
        else:
            is_correct = check_answer(predicted, expected)

        if is_correct:
            correct += 1

        # Progress every 10 or on errors
        if (i + 1) % 10 == 0 or (predicted is None) or not is_correct:
            status = "EXTRACTION_FAIL" if predicted is None else ("✓" if is_correct else f"✗ exp={expected[:20]} got={str(predicted)[:20]}")
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):5.1f}%  {status}")

        await asyncio.sleep(0.3)  # Rate limiting

    return MethodResult(
        method=method_name,
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        avg_time=round(sum(times) / len(times), 2),
        total_tokens=client.total_tokens,
        extraction_failures=extraction_failures
    )


async def run_benchmark():
    """Run full BBH benchmark."""
    print("\n" + "="*70)
    print("BBH (BIG-BENCH HARD) BENCHMARK")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Task: {BBH_TASK}")
    print(f"Problems: {N_PROBLEMS}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    problems = load_bbh_problems(BBH_TASK, N_PROBLEMS)

    # Run direct baseline
    client_direct = BenchmarkClient(MODEL)
    direct_result = await run_method(
        client_direct, problems, DIRECT_SYSTEM, "DIRECT"
    )

    # Run decomposition
    client_decomp = BenchmarkClient(MODEL)
    decomp_result = await run_method(
        client_decomp, problems, DECOMPOSITION_SYSTEM, "DECOMPOSITION"
    )

    # Results summary
    improvement = decomp_result.accuracy - direct_result.accuracy

    print("\n" + "="*70)
    print("BBH BENCHMARK RESULTS")
    print("="*70)
    print(f"\n{'Method':<20} {'Correct':>10} {'Accuracy':>10} {'Avg Time':>10} {'Tokens':>12}")
    print("-"*62)
    print(f"{'Direct':<20} {direct_result.correct:>6}/{direct_result.total:<3} {direct_result.accuracy:>9.1f}% {direct_result.avg_time:>9.2f}s {direct_result.total_tokens:>12,}")
    print(f"{'Decomposition':<20} {decomp_result.correct:>6}/{decomp_result.total:<3} {decomp_result.accuracy:>9.1f}% {decomp_result.avg_time:>9.2f}s {decomp_result.total_tokens:>12,}")
    print("-"*62)
    print(f"{'Improvement':<20} {decomp_result.correct - direct_result.correct:>+10} {improvement:>+9.1f}pp")
    print()

    if direct_result.extraction_failures > 0 or decomp_result.extraction_failures > 0:
        print(f"Extraction failures: Direct={direct_result.extraction_failures}, Decomp={decomp_result.extraction_failures}")

    # Conclusion
    if improvement > 10:
        conclusion = "STRONG VALIDATION: Decomposition excels on hard reasoning"
        symbol = "✅✅"
    elif improvement > 5:
        conclusion = "VALIDATED: Decomposition improves hard reasoning"
        symbol = "✅"
    elif improvement > 0:
        conclusion = "PARTIAL: Modest improvement"
        symbol = "⚠️"
    elif improvement > -5:
        conclusion = "INCONCLUSIVE: No significant difference"
        symbol = "⚠️"
    else:
        conclusion = "REGRESSION: Decomposition hurts accuracy"
        symbol = "❌"

    print(f"{symbol} {conclusion}")
    print("="*70)

    # Save results
    results = {
        "benchmark": "BBH",
        "task": BBH_TASK,
        "model": MODEL,
        "n_problems": N_PROBLEMS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "direct": asdict(direct_result),
        "decomposition": asdict(decomp_result),
        "improvement_pp": improvement,
        "conclusion": conclusion
    }

    results_dir = Path("experiments/results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"bbh_{BBH_TASK}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {results_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100, help="Number of problems")
    parser.add_argument("--model", type=str, default=MODEL, help="Model to use")
    parser.add_argument("--task", type=str, default=BBH_TASK, help="BBH task name")
    args = parser.parse_args()

    os.environ["BENCHMARK_N"] = str(args.n)
    os.environ["BENCHMARK_MODEL"] = args.model
    if args.task:
        os.environ["BBH_TASK"] = args.task

    asyncio.run(run_benchmark())
