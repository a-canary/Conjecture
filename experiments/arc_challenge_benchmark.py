#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
ARC-Challenge Standard Benchmark: Direct vs Decomposition

ARC (AI2 Reasoning Challenge) tests science question answering with reasoning.
Challenge set contains only questions answered incorrectly by retrieval algorithms.

Hypothesis: Decomposition should help (multi-step reasoning task)
Expected: Positive improvement over direct baseline
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
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")


# =============================================================================
# PROMPTS
# =============================================================================

DIRECT_SYSTEM = """You are a science question answering assistant.
Answer with just the letter (A, B, C, D, or E) corresponding to the correct option.
Give only the letter, nothing else."""

DECOMPOSITION_SYSTEM = """You are a careful reasoning assistant for science questions.
For each question:

1. ANALYZE: What scientific concept or principle is being tested?
2. EVALUATE: Consider each option and apply relevant knowledge
3. REASON: Use logical elimination to rule out incorrect answers
4. VERIFY: Check your reasoning against the question
5. ANSWER: State the letter of the correct answer

Always end with "Answer: X" where X is the letter."""


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
        max_tokens: int = 300
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

def load_arc_problems(limit: int) -> List[dict]:
    """Load ARC-Challenge test set."""
    print(f"Loading ARC-Challenge dataset (limit={limit})...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    problems = []
    for i, item in enumerate(ds):
        if i >= limit:
            break

        # ARC format: choices with labels (A, B, C, D, E)
        choices_dict = item["choices"]
        labels = choices_dict["label"]
        texts = choices_dict["text"]

        # Build choices map
        choices = {}
        for label, text in zip(labels, texts):
            choices[label] = text

        # Answer key
        answer_key = item["answerKey"]

        problems.append({
            "id": item["id"],
            "question": item["question"],
            "choices": choices,
            "answer": answer_key
        })

    print(f"Loaded {len(problems)} problems")
    return problems


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_answer(text: str) -> Optional[str]:
    """Extract letter answer from response."""
    if not text:
        return None

    text = text.strip().upper()

    # Look for "Answer: X"
    match = re.search(r'ANSWER[:\s]+([A-E])', text)
    if match:
        return match.group(1)

    # Look for standalone letter at start
    match = re.search(r'^([A-E])[\.\)\s]', text)
    if match:
        return match.group(1)

    # Look for "The answer is X"
    match = re.search(r'THE\s+ANSWER\s+IS\s+([A-E])', text)
    if match:
        return match.group(1)

    # Any standalone letter
    match = re.search(r'\b([A-E])\b', text)
    if match:
        return match.group(1)

    return None


def check_answer(predicted: Optional[str], expected: str) -> bool:
    """Check if predicted matches expected."""
    if predicted is None:
        return False
    # ARC uses numbers sometimes (1,2,3,4) - normalize
    return predicted.upper() == expected.upper()


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

        # Format question with choices
        choices_str = "\n".join(
            f"{label}. {text}" for label, text in sorted(prob["choices"].items())
        )
        prompt = f"Question: {prob['question']}\n\n{choices_str}"

        response, tokens = await client.generate(
            prompt=prompt,
            system=system_prompt,
            max_tokens=400 if method_name == "DECOMPOSITION" else 50
        )

        elapsed = time.time() - start
        times.append(elapsed)

        predicted = extract_answer(response)
        expected = prob["answer"]

        if predicted is None:
            extraction_failures += 1
            is_correct = False
        else:
            is_correct = check_answer(predicted, expected)

        if is_correct:
            correct += 1

        # Progress every 10 or on errors
        if (i + 1) % 10 == 0 or (predicted is None):
            status = "EXTRACTION_FAIL" if predicted is None else ("✓" if is_correct else f"✗ exp={expected} got={predicted}")
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
    """Run full ARC-Challenge benchmark."""
    print("\n" + "="*70)
    print("ARC-CHALLENGE STANDARD BENCHMARK")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Problems: {N_PROBLEMS}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    problems = load_arc_problems(N_PROBLEMS)

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
    print("ARC-CHALLENGE BENCHMARK RESULTS")
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
    if improvement > 5:
        conclusion = "VALIDATED: Decomposition improves reasoning"
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
        "benchmark": "ARC-Challenge",
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
    results_file = results_dir / f"arc_challenge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {results_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=100, help="Number of problems")
    parser.add_argument("--model", type=str, default=MODEL, help="Model to use")
    args = parser.parse_args()

    os.environ["BENCHMARK_N"] = str(args.n)
    os.environ["BENCHMARK_MODEL"] = args.model

    asyncio.run(run_benchmark())
