#!/usr/bin/env python3
"""
MMLU Alternatives Benchmark

Test 5 alternative prompting methods to find non-regressing approach for recall tasks:
1. Confidence-first: Answer then rate confidence
2. Minimal scaffolding: Brief think, then answer
3. Answer-first: Commit before reasoning
4. Selective (direct): Skip decomposition entirely
5. CoT-lite: One-line insight, then answer

Goal: Match or beat direct baseline (62%) without overthinking
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
# ALTERNATIVE PROMPTS
# =============================================================================

METHODS = {
    "direct": {
        "system": """You are a multiple choice question answering assistant.
Answer with just the letter (A, B, C, or D) corresponding to the correct option.
Give only the letter, nothing else.""",
        "user": "Question: {question}\n\n{choices}\n\nAnswer:"
    },

    "confidence_first": {
        "system": """You are a multiple choice question answering assistant.
First give your answer (A, B, C, or D), then rate your confidence (0.0-1.0).
Format: A (confidence: 0.8)""",
        "user": "Question: {question}\n\n{choices}\n\nAnswer and confidence:"
    },

    "minimal_scaffolding": {
        "system": """You are a multiple choice question answering assistant.
Think briefly (one line), then answer with the letter.""",
        "user": "Question: {question}\n\n{choices}\n\nBrief thought, then answer:"
    },

    "answer_first": {
        "system": """You are a multiple choice question answering assistant.
Format: Answer: X\nReasoning: [brief explanation]
Commit to your answer first, then explain.""",
        "user": "Question: {question}\n\n{choices}\n\nAnswer:"
    },

    "cot_lite": {
        "system": """You are a multiple choice question answering assistant.
Format: Key insight: [one line]\nAnswer: X""",
        "user": "Question: {question}\n\n{choices}\n\nKey insight and answer:"
    }
}


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
        max_tokens: int = 100
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

def load_mmlu_problems(limit: int) -> List[dict]:
    """Load MMLU test set."""
    print(f"Loading MMLU dataset (limit={limit})...")
    ds = load_dataset("cais/mmlu", "all", split="test")

    problems = []
    for i, item in enumerate(ds):
        if i >= limit:
            break

        choices = item["choices"]
        answer_idx = item["answer"]
        answer_letter = ["A", "B", "C", "D"][answer_idx]

        problems.append({
            "id": f"mmlu_{i}",
            "question": item["question"],
            "choices": choices,
            "answer": answer_letter,
            "subject": item["subject"]
        })

    print(f"Loaded {len(problems)} problems")
    return problems


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_answer(text: str, method: str) -> Optional[str]:
    """Extract letter answer from response."""
    if not text:
        return None

    text = text.strip().upper()

    # Method-specific extraction
    if method == "confidence_first":
        # Format: A (confidence: 0.8)
        match = re.search(r'^([A-D])\s*\(', text)
        if match:
            return match.group(1)

    elif method == "answer_first":
        # Format: Answer: X
        match = re.search(r'ANSWER[:\s]+([A-D])', text)
        if match:
            return match.group(1)

    elif method == "cot_lite":
        # Format: Key insight: ... Answer: X
        match = re.search(r'ANSWER[:\s]+([A-D])', text)
        if match:
            return match.group(1)

    # Generic extraction (works for all)
    # Look for standalone letter
    match = re.search(r'^([A-D])[\.\)\s]', text)
    if match:
        return match.group(1)

    # Look for "The answer is X"
    match = re.search(r'THE\s+ANSWER\s+IS\s+([A-D])', text)
    if match:
        return match.group(1)

    # Any standalone letter
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1)

    return None


def check_answer(predicted: Optional[str], expected: str) -> bool:
    """Check if predicted matches expected."""
    if predicted is None:
        return False
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
    method_name: str,
    method_config: dict
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

        # Format question
        choices_str = "\n".join(
            f"{chr(65+j)}. {c}" for j, c in enumerate(prob["choices"])
        )
        user_prompt = method_config["user"].format(
            question=prob["question"],
            choices=choices_str
        )

        response, tokens = await client.generate(
            prompt=user_prompt,
            system=method_config["system"],
            max_tokens=200 if method_name in ["answer_first", "cot_lite", "minimal_scaffolding"] else 50
        )

        elapsed = time.time() - start
        times.append(elapsed)

        predicted = extract_answer(response, method_name)
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
    """Run full MMLU alternatives benchmark."""
    print("\n" + "="*70)
    print("MMLU ALTERNATIVES BENCHMARK")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Problems: {N_PROBLEMS}")
    print(f"Methods: {len(METHODS)}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data once
    problems = load_mmlu_problems(N_PROBLEMS)

    # Run each method
    results = []
    for method_name, method_config in METHODS.items():
        client = BenchmarkClient(MODEL)
        result = await run_method(client, problems, method_name, method_config)
        results.append(result)

        # Brief pause between methods
        await asyncio.sleep(2)

    # Results summary
    print("\n" + "="*70)
    print("MMLU ALTERNATIVES RESULTS")
    print("="*70)
    print(f"\n{'Method':<20} {'Correct':>10} {'Accuracy':>10} {'vs Direct':>10} {'Ext Fail':>10}")
    print("-"*70)

    direct_acc = next((r.accuracy for r in results if r.method == "direct"), 0)

    for r in sorted(results, key=lambda x: x.accuracy, reverse=True):
        delta = r.accuracy - direct_acc
        symbol = "✓" if delta >= 0 else "✗"
        print(f"{r.method:<20} {r.correct:>6}/{r.total:<3} {r.accuracy:>9.1f}% {symbol}{delta:>+8.1f}pp {r.extraction_failures:>10}")

    # Best non-regressing method
    non_regressing = [r for r in results if r.accuracy >= direct_acc]
    if non_regressing:
        best = max(non_regressing, key=lambda x: x.accuracy)
        print(f"\n✅ BEST NON-REGRESSING: {best.method} ({best.accuracy:.1f}%, {best.accuracy - direct_acc:+.1f}pp vs direct)")
    else:
        print(f"\n❌ NO NON-REGRESSING METHOD FOUND (all worse than direct baseline)")

    # Save results
    results_data = {
        "benchmark": "MMLU_alternatives",
        "model": MODEL,
        "n_problems": N_PROBLEMS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "methods": [asdict(r) for r in results],
        "direct_baseline": direct_acc,
        "best_method": best.method if non_regressing else None
    }

    results_dir = Path("experiments/results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"mmlu_alternatives_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(results_data, indent=2))
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
