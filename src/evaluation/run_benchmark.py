#!/usr/bin/env python3
"""
Official Benchmark Runner using lm-evaluation-harness

Runs standardized benchmarks comparing:
1. Direct LLM (baseline)
2. Conjecture-enhanced LLM
3. Conjecture with accumulation

Usage:
    python -m src.evaluation.run_benchmark --tasks gsm8k --limit 100
"""

import argparse
import json
import asyncio
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
import httpx
import sys

# Load environment
from dotenv import load_dotenv
load_dotenv("/workspace/.env")

# Import robust answer extraction
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))
from answer_extraction import extract_answer, check_answer_match, AnswerType


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    task: str
    model: str
    method: str  # direct, conjecture, conjecture_accum
    correct: int
    total: int
    accuracy: float
    avg_time: float
    total_tokens: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ClaimMemory:
    """Memory for claim accumulation"""
    claims: List[Dict] = field(default_factory=list)

    def add(self, content: str, domain: str, confidence: float, is_correct: bool):
        self.claims.append({
            "content": content[:200],
            "domain": domain,
            "confidence": confidence,
            "is_correct": is_correct
        })

    def get_hints(self, domain: str, n: int = 3) -> str:
        correct = [c for c in self.claims if c["domain"] == domain and c["is_correct"]]
        top = sorted(correct, key=lambda x: x["confidence"], reverse=True)[:n]
        if not top:
            return ""
        return "Patterns from similar problems:\n" + "\n".join(f"- {c['content'][:80]}" for c in top) + "\n\n"


class LLMClient:
    """Unified LLM client with provider support"""

    def __init__(self, provider: str = "chutes", model: str = "deepseek-ai/DeepSeek-V3"):
        self.provider = provider
        self.model = model
        self.total_tokens = 0

        if provider == "cerebras":
            self.url = "https://api.cerebras.ai/v1/chat/completions"
            self.api_key = os.getenv("CEREBRAS_API_KEY")
        elif provider == "chutes":
            self.url = "https://llm.chutes.ai/v1/chat/completions"
            self.api_key = os.getenv("CHUTES_API_KEY")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def generate(self, prompt: str, max_tokens: int = 300) -> Tuple[str, int]:
        async with httpx.AsyncClient() as client:
            for attempt in range(3):
                try:
                    resp = await client.post(
                        self.url,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tokens,
                            "temperature": 0.1
                        },
                        timeout=120.0
                    )
                    if resp.status_code == 429:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    if resp.status_code == 200:
                        data = resp.json()
                        tokens = data.get("usage", {}).get("total_tokens", 0)
                        self.total_tokens += tokens
                        return data["choices"][0]["message"]["content"], tokens
                except Exception as e:
                    await asyncio.sleep(2)
        return "", 0


def load_gsm8k(limit: int = None) -> List[Dict]:
    """Load GSM8K dataset"""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        problems = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break
            # Extract answer from the solution
            solution = item["answer"]
            # GSM8K format: solution text ending with #### number
            match = re.search(r'####\s*(\-?[\d,\.]+)', solution)
            if match:
                answer = match.group(1).replace(",", "")
                problems.append({
                    "question": item["question"],
                    "answer": answer,
                    "full_solution": solution
                })
        return problems
    except Exception as e:
        print(f"Error loading GSM8K: {e}")
        return []


def load_mmlu(limit: int = None) -> List[Dict]:
    """Load MMLU dataset (subset)"""
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        problems = []
        for i, item in enumerate(ds):
            if limit and i >= limit:
                break
            choices = item["choices"]
            answer_idx = item["answer"]
            problems.append({
                "question": item["question"],
                "choices": choices,
                "answer": ["A", "B", "C", "D"][answer_idx],
                "subject": item["subject"]
            })
        return problems
    except Exception as e:
        print(f"Error loading MMLU: {e}")
        return []


def extract_answer_wrapper(response: str, expected: str) -> str:
    """
    Wrapper for robust extraction function.
    Infers answer type from expected value.
    """
    answer_type = None
    if expected in ["A", "B", "C", "D"]:
        answer_type = AnswerType.MULTIPLE_CHOICE
    else:
        answer_type = AnswerType.NUMERICAL
    return extract_answer(response, expected, answer_type)


def check_answer_wrapper(pred: str, expected: str) -> bool:
    """
    Wrapper for robust answer checking function.
    Infers answer type from expected value.
    """
    answer_type = None
    if expected in ["A", "B", "C", "D"]:
        answer_type = AnswerType.MULTIPLE_CHOICE
    else:
        answer_type = AnswerType.NUMERICAL
    return check_answer_match(pred, expected, answer_type)


async def run_direct(llm: LLMClient, problems: List[Dict], task: str) -> BenchmarkResult:
    """Run direct LLM benchmark"""
    print(f"\n{'='*60}\nDIRECT LLM: {task} ({len(problems)} problems)\n{'='*60}")

    correct = 0
    start_time = time.time()

    for i, p in enumerate(problems):
        if task == "gsm8k":
            prompt = f"Solve this math problem. Show your work and end with #### followed by the answer.\n\nProblem: {p['question']}\n\nSolution:"
        else:  # mmlu
            choices_str = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(p['choices']))
            prompt = f"Question: {p['question']}\n\n{choices_str}\n\nAnswer with just the letter (A, B, C, or D):"

        response, _ = await llm.generate(prompt, max_tokens=400 if task == "gsm8k" else 50)
        answer = extract_answer(response, p['answer'])

        if check_answer_match(answer, p['answer'], AnswerType.NUMERICAL):
            correct += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):.1f}%")

        await asyncio.sleep(0.1)

    elapsed = time.time() - start_time

    return BenchmarkResult(
        task=task,
        model=llm.model,
        method="direct",
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        avg_time=round(elapsed / len(problems), 2),
        total_tokens=llm.total_tokens
    )


async def run_conjecture(llm: LLMClient, problems: List[Dict], task: str, accumulate: bool = False) -> BenchmarkResult:
    """Run Conjecture-enhanced benchmark"""
    method = "conjecture_accum" if accumulate else "conjecture"
    print(f"\n{'='*60}\nCONJECTURE ({method}): {task} ({len(problems)} problems)\n{'='*60}")

    correct = 0
    memory = ClaimMemory() if accumulate else None
    start_time = time.time()

    for i, p in enumerate(problems):
        if task == "gsm8k":
            # Step 1: Decompose
            decompose_prompt = f"Analyze this problem. What are the key facts and what steps are needed?\n\nProblem: {p['question']}\n\nAnalysis:"
            analysis, _ = await llm.generate(decompose_prompt, max_tokens=200)

            # Step 2: Get hints if accumulating
            hints = memory.get_hints("math") if memory else ""

            # Step 3: Solve with context
            solve_prompt = f"{hints}Analysis: {analysis[:250]}\n\nProblem: {p['question']}\n\nSolve step by step. End with #### and the answer."
            response, _ = await llm.generate(solve_prompt, max_tokens=350)

        else:  # mmlu
            choices_str = "\n".join(f"{chr(65+j)}. {c}" for j, c in enumerate(p['choices']))

            # Step 1: Analyze
            analyze_prompt = f"Question: {p['question']}\n\n{choices_str}\n\nWhat key knowledge is needed to answer this? Be brief."
            analysis, _ = await llm.generate(analyze_prompt, max_tokens=100)

            # Step 2: Answer with context
            solve_prompt = f"Analysis: {analysis[:150]}\n\nQuestion: {p['question']}\n\n{choices_str}\n\nBased on the analysis, the answer is:"
            response, _ = await llm.generate(solve_prompt, max_tokens=50)

        answer = extract_answer(response, p['answer'])
        is_correct = check_answer_match(answer, p['answer'], AnswerType.NUMERICAL)

        if is_correct:
            correct += 1

        # Store claim if accumulating
        if memory:
            memory.add(
                content=f"{p['question'][:50]}... → {answer}",
                domain="math" if task == "gsm8k" else p.get("subject", "general"),
                confidence=0.9 if is_correct else 0.3,
                is_correct=is_correct
            )

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):.1f}%")

        await asyncio.sleep(0.1)

    elapsed = time.time() - start_time

    return BenchmarkResult(
        task=task,
        model=llm.model,
        method=method,
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        avg_time=round(elapsed / len(problems), 2),
        total_tokens=llm.total_tokens
    )


async def main():
    parser = argparse.ArgumentParser(description="Run official benchmarks")
    parser.add_argument("--tasks", type=str, default="gsm8k", help="Comma-separated task list: gsm8k,mmlu")
    parser.add_argument("--limit", type=int, default=50, help="Number of problems per task")
    parser.add_argument("--provider", type=str, default="chutes", help="LLM provider: cerebras, chutes")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-V3", help="Model name")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    results = []

    for task in tasks:
        print(f"\n{'#'*60}")
        print(f"# TASK: {task.upper()}")
        print(f"{'#'*60}")

        # Load dataset
        if task == "gsm8k":
            problems = load_gsm8k(args.limit)
        elif task == "mmlu":
            problems = load_mmlu(args.limit)
        else:
            print(f"Unknown task: {task}")
            continue

        if not problems:
            print(f"No problems loaded for {task}")
            continue

        print(f"Loaded {len(problems)} problems")

        # Run benchmarks
        llm1 = LLMClient(args.provider, args.model)
        direct_result = await run_direct(llm1, problems, task)
        results.append(direct_result)

        await asyncio.sleep(2)

        llm2 = LLMClient(args.provider, args.model)
        conjecture_result = await run_conjecture(llm2, problems, task, accumulate=False)
        results.append(conjecture_result)

        await asyncio.sleep(2)

        llm3 = LLMClient(args.provider, args.model)
        accum_result = await run_conjecture(llm3, problems, task, accumulate=True)
        results.append(accum_result)

    # Summary
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Task':<10} {'Method':<20} {'Accuracy':>10} {'Avg Time':>10} {'Tokens':>10}")
    print("-" * 80)

    for r in results:
        print(f"{r.task:<10} {r.method:<20} {r.accuracy:>9.1f}% {r.avg_time:>9.2f}s {r.total_tokens:>10}")

    # Calculate improvements
    print("\n" + "-" * 80)
    for task in tasks:
        task_results = [r for r in results if r.task == task]
        if len(task_results) >= 2:
            direct = next((r for r in task_results if r.method == "direct"), None)
            conjecture = next((r for r in task_results if r.method == "conjecture"), None)
            accum = next((r for r in task_results if r.method == "conjecture_accum"), None)

            if direct and conjecture:
                delta = conjecture.accuracy - direct.accuracy
                print(f"{task}: Conjecture vs Direct: {delta:+.1f}pp")
            if conjecture and accum:
                delta = accum.accuracy - conjecture.accuracy
                print(f"{task}: Accumulation vs Conjecture: {delta:+.1f}pp")

    print("=" * 80)

    # Save results
    output_dir = Path("/workspace/data/benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(output_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
