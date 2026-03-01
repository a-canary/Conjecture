#!/usr/bin/env python3
"""
Direct Core Hypothesis Validation
Uses direct HTTP calls to LLM provider for maximum reliability
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
import re
import httpx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config.unified_config import UnifiedConfig as Config


@dataclass
class MathProblem:
    """A math problem with ground truth answer"""

    id: str
    question: str
    expected: float


@dataclass
class Result:
    """Benchmark result"""

    id: str
    system: str
    response: str
    extracted: float
    expected: float
    correct: bool
    time: float
    error: str = None


class DirectValidator:
    """Direct HTTP-based validator"""

    def __init__(self):
        self.config = Config()
        self.provider = self.config.settings.get_primary_provider()

        if not self.provider:
            raise RuntimeError("No LLM provider configured")

        print(f"Provider: {self.provider.name} | Model: {self.provider.model}\n")

    def get_problems(self) -> List[MathProblem]:
        """GSM8K problems"""
        return [
            MathProblem(
                "janet_eggs",
                "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the rest at $2 per egg. How much does she make daily?",
                18.0,
            ),
            MathProblem(
                "pool_volume",
                "A pool is 50m long, 25m wide, and 2m deep. How many cubic meters of water?",
                2500.0,
            ),
            MathProblem(
                "fruit_cost",
                "Apples are $3/lb and oranges are $2/lb. John buys 5 lbs of apples and 8 lbs of oranges. Total cost?",
                31.0,
            ),
            MathProblem(
                "bus_riders",
                "A bus starts with 45 passengers. Stop 1: 12 off, 8 on. Stop 2: 15 off, 20 on. How many now?",
                46.0,
            ),
            MathProblem(
                "reading_pages",
                "Sarah reads a 480-page book in 8 days. Pages per day?",
                60.0,
            ),
        ]

    async def call_llm(self, prompt: str) -> str:
        """Direct LLM API call"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.provider.url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.provider.api}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.provider.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 500,
                    },
                )

                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                else:
                    return f"HTTP {resp.status_code}"

        except Exception as e:
            return f"Error: {str(e)}"

    def extract_number(self, text: str) -> float:
        """Extract numeric answer"""
        patterns = [
            r"answer is:?\s*\$?(\d+(?:\.\d+)?)",
            r"=\s*\$?(\d+(?:\.\d+)?)",
            r"\$(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*(?:dollars|meters|cubic|eggs|pages|passengers)",
        ]

        for pat in patterns:
            m = re.search(pat, text.lower())
            if m:
                return float(m.group(1))

        # Fallback: last number
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        return float(nums[-1]) if nums else 0.0

    async def run_test(self, problem: MathProblem, system: str) -> Result:
        """Run single test"""
        start = time.time()

        try:
            if system == "baseline":
                prompt = f"Solve this math problem. Give only the final numeric answer:\n\n{problem.question}\n\nAnswer:"
            else:  # conjecture
                prompt = f"""You are an evidence-based reasoning system. Solve step-by-step:

{problem.question}

1. Break down the problem
2. Show each calculation step
3. Verify your math
4. Give final numeric answer

Answer:"""

            response = await self.call_llm(prompt)
            elapsed = time.time() - start

            extracted = self.extract_number(response)
            correct = abs(extracted - problem.expected) < 0.01

            return Result(
                id=problem.id,
                system=system,
                response=response,
                extracted=extracted,
                expected=problem.expected,
                correct=correct,
                time=elapsed,
            )

        except Exception as e:
            return Result(
                id=problem.id,
                system=system,
                response="",
                extracted=0.0,
                expected=problem.expected,
                correct=False,
                time=time.time() - start,
                error=str(e),
            )

    async def run_benchmark(self) -> Dict:
        """Run full benchmark"""
        print("=" * 60)
        print("HYPOTHESIS VALIDATION - DIRECT API")
        print("=" * 60)
        print("H0: Claim-based prompting improves GSM8K accuracy\n")

        problems = self.get_problems()
        print(f"{len(problems)} problems\n")

        baseline_results = []
        conjecture_results = []

        for i, prob in enumerate(problems, 1):
            print(f"[{i}/{len(problems)}] {prob.id}")
            print(f"  Q: {prob.question[:60]}...")

            # Baseline
            print(f"  Baseline...", end="", flush=True)
            base = await self.run_test(prob, "baseline")
            baseline_results.append(base)
            mark = "PASS" if base.correct else "FAIL"
            print(f" {mark} ({base.time:.1f}s) -> {base.extracted}")

            # Conjecture
            print(f"  Conjecture...", end="", flush=True)
            conj = await self.run_test(prob, "conjecture")
            conjecture_results.append(conj)
            mark = "PASS" if conj.correct else "FAIL"
            print(f" {mark} ({conj.time:.1f}s) -> {conj.extracted}")
            print()

        # Metrics
        base_acc = sum(r.correct for r in baseline_results) / len(baseline_results)
        conj_acc = sum(r.correct for r in conjecture_results) / len(conjecture_results)
        base_time = statistics.mean(r.time for r in baseline_results)
        conj_time = statistics.mean(r.time for r in conjecture_results)

        improvement = conj_acc - base_acc
        improvement_pct = (improvement / base_acc * 100) if base_acc > 0 else 0
        overhead = conj_time - base_time
        overhead_pct = (overhead / base_time * 100) if base_time > 0 else 0

        summary = {
            "timestamp": datetime.now().isoformat(),
            "provider": self.provider.name,
            "model": self.provider.model,
            "problems": len(problems),
            "baseline": {
                "accuracy": base_acc,
                "correct": sum(r.correct for r in baseline_results),
                "total": len(baseline_results),
                "avg_time": base_time,
                "results": [asdict(r) for r in baseline_results],
            },
            "conjecture": {
                "accuracy": conj_acc,
                "correct": sum(r.correct for r in conjecture_results),
                "total": len(conjecture_results),
                "avg_time": conj_time,
                "results": [asdict(r) for r in conjecture_results],
            },
            "comparison": {
                "improvement": improvement,
                "improvement_pct": improvement_pct,
                "overhead": overhead,
                "overhead_pct": overhead_pct,
                "hypothesis_supported": improvement >= 0.05,
            },
        }

        # Results
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Baseline:    {base_acc:.1%} ({base_time:.1f}s avg)")
        print(f"Conjecture:  {conj_acc:.1%} ({conj_time:.1f}s avg)")
        print()
        print(f"Improvement: {improvement:+.1%} ({improvement_pct:+.1f}%)")
        print(f"Overhead:    {overhead:+.1f}s ({overhead_pct:+.1f}%)")
        print()

        if summary["comparison"]["hypothesis_supported"]:
            print("HYPOTHESIS SUPPORTED (>=5% improvement)")
        else:
            print(f"HYPOTHESIS NOT SUPPORTED ({improvement:.1%} < 5%)")
        print("=" * 60)

        return summary


async def main():
    validator = DirectValidator()
    summary = await validator.run_benchmark()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hypothesis_validation_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {filename}")
    return summary


if __name__ == "__main__":
    asyncio.run(main())
