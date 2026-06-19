#!/usr/bin/env python3
"""
HumanEval Benchmark — O-0008 Validation

Tests decomposition-based reasoning on code generation (HumanEval).
HumanEval contains 164 Python programming problems requiring multi-step
logical reasoning and code synthesis.

Dataset: openai/openai_humaneval (test split, full 164 problems)
n=100 problems by default

Expected pattern: HumanEval is a HARD REASONING + CODE SYNTHESIS task.
- Baseline likely 40-60% for DeepSeek-V3
- Decomposition should help: +5-12pp (complex logic decomposition)
- The O-0008 report predicted: "Potential benefit (multi-step logical reasoning)"

Note: This benchmark requires code execution. Tests are run in a subprocess
with a timeout for safety.

References:
- O-0008_VALIDATION_REPORT.md — hypothesis framing
- Chen et al. (2021) — HumanEval paper
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import signal
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
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "400"))
EXEC_TIMEOUT = 10  # seconds per test execution


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

DIRECT_SYSTEM = """You are an expert Python programmer.
Write clean, correct, and efficient Python code.
Return only the code — no explanation, no markdown."""

DECOMP_SYSTEM = """You are an expert Python programmer using structured decomposition.

For complex coding problems:
1. Understand the requirements: what is the input, output, and constraints?
2. Identify edge cases and special conditions
3. Break down the algorithm into steps
4. Implement each step methodically
5. Verify the solution handles all cases

Think through the problem structure before writing code."""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BenchmarkResult:
    task_id: str
    model_name: str
    using_conjecture: bool
    prompt: str
    response: str
    passed: bool
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
    messages.append({"role": "user", "content": f"Complete the following Python function:\n\n{prompt}"})

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
# CODE EXTRACTION
# =============================================================================

def extract_code(response: str, entry_point: str) -> str:
    """Extract Python code from model response. Handle markdown code blocks."""
    # Remove markdown code blocks
    response = re.sub(r'```python\n?', '', response)
    response = re.sub(r'```\n?', '', response)
    response = response.strip()

    # If response is just a function definition, return it
    if response.startswith("def "):
        return response

    # Try to find the function definition
    lines = response.split('\n')
    code_lines = []
    in_function = False
    for line in lines:
        if f"def {entry_point}" in line:
            in_function = True
        if in_function:
            code_lines.append(line)
            # Stop at next top-level definition or class
            if code_lines and not code_lines[-1].startswith(' ') and 'def ' in line:
                break

    if code_lines:
        return '\n'.join(code_lines)
    return response  # Return as-is if extraction fails


# =============================================================================
# CODE EXECUTION (sandboxed)
# =============================================================================

def run_tests(code: str, test_code: str, entry_point: str, timeout: int = EXEC_TIMEOUT) -> tuple[bool, Optional[str]]:
    """Run test cases against generated code. Returns (passed, error_message)."""
    try:
        # Create a temporary file with the combined code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.write('\n\n')
            f.write(test_code)
            f.write('\n\n')
            # Add check call
            f.write(f"\ncheck({entry_point})\n")
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tempfile.gettempdir(),
            )
            if result.returncode == 0:
                return True, None
            else:
                error = result.stderr or result.stdout
                # Truncate error message
                error = error[:500]
                return False, error
        finally:
            Path(temp_path).unlink(missing_ok=True)

    except subprocess.TimeoutExpired:
        return False, "Execution timeout"
    except Exception as e:
        return False, str(e)[:200]


# =============================================================================
# PROBLEM LOADING
# =============================================================================

def load_humaneval_problems(n: int = N_PROBLEMS) -> List[dict]:
    """Load HumanEval problems from test set."""
    ds = load_dataset("openai/openai_humaneval", split="test")
    problems = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        problems.append({
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "canonical_solution": item["canonical_solution"],
            "test": item["test"],
            "entry_point": item["entry_point"],
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
    """Evaluate a single HumanEval problem."""
    prompt = problem["prompt"]
    system = DECOMP_SYSTEM if use_conjecture else DIRECT_SYSTEM

    try:
        response, elapsed = await call_llm(prompt, system)
        code = extract_code(response, problem["entry_point"])
        passed, error = run_tests(code, problem["test"], problem["entry_point"])
    except Exception as e:
        response = ""
        elapsed = 0.0
        passed = False
        error = str(e)[:200]

    return BenchmarkResult(
        task_id=problem["task_id"],
        model_name=model,
        using_conjecture=use_conjecture,
        prompt=prompt[:200],
        response=response[:500],
        passed=passed,
        execution_time=elapsed,
        error=error,
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
            acc = sum(r.passed for r in results) / len(results) * 100
            print(f"  [{i+1}/{len(problems)}] pass rate: {acc:.1f}%")

    correct = sum(r.passed for r in results)
    accuracy = correct / len(results) * 100 if results else 0

    summary = BenchmarkSummary(
        benchmark_name="HumanEval",
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
    print(f"HumanEval Benchmark — O-0008 Validation")
    print(f"=" * 70)
    print(f"Model: {MODEL}")
    print(f"Problems: {N_PROBLEMS}")
    print(f"API: {API_URL}")

    problems = load_humaneval_problems(N_PROBLEMS)
    print(f"\nLoaded {len(problems)} HumanEval problems")

    print("\n--- Direct Prompting ---")
    direct_results, direct_summary = await run_evaluation(problems, use_conjecture=False)
    print(f"Direct pass rate: {direct_summary.accuracy:.1f}% ({direct_summary.correct_answers}/{direct_summary.total_tasks})")

    print("\n--- Decomposition (Conjecture) ---")
    decomp_results, decomp_summary = await run_evaluation(problems, use_conjecture=True)
    print(f"Conjecture pass rate: {decomp_summary.accuracy:.1f}% ({decomp_summary.correct_answers}/{decomp_summary.total_tasks})")

    improvement = decomp_summary.accuracy - direct_summary.accuracy
    print(f"\n--- Result ---")
    print(f"Improvement: {improvement:+.1f}pp")

    if improvement > 5:
        conclusion = "SUCCESS: Decomposition significantly improves HumanEval accuracy"
    elif improvement > 2:
        conclusion = "PROMISING: Decomposition shows mild improvement on HumanEval"
    elif improvement > -2:
        conclusion = "NEUTRAL: No significant difference on HumanEval"
    else:
        conclusion = "CHALLENGE: Decomposition hurts HumanEval accuracy"

    print(f"CONCLUSION: {conclusion}")

    results = {
        "benchmark": "HumanEval",
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
    results_file = results_dir / f"humaneval_o0008_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
