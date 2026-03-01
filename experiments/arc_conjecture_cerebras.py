#!/usr/bin/env python3
"""
ARC-AGI-2 Benchmark: Cerebras llama3.1-8b (bare vs +Conjecture)

Tests whether claim-based reasoning improves ARC pattern recognition.
"""
import asyncio
import json
import time
import os
import re
import httpx
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-hpr4pjyd895p4ktvpnn436exx49rr925f6dptjvmee5ycrx8")
CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
MODEL = "llama3.1-8b"

NUM_TASKS = 20  # Number of ARC tasks to test


@dataclass
class ARCTask:
    """ARC task structure"""
    task_id: str
    train: List[Dict]  # List of {input: grid, output: grid}
    test: List[Dict]   # List of {input: grid, output: grid}


@dataclass
class Claim:
    """A claim in the reasoning process"""
    id: str
    content: str
    confidence: float
    claim_type: str


@dataclass
class ConjectureSession:
    """Session for ARC reasoning"""
    claims: List[Claim] = field(default_factory=list)
    predicted_output: Optional[List[List[int]]] = None
    total_tokens: int = 0
    total_time: float = 0


def load_arc_tasks(num_tasks: int) -> List[ARCTask]:
    """Load ARC tasks from training set"""
    tasks = []
    data_dir = Path("/workspace/data/arc_agi2_repo/data/training")

    task_files = sorted(data_dir.glob("*.json"))[:num_tasks]

    for task_file in task_files:
        with open(task_file) as f:
            data = json.load(f)
            tasks.append(ARCTask(
                task_id=task_file.stem,
                train=data.get("train", []),
                test=data.get("test", [])
            ))

    return tasks


def grid_to_str(grid: List[List[int]]) -> str:
    """Convert grid to compact string"""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def format_arc_prompt_bare(task: ARCTask) -> str:
    """Format ARC task for bare LLM"""
    examples = []
    for i, ex in enumerate(task.train):
        examples.append(f"Example {i+1}:")
        examples.append(f"Input:\n{grid_to_str(ex['input'])}")
        examples.append(f"Output:\n{grid_to_str(ex['output'])}")
        examples.append("")

    test_input = task.test[0]["input"]

    return f"""Find the pattern in these input-output examples and apply it to the test input.

{chr(10).join(examples)}
Test Input:
{grid_to_str(test_input)}

Return ONLY the output grid as a JSON array of arrays (e.g., [[1,2],[3,4]]). No explanation."""


def parse_grid_response(response: str) -> Optional[List[List[int]]]:
    """Parse grid from LLM response"""
    if not response:
        return None

    # Try to find JSON array
    patterns = [
        r'\[\s*\[[\d\s,\[\]]+\]\s*\]',  # [[...]]
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                grid = json.loads(match.group())
                if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                    return grid
            except json.JSONDecodeError:
                continue

    # Try parsing entire response
    try:
        grid = json.loads(response.strip())
        if isinstance(grid, list):
            return grid
    except:
        pass

    return None


def grids_equal(g1: Optional[List[List[int]]], g2: List[List[int]]) -> bool:
    """Check if two grids are equal"""
    if g1 is None:
        return False
    if len(g1) != len(g2):
        return False
    for r1, r2 in zip(g1, g2):
        if len(r1) != len(r2):
            return False
        if r1 != r2:
            return False
    return True


async def query_cerebras(client: httpx.AsyncClient, prompt: str, max_tokens: int = 500) -> tuple[str, int, float]:
    """Query Cerebras API"""
    start = time.time()

    for attempt in range(3):
        try:
            resp = await client.post(
                CEREBRAS_URL,
                headers={"Authorization": f"Bearer {CEREBRAS_API_KEY}"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.1
                },
                timeout=60.0
            )
            elapsed = time.time() - start

            if resp.status_code == 429:
                await asyncio.sleep(2 * (attempt + 1))
                continue

            if resp.status_code != 200:
                return "", 0, elapsed

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return content, tokens, elapsed

        except Exception as e:
            await asyncio.sleep(1)

    return "", 0, time.time() - start


async def run_bare(client: httpx.AsyncClient, task: ARCTask) -> tuple[bool, float, int]:
    """Run bare LLM on ARC task"""
    prompt = format_arc_prompt_bare(task)
    response, tokens, elapsed = await query_cerebras(client, prompt, max_tokens=500)

    predicted = parse_grid_response(response)
    expected = task.test[0]["output"]
    correct = grids_equal(predicted, expected)

    return correct, elapsed, tokens


async def run_conjecture(client: httpx.AsyncClient, task: ARCTask) -> tuple[bool, float, int, int]:
    """Run Conjecture-enhanced reasoning on ARC task"""
    session = ConjectureSession()

    # Phase 1: Analyze patterns
    analyze_prompt = f"""Analyze the transformation pattern in these examples:

{chr(10).join(f"Example {i+1}: Input {len(ex['input'])}x{len(ex['input'][0])} -> Output {len(ex['output'])}x{len(ex['output'][0])}" for i, ex in enumerate(task.train))}

Training examples:
{chr(10).join(f"In: {json.dumps(ex['input'])} -> Out: {json.dumps(ex['output'])}" for ex in task.train[:2])}

List 3 key observations about:
1. Size changes (same, grow, shrink)
2. Color/value transformations
3. Spatial patterns (rotation, reflection, etc.)"""

    analysis, tokens1, time1 = await query_cerebras(client, analyze_prompt, max_tokens=200)
    session.total_tokens += tokens1
    session.total_time += time1
    session.claims.append(Claim("obs", analysis[:150] if analysis else "Unable to analyze", 0.7, "observation"))

    # Phase 2: Hypothesize transformation
    hyp_prompt = f"""Based on analysis:
{analysis[:200] if analysis else "Look for pattern"}

Examples:
{chr(10).join(f"In: {json.dumps(ex['input'])} -> Out: {json.dumps(ex['output'])}" for ex in task.train)}

What is the transformation rule? Be specific (e.g., "rotate 90 degrees", "flip horizontal", "fill enclosed regions")."""

    hypothesis, tokens2, time2 = await query_cerebras(client, hyp_prompt, max_tokens=150)
    session.total_tokens += tokens2
    session.total_time += time2
    session.claims.append(Claim("hyp", hypothesis[:150] if hypothesis else "Unknown", 0.6, "hypothesis"))

    # Phase 3: Apply transformation
    test_input = task.test[0]["input"]
    apply_prompt = f"""Apply this transformation rule:
{hypothesis[:200] if hypothesis else "Find the pattern"}

To this input:
{json.dumps(test_input)}

Training examples for reference:
{chr(10).join(f"{json.dumps(ex['input'])} -> {json.dumps(ex['output'])}" for ex in task.train[:2])}

Return ONLY the output grid as a JSON array. No explanation."""

    result, tokens3, time3 = await query_cerebras(client, apply_prompt, max_tokens=500)
    session.total_tokens += tokens3
    session.total_time += time3

    predicted = parse_grid_response(result)
    expected = task.test[0]["output"]
    correct = grids_equal(predicted, expected)

    return correct, session.total_time, session.total_tokens, len(session.claims)


async def benchmark_bare(client: httpx.AsyncClient, tasks: List[ARCTask]) -> dict:
    """Run bare benchmark"""
    print(f"\n{'='*60}")
    print(f"Testing: Bare {MODEL}")
    print(f"{'='*60}")

    correct = 0
    total_time = 0
    total_tokens = 0
    errors = 0

    for i, task in enumerate(tasks):
        try:
            is_correct, elapsed, tokens = await run_bare(client, task)

            if is_correct:
                correct += 1
            total_time += elapsed
            total_tokens += tokens

            status = "✓" if is_correct else "✗"
            print(f"  [{i+1:2d}/{len(tasks)}] {status} {task.task_id[:8]} ({elapsed:.2f}s)", flush=True)

            await asyncio.sleep(0.3)

        except Exception as e:
            errors += 1
            print(f"  [{i+1:2d}/{len(tasks)}] E {task.task_id[:8]} ({str(e)[:20]})", flush=True)

    accuracy = correct / len(tasks) * 100
    avg_time = total_time / len(tasks)

    print(f"\n  RESULT: {correct}/{len(tasks)} ({accuracy:.1f}%) avg={avg_time:.2f}s/task")

    return {
        "model": f"{MODEL} (bare)",
        "correct": correct,
        "total": len(tasks),
        "accuracy": round(accuracy, 1),
        "avg_time": round(avg_time, 2),
        "total_tokens": total_tokens,
        "errors": errors
    }


async def benchmark_conjecture(client: httpx.AsyncClient, tasks: List[ARCTask]) -> dict:
    """Run Conjecture-enhanced benchmark"""
    print(f"\n{'='*60}")
    print(f"Testing: {MODEL} + Conjecture")
    print(f"{'='*60}")

    correct = 0
    total_time = 0
    total_tokens = 0
    total_claims = 0
    errors = 0

    for i, task in enumerate(tasks):
        try:
            is_correct, elapsed, tokens, claims = await run_conjecture(client, task)

            if is_correct:
                correct += 1
            total_time += elapsed
            total_tokens += tokens
            total_claims += claims

            status = "✓" if is_correct else "✗"
            print(f"  [{i+1:2d}/{len(tasks)}] {status} {task.task_id[:8]} ({elapsed:.2f}s) claims={claims}", flush=True)

            await asyncio.sleep(0.3)

        except Exception as e:
            errors += 1
            print(f"  [{i+1:2d}/{len(tasks)}] E {task.task_id[:8]} ({str(e)[:20]})", flush=True)

    accuracy = correct / len(tasks) * 100
    avg_time = total_time / len(tasks)
    avg_claims = total_claims / len(tasks)

    print(f"\n  RESULT: {correct}/{len(tasks)} ({accuracy:.1f}%) avg={avg_time:.2f}s/task claims={avg_claims:.1f}/task")

    return {
        "model": f"{MODEL} + Conjecture",
        "correct": correct,
        "total": len(tasks),
        "accuracy": round(accuracy, 1),
        "avg_time": round(avg_time, 2),
        "total_tokens": total_tokens,
        "avg_claims": round(avg_claims, 1),
        "errors": errors
    }


async def main():
    print("ARC-AGI-2 Benchmark: Cerebras + Conjecture")
    print(f"Model: {MODEL}")
    print(f"Tasks: {NUM_TASKS}\n")

    # Load tasks
    print("Loading ARC tasks...")
    tasks = load_arc_tasks(NUM_TASKS)
    print(f"Loaded {len(tasks)} tasks\n")

    results = []

    async with httpx.AsyncClient() as client:
        # Run bare baseline
        bare_result = await benchmark_bare(client, tasks)
        results.append(bare_result)

        await asyncio.sleep(2)

        # Run Conjecture-enhanced
        conj_result = await benchmark_conjecture(client, tasks)
        results.append(conj_result)

    # Summary
    print("\n" + "="*80)
    print("COMPARISON: Bare vs Conjecture on ARC-AGI-2")
    print("="*80)

    bare = results[0]
    conj = results[1]

    improvement = conj["accuracy"] - bare["accuracy"]

    print(f"{'Configuration':<30} {'Accuracy':>10} {'Avg Time':>10} {'Tokens':>10}")
    print("-"*80)
    print(f"{'Bare llama3.1-8b':<30} {bare['accuracy']:>9.1f}% {bare['avg_time']:>9.2f}s {bare['total_tokens']:>10}")
    print(f"{'llama3.1-8b + Conjecture':<30} {conj['accuracy']:>9.1f}% {conj['avg_time']:>9.2f}s {conj['total_tokens']:>10}")
    print("-"*80)
    print(f"{'Improvement':<30} {improvement:>+9.1f}pp")
    print("="*80)

    # Save results
    results_file = Path("/workspace/data/arc_agi2_repo/conjecture_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "bare": bare,
            "conjecture": conj,
            "improvement_pp": round(improvement, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
