#!/usr/bin/env python3
"""
ARC-AGI-2 Benchmark: Cerebras llama3.1-8b (bare vs +Conjecture)

Tests pattern recognition on real ARC tasks.
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

NUM_TASKS = 10  # Number of ARC tasks to test


@dataclass
class ARCTask:
    """ARC-AGI task"""
    task_id: str
    train_examples: List[Dict]
    test_input: List[List[int]]
    test_output: List[List[int]]


@dataclass
class Claim:
    """A claim in the reasoning process"""
    id: str
    content: str
    confidence: float
    claim_type: str


@dataclass
class ConjectureSession:
    """Session for claim-based reasoning"""
    root_claim: Claim
    sub_claims: List[Claim] = field(default_factory=list)
    final_answer: Optional[List[List[int]]] = None
    total_tokens: int = 0
    total_time: float = 0


class CerebrasLLM:
    """Cerebras LLM client"""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def generate(self, prompt: str, max_tokens: int = 500) -> tuple[str, int, float]:
        """Generate response from Cerebras"""
        start = time.time()

        for attempt in range(3):
            try:
                resp = await self.client.post(
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
                    await asyncio.sleep(3 * (attempt + 1))
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


def format_grid(grid: List[List[int]]) -> str:
    """Format grid for display"""
    return "\n".join([" ".join(str(c) for c in row) for row in grid])


def parse_grid(response: str) -> Optional[List[List[int]]]:
    """Parse grid from LLM response"""
    try:
        # Try JSON array
        match = re.search(r'\[\[[\d,\s\[\]]+\]\]', response, re.DOTALL)
        if match:
            return json.loads(match.group())

        # Try to parse line-by-line
        lines = []
        for line in response.strip().split('\n'):
            nums = re.findall(r'\d+', line)
            if nums and len(nums) > 0:
                lines.append([int(n) for n in nums])
        if lines:
            return lines

    except Exception:
        pass
    return None


def load_arc_tasks(n: int = 10) -> List[ARCTask]:
    """Load ARC tasks from data directory"""
    tasks = []
    arc_dir = Path("/workspace/data/arc_agi2")

    if not arc_dir.exists():
        # Use sample tasks
        print("Using sample ARC tasks")
        return [
            ARCTask(
                task_id="sample_flip",
                train_examples=[
                    {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
                    {"input": [[5, 6], [7, 8]], "output": [[6, 5], [8, 7]]},
                ],
                test_input=[[9, 0], [1, 2]],
                test_output=[[0, 9], [2, 1]]
            ),
            ARCTask(
                task_id="sample_rotate",
                train_examples=[
                    {"input": [[1, 2, 3]], "output": [[1], [2], [3]]},
                    {"input": [[4, 5, 6]], "output": [[4], [5], [6]]},
                ],
                test_input=[[7, 8, 9]],
                test_output=[[7], [8], [9]]
            ),
            ARCTask(
                task_id="sample_fill",
                train_examples=[
                    {"input": [[0, 0], [0, 0]], "output": [[1, 1], [1, 1]]},
                    {"input": [[0, 0, 0]], "output": [[1, 1, 1]]},
                ],
                test_input=[[0, 0, 0], [0, 0, 0]],
                test_output=[[1, 1, 1], [1, 1, 1]]
            ),
        ]

    task_files = list(arc_dir.glob("*.json"))[:n]
    for task_file in task_files:
        with open(task_file) as f:
            data = json.load(f)

        tasks.append(ARCTask(
            task_id=task_file.stem,
            train_examples=data.get("train", []),
            test_input=data.get("test", [{}])[0].get("input", []),
            test_output=data.get("test", [{}])[0].get("output", [])
        ))

    return tasks


def format_task_prompt(task: ARCTask) -> str:
    """Format ARC task for bare LLM"""
    prompt = "Solve this pattern recognition task.\n\n"

    for i, ex in enumerate(task.train_examples):
        prompt += f"Example {i+1}:\n"
        prompt += f"Input:\n{format_grid(ex['input'])}\n"
        prompt += f"Output:\n{format_grid(ex['output'])}\n\n"

    prompt += f"Test Input:\n{format_grid(task.test_input)}\n\n"
    prompt += "What is the output? Return ONLY the grid as a JSON array [[...], [...]]."

    return prompt


async def benchmark_bare(client: httpx.AsyncClient, tasks: List[ARCTask]) -> dict:
    """Run bare llama3.1-8b benchmark on ARC tasks"""
    print(f"\n{'='*60}")
    print("Testing: Bare llama3.1-8b on ARC-AGI")
    print(f"{'='*60}")

    llm = CerebrasLLM(client)
    correct = 0
    total_time = 0
    total_tokens = 0

    for i, task in enumerate(tasks):
        prompt = format_task_prompt(task)
        response, tokens, elapsed = await llm.generate(prompt, max_tokens=300)

        total_time += elapsed
        total_tokens += tokens

        predicted = parse_grid(response)
        is_correct = predicted == task.test_output

        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"  [{i+1:2d}/{len(tasks)}] {status} task={task.task_id[:15]} ({elapsed:.2f}s)", flush=True)

        await asyncio.sleep(0.5)

    accuracy = correct / len(tasks) * 100 if tasks else 0
    avg_time = total_time / len(tasks) if tasks else 0

    print(f"\n  RESULT: {correct}/{len(tasks)} ({accuracy:.1f}%) avg={avg_time:.2f}s/task")

    return {
        "model": f"{MODEL} (bare)",
        "correct": correct,
        "total": len(tasks),
        "accuracy": round(accuracy, 1),
        "avg_time": round(avg_time, 2),
        "total_tokens": total_tokens
    }


class ConjectureARC:
    """Conjecture Framework for ARC tasks"""

    def __init__(self, llm: CerebrasLLM):
        self.llm = llm

    async def process_task(self, task: ARCTask) -> ConjectureSession:
        """Process ARC task with claim-based reasoning"""

        root = Claim(
            id="root",
            content=f"Solve ARC task {task.task_id}",
            confidence=0.0,
            claim_type="goal"
        )
        session = ConjectureSession(root_claim=root)

        # Claim 1: Pattern observation
        obs_prompt = f"""Analyze these input-output pairs:

{chr(10).join(f"Example {i+1}: Input {len(ex['input'])}x{len(ex['input'][0])} -> Output {len(ex['output'])}x{len(ex['output'][0])}" for i, ex in enumerate(task.train_examples))}

Detailed examples:
{chr(10).join(f"Input:\n{format_grid(ex['input'])}\nOutput:\n{format_grid(ex['output'])}" for ex in task.train_examples[:2])}

What transformation pattern do you observe? Be specific about:
1. Size changes
2. Value changes
3. Spatial operations (flip, rotate, etc.)"""

        obs_resp, tokens1, time1 = await self.llm.generate(obs_prompt, max_tokens=300)
        session.total_tokens += tokens1
        session.total_time += time1

        session.sub_claims.append(Claim(
            id="obs",
            content=obs_resp[:200] if obs_resp else "Unable to observe",
            confidence=0.8 if obs_resp else 0.3,
            claim_type="observation"
        ))

        # Claim 2: Hypothesis validation
        hyp_prompt = f"""Based on observation:
{obs_resp[:300] if obs_resp else "Analyze the pattern"}

Verify your hypothesis with ALL examples:
{chr(10).join(f"Example {i+1}: Does your rule transform {ex['input']} to {ex['output']}?" for i, ex in enumerate(task.train_examples))}

Is your hypothesis consistent? Rate confidence (0-100%)."""

        hyp_resp, tokens2, time2 = await self.llm.generate(hyp_prompt, max_tokens=200)
        session.total_tokens += tokens2
        session.total_time += time2

        session.sub_claims.append(Claim(
            id="hyp",
            content=hyp_resp[:200] if hyp_resp else "Unable to validate",
            confidence=0.7 if hyp_resp else 0.4,
            claim_type="hypothesis"
        ))

        # Claim 3: Apply transformation
        apply_prompt = f"""Based on analysis:
Observation: {obs_resp[:200] if obs_resp else "Pattern detected"}
Validation: {hyp_resp[:200] if hyp_resp else "Seems consistent"}

Apply the transformation to this test input:
{format_grid(task.test_input)}

Return ONLY the output grid as JSON array [[...], [...]]."""

        apply_resp, tokens3, time3 = await self.llm.generate(apply_prompt, max_tokens=200)
        session.total_tokens += tokens3
        session.total_time += time3

        session.final_answer = parse_grid(apply_resp)
        session.root_claim.confidence = 0.7 if session.final_answer else 0.3

        return session


async def benchmark_conjecture(client: httpx.AsyncClient, tasks: List[ARCTask]) -> dict:
    """Run llama3.1-8b + Conjecture benchmark on ARC tasks"""
    print(f"\n{'='*60}")
    print("Testing: llama3.1-8b + Conjecture on ARC-AGI")
    print(f"{'='*60}")

    llm = CerebrasLLM(client)
    framework = ConjectureARC(llm)

    correct = 0
    total_time = 0
    total_tokens = 0
    total_claims = 0

    for i, task in enumerate(tasks):
        session = await framework.process_task(task)

        is_correct = session.final_answer == task.test_output
        if is_correct:
            correct += 1

        total_time += session.total_time
        total_tokens += session.total_tokens
        total_claims += len(session.sub_claims)

        status = "✓" if is_correct else ("E" if not session.final_answer else "✗")
        claims = len(session.sub_claims)
        print(f"  [{i+1:2d}/{len(tasks)}] {status} task={task.task_id[:15]} ({session.total_time:.2f}s) claims={claims}", flush=True)

        await asyncio.sleep(0.5)

    accuracy = correct / len(tasks) * 100 if tasks else 0
    avg_time = total_time / len(tasks) if tasks else 0

    print(f"\n  RESULT: {correct}/{len(tasks)} ({accuracy:.1f}%) avg={avg_time:.2f}s/task")

    return {
        "model": f"{MODEL} + Conjecture",
        "correct": correct,
        "total": len(tasks),
        "accuracy": round(accuracy, 1),
        "avg_time": round(avg_time, 2),
        "total_tokens": total_tokens,
        "avg_claims": round(total_claims / len(tasks), 1) if tasks else 0
    }


async def main():
    print("ARC-AGI-2 Benchmark: Cerebras + Conjecture")
    print(f"Model: {MODEL}")
    print(f"Tasks: {NUM_TASKS}\n")

    # Load tasks
    tasks = load_arc_tasks(NUM_TASKS)
    print(f"Loaded {len(tasks)} ARC tasks\n")

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
    print("COMPARISON: Bare vs Conjecture (ARC-AGI)")
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
    results_file = Path("/workspace/data/arc_cerebras_results.json")
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
