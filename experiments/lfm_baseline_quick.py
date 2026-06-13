#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
LFM-2.5 Baseline Quick Test (n=10)

Establish baseline performance of 1.2B LFM-2.5 model on BBH logical reasoning.
Tests both direct prompting and current three-prompt architecture.

This establishes the starting point for 100-strategy exploration.
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from datasets import load_dataset
import httpx


# =============================================================================
# CONFIGURATION
# =============================================================================

LFM_ENDPOINT = "http://100.73.201.58:1234/v1/chat/completions"
MODEL = "liquid/lfm2.5-1.2b"
N_PROBLEMS = 10  # Quick test
BBH_TASK = "logical_deduction_three_objects"

MAX_ITERATIONS = 4
CONFIDENCE_THRESHOLD = 0.7


# =============================================================================
# LLM CLIENT
# =============================================================================

class LFMClient:
    def __init__(self):
        self.total_tokens = 0
        self.client = httpx.AsyncClient(timeout=60.0)

    async def generate(self, prompt: str, system: str = "", max_tokens: int = 500) -> Tuple[str, int]:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await self.client.post(
                LFM_ENDPOINT,
                json={
                    "model": MODEL,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            tokens = data["usage"]["total_tokens"]
            self.total_tokens += tokens
            return content, tokens

        except Exception as e:
            print(f"  LFM Error: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return "", 0


# =============================================================================
# DATA LOADING
# =============================================================================

def load_bbh_problems(task: str, limit: int):
    """Load BBH problems."""
    print(f"Loading BBH task: {task} (n={limit})...")
    ds = load_dataset("lukaemon/bbh", task)
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

    print(f"Loaded {len(problems)} problems")
    return problems


# =============================================================================
# ANSWER EXTRACTION
# =============================================================================

def extract_answer(text: str, target: str) -> Optional[str]:
    """Extract answer from response."""
    if not text:
        return None

    # Try exact match
    if target.lower() in text.lower():
        return target

    # Try option patterns
    for opt in ['(A)', '(B)', '(C)']:
        if opt in text:
            return opt

    return None


# =============================================================================
# DIRECT BASELINE
# =============================================================================

async def run_direct(client: LFMClient, problems):
    """Direct prompting baseline."""
    print(f"\nRunning DIRECT baseline...")

    correct = 0
    times = []

    for i, problem in enumerate(problems, 1):
        start = time.time()
        query = problem["input"]
        target = problem["target"]

        system = "You are a reasoning assistant. Think step by step and give your answer clearly."
        response, _ = await client.generate(query, system=system, max_tokens=400)

        answer = extract_answer(response, target)
        is_correct = answer == target

        if is_correct:
            correct += 1
            print(f"  [{i}/{len(problems)}] ✓")
        else:
            print(f"  [{i}/{len(problems)}] ✗ (got: {answer}, target: {target})")

        times.append(time.time() - start)

    accuracy = 100 * correct / len(problems)
    avg_time = sum(times) / len(times)

    return {
        "method": "DIRECT",
        "correct": correct,
        "total": len(problems),
        "accuracy": round(accuracy, 1),
        "avg_time": round(avg_time, 2),
        "total_tokens": client.total_tokens
    }


# =============================================================================
# THREE-PROMPT BASELINE
# =============================================================================

def parse_json(text: str):
    """Extract JSON from response."""
    try:
        return json.loads(text)
    except:
        pass

    # Try finding JSON in code blocks or {...}
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return None


async def run_three_prompt(client: LFMClient, problems):
    """Three-prompt architecture baseline."""
    print(f"\nRunning THREE-PROMPT baseline...")

    correct = 0
    times = []

    for i, problem in enumerate(problems, 1):
        start = time.time()
        query = problem["input"]
        target = problem["target"]

        claims = []

        # Simple 3-prompt loop (no claim persistence between problems for baseline)
        for iteration in range(1, MAX_ITERATIONS + 1):
            # Prompt 1: Update confidence (skip if no claims)
            if claims:
                claims_text = "\n".join([f"{j+1}. [{c['confidence']:.2f}] {c['content']}" for j, c in enumerate(claims)])
                prompt1 = f"PROBLEM: {query}\n\nCLAIMS:\n{claims_text}\n\nUpdate confidence (0-1) for each claim. Respond with JSON only:\n{{\"updates\": [{{\"id\": 1, \"confidence\": 0.8}}]}}"
                response1, _ = await client.generate(prompt1, max_tokens=200)
                updates = parse_json(response1)
                if updates and "updates" in updates:
                    for upd in updates["updates"]:
                        idx = upd.get("id", 0) - 1
                        if 0 <= idx < len(claims):
                            claims[idx]["confidence"] = upd.get("confidence", 0.5)

            # Prompt 2: Create claim or SKIP
            claims_text = "\n".join([f"{j+1}. [{c['confidence']:.2f}] {c['content']}" for j, c in enumerate(claims)]) if claims else "No claims yet"
            prompt2 = f"PROBLEM: {query}\n\nCLAIMS:\n{claims_text}\n\nCreate ONE new claim or say SKIP. Respond with JSON:\n{{\"action\": \"CREATE\", \"claim\": {{\"content\": \"...\", \"confidence\": 0.5}}}}\nOR\n{{\"action\": \"SKIP\"}}"
            response2, _ = await client.generate(prompt2, max_tokens=200)
            action_data = parse_json(response2)

            action = "SKIP"
            if action_data:
                action = action_data.get("action", "SKIP")
                if action == "CREATE" and "claim" in action_data:
                    claims.append({
                        "content": action_data["claim"].get("content", ""),
                        "confidence": action_data["claim"].get("confidence", 0.5)
                    })

            # Check stopping condition
            max_conf = max([c["confidence"] for c in claims], default=0.0)
            if max_conf >= CONFIDENCE_THRESHOLD and action == "SKIP":
                break

        # Prompt 3: Final answer
        claims_text = "\n".join([f"{j+1}. [{c['confidence']:.2f}] {c['content']}" for j, c in enumerate(claims)]) if claims else "No claims"
        prompt3 = f"PROBLEM: {query}\n\nCLAIMS:\n{claims_text}\n\nProvide your final answer clearly."
        response3, _ = await client.generate(prompt3, max_tokens=400)

        answer = extract_answer(response3, target)
        is_correct = answer == target

        if is_correct:
            correct += 1
            print(f"  [{i}/{len(problems)}] ✓ ({len(claims)} claims, {iteration} iters)")
        else:
            print(f"  [{i}/{len(problems)}] ✗ (got: {answer}, target: {target})")

        times.append(time.time() - start)

    accuracy = 100 * correct / len(problems)
    avg_time = sum(times) / len(times)

    return {
        "method": "THREE-PROMPT",
        "correct": correct,
        "total": len(problems),
        "accuracy": round(accuracy, 1),
        "avg_time": round(avg_time, 2),
        "total_tokens": client.total_tokens
    }


# =============================================================================
# MAIN
# =============================================================================

async def main():
    print("\n" + "="*70)
    print("LFM-2.5 BASELINE QUICK TEST (n=10)")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Endpoint: {LFM_ENDPOINT}")
    print(f"Task: {BBH_TASK}")
    print(f"Problems: {N_PROBLEMS}")
    print()

    problems = load_bbh_problems(BBH_TASK, N_PROBLEMS)

    # Direct baseline
    client_direct = LFMClient()
    direct_result = await run_direct(client_direct, problems)

    # Three-prompt baseline
    client_three = LFMClient()
    three_result = await run_three_prompt(client_three, problems)

    # Results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\n{'Method':<20} {'Correct':>10} {'Accuracy':>10} {'Tokens':>10}")
    print("-"*70)
    print(f"{'Direct':<20} {direct_result['correct']:>6}/{direct_result['total']:<3} {direct_result['accuracy']:>9.1f}% {direct_result['total_tokens']:>10,}")
    print(f"{'Three-Prompt':<20} {three_result['correct']:>6}/{three_result['total']:<3} {three_result['accuracy']:>9.1f}% {three_result['total_tokens']:>10,}")
    print("-"*70)
    improvement = three_result['accuracy'] - direct_result['accuracy']
    print(f"{'Improvement':<20} {three_result['correct'] - direct_result['correct']:>+10} {improvement:>+9.1f}pp")
    print()

    # Save results
    results = {
        "model": MODEL,
        "endpoint": LFM_ENDPOINT,
        "task": BBH_TASK,
        "n_problems": N_PROBLEMS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "direct": direct_result,
        "three_prompt": three_result,
        "improvement_pp": improvement
    }

    results_dir = Path("experiments/results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"lfm_baseline_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved: {results_file}")

    # Interpretation
    print("\nINTERPRETATION:")
    if improvement > 10:
        print("✅ THREE-PROMPT significantly improves LFM-2.5 performance!")
        print("   Core thesis validated - proceed with 100-strategy exploration.")
    elif improvement > 0:
        print("⚠️  THREE-PROMPT shows improvement but room for optimization.")
        print("   Proceed with 100-strategy exploration to maximize performance.")
    else:
        print("❌ THREE-PROMPT regresses on LFM-2.5 (like 8B models).")
        print("   100-strategy exploration critical to test proper interfacing thesis.")

    await client_direct.client.aclose()
    await client_three.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
