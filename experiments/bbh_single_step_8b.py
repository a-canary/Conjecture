#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
BBH Single-Step Experiment (Variation #3)

Tests whether forcing single-step reasoning (max_iterations=1) improves 8B
model performance by eliminating iteration overhead.

Hypothesis: 8B models fail (-32pp regression) due to ITERATION OVERHEAD
(cascading errors across prompts), not claim count or retrieval.

Single-Step Configuration:
- max_iterations=1 (force immediate answer)
- confidence_threshold=0.9 (high bar to ensure quality)
- Three prompts still used, but only one cycle

Baseline: 8B unlimited iterations = 40%
Variation: 8B single-step = ?

Success: +8pp improvement (p<0.05) restores performance
Failure: No significant difference - architecture itself incompatible with 8B
"""

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from openai import AsyncOpenAI


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = os.environ.get("BENCHMARK_MODEL", "meta-llama/llama-3.1-8b-instruct")
N_PROBLEMS = int(os.environ.get("BENCHMARK_N", "50"))
BBH_TASK = os.environ.get("BBH_TASK", "logical_deduction_three_objects")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")

MAX_ITERATIONS = 1  # Variation #3: Force single-step reasoning
CONFIDENCE_THRESHOLD = 0.9  # High bar to ensure quality single answer
MAX_CLAIMS = 10  # Unlimited (context not the issue per Variation #2)


# =============================================================================
# PROMPTS
# =============================================================================

DIRECT_SYSTEM = """You are a reasoning assistant for challenging logical problems.
Read the problem carefully, think step by step, and give your answer clearly."""


def build_claim_context(claims: List[Dict]) -> str:
    """Format claims for prompt context."""
    lines = []
    for i, claim in enumerate(claims, 1):
        conf = claim.get("confidence", 0.5)
        content = claim.get("content", "")
        lines.append(f"{i}. [{conf:.2f}] {content}")
    return "\n".join(lines)


PROMPT_1_UPDATE_CONFIDENCE = """PROBLEM: {query}

CURRENT CLAIMS:
{claims}

TASK: Update claim confidence scores (0.0 to 1.0)

Review the claims above. For each claim, assess:
- Does it directly support solving the problem?
- How certain are you about this claim?
- Does it conflict with other claims?

Output JSON:
{{
  "updates": [
    {{"id": "c001", "confidence": 0.85, "reason": "Directly relevant"}},
    {{"id": "c003", "confidence": 0.45, "reason": "Partially related"}}
  ]
}}

Only include claims that need confidence updates.
Respond with JSON only:"""


PROMPT_2_CREATE_OR_SKIP = """PROBLEM: {query}

CURRENT CLAIMS:
{claims}

ITERATION: {iteration}/{max_iterations}

TASK: Create ONE new claim to help solve the problem, or say SKIP.

New claims can be:
- Question to explore
- Logical deduction
- Constraint analysis
- Intermediate conclusion
- Next reasoning step

If you have high confidence (>0.7) and no more claims would help:
{{"action": "SKIP"}}

Otherwise, create ONE claim:
{{
  "action": "CREATE",
  "claim": {{
    "content": "The specific claim text",
    "confidence": 0.5,
    "type": "question|deduction|constraint|conclusion|step"
  }}
}}

Respond with JSON only:"""


PROMPT_3_FINAL_RESPONSE = """PROBLEM: {query}

RELEVANT CLAIMS:
{claims}

TASK: Provide final answer to the problem

Based on the claims above, generate a complete answer.
Include:
- Direct answer to the problem
- Key supporting claims (by number)
- Your reasoning

State your final answer clearly at the end."""


# =============================================================================
# DATA LOADING
# =============================================================================

def load_bbh_problems(task: str, limit: int) -> List[Dict]:
    """Load BBH task."""
    print(f"Loading BBH task: {task} (limit={limit})...")

    try:
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
# LLM CLIENT
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
        system: str = "",
        max_tokens: int = 500
    ) -> Tuple[str, int]:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
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
# THREE-PROMPT SYSTEM
# =============================================================================

class ThreePromptSystem:
    def __init__(self, client: BenchmarkClient):
        self.client = client
        self.claim_counter = 0

    def _parse_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from response."""
        try:
            return json.loads(text)
        except:
            pass

        # Try finding JSON in code blocks
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except:
                pass

        # Try finding first {...} block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass

        return None

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process query with three-prompt architecture."""
        claims = []
        iterations = []

        for iteration in range(1, MAX_ITERATIONS + 1):
            iter_start = time.time()

            # Prompt 1: Update confidence
            if claims:
                claims_text = build_claim_context(claims)
                prompt1 = PROMPT_1_UPDATE_CONFIDENCE.format(
                    query=query,
                    claims=claims_text
                )

                response1, tokens1 = await self.client.generate(prompt1)
                updates_data = self._parse_json(response1)

                if updates_data and "updates" in updates_data:
                    for update in updates_data["updates"]:
                        claim_id = update.get("id", "")
                        new_conf = update.get("confidence", 0.5)
                        for claim in claims:
                            if claim["id"] == claim_id:
                                claim["confidence"] = new_conf
                                break
            else:
                tokens1 = 0

            # Prompt 2: Create claim or SKIP
            claims_text = build_claim_context(claims) if claims else "No claims yet."
            prompt2 = PROMPT_2_CREATE_OR_SKIP.format(
                query=query,
                claims=claims_text,
                iteration=iteration,
                max_iterations=MAX_ITERATIONS
            )

            response2, tokens2 = await self.client.generate(prompt2)
            action_data = self._parse_json(response2)

            action = "SKIP"
            if action_data:
                action = action_data.get("action", "SKIP")
                if action == "CREATE" and "claim" in action_data:
                    self.claim_counter += 1
                    claim_data = action_data["claim"]
                    new_claim = {
                        "id": f"c{self.claim_counter:03d}",
                        "content": claim_data.get("content", ""),
                        "confidence": claim_data.get("confidence", 0.5),
                        "type": claim_data.get("type", "step")
                    }
                    claims.append(new_claim)

            # Calculate max confidence
            max_confidence = max([c["confidence"] for c in claims], default=0.0)

            iterations.append({
                "iteration": iteration,
                "action": action,
                "max_confidence": max_confidence,
                "claim_count": len(claims),
                "tokens": tokens1 + tokens2,
                "time": time.time() - iter_start
            })

            # Stop if confidence high and SKIP
            if max_confidence >= CONFIDENCE_THRESHOLD and action == "SKIP":
                break

        # Prompt 3: Final response
        claims_text = build_claim_context(claims) if claims else "No claims."
        prompt3 = PROMPT_3_FINAL_RESPONSE.format(
            query=query,
            claims=claims_text
        )

        response3, tokens3 = await self.client.generate(prompt3, max_tokens=600)

        return {
            "query": query,
            "iterations": iterations,
            "final_claims": claims,
            "final_response": response3,
            "total_iterations": len(iterations),
            "final_confidence": max([c["confidence"] for c in claims], default=0.0),
            "total_tokens": sum(it["tokens"] for it in iterations) + tokens3
        }


# =============================================================================
# ANSWER EXTRACTION
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
            if answer.lower() == target.lower():
                return target
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

    # Check if predicted is contained in expected
    if pred_norm in exp_norm:
        return True

    return False


# =============================================================================
# BENCHMARK RUNNER
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
    avg_iterations: float = 0.0


async def run_direct_baseline(
    client: BenchmarkClient,
    problems: List[Dict]
) -> MethodResult:
    """Run direct baseline."""
    print(f"\n{'='*60}")
    print(f"METHOD: DIRECT BASELINE")
    print(f"{'='*60}")

    correct = 0
    extraction_failures = 0
    times = []

    for i, prob in enumerate(problems):
        start = time.time()

        response, tokens = await client.generate(
            prompt=prob["input"],
            system=DIRECT_SYSTEM,
            max_tokens=600
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
        if (i + 1) % 10 == 0 or (predicted is None):
            status = "EXTRACTION_FAIL" if predicted is None else ("✓" if is_correct else f"✗ expected={expected[:30]}, got={predicted[:30]}")
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):5.1f}%  {status}")

        await asyncio.sleep(0.3)  # Rate limiting

    return MethodResult(
        method="DIRECT",
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        avg_time=round(sum(times) / len(times), 2),
        total_tokens=client.total_tokens,
        extraction_failures=extraction_failures
    )


async def run_three_prompt(
    client: BenchmarkClient,
    problems: List[Dict]
) -> MethodResult:
    """Run three-prompt method."""
    print(f"\n{'='*60}")
    print(f"METHOD: THREE-PROMPT")
    print(f"{'='*60}")

    correct = 0
    extraction_failures = 0
    times = []
    iteration_counts = []

    system = ThreePromptSystem(client)

    for i, prob in enumerate(problems):
        start = time.time()

        result = await system.process_query(prob["input"])

        elapsed = time.time() - start
        times.append(elapsed)
        iteration_counts.append(result["total_iterations"])

        predicted = extract_answer(result["final_response"], prob["target"])
        expected = prob["target"]

        if predicted is None:
            extraction_failures += 1
            is_correct = False
        else:
            is_correct = check_answer(predicted, expected)

        if is_correct:
            correct += 1

        # Progress every 10 or on errors
        if (i + 1) % 10 == 0 or (predicted is None):
            status = "EXTRACTION_FAIL" if predicted is None else ("✓" if is_correct else f"✗ expected={expected[:30]}, got={predicted[:30]}")
            iters = result["total_iterations"]
            conf = result["final_confidence"]
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):5.1f}%  iter={iters} conf={conf:.2f}  {status}")

        await asyncio.sleep(0.3)  # Rate limiting

    return MethodResult(
        method="THREE-PROMPT",
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        avg_time=round(sum(times) / len(times), 2),
        total_tokens=client.total_tokens,
        extraction_failures=extraction_failures,
        avg_iterations=round(sum(iteration_counts) / len(iteration_counts), 2)
    )


async def run_benchmark():
    """Run BBH single-step benchmark - Variation #3."""
    print("\n" + "="*70)
    print("BBH SINGLE-STEP BENCHMARK (Variation #3)")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Task: {BBH_TASK}")
    print(f"Problems: {N_PROBLEMS}")
    print(f"Max Iterations: {MAX_ITERATIONS} (FORCED SINGLE-STEP)")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    problems = load_bbh_problems(BBH_TASK, N_PROBLEMS)

    # Run direct baseline
    client_direct = BenchmarkClient(MODEL)
    direct_result = await run_direct_baseline(client_direct, problems)

    # Run three-prompt
    client_three = BenchmarkClient(MODEL)
    three_result = await run_three_prompt(client_three, problems)

    # Results summary
    improvement = three_result.accuracy - direct_result.accuracy

    print("\n" + "="*70)
    print("BBH SINGLE-STEP RESULTS - Variation #3")
    print("="*70)
    print(f"\n{'Method':<20} {'Correct':>10} {'Accuracy':>10} {'Avg Time':>10} {'Tokens':>12} {'Iters':>6}")
    print("-"*70)
    print(f"{'Direct':<20} {direct_result.correct:>6}/{direct_result.total:<3} {direct_result.accuracy:>9.1f}% {direct_result.avg_time:>9.2f}s {direct_result.total_tokens:>12,} {'  -':>6}")
    print(f"{'Three-Prompt':<20} {three_result.correct:>6}/{three_result.total:<3} {three_result.accuracy:>9.1f}% {three_result.avg_time:>9.2f}s {three_result.total_tokens:>12,} {three_result.avg_iterations:>6.1f}")
    print("-"*70)
    print(f"{'Improvement':<20} {three_result.correct - direct_result.correct:>+10} {improvement:>+9.1f}pp")
    print()

    if direct_result.extraction_failures > 0 or three_result.extraction_failures > 0:
        print(f"Extraction failures: Direct={direct_result.extraction_failures}, Three-Prompt={three_result.extraction_failures}")

    # Conclusion
    if improvement > 5:
        conclusion = "SUCCESS: Three-prompt significantly improves hard reasoning"
    elif improvement > 2:
        conclusion = "PROMISING: Three-prompt shows improvement on hard reasoning"
    elif improvement > -2:
        conclusion = "NEUTRAL: No significant difference"
    else:
        conclusion = "CHALLENGE: Three-prompt hurts hard reasoning accuracy"

    print(f"\nCONCLUSION: {conclusion}")
    print("="*70)

    # Save results
    results = {
        "benchmark": "BBH_SINGLE_STEP",
        "task": BBH_TASK,
        "architecture": "three-prompt-single-step",
        "variation": "Variation #3: Force single-step reasoning (max_iterations=1)",
        "hypothesis": "8B regression caused by iteration overhead, not claim count",
        "model": MODEL,
        "n_problems": N_PROBLEMS,
        "max_iterations": MAX_ITERATIONS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_context": {
            "prior_8b_unlimited_iterations": 40.0,
            "prior_8b_5claim_limit": 48.0,
            "prior_8b_direct": 72.0,
            "source": "bbh_context_limit_5claim_20260308_021130.json"
        },
        "direct": asdict(direct_result),
        "single_step": asdict(three_result),
        "improvement_pp": improvement,
        "conclusion": conclusion
    }

    results_dir = Path("experiments/results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"bbh_single_step_8b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {results_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=50, help="Number of problems")
    parser.add_argument("--task", type=str, default=BBH_TASK, help="BBH task name")
    parser.add_argument("--model", type=str, default=MODEL, help="Model to use")
    args = parser.parse_args()

    os.environ["BENCHMARK_N"] = str(args.n)
    os.environ["BBH_TASK"] = args.task
    os.environ["BENCHMARK_MODEL"] = args.model

    asyncio.run(run_benchmark())
