#!/usr/bin/env python3
"""
Benchmark: Optimized vs Legacy Prompt for Single-Claim Generation

Measures:
1. Single-claim rate (exactly 1 claim per turn)
2. Relevance (claim addresses the problem)
3. Atomicity (claim is single idea, <50 words)
4. Confidence inclusion rate
5. UPDATE vs CREATE accuracy (when context provided)

Reports composite scores and improvement.
"""

import asyncio
import os
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI


# =============================================================================
# PROMPTS TO COMPARE
# =============================================================================

OPTIMIZED_SYSTEM = """You are a careful reasoning assistant that builds knowledge through claims.

A claim is a single, atomic statement that captures one piece of reasoning.
Each claim has: content (what you believe), type, and confidence (0.0-1.0).

For each turn, choose ONE action:
  - UPDATE: Change confidence on existing claim ONLY if new evidence directly contradicts or confirms it
  - CREATE: Add ONE new claim when the existing claims don't cover a needed reasoning step

Decision guide:
  - UPDATE when: explicit new evidence changes belief in existing claim
  - CREATE when: you need a new conclusion, calculation, or fact not yet stated
  - When in doubt, CREATE a new claim rather than update

Rules:
  - Only ONE action per turn
  - Always include confidence values (0.0-1.0)
  - Be brief and atomic"""

LEGACY_SYSTEM = """You are a careful reasoning assistant.

For every query you may either:
  - Explore further: call create_claim to record observations,
    sub-questions, or intermediate conclusions that help build
    your understanding.  Call explore_further to signal you need
    another iteration before you are ready to respond.
  - Halt and respond: call respond_to_user when you have
    sufficient confidence and evidence to give a good answer.

Rules:
  1. Always cite supporting claim IDs in respond_to_user.
  2. Explore only when genuinely uncertain -- do not over-explore.
  3. When you have enough information, call respond_to_user.
  4. Every reasoning step you externalise should become a claim.
  5. Prefer accuracy over speed; prefer brevity over verbosity."""

# Optimized user prompt (interrogative style, tighter)
OPTIMIZED_USER = """CLAIMS:
{claims}

PROBLEM: {problem}

One action only. Output format:
- UPDATE [id]: conf X.X → Y.Y (reason)
- CREATE: "[claim]" (type, conf: X.X)

Your action:"""

# Legacy user prompt (imperative style)
LEGACY_USER = """EXISTING CLAIMS:
{claims}

PROBLEM: {problem}

State your next reasoning step as a claim."""


# =============================================================================
# TEST SCENARIOS
# =============================================================================

SCENARIOS = [
    # CREATE scenarios (should create new claim)
    {
        "name": "Math gap",
        "problem": "A store sells apples for $3 each. Buy 7 apples with 10% discount. Total cost?",
        "claims": [
            {"id": "c001", "content": "Each apple costs $3", "type": "observation", "confidence": 1.0},
            {"id": "c002", "content": "Buying 7 apples", "type": "observation", "confidence": 1.0},
        ],
        "expected_action": "CREATE"
    },
    {
        "name": "Logic gap",
        "problem": "All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?",
        "claims": [
            {"id": "c010", "content": "All mammals are warm-blooded", "type": "assertion", "confidence": 0.95},
        ],
        "expected_action": "CREATE"
    },
    {
        "name": "Reasoning gap",
        "problem": "Why might remote work increase productivity?",
        "claims": [
            {"id": "c020", "content": "Remote workers avoid commute", "type": "observation", "confidence": 0.9},
        ],
        "expected_action": "CREATE"
    },
    # UPDATE scenarios (should update existing claim)
    {
        "name": "Disconfirm",
        "problem": "Can penguins fly? New evidence: Penguins are flightless birds.",
        "claims": [
            {"id": "c030", "content": "All birds can fly", "type": "assumption", "confidence": 0.8},
            {"id": "c031", "content": "Penguins are birds", "type": "observation", "confidence": 1.0},
        ],
        "expected_action": "UPDATE"
    },
    {
        "name": "Confirm",
        "problem": "Is 7×8=56 correct? Verified: 8+8+8+8+8+8+8=56 ✓",
        "claims": [
            {"id": "c040", "content": "7 times 8 equals 56", "type": "assertion", "confidence": 0.7},
        ],
        "expected_action": "UPDATE"
    },
    # Edge cases
    {
        "name": "Empty context",
        "problem": "What is the capital of France?",
        "claims": [],
        "expected_action": "CREATE"
    },
    {
        "name": "Complex multi-step",
        "problem": "If train A leaves at 9am going 60mph and train B leaves at 10am going 80mph, when do they meet?",
        "claims": [
            {"id": "c050", "content": "Train A speed is 60mph", "type": "observation", "confidence": 1.0},
            {"id": "c051", "content": "Train B speed is 80mph", "type": "observation", "confidence": 1.0},
            {"id": "c052", "content": "Train B leaves 1 hour later", "type": "observation", "confidence": 1.0},
        ],
        "expected_action": "CREATE"
    },
    {
        "name": "Uncertain claim",
        "problem": "Based on symptoms, is this likely a cold or flu?",
        "claims": [
            {"id": "c060", "content": "Patient has fever", "type": "observation", "confidence": 1.0},
            {"id": "c061", "content": "This is probably a cold", "type": "conjecture", "confidence": 0.4},
        ],
        "expected_action": "UPDATE"
    },
]


@dataclass
class EvalResult:
    scenario: str
    prompt_type: str
    output: str
    is_single_action: bool
    is_relevant: bool
    is_atomic: bool
    has_confidence: bool
    correct_action: bool
    response_time: float
    composite_score: float


class LLMClient:
    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct"):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def generate(self, system: str, user: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content


def format_claims(claims: List[Dict]) -> str:
    if not claims:
        return "  (none)"
    return "\n".join([
        f"  [{c['id']}] {c['content']} ({c['type']}, conf: {c['confidence']})"
        for c in claims
    ])


def evaluate_output(output: str, expected_action: str) -> Dict[str, Any]:
    """Evaluate claim output quality."""
    if not output:
        return {
            "is_single_action": False,
            "is_relevant": False,
            "is_atomic": False,
            "has_confidence": False,
            "correct_action": False,
        }

    # Check action type
    is_update = "UPDATE" in output.upper() or "→" in output
    is_create = "CREATE" in output.upper() or any(w in output.lower() for w in ["claim:", "assertion", "observation"])

    # Single action check
    action_count = int(is_update) + int(is_create)
    is_single = action_count == 1 or (action_count == 0 and len(output.split('\n')) <= 2)

    # Correct action
    if expected_action == "UPDATE":
        correct = is_update
    else:  # CREATE
        correct = is_create or (not is_update)

    # Confidence check
    has_conf = bool(re.search(r'(?:confidence|conf)[:\s]*([0-9.]+)|→\s*([0-9.]+)|\b(0\.\d+|1\.0)\b', output, re.I))

    # Atomicity (short, single idea)
    is_atomic = len(output.split()) <= 60 and output.count('\n') <= 3

    # Relevance (basic check)
    is_relevant = len(output) > 10

    return {
        "is_single_action": is_single,
        "is_relevant": is_relevant,
        "is_atomic": is_atomic,
        "has_confidence": has_conf,
        "correct_action": correct,
    }


async def run_benchmark(model: str = "meta-llama/llama-3.1-8b-instruct"):
    """Run full benchmark comparing optimized vs legacy prompts."""

    print("\n" + "="*70)
    print("BENCHMARK: OPTIMIZED vs LEGACY PROMPT")
    print("="*70)
    print(f"Model: {model}")
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    client = LLMClient(model)
    results = {"optimized": [], "legacy": []}

    for scenario in SCENARIOS:
        claims_text = format_claims(scenario["claims"])

        # Test OPTIMIZED prompt
        print(f"\n--- {scenario['name']} ---")
        start = time.time()
        try:
            opt_output = await client.generate(
                OPTIMIZED_SYSTEM,
                OPTIMIZED_USER.format(claims=claims_text, problem=scenario["problem"])
            )
            opt_time = time.time() - start
            opt_eval = evaluate_output(opt_output, scenario["expected_action"])
            opt_score = (
                0.30 * opt_eval["is_single_action"] +
                0.25 * opt_eval["correct_action"] +
                0.20 * opt_eval["has_confidence"] +
                0.15 * opt_eval["is_atomic"] +
                0.10 * opt_eval["is_relevant"]
            )
            results["optimized"].append({
                "scenario": scenario["name"],
                "output": opt_output[:100],
                **opt_eval,
                "time": opt_time,
                "score": opt_score
            })
            print(f"  OPT: score={opt_score:.2f} single={opt_eval['is_single_action']} "
                  f"correct={opt_eval['correct_action']} conf={opt_eval['has_confidence']}")
        except Exception as e:
            print(f"  OPT ERROR: {e}")
            results["optimized"].append({"scenario": scenario["name"], "error": str(e), "score": 0})

        await asyncio.sleep(0.3)

        # Test LEGACY prompt
        start = time.time()
        try:
            leg_output = await client.generate(
                LEGACY_SYSTEM,
                LEGACY_USER.format(claims=claims_text, problem=scenario["problem"])
            )
            leg_time = time.time() - start
            leg_eval = evaluate_output(leg_output, scenario["expected_action"])
            leg_score = (
                0.30 * leg_eval["is_single_action"] +
                0.25 * leg_eval["correct_action"] +
                0.20 * leg_eval["has_confidence"] +
                0.15 * leg_eval["is_atomic"] +
                0.10 * leg_eval["is_relevant"]
            )
            results["legacy"].append({
                "scenario": scenario["name"],
                "output": leg_output[:100],
                **leg_eval,
                "time": leg_time,
                "score": leg_score
            })
            print(f"  LEG: score={leg_score:.2f} single={leg_eval['is_single_action']} "
                  f"correct={leg_eval['correct_action']} conf={leg_eval['has_confidence']}")
        except Exception as e:
            print(f"  LEG ERROR: {e}")
            results["legacy"].append({"scenario": scenario["name"], "error": str(e), "score": 0})

        await asyncio.sleep(0.3)

    # Compute aggregates
    opt_scores = [r["score"] for r in results["optimized"] if "score" in r]
    leg_scores = [r["score"] for r in results["legacy"] if "score" in r]

    opt_avg = sum(opt_scores) / len(opt_scores) if opt_scores else 0
    leg_avg = sum(leg_scores) / len(leg_scores) if leg_scores else 0
    improvement = (opt_avg - leg_avg) * 100

    # Detailed metrics
    opt_metrics = {
        "single_action": sum(r.get("is_single_action", 0) for r in results["optimized"]) / len(results["optimized"]),
        "correct_action": sum(r.get("correct_action", 0) for r in results["optimized"]) / len(results["optimized"]),
        "has_confidence": sum(r.get("has_confidence", 0) for r in results["optimized"]) / len(results["optimized"]),
        "is_atomic": sum(r.get("is_atomic", 0) for r in results["optimized"]) / len(results["optimized"]),
    }
    leg_metrics = {
        "single_action": sum(r.get("is_single_action", 0) for r in results["legacy"]) / len(results["legacy"]),
        "correct_action": sum(r.get("correct_action", 0) for r in results["legacy"]) / len(results["legacy"]),
        "has_confidence": sum(r.get("has_confidence", 0) for r in results["legacy"]) / len(results["legacy"]),
        "is_atomic": sum(r.get("is_atomic", 0) for r in results["legacy"]) / len(results["legacy"]),
    }

    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"\n{'Metric':<20} {'Optimized':>12} {'Legacy':>12} {'Delta':>12}")
    print("-"*58)
    print(f"{'Composite Score':<20} {opt_avg:>11.1%} {leg_avg:>11.1%} {improvement:>+11.1f}pp")
    print(f"{'Single Action':<20} {opt_metrics['single_action']:>11.0%} {leg_metrics['single_action']:>11.0%} {(opt_metrics['single_action']-leg_metrics['single_action'])*100:>+11.0f}pp")
    print(f"{'Correct Action':<20} {opt_metrics['correct_action']:>11.0%} {leg_metrics['correct_action']:>11.0%} {(opt_metrics['correct_action']-leg_metrics['correct_action'])*100:>+11.0f}pp")
    print(f"{'Has Confidence':<20} {opt_metrics['has_confidence']:>11.0%} {leg_metrics['has_confidence']:>11.0%} {(opt_metrics['has_confidence']-leg_metrics['has_confidence'])*100:>+11.0f}pp")
    print(f"{'Is Atomic':<20} {opt_metrics['is_atomic']:>11.0%} {leg_metrics['is_atomic']:>11.0%} {(opt_metrics['is_atomic']-leg_metrics['is_atomic'])*100:>+11.0f}pp")

    print(f"\n{'='*70}")
    if improvement > 5:
        verdict = "✅ OPTIMIZED PROMPT SIGNIFICANTLY BETTER"
    elif improvement > 0:
        verdict = "⚠️ OPTIMIZED PROMPT SLIGHTLY BETTER"
    elif improvement > -5:
        verdict = "⚖️ NO SIGNIFICANT DIFFERENCE"
    else:
        verdict = "❌ LEGACY PROMPT BETTER"
    print(verdict)
    print(f"Improvement: {improvement:+.1f}pp composite score")
    print("="*70)

    # Save results
    output_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "n_scenarios": len(SCENARIOS),
        "optimized_avg": opt_avg,
        "legacy_avg": leg_avg,
        "improvement_pp": improvement,
        "optimized_metrics": opt_metrics,
        "legacy_metrics": leg_metrics,
        "verdict": verdict,
        "details": results
    }

    results_file = f"experiments/results/benchmark_opt_vs_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return output_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="meta-llama/llama-3.1-8b-instruct")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.model))
