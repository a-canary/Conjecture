#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
Multi-Model Benchmark: Optimized Prompt for Single-Claim Generation

Tests the optimized prompt across multiple models and providers:
- Local LM Studio (liquid/lfm2.5-1.2b)
- OpenRouter small models (8B)
- OpenRouter medium models (14-32B)
- OpenRouter large models (70B+)

Reports benchmark scores for each model.
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODELS = {
    # Local LM Studio
    "lfm2.5-1.2b-local": {
        "provider": "lmstudio",
        "base_url": "http://100.73.201.58:1234/v1",
        "model": "liquid/lfm2.5-1.2b",  # Verified via /v1/models
        "size": "1.2B",
        "category": "tiny"
    },
    # OpenRouter tiny/small
    "lfm2.5-1.2b-or": {
        "provider": "openrouter",
        "model": "liquid/lfm-2.5-1.2b-instruct:free",
        "size": "1.2B",
        "category": "tiny"
    },
    "llama3.1-8b": {
        "provider": "openrouter",
        "model": "meta-llama/llama-3.1-8b-instruct",
        "size": "8B",
        "category": "small"
    },
    # Medium models
    "qwen3-14b": {
        "provider": "openrouter",
        "model": "qwen/qwen3-14b",
        "size": "14B",
        "category": "medium"
    },
    "mistral-small-24b": {
        "provider": "openrouter",
        "model": "mistralai/mistral-small-3.1-24b-instruct",
        "size": "24B",
        "category": "medium"
    },
    "qwen3-30b": {
        "provider": "openrouter",
        "model": "qwen/qwen3-30b-a3b",
        "size": "30B",
        "category": "medium"
    },
    # Large models
    "qwen2.5-72b": {
        "provider": "openrouter",
        "model": "qwen/qwen-2.5-72b-instruct",
        "size": "72B",
        "category": "large"
    },
    "qwen3-235b": {
        "provider": "openrouter",
        "model": "qwen/qwen3-235b-a22b",
        "size": "235B",
        "category": "large"
    },
    "deepseek-v3": {
        "provider": "openrouter",
        "model": "deepseek/deepseek-chat-v3-0324",
        "size": "685B",
        "category": "frontier"
    },
}


# =============================================================================
# OPTIMIZED PROMPT (winner from 32-variant batch)
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

OPTIMIZED_USER = """CLAIMS:
{claims}

PROBLEM: {problem}

One action only. Output format:
- UPDATE [id]: conf X.X → Y.Y (reason)
- CREATE: "[claim]" (type, conf: X.X)

Your action:"""


# =============================================================================
# TEST SCENARIOS
# =============================================================================

SCENARIOS = [
    # CREATE scenarios
    {
        "name": "Math calculation",
        "problem": "A store sells apples for $3 each. Buy 7 apples with 10% discount. Total cost?",
        "claims": [
            {"id": "c001", "content": "Each apple costs $3", "type": "observation", "confidence": 1.0},
            {"id": "c002", "content": "Buying 7 apples", "type": "observation", "confidence": 1.0},
        ],
        "expected": "CREATE"
    },
    {
        "name": "Logic deduction",
        "problem": "All mammals are warm-blooded. Whales are mammals. Are whales warm-blooded?",
        "claims": [
            {"id": "c010", "content": "All mammals are warm-blooded", "type": "assertion", "confidence": 0.95},
        ],
        "expected": "CREATE"
    },
    {
        "name": "Simple arithmetic",
        "problem": "What is 15 + 27?",
        "claims": [],
        "expected": "CREATE"
    },
    # UPDATE scenarios
    {
        "name": "Disconfirm evidence",
        "problem": "Can penguins fly? Evidence: Penguins are flightless birds.",
        "claims": [
            {"id": "c020", "content": "All birds can fly", "type": "assumption", "confidence": 0.8},
            {"id": "c021", "content": "Penguins are birds", "type": "observation", "confidence": 1.0},
        ],
        "expected": "UPDATE"
    },
    {
        "name": "Confirm calculation",
        "problem": "Is 7×8=56 correct? Verified: 8+8+8+8+8+8+8=56 ✓",
        "claims": [
            {"id": "c030", "content": "7 times 8 equals 56", "type": "assertion", "confidence": 0.7},
        ],
        "expected": "UPDATE"
    },
    # Edge cases
    {
        "name": "Capital city",
        "problem": "What is the capital of France?",
        "claims": [],
        "expected": "CREATE"
    },
    {
        "name": "Multi-step reasoning",
        "problem": "Train A leaves at 9am going 60mph, Train B leaves at 10am going 80mph. When do they meet?",
        "claims": [
            {"id": "c040", "content": "Train A speed is 60mph", "type": "observation", "confidence": 1.0},
            {"id": "c041", "content": "Train B speed is 80mph", "type": "observation", "confidence": 1.0},
        ],
        "expected": "CREATE"
    },
    {
        "name": "Uncertain conjecture",
        "problem": "Patient has fever. Is this a cold or flu?",
        "claims": [
            {"id": "c050", "content": "Patient has fever", "type": "observation", "confidence": 1.0},
            {"id": "c051", "content": "This is probably a cold", "type": "conjecture", "confidence": 0.4},
        ],
        "expected": "UPDATE"
    },
]


# =============================================================================
# CLIENT
# =============================================================================

class MultiModelClient:
    """Client that can connect to multiple providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config["provider"]
        self._client = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            if self.provider == "lmstudio":
                import httpx
                self._client = AsyncOpenAI(
                    api_key="lm-studio",
                    base_url=self.config["base_url"],
                    timeout=httpx.Timeout(60.0, connect=10.0),
                    http_client=httpx.AsyncClient(
                        timeout=httpx.Timeout(60.0, connect=10.0),
                        verify=False  # Local server may not have SSL
                    )
                )
            else:  # openrouter
                self._client = AsyncOpenAI(
                    api_key=os.environ.get("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1"
                )
        return self._client

    async def generate(self, system: str, user: str, timeout: float = 60.0) -> str:
        client = self._get_client()
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self.config["model"],
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    temperature=0.3,
                    max_tokens=200
                ),
                timeout=timeout
            )
            return response.choices[0].message.content
        except asyncio.TimeoutError:
            raise Exception(f"Timeout after {timeout}s")


# =============================================================================
# EVALUATION
# =============================================================================

def format_claims(claims: List[Dict]) -> str:
    if not claims:
        return "  (none)"
    return "\n".join([
        f"  [{c['id']}] {c['content']} ({c['type']}, conf: {c['confidence']})"
        for c in claims
    ])


def evaluate_output(output: str, expected: str) -> Dict[str, Any]:
    """Evaluate claim output quality."""
    if not output:
        return {
            "is_single_action": False,
            "is_relevant": False,
            "is_atomic": False,
            "has_confidence": False,
            "correct_action": False,
            "score": 0.0
        }

    # Check action type
    is_update = "UPDATE" in output.upper() or "→" in output
    is_create = "CREATE" in output.upper() or any(w in output.lower() for w in ["claim:", "assertion", "observation", "type:"])

    # Single action
    is_single = not (is_update and is_create)

    # Correct action
    if expected == "UPDATE":
        correct = is_update and not is_create
    else:
        correct = is_create or (not is_update and not is_create and len(output) > 10)

    # Confidence
    has_conf = bool(re.search(r'(?:confidence|conf)[:\s]*([0-9.]+)|→\s*([0-9.]+)|\b(0\.\d+|1\.0)\b', output, re.I))

    # Atomicity
    is_atomic = len(output.split()) <= 60 and output.count('\n') <= 4

    # Relevance
    is_relevant = len(output) > 10 and len(output) < 500

    # Composite score
    score = (
        0.30 * is_single +
        0.25 * correct +
        0.20 * has_conf +
        0.15 * is_atomic +
        0.10 * is_relevant
    )

    return {
        "is_single_action": is_single,
        "is_relevant": is_relevant,
        "is_atomic": is_atomic,
        "has_confidence": has_conf,
        "correct_action": correct,
        "score": score
    }


@dataclass
class ModelResult:
    model_name: str
    model_id: str
    size: str
    category: str
    n_scenarios: int
    composite_score: float
    single_action_rate: float
    correct_action_rate: float
    confidence_rate: float
    atomicity_rate: float
    avg_response_time: float
    error_rate: float
    errors: List[str]


async def benchmark_model(model_name: str, config: Dict[str, Any]) -> ModelResult:
    """Benchmark a single model."""
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name} ({config['size']})")
    print(f"{'='*60}")

    client = MultiModelClient(config)
    results = []
    errors = []
    times = []

    for scenario in SCENARIOS:
        claims_text = format_claims(scenario["claims"])
        user_prompt = OPTIMIZED_USER.format(claims=claims_text, problem=scenario["problem"])

        start = time.time()
        try:
            output = await client.generate(OPTIMIZED_SYSTEM, user_prompt)
            elapsed = time.time() - start
            times.append(elapsed)

            eval_result = evaluate_output(output, scenario["expected"])
            results.append(eval_result)

            status = "✅" if eval_result["score"] >= 0.7 else "⚠️" if eval_result["score"] >= 0.5 else "❌"
            print(f"  {status} {scenario['name']}: score={eval_result['score']:.2f} "
                  f"single={eval_result['is_single_action']} correct={eval_result['correct_action']} "
                  f"conf={eval_result['has_confidence']}")

        except Exception as e:
            elapsed = time.time() - start
            times.append(elapsed)
            errors.append(f"{scenario['name']}: {str(e)[:50]}")
            results.append({"score": 0, "is_single_action": False, "correct_action": False,
                           "has_confidence": False, "is_atomic": False})
            print(f"  ❌ {scenario['name']}: ERROR - {str(e)[:50]}")

        await asyncio.sleep(0.5)  # Rate limiting

    n = len(results)
    return ModelResult(
        model_name=model_name,
        model_id=config["model"],
        size=config["size"],
        category=config["category"],
        n_scenarios=n,
        composite_score=sum(r["score"] for r in results) / n if n > 0 else 0,
        single_action_rate=sum(r["is_single_action"] for r in results) / n if n > 0 else 0,
        correct_action_rate=sum(r["correct_action"] for r in results) / n if n > 0 else 0,
        confidence_rate=sum(r["has_confidence"] for r in results) / n if n > 0 else 0,
        atomicity_rate=sum(r["is_atomic"] for r in results) / n if n > 0 else 0,
        avg_response_time=sum(times) / len(times) if times else 0,
        error_rate=len(errors) / n if n > 0 else 0,
        errors=errors
    )


async def run_full_benchmark(model_filter: Optional[List[str]] = None):
    """Run benchmark across all models."""
    print("\n" + "="*70)
    print("MULTI-MODEL BENCHMARK: OPTIMIZED SINGLE-CLAIM PROMPT")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Models: {len(MODELS) if not model_filter else len(model_filter)}")
    print()

    all_results = []

    for model_name, config in MODELS.items():
        if model_filter and model_name not in model_filter:
            continue

        try:
            result = await benchmark_model(model_name, config)
            all_results.append(result)
        except Exception as e:
            print(f"  FATAL ERROR for {model_name}: {e}")
            all_results.append(ModelResult(
                model_name=model_name,
                model_id=config["model"],
                size=config["size"],
                category=config["category"],
                n_scenarios=0,
                composite_score=0,
                single_action_rate=0,
                correct_action_rate=0,
                confidence_rate=0,
                atomicity_rate=0,
                avg_response_time=0,
                error_rate=1.0,
                errors=[str(e)]
            ))

    # Sort by composite score
    all_results.sort(key=lambda x: x.composite_score, reverse=True)

    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Model':<20} {'Size':>8} {'Score':>8} {'Single':>8} {'Correct':>8} {'Conf':>8} {'Atomic':>8} {'Time':>8}")
    print("-"*88)

    for r in all_results:
        print(f"{r.model_name:<20} {r.size:>8} {r.composite_score:>7.1%} {r.single_action_rate:>7.0%} "
              f"{r.correct_action_rate:>7.0%} {r.confidence_rate:>7.0%} {r.atomicity_rate:>7.0%} "
              f"{r.avg_response_time:>7.2f}s")

    # Category averages
    print("\n" + "="*70)
    print("CATEGORY AVERAGES")
    print("="*70)
    categories = {}
    for r in all_results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r.composite_score)

    for cat in ["tiny", "small", "medium", "large", "frontier"]:
        if cat in categories:
            scores = categories[cat]
            avg = sum(scores) / len(scores)
            print(f"  {cat.upper():<12} avg={avg:.1%} (n={len(scores)})")

    # Best model
    if all_results:
        best = all_results[0]
        print(f"\n🏆 BEST MODEL: {best.model_name} ({best.size}) - {best.composite_score:.1%}")

    # Save results
    output_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_scenarios": len(SCENARIOS),
        "results": [asdict(r) for r in all_results],
        "best_model": all_results[0].model_name if all_results else None
    }

    results_dir = Path("experiments/results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"benchmark_multimodel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved to: {results_file}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", "-m", nargs="+", help="Specific models to test")
    parser.add_argument("--local-only", action="store_true", help="Only test local LM Studio")
    parser.add_argument("--small-only", action="store_true", help="Only test small models")
    args = parser.parse_args()

    if args.local_only:
        models = ["lfm2.5-1.2b-local"]
    elif args.small_only:
        models = ["lfm2.5-1.2b-local", "lfm2.5-1.2b-or", "llama3.1-8b"]
    elif args.models:
        models = args.models
    else:
        models = None  # All models

    asyncio.run(run_full_benchmark(models))
