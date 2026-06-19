#!/usr/bin/env python3
# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
32-Variant Prompt Batch Evaluation for Tiny Model Single-Claim Generation

Tests 32 systematically varied prompts to find optimal configuration for
small models to produce exactly 1 high-quality claim per turn.

Dimensions varied:
- System prompt style (4): minimal, detailed, role-based, constraint-focused
- Instruction format (4): imperative, interrogative, templated, example-based
- Output format (2): natural language, JSON structure

4 x 4 x 2 = 32 combinations

Usage:
    python experiments/prompt_batch_32.py --variant 0  # Run variant 0
    python experiments/prompt_batch_32.py --all        # Run all (sequential)
"""

import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI


# =============================================================================
# PROMPT VARIATION DEFINITIONS
# =============================================================================

# Dimension 1: System Prompt Styles (4 variants)
SYSTEM_PROMPTS = {
    "minimal": "You are a reasoning assistant. Output exactly one claim per response.",

    "detailed": """You are a careful reasoning assistant that builds knowledge through claims.

A claim is a single, atomic statement that captures one piece of reasoning.
Each claim must have: content (what you believe), type (observation/assertion/assumption), and confidence (0.0-1.0).

Output exactly ONE claim per response. Do not combine multiple ideas into one claim.""",

    "role": """You are a Socratic philosopher who builds arguments through precise claims.

Your role is to examine problems by stating exactly one well-formed claim at a time.
Each claim should be clear, specific, and independently verifiable.
Never rush - quality over quantity. One claim only.""",

    "constraint": """STRICT RULES:
1. Output EXACTLY ONE claim
2. Claim must be atomic (one idea only)
3. Claim must be falsifiable
4. Include confidence level
5. NO additional text or explanation
6. Violation of these rules is a critical failure"""
}

# Dimension 2: Instruction Formats (4 variants)
INSTRUCTION_FORMATS = {
    "imperative": """State your next claim about: {problem}

Claim:""",

    "interrogative": """What is the single most important claim you can make about: {problem}

Your claim:""",

    "templated": """Problem: {problem}

Complete exactly ONE claim:
Content: [your claim here]
Type: [observation|assertion|assumption|conjecture]
Confidence: [0.0-1.0]""",

    "example": """Problem: What is 2+2?
Claim: "The sum of 2 and 2 equals 4" (assertion, confidence: 1.0)

Problem: {problem}
Claim:"""
}

# Dimension 3: Output Format (2 variants)
OUTPUT_FORMATS = {
    "natural": "",  # No additional format constraint

    "json": """

Output as JSON: {{"content": "...", "type": "...", "confidence": 0.0-1.0}}"""
}

# Additional modifiers to test (applied to some variants)
MODIFIERS = {
    "brevity": "\n\nBe extremely concise. Maximum 20 words per claim.",
    "reasoning": "\n\nBriefly explain your reasoning before stating the claim.",
    "confidence_first": "\n\nStart with your confidence level, then state the claim.",
    "none": ""
}


@dataclass
class PromptVariant:
    """A specific prompt configuration."""
    id: int
    system_key: str
    instruction_key: str
    output_key: str
    modifier_key: str

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPTS[self.system_key] + MODIFIERS[self.modifier_key]

    def format_user_prompt(self, problem: str) -> str:
        base = INSTRUCTION_FORMATS[self.instruction_key].format(problem=problem)
        return base + OUTPUT_FORMATS[self.output_key]

    @property
    def name(self) -> str:
        return f"v{self.id:02d}_{self.system_key}_{self.instruction_key}_{self.output_key}"


def generate_all_variants() -> List[PromptVariant]:
    """Generate all 32 prompt variants."""
    variants = []
    variant_id = 0

    # Core 32 combinations (4 x 4 x 2)
    for sys_key in SYSTEM_PROMPTS:
        for inst_key in INSTRUCTION_FORMATS:
            for out_key in OUTPUT_FORMATS:
                # Cycle through modifiers
                mod_key = list(MODIFIERS.keys())[variant_id % len(MODIFIERS)]
                variants.append(PromptVariant(
                    id=variant_id,
                    system_key=sys_key,
                    instruction_key=inst_key,
                    output_key=out_key,
                    modifier_key=mod_key
                ))
                variant_id += 1

    return variants


# =============================================================================
# TEST PROBLEMS
# =============================================================================

TEST_PROBLEMS = [
    # Math reasoning
    "A store sells apples for $2 each. If I buy 5 apples and pay with $20, how much change do I get?",
    "What is 15% of 80?",
    "If a train travels 60 miles in 1 hour, how far does it travel in 2.5 hours?",

    # Logic
    "All cats are animals. Whiskers is a cat. What can we conclude?",
    "If it rains, the ground gets wet. The ground is wet. Did it rain?",
    "There are 3 boxes: one has only apples, one has only oranges, one has both. All labels are wrong. You pick one fruit from the 'Both' box. How do you determine all box contents?",

    # General reasoning
    "Why might a company choose to hire remote workers?",
    "What are the trade-offs between speed and accuracy in decision making?",

    # Decomposition
    "How would you plan a birthday party for 20 people?",
    "What steps are needed to debug a failing software test?",
]


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class ClaimEvaluation:
    """Evaluation of a single claim output."""
    problem: str
    raw_output: str
    has_exactly_one_claim: bool
    claim_is_relevant: bool
    claim_is_atomic: bool
    has_confidence: bool
    confidence_value: Optional[float]
    word_count: int
    response_time: float
    error: Optional[str] = None


class LLMClient:
    """Minimal async client."""

    def __init__(self, model: str = "meta-llama/llama-3.1-8b-instruct"):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = model
        self._client = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def generate(self, prompt: str, system_prompt: str, max_tokens: int = 200) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


def evaluate_claim_output(problem: str, output: str, response_time: float) -> ClaimEvaluation:
    """Evaluate quality of claim output."""
    if not output:
        return ClaimEvaluation(
            problem=problem,
            raw_output="",
            has_exactly_one_claim=False,
            claim_is_relevant=False,
            claim_is_atomic=False,
            has_confidence=False,
            confidence_value=None,
            word_count=0,
            response_time=response_time,
            error="Empty output"
        )

    # Count potential claims (sentences ending in period, or JSON objects)
    claim_markers = len(re.findall(r'[.!?]\s*(?=[A-Z"]|$)', output))
    json_objects = len(re.findall(r'\{[^}]+\}', output))

    # For JSON format, count objects
    if json_objects > 0:
        has_one_claim = json_objects == 1
    else:
        # For natural language, be more lenient - look for single statement
        has_one_claim = claim_markers <= 2 and len(output.split('\n\n')) <= 2

    # Check for confidence
    conf_match = re.search(r'(?:confidence|conf)[:\s]*([0-9.]+)', output, re.I)
    if not conf_match:
        conf_match = re.search(r'"confidence"[:\s]*([0-9.]+)', output)
    if not conf_match:
        conf_match = re.search(r'\b(0\.\d+|1\.0?)\b', output)

    has_confidence = conf_match is not None
    confidence_value = float(conf_match.group(1)) if conf_match else None

    # Check relevance (simple keyword overlap)
    problem_words = set(problem.lower().split())
    output_words = set(output.lower().split())
    overlap = len(problem_words & output_words)
    is_relevant = overlap >= 2

    # Check atomicity (single idea = relatively short)
    word_count = len(output.split())
    is_atomic = word_count <= 50

    return ClaimEvaluation(
        problem=problem,
        raw_output=output,
        has_exactly_one_claim=has_one_claim,
        claim_is_relevant=is_relevant,
        claim_is_atomic=is_atomic,
        has_confidence=has_confidence,
        confidence_value=confidence_value,
        word_count=word_count,
        response_time=response_time
    )


async def evaluate_variant(
    variant: PromptVariant,
    client: LLMClient,
    problems: List[str]
) -> Dict[str, Any]:
    """Evaluate a single prompt variant on all problems."""

    results = []
    for problem in problems:
        start = time.time()
        try:
            output = await client.generate(
                prompt=variant.format_user_prompt(problem),
                system_prompt=variant.system_prompt
            )
            elapsed = time.time() - start
            eval_result = evaluate_claim_output(problem, output, elapsed)
            results.append(eval_result)
        except Exception as e:
            results.append(ClaimEvaluation(
                problem=problem,
                raw_output="",
                has_exactly_one_claim=False,
                claim_is_relevant=False,
                claim_is_atomic=False,
                has_confidence=False,
                confidence_value=None,
                word_count=0,
                response_time=time.time() - start,
                error=str(e)
            ))

        await asyncio.sleep(0.5)  # Rate limiting

    # Compute aggregate scores
    n = len(results)
    scores = {
        "variant_id": variant.id,
        "variant_name": variant.name,
        "system_key": variant.system_key,
        "instruction_key": variant.instruction_key,
        "output_key": variant.output_key,
        "modifier_key": variant.modifier_key,
        "n_problems": n,
        "single_claim_rate": sum(r.has_exactly_one_claim for r in results) / n,
        "relevance_rate": sum(r.claim_is_relevant for r in results) / n,
        "atomicity_rate": sum(r.claim_is_atomic for r in results) / n,
        "confidence_rate": sum(r.has_confidence for r in results) / n,
        "avg_word_count": sum(r.word_count for r in results) / n,
        "avg_response_time": sum(r.response_time for r in results) / n,
        "error_rate": sum(1 for r in results if r.error) / n,
        # Composite score: weighted average
        "composite_score": (
            0.35 * sum(r.has_exactly_one_claim for r in results) / n +
            0.25 * sum(r.claim_is_relevant for r in results) / n +
            0.20 * sum(r.claim_is_atomic for r in results) / n +
            0.20 * sum(r.has_confidence for r in results) / n
        ),
    }

    return scores


async def run_single_variant(variant_id: int, model: str = "meta-llama/llama-3.1-8b-instruct"):
    """Run evaluation for a single variant."""
    variants = generate_all_variants()
    if variant_id >= len(variants):
        print(f"Error: variant_id {variant_id} out of range (0-{len(variants)-1})")
        return

    variant = variants[variant_id]
    client = LLMClient(model=model)

    print(f"\n{'='*60}")
    print(f"EVALUATING VARIANT {variant.id}: {variant.name}")
    print(f"{'='*60}")
    print(f"System: {variant.system_key}")
    print(f"Instruction: {variant.instruction_key}")
    print(f"Output: {variant.output_key}")
    print(f"Modifier: {variant.modifier_key}")
    print(f"Model: {model}")
    print()

    results = await evaluate_variant(variant, client, TEST_PROBLEMS)

    print(f"\nRESULTS:")
    print(f"  Single-claim rate: {results['single_claim_rate']:.1%}")
    print(f"  Relevance rate:    {results['relevance_rate']:.1%}")
    print(f"  Atomicity rate:    {results['atomicity_rate']:.1%}")
    print(f"  Confidence rate:   {results['confidence_rate']:.1%}")
    print(f"  Composite score:   {results['composite_score']:.3f}")
    print(f"  Avg word count:    {results['avg_word_count']:.1f}")
    print(f"  Avg response time: {results['avg_response_time']:.2f}s")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"prompt_v{variant_id:02d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to: {results_file}")

    return results


async def run_all_variants(model: str = "meta-llama/llama-3.1-8b-instruct"):
    """Run all 32 variants sequentially."""
    variants = generate_all_variants()
    all_results = []

    print(f"\n{'='*60}")
    print(f"BATCH EVALUATION: 32 PROMPT VARIANTS")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Problems: {len(TEST_PROBLEMS)}")
    print()

    client = LLMClient(model=model)

    for variant in variants:
        print(f"[{variant.id+1}/32] Testing {variant.name}...", end=" ", flush=True)
        results = await evaluate_variant(variant, client, TEST_PROBLEMS)
        all_results.append(results)
        print(f"score={results['composite_score']:.3f}")

    # Sort by composite score
    all_results.sort(key=lambda x: x['composite_score'], reverse=True)

    print(f"\n{'='*60}")
    print("TOP 5 VARIANTS")
    print(f"{'='*60}")
    for i, r in enumerate(all_results[:5]):
        print(f"{i+1}. {r['variant_name']}: {r['composite_score']:.3f}")
        print(f"   single={r['single_claim_rate']:.0%} rel={r['relevance_rate']:.0%} "
              f"atom={r['atomicity_rate']:.0%} conf={r['confidence_rate']:.0%}")

    print(f"\n{'='*60}")
    print("WORST 3 VARIANTS")
    print(f"{'='*60}")
    for r in all_results[-3:]:
        print(f"  {r['variant_name']}: {r['composite_score']:.3f}")

    # Save all results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"prompt_batch32_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nAll results saved to: {results_file}")

    return all_results


def print_variants():
    """Print all 32 variants for reference."""
    variants = generate_all_variants()
    print(f"\n32 PROMPT VARIANTS")
    print("="*60)
    for v in variants:
        print(f"v{v.id:02d}: sys={v.system_key:12s} inst={v.instruction_key:14s} "
              f"out={v.output_key:8s} mod={v.modifier_key}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", "-v", type=int, help="Run single variant by ID (0-31)")
    parser.add_argument("--all", "-a", action="store_true", help="Run all 32 variants")
    parser.add_argument("--list", "-l", action="store_true", help="List all variants")
    parser.add_argument("--model", "-m", default="meta-llama/llama-3.1-8b-instruct",
                        help="Model to test")
    args = parser.parse_args()

    if args.list:
        print_variants()
    elif args.variant is not None:
        asyncio.run(run_single_variant(args.variant, args.model))
    elif args.all:
        asyncio.run(run_all_variants(args.model))
    else:
        print("Usage: python prompt_batch_32.py --variant N | --all | --list")
        print_variants()
