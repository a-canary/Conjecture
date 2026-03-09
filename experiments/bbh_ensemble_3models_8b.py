#!/usr/bin/env python3
"""
BBH Three-Model Ensemble Experiment (Variation #1)

Tests whether ensemble voting across 3 diverse 8B models can compensate for
architectural incompatibility of three-prompt architecture with small models.

Hypothesis: Individual 8B models fail in different ways. Majority voting across
diverse models (Llama-3.1-8B, Qwen2.5-7B, Mistral-7B) should cover individual
failure modes and improve aggregate accuracy.

Ensemble Strategy:
- Run same problem on 3 different 8B models independently
- Each model uses three-prompt architecture (max_iterations=4)
- Majority vote determines final answer (2/3 agreement required)
- If no majority (3-way split), use highest-confidence model

Baseline: Single 8B three-prompt = 40-58%
Variation: 3-model ensemble = ?

Success: +15pp improvement (p<0.05) restores viability
Failure: No significant improvement → conclude 8B architecturally incompatible
"""

import asyncio
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from openai import AsyncOpenAI
from scipy import stats


# =============================================================================
# CONFIGURATION
# =============================================================================

# Three diverse 8B models for ensemble
ENSEMBLE_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mistral-7b-instruct"
]

N_PROBLEMS = int(os.environ.get("BENCHMARK_N", "50"))
BBH_TASK = os.environ.get("BBH_TASK", "logical_deduction_three_objects")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")

MAX_ITERATIONS = 4
CONFIDENCE_THRESHOLD = 0.7
MAX_CLAIMS = 10


# =============================================================================
# PROMPTS (Same as baseline three-prompt)
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
            print(f"  API error ({self.model}): {e}")
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

    # Try option patterns (A), (B), (C)
    option_match = re.search(r'\(([A-C])\)', text)
    if option_match:
        return f"({option_match.group(1)})"

    # Try "answer is X" patterns
    answer_match = re.search(r'answer is:?\s*\(([A-C])\)', text, re.IGNORECASE)
    if answer_match:
        return f"({answer_match.group(1)})"

    # Last line heuristic
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        for opt in ['(A)', '(B)', '(C)']:
            if opt in last_line:
                return opt

    return None


# =============================================================================
# ENSEMBLE VOTING
# =============================================================================

def ensemble_vote(results: List[Dict], target: str) -> Tuple[Optional[str], float]:
    """
    Perform majority voting across ensemble results.

    Returns:
        (answer, confidence) tuple
    """
    # Extract answers and confidences
    answers = []
    confidences = {}

    for result in results:
        answer = extract_answer(result["final_response"], target)
        confidence = result["final_confidence"]

        if answer:
            answers.append(answer)
            confidences[answer] = max(confidences.get(answer, 0.0), confidence)

    if not answers:
        return None, 0.0

    # Count votes
    vote_counts = Counter(answers)
    most_common = vote_counts.most_common()

    # Majority vote (2/3 or 3/3)
    if len(most_common) > 0:
        winner, count = most_common[0]

        # If majority exists (2+ votes), return winner
        if count >= 2:
            return winner, confidences.get(winner, 0.7)

        # No majority (3-way split): use highest confidence
        best_answer = max(answers, key=lambda a: confidences.get(a, 0.0))
        return best_answer, confidences.get(best_answer, 0.5)

    return None, 0.0


# =============================================================================
# BENCHMARK EXECUTION
# =============================================================================

@dataclass
class BenchmarkResult:
    method: str
    correct: int
    total: int
    accuracy: float
    avg_time: float
    total_tokens: int
    extraction_failures: int
    avg_iterations: float


async def run_ensemble(problems: List[Dict]) -> BenchmarkResult:
    """Run ensemble three-prompt across 3 models."""
    print(f"\nRunning ENSEMBLE three-prompt ({len(ENSEMBLE_MODELS)} models)...")

    correct = 0
    times = []
    extraction_failures = 0
    iteration_counts = []
    total_tokens = 0

    for i, problem in enumerate(problems, 1):
        prob_start = time.time()
        query = problem["input"]
        target = problem["target"]

        # Run all 3 models in parallel
        clients = [BenchmarkClient(model) for model in ENSEMBLE_MODELS]
        systems = [ThreePromptSystem(client) for client in clients]

        print(f"  [{i}/{len(problems)}] Running ensemble (3 models)...", end="", flush=True)

        tasks = [system.process_query(query) for system in systems]
        results = await asyncio.gather(*tasks)

        # Ensemble voting
        ensemble_answer, ensemble_conf = ensemble_vote(results, target)

        # Check correctness
        is_correct = ensemble_answer == target if ensemble_answer else False
        if is_correct:
            correct += 1
            print(f" ✓ (ensemble: {ensemble_answer}, conf: {ensemble_conf:.2f})")
        else:
            print(f" ✗ (ensemble: {ensemble_answer}, target: {target})")
            if not ensemble_answer:
                extraction_failures += 1

        prob_time = time.time() - prob_start
        times.append(prob_time)

        # Aggregate stats
        avg_iters = sum(r["total_iterations"] for r in results) / len(results)
        iteration_counts.append(avg_iters)

        model_tokens = sum(c.total_tokens for c in clients)
        total_tokens += model_tokens

    return BenchmarkResult(
        method="ENSEMBLE-3-MODELS",
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        avg_time=round(sum(times) / len(times), 2),
        total_tokens=total_tokens,
        extraction_failures=extraction_failures,
        avg_iterations=round(sum(iteration_counts) / len(iteration_counts), 2)
    )


async def run_benchmark():
    """Run BBH three-model ensemble benchmark - Variation #1."""
    print("\n" + "="*70)
    print("BBH THREE-MODEL ENSEMBLE BENCHMARK (Variation #1)")
    print("="*70)
    print(f"Models: {', '.join(ENSEMBLE_MODELS)}")
    print(f"Task: {BBH_TASK}")
    print(f"Problems: {N_PROBLEMS}")
    print(f"Max Iterations: {MAX_ITERATIONS}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Voting Strategy: Majority (2/3) or highest confidence")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Load data
    problems = load_bbh_problems(BBH_TASK, N_PROBLEMS)

    # Run ensemble
    ensemble_result = await run_ensemble(problems)

    # Results summary
    print("\n" + "="*70)
    print("BBH THREE-MODEL ENSEMBLE RESULTS - Variation #1")
    print("="*70)
    print(f"\n{'Method':<25} {'Correct':>10} {'Accuracy':>10} {'Avg Time':>10} {'Tokens':>12} {'Iters':>6}")
    print("-"*70)
    print(f"{'Ensemble (3 models)':<25} {ensemble_result.correct:>6}/{ensemble_result.total:<3} {ensemble_result.accuracy:>9.1f}% {ensemble_result.avg_time:>9.2f}s {ensemble_result.total_tokens:>12,} {ensemble_result.avg_iterations:>6.1f}")
    print("-"*70)
    print()

    # Baseline comparisons
    print("BASELINE COMPARISONS:")
    print(f"  Single 8B unlimited iterations: 40.0%")
    print(f"  Single 8B 5-claim limit:        48.0%")
    print(f"  Single 8B single-step:          58.0%")
    print(f"  Ensemble (3 models):            {ensemble_result.accuracy:.1f}%")
    print()

    if ensemble_result.extraction_failures > 0:
        print(f"Extraction failures: {ensemble_result.extraction_failures}")

    # Statistical analysis
    baseline_accuracy = 0.40  # Unlimited iterations baseline
    ensemble_accuracy = ensemble_result.accuracy / 100

    # Two-proportion z-test
    n1, p1 = N_PROBLEMS, ensemble_accuracy
    n2, p2 = 50, baseline_accuracy

    p_pooled = (p1*n1 + p2*n2) / (n1 + n2)
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    z = (p1 - p2) / se
    p_value = 2 * stats.norm.cdf(-abs(z))

    improvement = (p1 - p2) * 100

    print("STATISTICAL ANALYSIS:")
    print(f"  Ensemble vs baseline: {improvement:+.1f}pp")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant (p<0.05)? {'YES' if p_value < 0.05 else 'NO'}")
    print()

    # Conclusion
    if improvement > 15 and p_value < 0.05:
        conclusion = "SUCCESS: Ensemble significantly improves 8B performance"
    elif improvement > 5 and p_value < 0.05:
        conclusion = "PROMISING: Ensemble shows improvement but still below direct baseline"
    elif p_value < 0.10:
        conclusion = "MARGINAL: Ensemble shows trend but not statistically significant"
    else:
        conclusion = "FAILED: Ensemble does not significantly improve 8B performance"

    print(f"CONCLUSION: {conclusion}")
    print("="*70)

    # Save results
    results = {
        "benchmark": "BBH_ENSEMBLE",
        "task": BBH_TASK,
        "architecture": "three-prompt-ensemble-3models",
        "variation": "Variation #1: Three-model ensemble voting",
        "hypothesis": "Model diversity compensates for 8B architectural incompatibility",
        "ensemble_models": ENSEMBLE_MODELS,
        "voting_strategy": "Majority (2/3) or highest confidence fallback",
        "n_problems": N_PROBLEMS,
        "max_iterations": MAX_ITERATIONS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_context": {
            "single_8b_unlimited": 40.0,
            "single_8b_5claim": 48.0,
            "single_8b_single_step": 58.0,
            "single_8b_direct": "72-90% (high variance)",
            "source": "bbh_single_step_8b_20260308_022125.json"
        },
        "ensemble": asdict(ensemble_result),
        "improvement_vs_baseline_pp": improvement,
        "p_value": p_value,
        "conclusion": conclusion
    }

    results_dir = Path("experiments/results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"bbh_ensemble_3models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {results_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=50, help="Number of problems")
    parser.add_argument("--task", type=str, default=BBH_TASK, help="BBH task name")
    args = parser.parse_args()

    os.environ["BENCHMARK_N"] = str(args.n)
    os.environ["BBH_TASK"] = args.task

    asyncio.run(run_benchmark())
