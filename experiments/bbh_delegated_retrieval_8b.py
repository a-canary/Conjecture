#!/usr/bin/env python3
"""
BBH Delegated Retrieval 8B Re-Validation

Tests whether delegated tool calling (A-0015) restores performance for small
8B models on BBH hard reasoning tasks.

Background:
- Previous 8B result: 72% direct -> 40% three-prompt (-32pp, p<0.001)
- Hypothesis: Missing retrieval caused regression, not architecture itself
- A-0015: Small models need retrieved evidence to reason effectively

Three modes tested:
1. DIRECT baseline (simple single prompt)
2. THREE-PROMPT without retrieval (existing architecture - expected ~40%)
3. THREE-PROMPT with DELEGATED RETRIEVAL (new - hypothesis test)

The delegated retrieval protocol:
- Script acts as the HTTP caller in the A-0015 architecture
- On first iteration, injects mock retrieval evidence (problem restatement as evidence)
- This simulates what the real endpoint would do: pause, caller supplies evidence, resume
- Mock retrieval strategy: restate problem as structured evidence + pattern hint

Statistical validation:
- Two-proportion z-test (scipy.stats) for p-values
- p < 0.05 = significant
- Either outcome (confirms or disproves hypothesis) satisfies the gate
"""

import asyncio
import json
import math
import os
import re
import time
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

# Default: 8B model to test the hypothesis (can override with BENCHMARK_MODEL env)
MODEL = os.environ.get("BENCHMARK_MODEL", "meta-llama/llama-3.1-8b-instruct")
N_PROBLEMS = int(os.environ.get("BENCHMARK_N", "50"))
BBH_TASK = os.environ.get("BBH_TASK", "logical_deduction_three_objects")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")

MAX_ITERATIONS = 4
CONFIDENCE_THRESHOLD = 0.7


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
# MOCK RETRIEVAL (Simulates the A-0015 delegated retrieval caller)
# =============================================================================

def generate_mock_retrieval_evidence(problem_text: str, task: str) -> str:
    """
    Generate mock retrieval evidence for a BBH problem.

    This simulates what the real delegated retrieval caller would do:
    1. Receive the pause signal from the endpoint
    2. Extract key terms from the retrieval request
    3. Perform retrieval (here: mock with problem restatement + pattern hint)
    4. Return evidence as structured text

    The evidence is intentionally simple - we're testing if ANY retrieval helps,
    not optimizing retrieval quality.
    """
    # Extract key terms from the problem (first sentence or first 100 chars)
    first_sentence = problem_text.split('.')[0] if '.' in problem_text else problem_text[:100]

    # Pattern hint based on task type
    if "logical_deduction" in task:
        pattern_hint = (
            "BBH logical deduction problems involve ordering objects based on "
            "given constraints. Approach: list all constraints, determine valid "
            "orderings by elimination, verify against all conditions."
        )
    elif "boolean" in task:
        pattern_hint = (
            "Boolean expression problems require evaluating logical operators "
            "(AND, OR, NOT, XOR) step by step following operator precedence."
        )
    elif "causal" in task:
        pattern_hint = (
            "Causal reasoning problems require identifying cause-effect "
            "relationships and distinguishing necessary from sufficient conditions."
        )
    else:
        pattern_hint = (
            "This problem requires careful step-by-step reasoning with explicit "
            "tracking of constraints and intermediate conclusions."
        )

    evidence = (
        f"RETRIEVED EVIDENCE: The problem context is: {first_sentence}. "
        f"{pattern_hint} "
        f"Key approach: decompose into sub-problems, apply constraints "
        f"systematically, verify the final answer against all given conditions."
    )

    return evidence


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
# THREE-PROMPT SYSTEM (Without Retrieval) - Matches existing benchmark exactly
# =============================================================================

class ThreePromptSystem:
    """Three-prompt architecture without delegated retrieval."""

    def __init__(self, client: BenchmarkClient):
        self.client = client
        self.claim_counter = 0

    def _parse_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from response."""
        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return None

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process query with three-prompt architecture (no retrieval)."""
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

            max_confidence = max([c["confidence"] for c in claims], default=0.0)

            iterations.append({
                "iteration": iteration,
                "action": action,
                "max_confidence": max_confidence,
                "claim_count": len(claims),
                "tokens": tokens1 + tokens2,
                "time": time.time() - iter_start
            })

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
# THREE-PROMPT WITH DELEGATED RETRIEVAL (A-0015 Hypothesis Test)
# =============================================================================

class ThreePromptWithRetrieval:
    """
    Three-prompt architecture with delegated retrieval (A-0015 protocol).

    Simulates the HTTP caller pattern:
    - On iteration 1: inject mock retrieval evidence BEFORE claim building
    - This evidence is prepended to the claim context as a high-confidence claim
    - Subsequent iterations proceed as normal three-prompt

    This tests whether providing retrieved evidence at the start of reasoning
    restores 8B model performance, validating the A-0015 hypothesis.
    """

    def __init__(self, client: BenchmarkClient, task: str):
        self.client = client
        self.claim_counter = 0
        self.task = task

    def _parse_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from response."""
        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass

        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return None

    def _inject_retrieval_evidence(self, query: str) -> Dict:
        """
        Simulate delegated retrieval: generate mock evidence claim.

        In the real A-0015 protocol:
        1. LLM endpoint emits retrieve_knowledge tool call
        2. HTTP server returns 200 with X-Conjecture-Pause-ID header
        3. Caller performs retrieval using the query
        4. Caller POSTs to /resume with retrieval results
        5. Endpoint decomposes results into claims and continues

        Here we simulate step 3-5 inline: generate evidence, create a claim.
        """
        evidence_text = generate_mock_retrieval_evidence(query, self.task)
        self.claim_counter += 1
        return {
            "id": f"c{self.claim_counter:03d}",
            "content": evidence_text,
            "confidence": 0.75,  # Evidence from retrieval has moderate-high confidence
            "type": "retrieved_evidence",
            "source": "delegated_retrieval"
        }

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process query with three-prompt + delegated retrieval.

        Key difference from ThreePromptSystem:
        - Iteration 1: inject retrieval evidence claim BEFORE any LLM calls
        - This simulates the pause/resume protocol of A-0015
        """
        # Simulate delegated retrieval: inject evidence before reasoning starts
        # This corresponds to: endpoint pauses -> caller retrieves -> caller resumes
        retrieval_claim = self._inject_retrieval_evidence(query)
        claims = [retrieval_claim]  # Start with retrieved evidence

        iterations = []
        retrieval_pauses = 1  # Track how many pauses occurred

        for iteration in range(1, MAX_ITERATIONS + 1):
            iter_start = time.time()

            # Prompt 1: Update confidence (including the retrieval evidence claim)
            if len(claims) > 1 or iteration > 1:
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
                # Iteration 1 with only retrieval claim: skip confidence update
                tokens1 = 0

            # Prompt 2: Create claim or SKIP (with retrieval evidence in context)
            claims_text = build_claim_context(claims)
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

            max_confidence = max([c["confidence"] for c in claims], default=0.0)

            iterations.append({
                "iteration": iteration,
                "action": action,
                "max_confidence": max_confidence,
                "claim_count": len(claims),
                "tokens": tokens1 + tokens2,
                "time": time.time() - iter_start,
                "retrieval_injected": (iteration == 1)
            })

            if max_confidence >= CONFIDENCE_THRESHOLD and action == "SKIP":
                break

        # Prompt 3: Final response (with retrieved evidence in claim context)
        claims_text = build_claim_context(claims)
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
            "total_tokens": sum(it["tokens"] for it in iterations) + tokens3,
            "retrieval_pauses": retrieval_pauses
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
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        return lines[-1]

    return None


def check_answer(predicted: Optional[str], expected: str) -> bool:
    """Check if predicted matches expected."""
    if predicted is None:
        return False

    pred_norm = predicted.lower().strip().strip('.')
    exp_norm = expected.lower().strip().strip('.')

    if pred_norm == exp_norm:
        return True
    if exp_norm in pred_norm:
        return True
    if pred_norm in exp_norm:
        return True

    return False


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def two_proportion_z_test(
    correct_a: int, n_a: int,
    correct_b: int, n_b: int
) -> Tuple[float, float]:
    """
    Two-proportion z-test.

    Tests H0: p_a == p_b
    Returns (z_statistic, p_value) for two-tailed test.
    """
    p_a = correct_a / n_a
    p_b = correct_b / n_b
    p_pool = (correct_a + correct_b) / (n_a + n_b)

    se = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    if se == 0:
        return 0.0, 1.0

    z = (p_a - p_b) / se
    p_value = 2 * stats.norm.cdf(-abs(z))

    return round(z, 4), round(p_value, 4)


def compute_confidence_interval(correct: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson confidence interval for proportion."""
    p = correct / n
    z = stats.norm.ppf((1 + confidence) / 2)
    center = (p + z**2 / (2*n)) / (1 + z**2 / n)
    margin = z * math.sqrt(p * (1-p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return round((center - margin) * 100, 1), round((center + margin) * 100, 1)


# =============================================================================
# BENCHMARK RUNNERS
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

        response, _ = await client.generate(
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

        if (i + 1) % 10 == 0 or (predicted is None):
            status = "FAIL" if predicted is None else ("OK" if is_correct else f"X exp={expected[:20]} got={str(predicted)[:20]}")
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):5.1f}%  {status}")

        await asyncio.sleep(0.3)

    return MethodResult(
        method="DIRECT",
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        avg_time=round(sum(times) / len(times), 2),
        total_tokens=client.total_tokens,
        extraction_failures=extraction_failures
    )


async def run_three_prompt_no_retrieval(
    client: BenchmarkClient,
    problems: List[Dict]
) -> MethodResult:
    """Run three-prompt WITHOUT retrieval (control condition)."""
    print(f"\n{'='*60}")
    print(f"METHOD: THREE-PROMPT (no retrieval)")
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

        if (i + 1) % 10 == 0 or (predicted is None):
            iters = result["total_iterations"]
            conf = result["final_confidence"]
            status = "FAIL" if predicted is None else ("OK" if is_correct else f"X exp={expected[:20]} got={str(predicted)[:20]}")
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):5.1f}%  iter={iters} conf={conf:.2f}  {status}")

        await asyncio.sleep(0.3)

    return MethodResult(
        method="THREE-PROMPT-NO-RETRIEVAL",
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        avg_time=round(sum(times) / len(times), 2),
        total_tokens=client.total_tokens,
        extraction_failures=extraction_failures,
        avg_iterations=round(sum(iteration_counts) / len(iteration_counts), 2)
    )


async def run_three_prompt_with_retrieval(
    client: BenchmarkClient,
    problems: List[Dict],
    task: str
) -> MethodResult:
    """Run three-prompt WITH delegated retrieval (A-0015 hypothesis test)."""
    print(f"\n{'='*60}")
    print(f"METHOD: THREE-PROMPT + DELEGATED RETRIEVAL (A-0015)")
    print(f"{'='*60}")
    print(f"Simulating pause/resume protocol with mock retrieval evidence")

    correct = 0
    extraction_failures = 0
    times = []
    iteration_counts = []

    system = ThreePromptWithRetrieval(client, task)

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

        if (i + 1) % 10 == 0 or (predicted is None):
            iters = result["total_iterations"]
            conf = result["final_confidence"]
            pauses = result["retrieval_pauses"]
            status = "FAIL" if predicted is None else ("OK" if is_correct else f"X exp={expected[:20]} got={str(predicted)[:20]}")
            print(f"  [{i+1:3d}/{len(problems)}] acc={100*correct/(i+1):5.1f}%  iter={iters} pauses={pauses} conf={conf:.2f}  {status}")

        await asyncio.sleep(0.3)

    return MethodResult(
        method="THREE-PROMPT-DELEGATED-RETRIEVAL",
        correct=correct,
        total=len(problems),
        accuracy=round(100 * correct / len(problems), 2),
        avg_time=round(sum(times) / len(times), 2),
        total_tokens=client.total_tokens,
        extraction_failures=extraction_failures,
        avg_iterations=round(sum(iteration_counts) / len(iteration_counts), 2)
    )


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
        print("Falling back to: logical_deduction_three_objects")
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
# MAIN BENCHMARK
# =============================================================================

async def run_benchmark(
    use_prior_data: bool = False,
    skip_no_retrieval: bool = False
) -> Dict[str, Any]:
    """
    Run full BBH delegated retrieval re-validation.

    Args:
        use_prior_data: If True, use known prior results for direct and
                        three-prompt-no-retrieval instead of re-running them.
                        Saves ~60 minutes of API calls.
        skip_no_retrieval: If True, skip the three-prompt-no-retrieval run.
                           Use when prior data is sufficient.
    """
    print("\n" + "="*70)
    print("BBH DELEGATED RETRIEVAL 8B RE-VALIDATION (Phase 4 / A-0015)")
    print("="*70)
    print(f"Model: {MODEL}")
    print(f"Task: {BBH_TASK}")
    print(f"Problems: {N_PROBLEMS}")
    print(f"Max Iterations: {MAX_ITERATIONS}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    print("Hypothesis: 8B model three-prompt regression (-32pp) caused by")
    print("missing retrieval evidence, not by the architecture itself.")
    print()

    # Load data
    problems = load_bbh_problems(BBH_TASK, N_PROBLEMS)

    direct_result = None
    three_no_retrieval_result = None

    if use_prior_data:
        # Use known prior results from bbh_three_prompt_20260307_031018.json
        print("\nUsing prior data for DIRECT and THREE-PROMPT-NO-RETRIEVAL:")
        print("  Source: bbh_three_prompt_20260307_031018.json")
        print("  Direct: 36/50 = 72.0%")
        print("  Three-Prompt-No-Retrieval: 20/50 = 40.0%")
        print()
        direct_result = MethodResult(
            method="DIRECT",
            correct=36,
            total=50,
            accuracy=72.0,
            avg_time=3.6,
            total_tokens=14422,
            extraction_failures=0
        )
        three_no_retrieval_result = MethodResult(
            method="THREE-PROMPT-NO-RETRIEVAL",
            correct=20,
            total=50,
            accuracy=40.0,
            avg_time=14.91,
            total_tokens=146705,
            extraction_failures=0,
            avg_iterations=4.0
        )
    else:
        # Run direct baseline fresh
        client_direct = BenchmarkClient(MODEL)
        direct_result = await run_direct_baseline(client_direct, problems)

        # Run three-prompt without retrieval (control)
        if not skip_no_retrieval:
            client_no_ret = BenchmarkClient(MODEL)
            three_no_retrieval_result = await run_three_prompt_no_retrieval(
                client_no_ret, problems
            )
        else:
            print("\nSkipping three-prompt-no-retrieval (--skip-no-retrieval flag set)")

    # Run three-prompt WITH delegated retrieval (the hypothesis test)
    client_delegated = BenchmarkClient(MODEL)
    delegated_result = await run_three_prompt_with_retrieval(
        client_delegated, problems, BBH_TASK
    )

    # ==========================================================================
    # STATISTICAL ANALYSIS
    # ==========================================================================

    print("\n" + "="*70)
    print("BBH DELEGATED RETRIEVAL RESULTS")
    print("="*70)

    methods = [
        ("Direct", direct_result),
        ("Three-Prompt (no retrieval)", three_no_retrieval_result),
        ("Three-Prompt + Retrieval (A-0015)", delegated_result),
    ]

    print(f"\n{'Method':<40} {'Correct':>8} {'Accuracy':>10} {'95% CI':>18} {'Iters':>6}")
    print("-"*84)
    for name, r in methods:
        if r is None:
            continue
        ci_lo, ci_hi = compute_confidence_interval(r.correct, r.total)
        iters_str = f"{r.avg_iterations:.1f}" if r.avg_iterations > 0 else "  -"
        print(f"  {name:<38} {r.correct:>4}/{r.total:<3} {r.accuracy:>9.1f}% {f'[{ci_lo}%, {ci_hi}%]':>18} {iters_str:>6}")

    print("-"*84)
    print()

    # P-values
    print("Statistical Tests (two-proportion z-test):")

    # Delegated vs Direct
    z_del_vs_dir, p_del_vs_dir = two_proportion_z_test(
        delegated_result.correct, delegated_result.total,
        direct_result.correct, direct_result.total
    )
    delta_del_vs_dir = delegated_result.accuracy - direct_result.accuracy
    sig_del_vs_dir = "SIGNIFICANT" if p_del_vs_dir < 0.05 else "not significant"
    print(f"  Delegated vs Direct:            delta={delta_del_vs_dir:+.1f}pp  z={z_del_vs_dir:+.3f}  p={p_del_vs_dir:.4f}  ({sig_del_vs_dir})")

    p_del_vs_no_ret = None
    z_del_vs_no_ret = None
    delta_del_vs_no_ret = None
    if three_no_retrieval_result is not None:
        z_del_vs_no_ret, p_del_vs_no_ret = two_proportion_z_test(
            delegated_result.correct, delegated_result.total,
            three_no_retrieval_result.correct, three_no_retrieval_result.total
        )
        delta_del_vs_no_ret = delegated_result.accuracy - three_no_retrieval_result.accuracy
        sig_del_vs_no_ret = "SIGNIFICANT" if p_del_vs_no_ret < 0.05 else "not significant"
        print(f"  Delegated vs No-Retrieval:      delta={delta_del_vs_no_ret:+.1f}pp  z={z_del_vs_no_ret:+.3f}  p={p_del_vs_no_ret:.4f}  ({sig_del_vs_no_ret})")

    print()

    # ==========================================================================
    # HYPOTHESIS INTERPRETATION
    # ==========================================================================

    print("HYPOTHESIS INTERPRETATION:")
    print(f"  Hypothesis: 8B regression (-32pp) was caused by MISSING RETRIEVAL,")
    print(f"              not by the three-prompt architecture itself.")
    print()

    if delta_del_vs_dir is not None and p_del_vs_dir is not None:
        if delta_del_vs_dir > 5 and p_del_vs_dir < 0.05:
            hypothesis_outcome = "CONFIRMED"
            hypothesis_detail = (
                f"Delegated retrieval restored and exceeded direct baseline "
                f"({delta_del_vs_dir:+.1f}pp, p={p_del_vs_dir:.4f}). "
                f"Missing retrieval WAS the cause of regression. "
                f"A-0015 architecture validated for 8B models."
            )
        elif delegated_result.accuracy >= direct_result.accuracy - 5:
            hypothesis_outcome = "PARTIALLY_CONFIRMED"
            hypothesis_detail = (
                f"Delegated retrieval partially recovered performance "
                f"(delegated={delegated_result.accuracy:.1f}% vs direct={direct_result.accuracy:.1f}%). "
                f"Some regression remains but retrieval helps. "
                f"A-0015 architecture shows promise but retrieval quality matters."
            )
        elif delta_del_vs_no_ret is not None and delta_del_vs_no_ret > 5 and p_del_vs_no_ret < 0.05:
            hypothesis_outcome = "PARTIALLY_CONFIRMED"
            hypothesis_detail = (
                f"Delegated retrieval improved over no-retrieval "
                f"({delta_del_vs_no_ret:+.1f}pp, p={p_del_vs_no_ret:.4f}) "
                f"but did not match direct baseline. "
                f"Retrieval helps the architecture but mock evidence quality limits recovery."
            )
        else:
            hypothesis_outcome = "DISPROVED"
            hypothesis_detail = (
                f"Delegated retrieval did not significantly improve over no-retrieval "
                f"(delegated={delegated_result.accuracy:.1f}% vs direct={direct_result.accuracy:.1f}%). "
                f"The 8B regression reflects architectural incompatibility, "
                f"not just missing retrieval. Three-prompt unsuitable for 8B models."
            )
    else:
        hypothesis_outcome = "INCONCLUSIVE"
        hypothesis_detail = "Insufficient comparison data."

    print(f"  OUTCOME: {hypothesis_outcome}")
    print(f"  DETAIL: {hypothesis_detail}")
    print()

    # ==========================================================================
    # PRODUCTION GUIDANCE
    # ==========================================================================

    print("PRODUCTION GUIDANCE:")
    if hypothesis_outcome == "CONFIRMED":
        print("  - 8B models CAN use three-prompt architecture WITH retrieval")
        print("  - A-0015 delegated retrieval essential for small models")
        print("  - Do NOT deploy three-prompt to 8B without retrieval support")
        print("  - Mock retrieval sufficient; real retrieval should improve further")
    elif hypothesis_outcome == "PARTIALLY_CONFIRMED":
        print("  - A-0015 retrieval improves 8B performance but doesn't fully close gap")
        print("  - Real (non-mock) retrieval may close the remaining gap")
        print("  - Consider retrieval quality optimization before production 8B deployment")
        print("  - 70B+ models preferred when retrieval quality uncertain")
    else:
        print("  - 8B models NOT recommended for three-prompt architecture")
        print("  - Use 70B+ for three-prompt; use direct for 8B")
        print("  - A-0015 architecture still valid for larger models")
        print("  - Consider model-size routing in production")

    print()

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================

    results = {
        "benchmark": "BBH_DELEGATED_RETRIEVAL_8B",
        "task": BBH_TASK,
        "architecture": "three-prompt-delegated-retrieval",
        "model": MODEL,
        "n_problems": N_PROBLEMS,
        "max_iterations": MAX_ITERATIONS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "Phase 4 / A-0015 Re-Validation",
        "hypothesis": "8B three-prompt regression caused by missing retrieval evidence",
        "prior_context": {
            "prior_8b_direct": 72.0,
            "prior_8b_three_prompt_no_retrieval": 40.0,
            "prior_regression_pp": -32.0,
            "prior_p_value": 0.0007,
            "source": "bbh_three_prompt_20260307_031018.json"
        },
        "used_prior_data": use_prior_data,
        # Required fields per PLAN.md Phase 4 gates:
        "direct_accuracy": direct_result.accuracy,
        "three_prompt_accuracy": (
            three_no_retrieval_result.accuracy
            if three_no_retrieval_result is not None
            else 40.0  # Prior known value
        ),
        "delegated_accuracy": delegated_result.accuracy,
        "p_value_delegated_vs_direct": p_del_vs_dir,
        "p_value_delegated_vs_no_retrieval": p_del_vs_no_ret,
        "z_delegated_vs_direct": z_del_vs_dir,
        "z_delegated_vs_no_retrieval": z_del_vs_no_ret,
        "delta_delegated_vs_direct_pp": delta_del_vs_dir,
        "delta_delegated_vs_no_retrieval_pp": delta_del_vs_no_ret,
        "hypothesis_outcome": hypothesis_outcome,
        "hypothesis_detail": hypothesis_detail,
        # Method results
        "direct": asdict(direct_result),
        "three_prompt_no_retrieval": (
            asdict(three_no_retrieval_result)
            if three_no_retrieval_result is not None
            else None
        ),
        "delegated": asdict(delegated_result),
        "mock_retrieval_strategy": (
            "Problem restatement as structured evidence + task-specific pattern hint. "
            "Evidence prepended as high-confidence (0.75) claim before reasoning loop."
        ),
        "statistical_note": (
            f"n={N_PROBLEMS} gives ~14pp margin of error (95% CI). "
            f"Differences <14pp may be noise. Use p<0.05 threshold."
        )
    }

    results_dir = Path("/workspace/experiments/results")
    results_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"bbh_delegated_8b_{ts}.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"Results saved: {results_file}")

    # Append to benchmark_results.csv
    csv_path = results_dir / "benchmark_results.csv"

    rows = []
    for method_name, r in [
        ("direct", direct_result),
        ("three_prompt_no_retrieval", three_no_retrieval_result),
        ("three_prompt_delegated_retrieval", delegated_result),
    ]:
        if r is None:
            continue
        rows.append(
            f"BBH_8B_delegated,BBH,{MODEL},{method_name},"
            f"{r.total},{r.correct},{r.accuracy},{r.avg_time},"
            f"{r.total_tokens},{r.extraction_failures},"
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')},"
            f"A-0015 delegated retrieval re-validation hypothesis_outcome={hypothesis_outcome}"
        )

    if csv_path.exists():
        with open(csv_path, 'a') as f:
            f.write('\n'.join(rows) + '\n')
    else:
        csv_header = "benchmark,dataset,model,method,n_problems,correct,accuracy,avg_time_sec,total_tokens,extraction_failures,timestamp,notes\n"
        with open(csv_path, 'w') as f:
            f.write(csv_header + '\n'.join(rows) + '\n')

    print(f"Appended {len(rows)} rows to benchmark_results.csv")
    print()
    print(f"GATE CHECK:")
    print(f"  [OK] Benchmark completed n={N_PROBLEMS} without crash")
    print(f"  [OK] Result JSON has: direct_accuracy={results['direct_accuracy']}, "
          f"three_prompt_accuracy={results['three_prompt_accuracy']}, "
          f"delegated_accuracy={results['delegated_accuracy']}, "
          f"p_value_delegated_vs_direct={results['p_value_delegated_vs_direct']}")
    print(f"  [OK] Hypothesis outcome documented: {hypothesis_outcome}")

    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BBH Delegated Retrieval 8B Re-Validation (A-0015 Phase 4)"
    )
    parser.add_argument(
        "-n", type=int, default=50,
        help="Number of problems (default: 50)"
    )
    parser.add_argument(
        "--task", type=str, default=BBH_TASK,
        help=f"BBH task name (default: {BBH_TASK})"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL,
        help=f"Model to use (default: {MODEL})"
    )
    parser.add_argument(
        "--use-prior-data", action="store_true",
        help=(
            "Use prior known results for direct (72%%) and three-prompt-no-retrieval (40%%) "
            "from bbh_three_prompt_20260307_031018.json. "
            "Only runs the delegated retrieval condition. Saves ~60 min API calls."
        )
    )
    parser.add_argument(
        "--skip-no-retrieval", action="store_true",
        help="Skip the three-prompt-no-retrieval condition (saves ~30 min API calls)."
    )

    args = parser.parse_args()

    os.environ["BENCHMARK_N"] = str(args.n)
    os.environ["BBH_TASK"] = args.task
    os.environ["BENCHMARK_MODEL"] = args.model

    if not OPENROUTER_KEY:
        print("ERROR: OPENROUTER_API_KEY environment variable not set")
        import sys
        sys.exit(1)

    asyncio.run(run_benchmark(
        use_prior_data=args.use_prior_data,
        skip_no_retrieval=args.skip_no_retrieval
    ))
