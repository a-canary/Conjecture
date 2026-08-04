# Copyright 2025 a-canary
# SPDX-License-Identifier: Apache-2.0
"""
DeepEval Benchmark Suite for Conjecture
O-0008: 10 benchmarks, 40+ samples each, >= Direct on ALL, +20pp on 5
Benchmarks: GSM8K, LogiQA, TruthfulQA, 7 BigBenchHard reasoning tasks
Target: OSS models (20B class) where Conjecture should add value
Per O-0006: Uses 1 persistent session for claim accumulation across test cases.
Outputs to STATS.yaml

Baseline Caching: Direct model scores are cached to avoid redundant API calls.
Only update baseline cache when fixing benchmark bugs or parser issues.
"""

import argparse
import asyncio
import yaml
import json
import os
import sys
import re
sys.path.insert(0, '/workspace')

BASELINE_CACHE_FILE = "/workspace/benchmarks/baseline_cache.json"

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from deepeval.benchmarks import GSM8K, MathQA, HellaSwag, LogiQA, TruthfulQA, BoolQ, BigBenchHard, MMLU, Winogrande
    from deepeval.benchmarks.gsm8k.template import GSM8KTemplate
    from deepeval.benchmarks.math_qa.template import MathQATemplate
    from deepeval.benchmarks.hellaswag.template import HellaSwagTemplate
    from deepeval.benchmarks.logi_qa.template import LogiQATemplate
    from deepeval.benchmarks.truthful_qa.template import TruthfulQATemplate
    from deepeval.benchmarks.bool_q.template import BoolQTemplate
    from deepeval.benchmarks.big_bench_hard.template import BigBenchHardTemplate
    from deepeval.benchmarks.big_bench_hard.task import BigBenchHardTask
    from deepeval.benchmarks.mmlu.template import MMLUTemplate
    from deepeval.benchmarks.mmlu.task import MMLUTask
    from deepeval.benchmarks.winogrande.template import WinograndeTemplate
    from deepeval.models import GPTModel
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

from benchmarks.answer_extraction import extract_answer, check_answer_match, AnswerType


def create_chutes_model(api_key: str = None, model: str = "openai/gpt-oss-20b"):
    """Create DeepEval model using Chutes.ai endpoint"""
    api_key = api_key or os.environ.get("CHUTES_API_KEY")
    if not api_key:
        raise ValueError("CHUTES_API_KEY required. Set env var or pass api_key")
    return GPTModel(
        model=model,
        api_key=api_key,
        base_url="https://llm.chutes.ai/v1"
    )


def create_openrouter_model(api_key: str = None, model: str = "meta-llama/llama-3.1-8b-instruct"):
    """Create DeepEval model using OpenRouter endpoint"""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY required. Set env var or pass api_key")
    return GPTModel(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )


def _call_model(model, prompt: str) -> str:
    """Call a model and return the response text, handling (text, usage) tuples."""
    result = model.generate(prompt)
    if isinstance(result, tuple):
        return result[0]
    return str(result)


def load_baseline_cache() -> dict:
    """Load cached baseline scores. Only update when fixing benchmark/parser bugs."""
    if os.path.exists(BASELINE_CACHE_FILE):
        with open(BASELINE_CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_baseline_cache(cache: dict):
    """Save baseline scores to cache."""
    with open(BASELINE_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def get_cached_baseline(benchmark: str, model: str, n_samples: int) -> Optional[float]:
    """Get cached baseline score if available."""
    cache = load_baseline_cache()
    key = f"{benchmark}:{model}:{n_samples}"
    if key in cache:
        return cache[key]["score"]
    return None


def set_cached_baseline(benchmark: str, model: str, n_samples: int, score: float):
    """Cache a baseline score."""
    cache = load_baseline_cache()
    key = f"{benchmark}:{model}:{n_samples}"
    cache[key] = {
        "score": score,
        "timestamp": datetime.now().isoformat(),
        "samples": n_samples
    }
    save_baseline_cache(cache)


def extract_gsm8k_answer(response: str) -> str:
    """Extract numeric answer from GSM8K response.

    GSM8K answers are numbers. Look for:
    - #### followed by number
    - "answer is X"
    - Final number in response
    """
    # Pattern 1: #### number
    match = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', response)
    if match:
        return match.group(1).replace(',', '')

    # Pattern 2: "the answer is X" or "= X"
    match = re.search(r'(?:answer\s+is|=)\s*(-?\d[\d,]*\.?\d*)', response, re.I)
    if match:
        return match.group(1).replace(',', '')

    # Pattern 3: Last number in response
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', response)
    if numbers:
        return numbers[-1].replace(',', '')

    return ""


def extract_mathqa_answer(response: str) -> str:
    """Extract multiple choice answer (a-e) from MathQA response."""
    # Look for explicit choice markers
    match = re.search(r'\b([a-e])\s*\)', response, re.I)
    if match:
        return match.group(1).lower()

    # Look for "answer is X" pattern
    match = re.search(r'answer\s+is\s*[:\s]*([a-e])', response, re.I)
    if match:
        return match.group(1).lower()

    # Look for standalone letter at end
    match = re.search(r'\b([a-e])\s*$', response.strip(), re.I)
    if match:
        return match.group(1).lower()

    return ""


def extract_hellaswag_answer(response: str) -> str:
    """Extract multiple choice answer (A-D) from HellaSwag response."""
    # Look for explicit choice
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1)

    return ""


def extract_logiqa_answer(response: str) -> str:
    """Extract multiple choice answer (A-D) from LogiQA response."""
    # Look for explicit choice markers
    match = re.search(r'\b([A-D])\s*[\.\):]', response)
    if match:
        return match.group(1).upper()

    # Look for "answer is X" pattern
    match = re.search(r'answer\s+is\s*[:\s]*([A-D])', response, re.I)
    if match:
        return match.group(1).upper()

    # Look for standalone letter
    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1).upper()

    return ""


def extract_truthfulqa_answer(response: str) -> str:
    """Extract multiple choice answer from TruthfulQA response.

    TruthfulQA MC1 uses numeric answers (1, 2, 3, 4, 5, etc.)
    """
    # Pattern 1: "answer is X" with number
    match = re.search(r'answer\s+is\s*[:\s]*(\d+)', response, re.I)
    if match:
        return match.group(1)

    # Pattern 2: Number at end of response
    match = re.search(r'\b(\d+)\s*[\.!]?\s*$', response.strip())
    if match:
        return match.group(1)

    # Pattern 3: "option X" or "choice X"
    match = re.search(r'(?:option|choice)\s*[:\s]*(\d+)', response, re.I)
    if match:
        return match.group(1)

    # Pattern 4: First standalone number 1-9
    match = re.search(r'\b([1-9])\b', response)
    if match:
        return match.group(1)

    return ""


def extract_boolq_answer(response: str) -> str:
    """Extract boolean answer from BoolQ response.

    BoolQ expects Yes/No answers.
    """
    response_lower = response.lower()

    # Check for yes/no first (BoolQ format)
    if re.search(r'\byes\b', response_lower):
        return "Yes"
    if re.search(r'\bno\b', response_lower):
        return "No"

    # Also check for true/false
    if re.search(r'\btrue\b', response_lower):
        return "Yes"
    if re.search(r'\bfalse\b', response_lower):
        return "No"

    return ""


def extract_bbh_answer(response: str) -> str:
    """Extract answer from BigBenchHard response.

    BBH uses various formats - try common patterns.
    """
    # Pattern 1: "answer is NUMBER" (handles negative)
    match = re.search(r'answer\s+is\s*[:\s]*(-?\d+)', response, re.I)
    if match:
        return match.group(1)

    # Pattern 2: "= NUMBER" at end of calculation
    match = re.search(r'=\s*(-?\d+)\s*$', response.strip())
    if match:
        return match.group(1)

    # Pattern 3: Final number in response (for arithmetic)
    match = re.search(r'(-?\d+)\s*[\.!]?\s*$', response.strip())
    if match:
        return match.group(1)

    # Pattern 4: True/False for boolean tasks
    if re.search(r'\btrue\b', response.lower()):
        return "True"
    if re.search(r'\bfalse\b', response.lower()):
        return "False"

    # Pattern 5: Valid/Invalid for formal fallacies
    if re.search(r'\bvalid\b', response.lower()):
        return "valid"
    if re.search(r'\binvalid\b', response.lower()):
        return "invalid"

    # Pattern 6: Multiple choice (A), (B), etc.
    match = re.search(r'\(([A-E])\)', response)
    if match:
        return f"({match.group(1)})"

    # Pattern 7: Last word/phrase
    words = response.strip().split()
    if words:
        return words[-1].strip('.,!?')

    return ""


def extract_mmlu_answer(response: str) -> str:
    """Extract multiple choice answer from MMLU response."""
    # Look for explicit choice markers
    match = re.search(r'\b([A-D])\s*[\.\):]', response)
    if match:
        return match.group(1).upper()

    match = re.search(r'answer\s+is\s*[:\s]*([A-D])', response, re.I)
    if match:
        return match.group(1).upper()

    match = re.search(r'\b([A-D])\b', response)
    if match:
        return match.group(1).upper()

    return ""


def extract_winogrande_answer(response: str) -> str:
    """Extract A or B answer from Winogrande response."""
    # Look for explicit markers
    match = re.search(r'answer\s+is\s*[:\s]*([AB])', response, re.I)
    if match:
        return match.group(1).upper()

    match = re.search(r'\b([AB])\s*[\.\):]', response)
    if match:
        return match.group(1).upper()

    match = re.search(r'\b([AB])\b', response)
    if match:
        return match.group(1).upper()

    return ""


class ConjectureModel:
    """Wrapper that adds Conjecture enhancement to any base model.

    Per O-0006: Uses persistent session for claim accumulation across test cases.
    Claims learned during benchmark run persist and can enhance later queries.
    """

    def __init__(self, base_model, use_endpoint: bool = True):
        self.base_model = base_model
        self.use_endpoint = use_endpoint
        self._endpoint = None
        self._session_id = None
        self._loop = None

    def _get_loop(self):
        """Get or create event loop for async operations."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def initialize_session(self, session_id: str = "benchmark_session"):
        """Initialize Conjecture endpoint with persistent session.

        Per O-0006: 1 persistent session for claim accumulation.
        """
        if not self.use_endpoint:
            return

        try:
            from src.endpoint.conjecture_endpoint import ConjectureEndpoint
            self._endpoint = ConjectureEndpoint(db_path="data/benchmark.db")
            loop = self._get_loop()
            loop.run_until_complete(self._endpoint.initialize())
            self._endpoint.start_session(session_id=session_id, metadata={"type": "benchmark"})
            self._session_id = session_id
            print(f"  [Session: {session_id}, claims: 0]")
        except Exception as e:
            print(f"  [Endpoint init failed: {e}, using prompt-only mode]")
            self._endpoint = None

    def generate(self, prompt: str, problem_type: str = None, **kwargs) -> str:
        """Enhanced generation with step-by-step reasoning.

        If endpoint is available, uses claim context from persistent session.
        """
        # Build enhanced prompt based on problem type
        if problem_type == "math":
            enhanced = f"""Solve this step-by-step. Show all work clearly.
After solving, verify your answer by checking it makes sense.
Write your final numeric answer after ####

{prompt}"""
        elif problem_type == "commonsense":
            enhanced = f"""Think through each option carefully.
Consider what makes logical sense given the context.
State your chosen answer clearly.

{prompt}"""
        elif problem_type == "logic":
            enhanced = f"""Analyze this logical reasoning problem step-by-step.
1. Identify the premises and conclusion structure
2. Check for valid logical relationships
3. Eliminate incorrect options systematically
4. State your final answer (A, B, C, or D)

{prompt}"""
        elif problem_type == "verification":
            enhanced = f"""Evaluate the truthfulness of this claim carefully.
1. Consider what factual information is relevant
2. Check for common misconceptions or false beliefs
3. Identify the most accurate answer
4. State your final answer clearly

{prompt}"""
        else:
            enhanced = f"""Think step-by-step. Show your reasoning clearly.
After working through the problem, verify your answer makes sense.

{prompt}"""

        # If endpoint available, prepend claim context
        if self._endpoint and self._endpoint.claim_count() > 0:
            try:
                loop = self._get_loop()
                # Get relevant claims from session
                search_resp = loop.run_until_complete(
                    self._endpoint.search_claims(query=prompt[:200], limit=5)
                )
                if search_resp.success and search_resp.data.get("claims"):
                    claims = search_resp.data["claims"]
                    from src.endpoint.llm_client import build_claim_context
                    context = build_claim_context(claims)
                    if context:
                        enhanced = f"{context}\n\n{enhanced}"
            except Exception:
                pass  # Fall back to prompt-only mode

        return _call_model(self.base_model, enhanced)

    def close(self):
        """Close endpoint and report session stats."""
        if self._endpoint:
            count = self._endpoint.claim_count()
            print(f"  [Session ended: {count} claims accumulated]")
            loop = self._get_loop()
            loop.run_until_complete(self._endpoint.close())
            self._endpoint = None

    def get_model_name(self) -> str:
        base_name = getattr(self.base_model, 'model_name', None) or getattr(self.base_model, '_model_name', None) or 'unknown'
        return f"{base_name}+Conjecture"


@dataclass
class BenchmarkResult:
    name: str
    sample_count: int
    baseline_score: float
    conjecture_score: float
    delta: float
    timestamp: str
    error: Optional[str] = None


class DeepEvalSuite:
    """Run DeepEval benchmarks comparing baseline vs Conjecture using direct answer extraction.

    Baseline Caching: Use use_baseline_cache=True to skip baseline API calls when cached.
    Only set refresh_baseline=True when fixing benchmark bugs or parser issues.
    """

    def __init__(self, base_model=None, use_baseline_cache: bool = True):
        self.base_model = base_model
        self.conjecture_model = ConjectureModel(base_model) if base_model else None
        self.results: List[BenchmarkResult] = []
        self.stats_path = Path(__file__).parent.parent / "STATS.yaml"
        self.use_baseline_cache = use_baseline_cache

    def _get_model_name(self) -> str:
        if not self.base_model:
            return "none"
        return (
            getattr(self.base_model, 'model_name', None)
            or getattr(self.base_model, '_model_name', None)
            or type(self.base_model).__name__
        )

    def run_gsm8k(self, n_samples: int = 20) -> BenchmarkResult:
        """GSM8K: Grade school math — where CoT should help most"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("GSM8K", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        model_name = self._get_model_name()

        # Check baseline cache
        cached_baseline = None
        if self.use_baseline_cache:
            cached_baseline = get_cached_baseline("GSM8K", model_name, n_samples)
            if cached_baseline is not None:
                print(f"  GSM8K: Using cached baseline {cached_baseline:.1f}%")

        try:
            gsm_bench = GSM8K(n_problems=n_samples, n_shots=5, enable_cot=True)
            goldens = gsm_bench.load_benchmark_dataset()[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                prompt = GSM8KTemplate.generate_output(
                    input=golden.input,
                    train_set=gsm_bench.shots_dataset,
                    n_shots=5,
                    enable_cot=True,
                )
                expected = golden.expected_output  # Numeric answer

                # Baseline (skip if cached)
                if cached_baseline is None:
                    try:
                        baseline_response = _call_model(self.base_model, prompt)
                        extracted = extract_gsm8k_answer(baseline_response)
                        try:
                            if abs(float(extracted) - float(expected)) < 0.01:
                                baseline_correct += 1
                        except ValueError:
                            pass
                    except Exception:
                        pass

                # Conjecture
                try:
                    conj_response = self.conjecture_model.generate(prompt, problem_type="math")
                    extracted_c = extract_gsm8k_answer(conj_response)
                    try:
                        if abs(float(extracted_c) - float(expected)) < 0.01:
                            conj_correct += 1
                    except ValueError:
                        pass
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    bl = cached_baseline if cached_baseline else baseline_correct
                    print(f"  GSM8K: {i+1}/{total} done (baseline {bl}, conj {conj_correct})")

            # Use cached or computed baseline
            if cached_baseline is not None:
                baseline_score = cached_baseline
            else:
                baseline_score = baseline_correct / total * 100
                set_cached_baseline("GSM8K", model_name, n_samples, baseline_score)

            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "GSM8K", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("GSM8K", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_mathqa(self, n_samples: int = 20) -> BenchmarkResult:
        """MathQA: Multiple choice math reasoning"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("MathQA", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            mathqa_bench = MathQA(n_problems_per_task=n_samples, n_shots=5)
            task = mathqa_bench.tasks[0]
            goldens = mathqa_bench.load_benchmark_dataset(task)[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                prompt = MathQATemplate.generate_output(
                    input=golden.input,
                    n_shots=5,
                )
                expected = golden.expected_output.lower()  # a, b, c, d, or e

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_mathqa_answer(baseline_response)
                    if extracted == expected:
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture
                try:
                    conj_response = self.conjecture_model.generate(prompt, problem_type="math")
                    extracted_c = extract_mathqa_answer(conj_response)
                    if extracted_c == expected:
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  MathQA: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "MathQA", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("MathQA", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_hellaswag(self, n_samples: int = 20) -> BenchmarkResult:
        """HellaSwag: Commonsense reasoning - sentence completion"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("HellaSwag", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            hella_bench = HellaSwag(n_problems_per_task=n_samples, n_shots=5)
            task = hella_bench.tasks[0]
            goldens = hella_bench.load_benchmark_dataset(task)[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                prompt = HellaSwagTemplate.generate_output(
                    input=golden.input,
                    train_set=hella_bench.shots_dataset,
                    task=task,
                    n_shots=5,
                )
                expected = golden.expected_output  # A, B, C, or D

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_hellaswag_answer(baseline_response)
                    if extracted == expected:
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture
                try:
                    conj_response = self.conjecture_model.generate(prompt, problem_type="commonsense")
                    extracted_c = extract_hellaswag_answer(conj_response)
                    if extracted_c == expected:
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  HellaSwag: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "HellaSwag", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("HellaSwag", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_logiqa(self, n_samples: int = 20) -> BenchmarkResult:
        """LogiQA: Logical reasoning - multiple choice"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("LogiQA", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            logiqa_bench = LogiQA(n_problems_per_task=n_samples, n_shots=3)
            task = logiqa_bench.tasks[0]  # Categorical Reasoning
            goldens = logiqa_bench.load_benchmark_dataset(task)[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                prompt = LogiQATemplate.generate_output(
                    input=golden.input,
                    n_shots=3,
                )
                expected = golden.expected_output  # A, B, C, or D

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_logiqa_answer(baseline_response)
                    if extracted == expected:
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture - use "logic" problem type for reasoning enhancement
                try:
                    conj_response = self.conjecture_model.generate(prompt, problem_type="logic")
                    extracted_c = extract_logiqa_answer(conj_response)
                    if extracted_c == expected:
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  LogiQA: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "LogiQA", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("LogiQA", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_truthfulqa(self, n_samples: int = 20) -> BenchmarkResult:
        """TruthfulQA: Truth and factuality - multiple choice"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("TruthfulQA", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode
            truthqa_bench = TruthfulQA(n_problems_per_task=n_samples, mode=TruthfulQAMode.MC1)
            task = truthqa_bench.tasks[0]  # Language task
            goldens = truthqa_bench.load_benchmark_dataset(task, TruthfulQAMode.MC1)[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                prompt = TruthfulQATemplate.generate_output(
                    input=golden.input,
                    mode=TruthfulQAMode.MC1,
                )
                expected = golden.expected_output  # Answer number (1-4)

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_truthfulqa_answer(baseline_response)
                    if extracted == expected:
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture - use verification enhancement
                try:
                    conj_response = self.conjecture_model.generate(prompt, problem_type="verification")
                    extracted_c = extract_truthfulqa_answer(conj_response)
                    if extracted_c == expected:
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  TruthfulQA: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "TruthfulQA", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("TruthfulQA", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_boolq(self, n_samples: int = 20) -> BenchmarkResult:
        """BoolQ: Boolean question answering"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("BoolQ", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            boolq_bench = BoolQ(n_problems=n_samples, n_shots=5)
            goldens = boolq_bench.load_benchmark_dataset()[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                prompt = BoolQTemplate.generate_output(
                    input=golden.input,
                    n_shots=5,
                )
                expected = golden.expected_output  # "Yes" or "No"

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_boolq_answer(baseline_response)
                    if extracted == expected:
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture - use verification for fact-checking
                try:
                    conj_response = self.conjecture_model.generate(prompt, problem_type="verification")
                    extracted_c = extract_boolq_answer(conj_response)
                    if extracted_c == expected:
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  BoolQ: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "BoolQ", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("BoolQ", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_bbh_math(self, n_samples: int = 20) -> BenchmarkResult:
        """BigBenchHard: Multistep arithmetic - math reasoning"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("BBH-Math", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            # Use multistep_arithmetic_two - harder math that benefits from CoT
            bbh_bench = BigBenchHard(
                tasks=[BigBenchHardTask.MULTISTEP_ARITHMETIC_TWO],
                n_problems_per_task=n_samples,
                n_shots=3,
                enable_cot=True
            )
            task = BigBenchHardTask.MULTISTEP_ARITHMETIC_TWO
            goldens = bbh_bench.load_benchmark_dataset(task)[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                prompt = BigBenchHardTemplate.generate_output(
                    input=golden.input,
                    task=task,
                    n_shots=3,
                    enable_cot=True,
                )
                expected = golden.expected_output  # numeric answer e.g., "24", "-50"

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_bbh_answer(baseline_response)
                    if extracted == expected:  # exact match for numbers
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture - use math enhancement for arithmetic
                try:
                    conj_response = self.conjecture_model.generate(prompt, problem_type="math")
                    extracted_c = extract_bbh_answer(conj_response)
                    if extracted_c == expected:  # exact match for numbers
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  BBH-Math: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "BBH-Math", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("BBH-Math", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_bbh_object_counting(self, n_samples: int = 20) -> BenchmarkResult:
        """BigBenchHard: Object counting - where Conjecture excels (+80pp)"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("BBH-ObjectCount", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            bbh_bench = BigBenchHard(
                tasks=[BigBenchHardTask.OBJECT_COUNTING],
                n_problems_per_task=n_samples,
                n_shots=3,
                enable_cot=True
            )
            task = BigBenchHardTask.OBJECT_COUNTING
            goldens = bbh_bench.load_benchmark_dataset(task)[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                prompt = BigBenchHardTemplate.generate_output(
                    input=golden.input,
                    task=task,
                    n_shots=3,
                    enable_cot=True,
                )
                expected = golden.expected_output  # numeric answer

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_bbh_answer(baseline_response)
                    if extracted == expected:
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture - use math enhancement for counting
                try:
                    conj_response = self.conjecture_model.generate(prompt, problem_type="math")
                    extracted_c = extract_bbh_answer(conj_response)
                    if extracted_c == expected:
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  BBH-ObjectCount: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "BBH-ObjectCount", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("BBH-ObjectCount", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_bbh_logical_deduction(self, n_samples: int = 20) -> BenchmarkResult:
        """BigBenchHard: Logical deduction with 3 objects"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("BBH-Logic", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            bbh_bench = BigBenchHard(
                tasks=[BigBenchHardTask.LOGICAL_DEDUCTION_THREE_OBJECTS],
                n_problems_per_task=n_samples,
                n_shots=3,
                enable_cot=True
            )
            task = BigBenchHardTask.LOGICAL_DEDUCTION_THREE_OBJECTS
            goldens = bbh_bench.load_benchmark_dataset(task)[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                prompt = BigBenchHardTemplate.generate_output(
                    input=golden.input,
                    task=task,
                    n_shots=3,
                    enable_cot=True,
                )
                expected = golden.expected_output

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_bbh_answer(baseline_response)
                    if extracted.lower() == expected.lower():
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture - use logic enhancement
                try:
                    conj_response = self.conjecture_model.generate(prompt, problem_type="logic")
                    extracted_c = extract_bbh_answer(conj_response)
                    if extracted_c.lower() == expected.lower():
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  BBH-Logic: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "BBH-Logic", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("BBH-Logic", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_bbh_navigate(self, n_samples: int = 20) -> BenchmarkResult:
        """BigBenchHard: Navigation/spatial reasoning"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("BBH-Navigate", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            bbh_bench = BigBenchHard(
                tasks=[BigBenchHardTask.NAVIGATE],
                n_problems_per_task=n_samples,
                n_shots=3,
                enable_cot=True
            )
            task = BigBenchHardTask.NAVIGATE
            goldens = bbh_bench.load_benchmark_dataset(task)[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                prompt = BigBenchHardTemplate.generate_output(
                    input=golden.input,
                    task=task,
                    n_shots=3,
                    enable_cot=True,
                )
                expected = golden.expected_output

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_bbh_answer(baseline_response)
                    if extracted.lower() == expected.lower():
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture
                try:
                    conj_response = self.conjecture_model.generate(prompt, problem_type="logic")
                    extracted_c = extract_bbh_answer(conj_response)
                    if extracted_c.lower() == expected.lower():
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  BBH-Navigate: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "BBH-Navigate", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("BBH-Navigate", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_bbh_date(self, n_samples: int = 20) -> BenchmarkResult:
        """BigBenchHard: Date understanding/temporal reasoning"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("BBH-Date", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            bbh_bench = BigBenchHard(
                tasks=[BigBenchHardTask.DATE_UNDERSTANDING],
                n_problems_per_task=n_samples,
                n_shots=3,
                enable_cot=True
            )
            task = BigBenchHardTask.DATE_UNDERSTANDING
            goldens = bbh_bench.load_benchmark_dataset(task)[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                prompt = BigBenchHardTemplate.generate_output(
                    input=golden.input,
                    task=task,
                    n_shots=3,
                    enable_cot=True,
                )
                expected = golden.expected_output

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_bbh_answer(baseline_response)
                    if extracted.lower() == expected.lower():
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture
                try:
                    conj_response = self.conjecture_model.generate(prompt, problem_type="logic")
                    extracted_c = extract_bbh_answer(conj_response)
                    if extracted_c.lower() == expected.lower():
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  BBH-Date: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "BBH-Date", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("BBH-Date", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_bbh_tracking(self, n_samples: int = 20) -> BenchmarkResult:
        """BigBenchHard: Tracking shuffled objects"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("BBH-Tracking", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            bbh_bench = BigBenchHard(
                tasks=[BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS],
                n_problems_per_task=n_samples,
                n_shots=3,
                enable_cot=True
            )
            task = BigBenchHardTask.TRACKING_SHUFFLED_OBJECTS_THREE_OBJECTS
            goldens = bbh_bench.load_benchmark_dataset(task)[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                prompt = BigBenchHardTemplate.generate_output(
                    input=golden.input,
                    task=task,
                    n_shots=3,
                    enable_cot=True,
                )
                expected = golden.expected_output

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_bbh_answer(baseline_response)
                    if extracted.lower() == expected.lower():
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture
                try:
                    conj_response = self.conjecture_model.generate(prompt, problem_type="logic")
                    extracted_c = extract_bbh_answer(conj_response)
                    if extracted_c.lower() == expected.lower():
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  BBH-Tracking: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "BBH-Tracking", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("BBH-Tracking", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_bbh_web_of_lies(self, n_samples: int = 20) -> BenchmarkResult:
        """BigBenchHard: Web of lies - truth/lie deduction"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("BBH-WebOfLies", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            bbh_bench = BigBenchHard(
                tasks=[BigBenchHardTask.WEB_OF_LIES],
                n_problems_per_task=n_samples,
                n_shots=3,
                enable_cot=True
            )
            task = BigBenchHardTask.WEB_OF_LIES
            goldens = bbh_bench.load_benchmark_dataset(task)[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                prompt = BigBenchHardTemplate.generate_output(
                    input=golden.input,
                    task=task,
                    n_shots=3,
                    enable_cot=True,
                )
                expected = golden.expected_output

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_bbh_answer(baseline_response)
                    if extracted.lower() == expected.lower():
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture
                try:
                    conj_response = self.conjecture_model.generate(prompt, problem_type="logic")
                    extracted_c = extract_bbh_answer(conj_response)
                    if extracted_c.lower() == expected.lower():
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  BBH-WebOfLies: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "BBH-WebOfLies", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("BBH-WebOfLies", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_mmlu_hard(self, n_samples: int = 20) -> BenchmarkResult:
        """MMLU-Hard: College math + formal logic tasks"""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("MMLU-Hard", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            # Use hard MMLU tasks: college math, formal logic, abstract algebra
            hard_tasks = [
                MMLUTask.COLLEGE_MATHEMATICS,
                MMLUTask.FORMAL_LOGIC,
                MMLUTask.ABSTRACT_ALGEBRA,
            ]
            mmlu_bench = MMLU(
                tasks=hard_tasks,
                n_problems_per_task=n_samples // 3 + 1,  # Distribute across tasks
                n_shots=5
            )

            baseline_correct = 0
            conj_correct = 0
            total = 0

            for task in hard_tasks:
                goldens = mmlu_bench.load_benchmark_dataset(task)[:n_samples // 3]
                total += len(goldens)

                for i, golden in enumerate(goldens):
                    prompt = MMLUTemplate.generate_output(
                        input=golden.input,
                        train_set=mmlu_bench.shots_dataset,
                        task=task,
                        n_shots=5,
                    )
                    expected = golden.expected_output  # A, B, C, or D

                    # Baseline
                    try:
                        baseline_response = _call_model(self.base_model, prompt)
                        extracted = extract_mmlu_answer(baseline_response)
                        if extracted == expected:
                            baseline_correct += 1
                    except Exception:
                        pass

                    # Conjecture - use math enhancement
                    try:
                        conj_response = self.conjecture_model.generate(prompt, problem_type="math")
                        extracted_c = extract_mmlu_answer(conj_response)
                        if extracted_c == expected:
                            conj_correct += 1
                    except Exception:
                        pass

            print(f"  MMLU-Hard: {total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100 if total > 0 else 0
            conj_score = conj_correct / total * 100 if total > 0 else 0

            return BenchmarkResult(
                "MMLU-Hard", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("MMLU-Hard", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_winogrande(self, n_samples: int = 20) -> BenchmarkResult:
        """Winogrande: Commonsense reasoning with pronoun resolution."""
        if not DEEPEVAL_AVAILABLE or not self.base_model:
            return BenchmarkResult("Winogrande", 0, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), "Model not configured")

        try:
            wg_bench = Winogrande(n_problems=n_samples, n_shots=5)
            goldens = wg_bench.load_benchmark_dataset()[:n_samples]

            baseline_correct = 0
            conj_correct = 0
            total = len(goldens)

            for i, golden in enumerate(goldens):
                prompt = WinograndeTemplate.generate_output(
                    input=golden.input,
                    n_shots=5,
                )
                expected = golden.expected_output  # A or B

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_winogrande_answer(baseline_response)
                    if extracted == expected:
                        baseline_correct += 1
                except Exception:
                    pass

                # Conjecture - use logic enhancement for reasoning
                try:
                    conj_response = self.conjecture_model.generate(prompt, problem_type="logic")
                    extracted_c = extract_winogrande_answer(conj_response)
                    if extracted_c == expected:
                        conj_correct += 1
                except Exception:
                    pass

                if (i + 1) % 5 == 0:
                    print(f"  Winogrande: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
            conj_score = conj_correct / total * 100

            return BenchmarkResult(
                "Winogrande", total, baseline_score, conj_score,
                conj_score - baseline_score, datetime.now().isoformat()
            )
        except Exception as e:
            return BenchmarkResult("Winogrande", n_samples, 0.0, 0.0, 0.0,
                datetime.now().isoformat(), str(e))

    def run_full_suite(self, n_samples: int = 20, session_id: str = "benchmark_session") -> Dict[str, BenchmarkResult]:
        """Run all benchmarks sequentially with persistent session.

        Per O-0006: Uses 1 persistent session for claim accumulation across test cases.
        Claims from earlier problems may enhance later problem solving.
        """
        # Initialize persistent session for Conjecture model
        if self.conjecture_model:
            self.conjecture_model.initialize_session(session_id=session_id)

        try:
            # O-0008: 10 benchmarks, >= Direct on ALL, +20pp on 5
            # Use reasoning-focused tasks where Conjecture adds value
            results = {
                "GSM8K": self.run_gsm8k(n_samples),
                "LogiQA": self.run_logiqa(n_samples),
                "BBH-Math": self.run_bbh_math(n_samples),
                "BBH-ObjectCount": self.run_bbh_object_counting(n_samples),
                "TruthfulQA": self.run_truthfulqa(n_samples),
                "HellaSwag": self.run_hellaswag(n_samples),
                "BoolQ": self.run_boolq(n_samples),
                "BBH-Date": self.run_bbh_date(n_samples),
                "BBH-WebOfLies": self.run_bbh_web_of_lies(n_samples),
                "Winogrande": self.run_winogrande(n_samples),
            }
            self.results = list(results.values())
            return results
        finally:
            # Close session and report stats
            if self.conjecture_model:
                self.conjecture_model.close()

    def update_stats_yaml(self, key: str = "deepeval_benchmarks", session_claims: int = 0) -> dict:
        """Update STATS.yaml with results"""
        stats = {}
        if self.stats_path.exists():
            with open(self.stats_path) as f:
                stats = yaml.safe_load(f) or {}

        stats[key] = {
            "last_run": datetime.now().isoformat(),
            "model": self._get_model_name(),
            "session_claims": session_claims,  # Per O-0006: track claim accumulation
            "benchmarks": {
                r.name: {
                    "sample_count": r.sample_count,
                    "baseline_score": round(r.baseline_score, 2),
                    "conjecture_score": round(r.conjecture_score, 2),
                    "delta": round(r.delta, 2),
                    "error": r.error
                } for r in self.results
            }
        }

        valid = [r for r in self.results if r.error is None]
        if valid:
            stats[key]["summary"] = {
                "avg_baseline": round(sum(r.baseline_score for r in valid) / len(valid), 2),
                "avg_conjecture": round(sum(r.conjecture_score for r in valid) / len(valid), 2),
                "avg_delta": round(sum(r.delta for r in valid) / len(valid), 2),
                "benchmarks_passed": len(valid),
                "benchmarks_failed": len(self.results) - len(valid)
            }

        with open(self.stats_path, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
        return stats


def tally_paired(cases: List[dict]) -> dict:
    """Raw tallies for a paired run. No statistics, no gate (tracer only).

    Each case: {"routed_correct": bool, "direct_correct": bool, "strategy": str,
                "routed_error": str|None, "direct_error": str|None}

    n_errored counts cases where either arm raised. Those cases still appear in
    n with correct=False, so read scoring against n_clean, not n — an infra
    failure is not a benchmark miss (UM-0500).
    """
    tallies = {
        "n": len(cases),
        "n_errored": sum(1 for c in cases
                         if c.get("routed_error") or c.get("direct_error")),
        "routed_correct": sum(1 for c in cases if c["routed_correct"]),
        "direct_correct": sum(1 for c in cases if c["direct_correct"]),
        "by_strategy": {},
    }
    tallies["n_clean"] = tallies["n"] - tallies["n_errored"]
    for c in cases:
        s = tallies["by_strategy"].setdefault(
            c["strategy"], {"n": 0, "routed_correct": 0, "direct_correct": 0})
        s["n"] += 1
        s["routed_correct"] += c["routed_correct"]
        s["direct_correct"] += c["direct_correct"]
    return tallies


def _gsm8k_correct(response: str, expected: str) -> bool:
    try:
        return abs(float(extract_gsm8k_answer(response)) - float(expected)) < 0.01
    except (ValueError, TypeError):
        return False


def _load_paired_gsm8k(n_samples: int):
    # ponytail: installed deepeval 2.6.7 loads the retired bare "gsm8k" HF id,
    # which current huggingface_hub rejects; rewrite to the canonical
    # "openai/gsm8k". Drop when deepeval is upgraded.
    import datasets as _datasets
    _orig_load = _datasets.load_dataset
    _datasets.load_dataset = lambda path, *a, **k: _orig_load(
        "openai/gsm8k" if path == "gsm8k" else path, *a, **k)
    try:
        gsm_bench = GSM8K(n_problems=n_samples, n_shots=5, enable_cot=True)
        goldens = gsm_bench.load_benchmark_dataset()[:n_samples]
    finally:
        _datasets.load_dataset = _orig_load  # don't leak the patch past loading

    def prompt_fn(golden):
        return GSM8KTemplate.generate_output(
            input=golden.input, train_set=gsm_bench.shots_dataset,
            n_shots=5, enable_cot=True)

    return goldens, prompt_fn, _gsm8k_correct


def _load_paired_logiqa(n_samples: int):
    logiqa_bench = LogiQA(n_problems_per_task=n_samples, n_shots=3)
    task = logiqa_bench.tasks[0]  # Categorical Reasoning
    goldens = logiqa_bench.load_benchmark_dataset(task)[:n_samples]

    def prompt_fn(golden):
        return LogiQATemplate.generate_output(input=golden.input, n_shots=3)

    return goldens, prompt_fn, lambda resp, exp: extract_logiqa_answer(resp) == exp


def _load_paired_truthfulqa(n_samples: int):
    from deepeval.benchmarks.truthful_qa.mode import TruthfulQAMode
    truthqa_bench = TruthfulQA(n_problems_per_task=n_samples, mode=TruthfulQAMode.MC1)
    task = truthqa_bench.tasks[0]  # Language task
    goldens = truthqa_bench.load_benchmark_dataset(task, TruthfulQAMode.MC1)[:n_samples]

    def prompt_fn(golden):
        return TruthfulQATemplate.generate_output(
            input=golden.input, mode=TruthfulQAMode.MC1)

    return goldens, prompt_fn, lambda resp, exp: extract_truthfulqa_answer(resp) == exp


# stdlib-only module; safe to import at module level (the rest of paired_stats
# is still imported lazily inside compute_paired_verdict).
from benchmarks.paired_stats import (  # noqa: E402
    N_REQUIRED as GATE_N_REQUIRED, PINNED_SMALL_MODELS, reduce_verdicts)

# The paired-gate benchmark set: the reasoning-class and recall-class
# benchmarks already in the suite (O-0009 evidence classes). expected_strategies
# is the router's ground truth on that benchmark — router_accuracy = share of
# clean cases classified into it. class "recall" disables the anti-cowardice
# half of the verdict (staying on the cheap path is correct there).
PAIRED_BENCHMARKS = {
    "gsm8k": {
        "display": "GSM8K",
        "class": "reasoning",
        "expected_strategies": frozenset({"math"}),
        "prompt_template": "gsm8k-5shot-cot (deepeval GSM8KTemplate, shared by both arms)",
        "load": _load_paired_gsm8k,
    },
    "logiqa": {
        "display": "LogiQA",
        "class": "reasoning",
        "expected_strategies": frozenset({"reasoning"}),
        "prompt_template": "logiqa-3shot (deepeval LogiQATemplate, shared by both arms)",
        "load": _load_paired_logiqa,
    },
    "truthfulqa": {
        "display": "TruthfulQA",
        "class": "recall",
        "expected_strategies": frozenset({"recall"}),
        "prompt_template": "truthfulqa-mc1 (deepeval TruthfulQATemplate, shared by both arms)",
        "load": _load_paired_truthfulqa,
    },
}

# Kept for older callers/artifacts that reference the GSM8K template string.
PAIRED_PROMPT_TEMPLATE = PAIRED_BENCHMARKS["gsm8k"]["prompt_template"]


def run_paired_benchmark(bench_key: str, base_model, n_samples: int = 10) -> List[dict]:
    """Paired routed-vs-direct comparison on one PAIRED_BENCHMARKS entry.

    Each case runs BOTH arms in the same invocation, over the same case set,
    with ONE shared prompt (the benchmark's deepeval template output):
      routed — endpoint.evaluate() on the shipped O-0009 path; the strategy is
               classified from the bare question (what a user would type), the
               evaluated prompt is the shared template
      direct — the same shared prompt via _call_model on base_model

    Serving-model pinning is the caller's job (main() pins llm_client's
    DEFAULT/TOOL_CAPABLE_MODEL to --model before this runs); this function
    records the model each routed call REPORTS serving in
    case["routed_model"], so pinning is audited, not assumed. Verdict emission
    refuses when the recorded identities don't match — see compute_paired_verdict.

    Baseline cache is bypassed by construction (never read, never written).
    Returns per-case records; caller prints tallies via tally_paired().
    """
    from src.endpoint.conjecture_endpoint import ConjectureEndpoint
    from src.agent.task_router import classify_query

    spec = PAIRED_BENCHMARKS[bench_key]
    goldens, prompt_fn, correct_fn = spec["load"](n_samples)

    endpoint = ConjectureEndpoint(db_path="data/paired_benchmark.db")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    cases = []
    try:
        loop.run_until_complete(endpoint.initialize())
        endpoint.start_session(session_id=f"paired_{bench_key}",
                               metadata={"type": "paired_benchmark"})

        for i, golden in enumerate(goldens):
            prompt = prompt_fn(golden)
            expected = golden.expected_output
            # Classify once here and hand the result to evaluate() via route=,
            # so the recorded strategy IS the one used rather than a re-derived
            # guess (classify_query can fall through to an LLM call, so calling
            # it twice is neither free nor guaranteed to agree with itself).
            # Classify the bare question (what a user would type), evaluate the
            # shared template — routing signal and measured prompt stay separate.
            query_type = classify_query(golden.input)
            strategy = query_type.value

            def _correct(response: str) -> bool:
                return correct_fn(response, expected)

            case = {"index": i, "strategy": strategy,
                    "routed_correct": False, "direct_correct": False,
                    "routed_error": None, "direct_error": None,
                    "routed_model": None}

            try:
                resp = loop.run_until_complete(
                    endpoint.evaluate(query=prompt, route=query_type))
                if resp.success:
                    case["routed_correct"] = _correct(resp.data.get("response", ""))
                    case["routed_model"] = resp.data.get("model")
                else:
                    case["routed_error"] = resp.message
            except Exception as e:
                case["routed_error"] = str(e)

            try:
                case["direct_correct"] = _correct(_call_model(base_model, prompt))
            except Exception as e:
                case["direct_error"] = str(e)

            cases.append(case)
            print(f"  paired {spec['display']} {i+1}/{len(goldens)}: strategy={strategy} "
                  f"routed={'Y' if case['routed_correct'] else 'n'} "
                  f"direct={'Y' if case['direct_correct'] else 'n'}"
                  + (f" [routed_error: {case['routed_error']}]" if case["routed_error"] else "")
                  + (f" [direct_error: {case['direct_error']}]" if case["direct_error"] else ""))
    finally:
        try:
            loop.run_until_complete(endpoint.close())
        except Exception as e:  # init may have failed before there was anything to close
            print(f"  [endpoint close failed: {e}]")
        asyncio.set_event_loop(None)
        loop.close()

    return cases


def print_paired_tallies(cases: List[dict], routed_model: str = None,
                         direct_model: str = None, bench_key: str = "gsm8k"):
    t = tally_paired(cases)
    errored = [c for c in cases if c.get("routed_error") or c.get("direct_error")]
    display = PAIRED_BENCHMARKS[bench_key]["display"]
    print(f"\nPaired {display} raw tallies (no statistics, no gate):")
    if routed_model or direct_model:
        served = sorted({str(c.get("routed_model")) for c in cases
                         if not c.get("routed_error")})
        print(f"  pinned model: routed={routed_model} direct={direct_model}")
        print(f"  routed arm reported serving: {served or ['(none)']}")
    print(f"  n={t['n']}  n_clean={t['n_clean']}  n_errored={t['n_errored']}")
    print(f"  routed_correct={t['routed_correct']}  direct_correct={t['direct_correct']}"
          f"  (errored cases count as incorrect — score against n_clean)")
    for name, s in t["by_strategy"].items():
        print(f"  strategy={name}: n={s['n']} routed={s['routed_correct']} direct={s['direct_correct']}")
    if errored:
        print(f"  cases with arm errors: {len(errored)} (see per-case log above)")


def arm_identity_mismatch(cases: List[dict], pinned_model: str) -> List[str]:
    """Reasons the two arms cannot be treated as the same model ([] = matched).

    The direct arm calls pinned_model by construction; the routed arm's
    identity is whatever each evaluate() call REPORTED serving
    (case["routed_model"]). Unreported or different → mismatch.
    """
    clean = [c for c in cases
             if not c.get("routed_error") and not c.get("direct_error")]
    reasons = []
    unreported = sum(1 for c in clean
                     if c.get("routed_model") in (None, "unknown"))
    if unreported:
        reasons.append(f"{unreported}/{len(clean)} clean cases did not report "
                       "a served model for the routed arm")
    others = sorted({c["routed_model"] for c in clean
                     if c.get("routed_model") not in (None, "unknown")
                     and c["routed_model"] != pinned_model})
    if others:
        reasons.append(f"routed arm served {others}, pinned model is {pinned_model}")
    return reasons


def compute_paired_verdict(cases: List[dict], pinned_model: str = None,
                           frozen_tolerance: float = None,
                           bench_key: str = "gsm8k") -> dict:
    """Wire paired_stats over the tracer's per-case records: delta, CI,
    tolerance, cowardice inputs, and the four-way verdict.

    router_accuracy: each PAIRED_BENCHMARKS entry declares the router's ground
    truth for its cases (expected_strategies) — accuracy is the share of clean
    cases classified into that set. A recall-class benchmark also disables the
    anti-cowardice half: staying on the cheap path is the CORRECT behavior
    there per O-0009, so scoring it as cowardice would fail a working router.
    """
    from benchmarks.paired_stats import (
        paired_delta, non_inferiority_tolerance, cowardice_metrics, verdict)

    stats = paired_delta(cases)
    if frozen_tolerance is not None:
        tol = {"tolerance": frozen_tolerance,
               "basis": "frozen via --tolerance (reference-run margin)"}
    else:
        tol = non_inferiority_tolerance(stats)
    cow = cowardice_metrics(cases)
    clean = [c for c in cases
             if not c.get("routed_error") and not c.get("direct_error")]
    spec = PAIRED_BENCHMARKS[bench_key]
    expected = spec["expected_strategies"]
    router_accuracy = (sum(1 for c in clean if c["strategy"] in expected) / len(clean)
                       if clean else 0.0)
    out = {"stats": stats, "tolerance": tol, "cowardice": cow,
           "router_accuracy": router_accuracy, "pinned_model": pinned_model,
           "benchmark": bench_key, "benchmark_class": spec["class"],
           "arm_mismatch": []}
    if pinned_model is not None:
        mismatch = arm_identity_mismatch(cases, pinned_model)
        if mismatch:
            # Refuse: a verdict is a statement about ONE model; if the arms
            # weren't provably that model, there is no verdict to emit.
            out.update(arm_mismatch=mismatch,
                       verdict="refused-arm-mismatch", reasons=mismatch)
            return out
    v = verdict(stats, tol["tolerance"] or 0.0,
                cow["non_direct_share"], cow["reasoning_uplift"], router_accuracy,
                apply_cowardice=spec["class"] != "recall")
    out.update(**v)
    return out


def print_paired_verdict(r: dict):
    s, t = r["stats"], r["tolerance"]
    print("\nPaired verdict (do-no-harm gate statistics):")
    if s["delta"] is None:
        print("  no clean cases — no statistics computable")
    else:
        print(f"  delta={s['delta']:+.4f}  95% CI=[{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]"
              f"  (n_clean={s['n_clean']}, n_required={s['n_required']})")
        tol_str = f"{t['tolerance']:.4f}" if t["tolerance"] is not None else "n/a"
        print(f"  tolerance={tol_str}  basis: {t['basis']}")
        print(f"  non_direct_share={r['cowardice']['non_direct_share']:.4f}"
              f"  reasoning_uplift={r['cowardice']['reasoning_uplift']}"
              f"  (n_reasoning={r['cowardice']['n_reasoning']})"
              f"  router_accuracy={r['router_accuracy']:.4f}")
    print(f"  VERDICT: {r['verdict']}")
    for reason in r["reasons"]:
        print(f"    - {reason}")


def write_paired_artifact(payload: dict, results_dir: str = None,
                          bench_key: str = "gsm8k") -> str:
    """Persist the full paired run to a timestamped JSON artifact.

    benchmarks/results/ is gitignored (run output, not source): the artifact
    survives the terminal session on the machine that produced it, and
    STATS.yaml (tracked) records its path, model, prompt template and verdict
    so a committed run is auditable even where the JSON isn't checked in.
    """
    results_dir = results_dir or str(Path(__file__).parent / "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(
        results_dir,
        f"paired_{bench_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def update_paired_stats_yaml(payload: dict, artifact_path: str,
                             stats_path: str = None, key: str = "paired_gsm8k") -> dict:
    """Write the machine-readable summary of a paired run into STATS.yaml."""
    stats_path = stats_path or str(Path(__file__).parent.parent / "STATS.yaml")
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = yaml.safe_load(f) or {}
    v = payload["verdict_record"]
    stats[key] = {
        "last_run": payload["timestamp"],
        "artifact": artifact_path,
        "model": payload["model"],
        "provider": payload["provider"],
        "prompt_template": payload["prompt_template"],
        "n": payload["n"],
        "n_clean": v["stats"]["n_clean"],
        "delta": v["stats"]["delta"],
        "ci_low": v["stats"]["ci_low"],
        "ci_high": v["stats"]["ci_high"],
        "tolerance": v["tolerance"]["tolerance"],
        "tolerance_basis": v["tolerance"]["basis"],
        "router_accuracy": v["router_accuracy"],
        "verdict": v["verdict"],
        "reasons": v["reasons"],
    }
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    return stats


# Release-time cadence: the gate is only evidence while it is recent. A gate
# result older than this is stale — the code it certified has moved on.
GATE_MAX_AGE_DAYS = 90
GATE_STATS_KEY = "paired_gate"


def update_paired_gate_yaml(gate: dict, stats_path: str = None,
                            key: str = GATE_STATS_KEY) -> dict:
    """Write the reduced across-benchmark gate result into STATS.yaml.

    This is the row the release cadence hook reads: one verdict for the whole
    small-model do-no-harm gate, reduced by worst case across benchmarks.
    """
    stats_path = stats_path or str(Path(__file__).parent.parent / "STATS.yaml")
    stats = {}
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = yaml.safe_load(f) or {}
    stats[key] = gate
    with open(stats_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    return stats


def check_release_gate(stats_path: str = None, now: datetime = None,
                       max_age_days: int = GATE_MAX_AGE_DAYS,
                       key: str = GATE_STATS_KEY) -> dict:
    """Release-time cadence check: is there a fresh, passing gate on record?

    Does NOT run the benchmark (no API keys at release time, and a gate run is
    hours of paid calls). It checks that a run HAPPENED, recently, on the
    pinned model set, and passed — absent evidence blocks the release rather
    than being read as consent (UM-0500).

    Returns {"ok": bool, "reasons": [...]}; caller maps ok=False to exit 1.
    """
    now = now or datetime.now()
    stats_path = stats_path or str(Path(__file__).parent.parent / "STATS.yaml")
    if not os.path.exists(stats_path):
        return {"ok": False, "reasons": [f"{stats_path} missing — no gate on record"]}
    with open(stats_path) as f:
        stats = yaml.safe_load(f) or {}
    gate = stats.get(key)
    if not gate:
        return {"ok": False,
                "reasons": [f"no '{key}' entry in {stats_path} — gate never run"]}
    # Freshness is per model: model A's pass from 89 days ago must not be kept
    # alive by model B passing yesterday. Each pinned model carries its own
    # pass timestamp; a later failing run on a model deletes its entry.
    passes = gate.get("model_passes") or {}
    reasons = []
    for m in PINNED_SMALL_MODELS:
        ts = passes.get(m)
        if not ts:
            reasons.append(f"pinned model has no passing gate on record: {m}")
            continue
        try:
            age = (now - datetime.fromisoformat(ts)).days
        except (TypeError, ValueError):
            reasons.append(f"unreadable pass timestamp for {m}: {ts!r}")
            continue
        if age > max_age_days:
            reasons.append(f"gate pass for {m} is {age}d old "
                           f"(max {max_age_days}d) — stale evidence")
    return {"ok": not reasons, "reasons": reasons}


def main():
    parser = argparse.ArgumentParser(
        description="DeepEval Benchmark Suite — O-0008: 10 benchmarks via Chutes.ai"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
        help="Model ID on Chutes.ai (default: openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of samples per benchmark (default: 20)",
    )
    parser.add_argument(
        "--stats-key",
        default=None,
        help="Key to use in STATS.yaml (default: deepeval_oss)",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Run specific benchmark: gsm8k, logiqa, bbh-math, bbh-object, truthfulqa",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "chutes", "openrouter"],
        help="API provider: auto (try openrouter first), chutes, or openrouter",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Paired routed-vs-direct gate across PAIRED_BENCHMARKS: both arms "
             "per case, same run, baseline cache bypassed, per-benchmark "
             "verdicts reduced by worst case",
    )
    parser.add_argument(
        "--paired-benchmarks",
        default=",".join(PAIRED_BENCHMARKS),
        help="Comma-separated paired benchmarks to run "
             f"(default: all — {', '.join(PAIRED_BENCHMARKS)})",
    )
    parser.add_argument(
        "--check-gate",
        action="store_true",
        help="Release-time cadence check: exit non-zero unless STATS.yaml "
             "records a fresh, passing gate on the pinned model set. Runs no "
             "benchmarks and needs no API key.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Frozen non-inferiority tolerance (proportion, e.g. 0.05) from a "
             "reference run; default derives it from this run's measured sd",
    )
    parser.add_argument(
        "--refresh-baseline",
        action="store_true",
        help="Re-run baseline tests (use when fixing benchmark/parser bugs)",
    )
    args = parser.parse_args()

    if args.check_gate:
        r = check_release_gate()
        print("Release gate check (small-model do-no-harm, O-0008 amendment):")
        for reason in r["reasons"]:
            print(f"  - {reason}")
        print(f"  {'OK' if r['ok'] else 'BLOCKED'}")
        raise SystemExit(0 if r["ok"] else 1)

    print("DeepEval Benchmark Suite (OSS Models)")
    print("=" * 50)
    print(f"N     : {args.n} samples per benchmark")
    print("O-0008: 10 benchmarks, >= Direct on ALL, +20pp on 5")

    if args.paired:
        # Runs before deepeval provider selection: installed deepeval's
        # GPTModel validates model names against OpenAI's list and rejects
        # OSS ids, and paired mode only needs .generate() for the direct arm.
        import openai

        # Pin BOTH arms to --model: env for anything not yet imported, module
        # attrs for the already-imported constants (evaluator re-imports them
        # per call, so the patch takes effect).
        os.environ["CONJECTURE_DEFAULT_MODEL"] = args.model
        os.environ["CONJECTURE_TOOL_MODEL"] = args.model
        import src.endpoint.llm_client as _llm_client
        _llm_client.DEFAULT_MODEL = args.model
        _llm_client.TOOL_CAPABLE_MODEL = args.model

        chutes_key = os.environ.get("CHUTES_API_KEY")
        if args.provider in ("auto", "chutes") and chutes_key:
            # CHUTES_BASE_URL honored so both arms hit the same provider,
            # mirroring llm_client's override
            base_url = os.environ.get("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")
            key = chutes_key
            provider = base_url
        else:
            # The routed arm serves through llm_client, which is Chutes-only
            # (CHUTES_API_KEY / CHUTES_BASE_URL). Running the direct arm on
            # OpenRouter would put the arms on different serving stacks under
            # one model name — a confound arm_identity_mismatch cannot see.
            print("ERROR: paired mode requires CHUTES_API_KEY — the routed arm "
                  "serves via llm_client (Chutes only); a non-Chutes direct arm "
                  "would compare two different serving stacks")
            return
        if args.tolerance is not None and not 0 <= args.tolerance <= 1:
            print(f"ERROR: --tolerance must be a proportion in [0,1], got {args.tolerance}")
            return
        bench_keys = [b.strip().lower() for b in args.paired_benchmarks.split(",")
                      if b.strip()]
        unknown = [b for b in bench_keys if b not in PAIRED_BENCHMARKS]
        if unknown:
            print(f"ERROR: unknown paired benchmark(s) {unknown}; "
                  f"available: {', '.join(PAIRED_BENCHMARKS)}")
            return

        class _DirectArmModel:
            # ponytail: minimal .generate() shim, only interface _call_model needs
            def __init__(self):
                self.client = openai.OpenAI(api_key=key, base_url=base_url)

            def generate(self, prompt: str) -> str:
                r = self.client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return r.choices[0].message.content or ""

        print(f"Provider: {provider}")
        print(f"Model   : {args.model} (BOTH arms, pinned)")
        if args.model not in PINNED_SMALL_MODELS:
            print(f"  NOTE: {args.model} is not in the pinned gate model set "
                  f"{list(PINNED_SMALL_MODELS)} — this run informs, it does not "
                  "discharge the O-0008 gate (amend CHOICES.md to add a model)")
        print(f"Benchmarks: {', '.join(bench_keys)}")
        print(f"n={args.n} per benchmark (gate bar: n>={GATE_N_REQUIRED} clean each)")
        print("\nPaired mode: routed endpoint vs direct, baseline cache BYPASSED")
        print("Other suite flags (--refresh-baseline etc.) do not apply in paired mode")

        direct_arm = _DirectArmModel()
        per_benchmark, artifacts = {}, {}
        for bench_key in bench_keys:
            spec = PAIRED_BENCHMARKS[bench_key]
            print(f"\n=== {spec['display']} ({spec['class']}-class) ===")
            print(f"Prompt  : {spec['prompt_template']}")
            cases = run_paired_benchmark(bench_key, direct_arm, n_samples=args.n)
            print_paired_tallies(cases, routed_model=args.model,
                                 direct_model=args.model, bench_key=bench_key)
            verdict_record = compute_paired_verdict(
                cases, pinned_model=args.model, frozen_tolerance=args.tolerance,
                bench_key=bench_key)
            print_paired_verdict(verdict_record)
            payload = {
                "timestamp": datetime.now().isoformat(),
                "benchmark": spec["display"],
                "benchmark_class": spec["class"],
                "provider": provider,
                "model": args.model,
                "prompt_template": spec["prompt_template"],
                "n": args.n,
                "frozen_tolerance_arg": args.tolerance,
                "cases": cases,
                "tallies": tally_paired(cases),
                "verdict_record": verdict_record,
            }
            artifacts[bench_key] = write_paired_artifact(payload, bench_key=bench_key)
            update_paired_stats_yaml(payload, artifacts[bench_key],
                                     key=f"paired_{bench_key}")
            per_benchmark[bench_key] = verdict_record["verdict"]
            print(f"  artifact: {artifacts[bench_key]}")

        # The gate stands only if EVERY benchmark stands (O-0008 "no regressions").
        gate = reduce_verdicts(per_benchmark)
        print("\n" + "=" * 50)
        print(f"GATE VERDICT (worst case across {len(per_benchmark)} benchmarks): "
              f"{gate['verdict']}")
        for reason in gate["reasons"]:
            print(f"  - {reason}")
        # One invocation gates one model. The release check requires a FRESH
        # pass per pinned model, so passes accumulate as model -> timestamp;
        # a failing run deletes that model's prior pass.
        stats_path = str(Path(__file__).parent.parent / "STATS.yaml")
        prior = {}
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                prior = (yaml.safe_load(f) or {}).get(
                    args.stats_key or GATE_STATS_KEY) or {}
        model_passes = dict(prior.get("model_passes") or {})
        if gate["verdict"] == "pass":
            model_passes[args.model] = datetime.now().isoformat()
        else:
            model_passes.pop(args.model, None)
        update_paired_gate_yaml({
            "last_run": datetime.now().isoformat(),
            "verdict": gate["verdict"],
            "reasons": gate["reasons"],
            "per_benchmark": gate["per_benchmark"],
            "n": args.n,
            "n_required": GATE_N_REQUIRED,
            "model_passes": model_passes,
            "pinned_models": list(PINNED_SMALL_MODELS),
            "artifacts": artifacts,
        }, key=args.stats_key or GATE_STATS_KEY)
        print(f"STATS.yaml updated (per-benchmark keys + "
              f"'{args.stats_key or GATE_STATS_KEY}')")
        return

    # Select provider
    model = None
    if args.provider in ("auto", "openrouter"):
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            # Use OpenRouter with specified model or default
            or_model = args.model if args.model != "openai/gpt-oss-20b" else "meta-llama/llama-3.1-8b-instruct"
            model = create_openrouter_model(openrouter_key, model=or_model)
            print(f"Provider: OpenRouter")
            print(f"Model   : {or_model}")
        elif args.provider == "openrouter":
            print("ERROR: Set OPENROUTER_API_KEY environment variable")
            return

    if model is None and args.provider in ("auto", "chutes"):
        chutes_key = os.environ.get("CHUTES_API_KEY")
        if chutes_key:
            model = create_chutes_model(chutes_key, model=args.model)
            print(f"Provider: Chutes.ai")
            print(f"Model   : {args.model}")
        elif args.provider == "chutes":
            print("ERROR: Set CHUTES_API_KEY environment variable")
            return

    if model is None:
        print("ERROR: No API key found. Set OPENROUTER_API_KEY or CHUTES_API_KEY")
        return

    use_cache = not args.refresh_baseline
    suite = DeepEvalSuite(base_model=model, use_baseline_cache=use_cache)
    print(f"\nRunning benchmarks ({args.n} samples each)...")
    print("Per O-0006: Using 1 persistent session for claim accumulation")
    if use_cache:
        print("Baseline cache: ENABLED (use --refresh-baseline to re-run baseline tests)\n")
    else:
        print("Baseline cache: DISABLED (re-running baseline tests)\n")

    session_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Run specific benchmark or full suite
    if args.benchmark:
        # Initialize session for single benchmark
        if suite.conjecture_model:
            suite.conjecture_model.initialize_session(session_id=session_id)

        benchmark_map = {
            "gsm8k": suite.run_gsm8k,
            "logiqa": suite.run_logiqa,
            "bbh-math": suite.run_bbh_math,
            "bbh-object": suite.run_bbh_object_counting,
            "bbh-logic": suite.run_bbh_logical_deduction,
            "bbh-navigate": suite.run_bbh_navigate,
            "bbh-date": suite.run_bbh_date,
            "bbh-tracking": suite.run_bbh_tracking,
            "bbh-lies": suite.run_bbh_web_of_lies,
            "truthfulqa": suite.run_truthfulqa,
            "winogrande": suite.run_winogrande,
        }

        if args.benchmark.lower() not in benchmark_map:
            print(f"Unknown benchmark: {args.benchmark}")
            print(f"Available: {', '.join(benchmark_map.keys())}")
            return

        result = benchmark_map[args.benchmark.lower()](args.n)
        results = {args.benchmark.upper(): result}

        # Close session
        if suite.conjecture_model:
            suite.conjecture_model.close()
    else:
        results = suite.run_full_suite(n_samples=args.n, session_id=session_id)

    # Get final claim count from session
    session_claims = 0
    if suite.conjecture_model and suite.conjecture_model._endpoint:
        session_claims = suite.conjecture_model._endpoint.claim_count()

    print("\nResults:")
    print("-" * 50)
    for name, r in results.items():
        if r.error:
            print(f"{name}: ERROR - {r.error}")
        else:
            print(f"{name}: baseline={r.baseline_score:.1f}%  conjecture={r.conjecture_score:.1f}%  delta={r.delta:+.1f}pp")

    stats_key = args.stats_key or "deepeval_oss"
    suite.update_stats_yaml(key=stats_key, session_claims=session_claims)
    print(f"\nSTATS.yaml updated (key: {stats_key}, session_claims: {session_claims})")


if __name__ == "__main__":
    main()
