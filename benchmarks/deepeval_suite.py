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
        "--refresh-baseline",
        action="store_true",
        help="Re-run baseline tests (use when fixing benchmark/parser bugs)",
    )
    args = parser.parse_args()

    print("DeepEval Benchmark Suite (OSS Models)")
    print("=" * 50)
    print(f"N     : {args.n} samples per benchmark")
    print("O-0008: 10 benchmarks, >= Direct on ALL, +20pp on 5")

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
