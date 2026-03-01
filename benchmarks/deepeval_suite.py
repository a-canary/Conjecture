"""
DeepEval Benchmark Suite for Conjecture
Benchmarks: GSM8K (math), MathQA (math reasoning), HellaSwag (commonsense)
Target: OSS models (20B class) where Conjecture should add value
Per O-0006: Uses 1 persistent session for claim accumulation across test cases.
Outputs to STATS.yaml
"""

import argparse
import asyncio
import yaml
import os
import sys
import re
sys.path.insert(0, '/workspace')

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from deepeval.benchmarks import GSM8K, MathQA, HellaSwag, LogiQA, TruthfulQA
    from deepeval.benchmarks.gsm8k.template import GSM8KTemplate
    from deepeval.benchmarks.math_qa.template import MathQATemplate
    from deepeval.benchmarks.hellaswag.template import HellaSwagTemplate
    from deepeval.benchmarks.logi_qa.template import LogiQATemplate
    from deepeval.benchmarks.truthful_qa.template import TruthfulQATemplate
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


def _call_model(model, prompt: str) -> str:
    """Call a model and return the response text, handling (text, usage) tuples."""
    result = model.generate(prompt)
    if isinstance(result, tuple):
        return result[0]
    return str(result)


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

    TruthfulQA MC1 uses numeric answers (1, 2, 3, 4, etc.)
    """
    # Pattern 1: "answer is X" with number
    match = re.search(r'answer\s+is\s*[:\s]*(\d+)', response, re.I)
    if match:
        return match.group(1)

    # Pattern 2: Number at end of response
    match = re.search(r'\b([1-4])\s*$', response.strip())
    if match:
        return match.group(1)

    # Pattern 3: "option X" or "choice X"
    match = re.search(r'(?:option|choice)\s*[:\s]*(\d+)', response, re.I)
    if match:
        return match.group(1)

    # Pattern 4: First standalone number 1-4
    match = re.search(r'\b([1-4])\b', response)
    if match:
        return match.group(1)

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
    """Run DeepEval benchmarks comparing baseline vs Conjecture using direct answer extraction"""

    def __init__(self, base_model=None):
        self.base_model = base_model
        self.conjecture_model = ConjectureModel(base_model) if base_model else None
        self.results: List[BenchmarkResult] = []
        self.stats_path = Path(__file__).parent.parent / "STATS.yaml"

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

                # Baseline
                try:
                    baseline_response = _call_model(self.base_model, prompt)
                    extracted = extract_gsm8k_answer(baseline_response)
                    # Numeric comparison (handle floats)
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
                    print(f"  GSM8K: {i+1}/{total} done (baseline {baseline_correct}, conj {conj_correct})")

            baseline_score = baseline_correct / total * 100
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

    def run_full_suite(self, n_samples: int = 20, session_id: str = "benchmark_session") -> Dict[str, BenchmarkResult]:
        """Run all benchmarks sequentially with persistent session.

        Per O-0006: Uses 1 persistent session for claim accumulation across test cases.
        Claims from earlier problems may enhance later problem solving.
        """
        # Initialize persistent session for Conjecture model
        if self.conjecture_model:
            self.conjecture_model.initialize_session(session_id=session_id)

        try:
            results = {
                "GSM8K": self.run_gsm8k(n_samples),
                "MathQA": self.run_mathqa(n_samples),
                "HellaSwag": self.run_hellaswag(n_samples),
                "LogiQA": self.run_logiqa(n_samples),
                "TruthfulQA": self.run_truthfulqa(n_samples),
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
        description="DeepEval Benchmark Suite — GSM8K, MathQA, HellaSwag via Chutes.ai"
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
    args = parser.parse_args()

    print("DeepEval Benchmark Suite (OSS Models)")
    print("=" * 50)
    print(f"Model : {args.model}")
    print(f"N     : {args.n} samples per benchmark")
    print("Benchmarks: GSM8K, MathQA, HellaSwag")

    api_key = os.environ.get("CHUTES_API_KEY")
    if not api_key:
        print("ERROR: Set CHUTES_API_KEY environment variable")
        print("  export CHUTES_API_KEY=your_key_here")
        return

    model = create_chutes_model(api_key, model=args.model)

    suite = DeepEvalSuite(base_model=model)
    print(f"\nRunning benchmarks ({args.n} samples each)...")
    print("Per O-0006: Using 1 persistent session for claim accumulation\n")

    session_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
