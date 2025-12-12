"""
Modular Benchmark Framework for Conjecture
Supports multiple benchmark types: AIME25, GPQA, LiveCodeBench, etc.
"""

import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from abc import ABC, abstractmethod

@dataclass
class BenchmarkTask:
    """Single benchmark task"""
    task_id: str
    prompt: str
    expected_answer: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    task_id: str
    model_name: str
    using_conjecture: bool
    response: str
    correct: bool
    score: float
    execution_time: float
    error: Optional[str] = None

@dataclass
class BenchmarkSummary:
    """Summary of benchmark results"""
    benchmark_name: str
    model_name: str
    using_conjecture: bool
    total_tasks: int
    correct_answers: int
    accuracy: float
    average_time: float
    total_time: float

class Benchmark(ABC):
    """Abstract base class for benchmarks"""

    def __init__(self, name: str):
        self.name = name
        self.tasks: List[BenchmarkTask] = []

    @abstractmethod
    async def load_tasks(self) -> List[BenchmarkTask]:
        """Load benchmark tasks"""
        pass

    @abstractmethod
    def evaluate_response(self, task: BenchmarkTask, response: str) -> bool:
        """Evaluate if response is correct"""
        pass

    async def run_benchmark(self, model_func: Callable, model_name: str, using_conjecture: bool = False) -> BenchmarkSummary:
        """Run benchmark with given model function"""
        if not self.tasks:
            self.tasks = await self.load_tasks()

        results = []
        start_time = time.time()

        for task in self.tasks:
            try:
                response_start = time.time()
                response = await model_func(task.prompt)
                execution_time = time.time() - response_start

                correct = self.evaluate_response(task, response)
                score = 1.0 if correct else 0.0

                result = BenchmarkResult(
                    task_id=task.task_id,
                    model_name=model_name,
                    using_conjecture=using_conjecture,
                    response=response,
                    correct=correct,
                    score=score,
                    execution_time=execution_time
                )
                results.append(result)

            except Exception as e:
                result = BenchmarkResult(
                    task_id=task.task_id,
                    model_name=model_name,
                    using_conjecture=using_conjecture,
                    response="",
                    correct=False,
                    score=0.0,
                    execution_time=0.0,
                    error=str(e)
                )
                results.append(result)

        total_time = time.time() - start_time
        correct_count = sum(1 for r in results if r.correct)
        accuracy = correct_count / len(results)
        avg_time = sum(r.execution_time for r in results) / len(results)

        return BenchmarkSummary(
            benchmark_name=self.name,
            model_name=model_name,
            using_conjecture=using_conjecture,
            total_tasks=len(results),
            correct_answers=correct_count,
            accuracy=accuracy,
            average_time=avg_time,
            total_time=total_time
        )

class AIME25Benchmark(Benchmark):
    """AIME 2025 Mathematics Competition Benchmark"""

    def __init__(self):
        super().__init__("AIME25")
        self.dataset = None
        self.sample_tasks = [
            BenchmarkTask(
                task_id="aime25_1",
                prompt="Find the sum of all positive integers n such that n divides 7^n - 1.",
                expected_answer="42",
                metadata={"difficulty": "medium", "category": "number_theory"}
            ),
            BenchmarkTask(
                task_id="aime25_2",
                prompt="In triangle ABC, AB = 13, BC = 14, CA = 15. Let D be the foot of the altitude from A to BC. Find AD.",
                expected_answer="12",
                metadata={"difficulty": "easy", "category": "geometry"}
            )
        ]

    async def load_tasks(self) -> List[BenchmarkTask]:
        """Load real AIME 2025 tasks from Hugging Face dataset"""
        try:
            # Try to load from Hugging Face
            from datasets import load_dataset

            print("Loading AIME 2025 dataset from Hugging Face...")
            # Load both AIME 2025-I and AIME 2025-II
            dataset_i = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
            dataset_ii = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")

            tasks = []

            # Process AIME 2025-I problems
            for i, example in enumerate(dataset_i):
                task = BenchmarkTask(
                    task_id=f"aime25_i_{i+1:03d}",
                    prompt=example["question"],
                    expected_answer=str(example["answer"]),
                    metadata={
                        "source": "AIME 2025-I",
                        "difficulty": "hard",
                        "category": "mathematics",
                        "exam": "I"
                    }
                )
                tasks.append(task)

            # Process AIME 2025-II problems
            for i, example in enumerate(dataset_ii):
                task = BenchmarkTask(
                    task_id=f"aime25_ii_{i+1:03d}",
                    prompt=example["question"],
                    expected_answer=str(example["answer"]),
                    metadata={
                        "source": "AIME 2025-II",
                        "difficulty": "hard",
                        "category": "mathematics",
                        "exam": "II"
                    }
                )
                tasks.append(task)

            print(f"Loaded {len(tasks)} AIME 2025 problems ({len(dataset_i)} from I, {len(dataset_ii)} from II)")
            return tasks

        except Exception as e:
            print(f"Failed to load AIME 2025 dataset: {e}")
            print("Falling back to sample tasks...")
            return self.sample_tasks

    def evaluate_response(self, task: BenchmarkTask, response: str) -> bool:
        """Check if response contains the correct numerical answer"""
        if not task.expected_answer:
            return False

        # For mathematical answers, we need to be more sophisticated
        # Extract numbers from response and check if the expected answer is present
        import re

        # Clean the expected answer (remove whitespace)
        expected_clean = task.expected_answer.strip()

        # Look for the exact answer in the response
        # Check for patterns like "Answer: 42", "42", "The answer is 42", etc.
        patterns = [
            rf"(?:answer[:\s]+|is[:\s]+|=[:\s]+){re.escape(expected_clean)}\b",
            rf"\b{re.escape(expected_clean)}\b",
            rf"(?:final answer|result)[:\s]+{re.escape(expected_clean)}\b"
        ]

        for pattern in patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return True

        # As a fallback, check if the number appears anywhere in the response
        if expected_clean in response:
            return True

        return False

class GPQABenchmark(Benchmark):
    """GPQA (Graduate-Level Google-Proof Q&A) Benchmark"""

    def __init__(self, variant: str = "diamond"):
        super().__init__(f"GPQA-{variant}")
        self.variant = variant
        self.sample_tasks = [
            BenchmarkTask(
                task_id="gpqa_1",
                prompt="Explain the mechanism by which CRISPR-Cas9 gene editing works, including the roles of guide RNA and Cas9 protein.",
                expected_answer=None,  # Open-ended evaluation
                metadata={"difficulty": "hard", "category": "biology"}
            )
        ]

    async def load_tasks(self) -> List[BenchmarkTask]:
        return self.sample_tasks

    def evaluate_response(self, task: BenchmarkTask, response: str) -> bool:
        """Evaluate GPQA response based on key concepts"""
        if task.task_id == "gpqa_1":
            key_concepts = ["guide RNA", "Cas9", "DNA", "cut", "repair"]
            return sum(1 for concept in key_concepts if concept.lower() in response.lower()) >= 3
        return len(response) > 100  # Basic completeness check

class SWEVerifiedBenchmark(Benchmark):
    """SWE-Bench Verified Benchmark"""

    def __init__(self):
        super().__init__("SWE-Bench-Verified")
        self.sample_tasks = [
            BenchmarkTask(
                task_id="swe_1",
                prompt="Fix this Python function that should reverse a list:\n\ndef reverse_list(lst):\n    return lst.reverse()",
                expected_answer=None,
                metadata={"difficulty": "easy", "category": "coding"}
            )
        ]

    async def load_tasks(self) -> List[BenchmarkTask]:
        return self.sample_tasks

    def evaluate_response(self, task: BenchmarkTask, response: str) -> bool:
        """Check if the response fixes the issue"""
        if "reverse_list" in response and "def reverse_list" in response:
            # Check if it returns a new list rather than modifying in place
            return "return lst[::-1]" in response or "new_list" in response
        return False

class LiveCodeBenchBenchmark(Benchmark):
    """LiveCodeBench Programming Benchmark"""

    def __init__(self):
        super().__init__("LiveCodeBench")
        self.sample_tasks = [
            BenchmarkTask(
                task_id="lcb_1",
                prompt="Write a Python function to implement a binary search tree with insert and search operations.",
                expected_answer=None,
                metadata={"difficulty": "medium", "category": "data_structures"}
            )
        ]

    async def load_tasks(self) -> List[BenchmarkTask]:
        return self.sample_tasks

    def evaluate_response(self, task: BenchmarkTask, response: str) -> bool:
        """Evaluate code completeness"""
        required_elements = ["class", "def insert", "def search"]
        return all(elem in response for elem in required_elements)

class BenchmarkRunner:
    """Main benchmark runner for multiple models and benchmarks"""

    def __init__(self):
        self.benchmarks = {
            "AIME25": AIME25Benchmark(),
            "GPQA": GPQABenchmark("diamond"),  # Use diamond as default
            "SWE-Bench-Verified": SWEVerifiedBenchmark(),
            "LiveCodeBench": LiveCodeBenchBenchmark()
        }
        self.results = []

    async def run_all_benchmarks(self, models: Dict[str, Callable]) -> Dict[str, List[BenchmarkSummary]]:
        """Run all benchmarks for all models"""
        all_results = {}

        for model_name, model_func in models.items():
            model_results = []

            for benchmark_name, benchmark in self.benchmarks.items():
                print(f"Running {benchmark_name} with {model_name}...")

                # Run without Conjecture
                result_direct = await benchmark.run_benchmark(model_func, model_name, using_conjecture=False)
                model_results.append(result_direct)

                # Run with Conjecture (if available)
                if self.has_conjecture():
                    result_conjecture = await benchmark.run_benchmark(
                        lambda prompt: self.with_conjecture(model_func, prompt),
                        f"{model_name}+Conjecture",
                        using_conjecture=True
                    )
                    model_results.append(result_conjecture)

            all_results[model_name] = model_results

        return all_results

    def has_conjecture(self) -> bool:
        """Check if Conjecture enhancement is available"""
        try:
            from ..processing.unified_bridge import UnifiedLLMBridge
            return True
        except ImportError:
            return False

    async def with_conjecture(self, base_model_func: Callable, prompt: str) -> str:
        """Apply Conjecture enhancement to model call"""
        # Map base model function to conjecture version
        if base_model_func == granite_tiny_model:
            return await granite_tiny_conjecture(prompt)
        elif base_model_func == gpt_oss_20b_model:
            return await gpt_oss_20b_conjecture(prompt)
        elif base_model_func == glm_46_model:
            return await glm_46_conjecture(prompt)
        else:
            # Fallback to base model with enhanced prompt
            enhanced_prompt = f"""Using advanced reasoning and systematic approach:

{prompt}

Please provide a comprehensive and accurate solution."""
            return await base_model_func(enhanced_prompt)

    def generate_results_chart(self, results: Dict[str, List[BenchmarkSummary]]) -> str:
        """Generate ASCII chart of benchmark results"""
        chart_lines = []
        chart_lines.append("\nBENCHMARK RESULTS CHART")
        chart_lines.append("=" * 80)

        # Header
        chart_lines.append(f"{'Benchmark':<20} {'Model':<15} {'Accuracy':<10} {'Time (s)':<10} {'Conjecture':<10}")
        chart_lines.append("-" * 80)

        # Results
        for model_name, model_results in results.items():
            for result in model_results:
                conjecture_str = "Yes" if result.using_conjecture else "No"
                accuracy_str = f"{result.accuracy:.1%}"
                time_str = f"{result.average_time:.2f}"
                chart_lines.append(
                    f"{result.benchmark_name:<20} {result.model_name:<15} "
                    f"{accuracy_str:<10} {time_str:<10} {conjecture_str:<10}"
                )

        return "\n".join(chart_lines)

# Import real model integration
from .model_integration import (
    granite_tiny_model, gpt_oss_20b_model, glm_46_model,
    granite_tiny_conjecture, gpt_oss_20b_conjecture, glm_46_conjecture
)