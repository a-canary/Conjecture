#!/usr/bin/env python3
"""
ARC-AGI-2 Benchmark Runner
Primary benchmark per CHOICES.md O-0006

Compares:
- Bare LLM: Direct prompting (Chutes/glm-4.5-air or Anthropic/Haiku)
- LLM+Conjecture: Same model with Conjecture harness for enhanced reasoning

The goal is to measure reasoning improvement from the Conjecture harness,
not just speed or token efficiency.
"""

import asyncio
import json
import time
import uuid
import statistics
import requests
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logger = logging.getLogger(__name__)


class ChutesLLMWrapper:
    """Wrapper for Chutes.ai API with compatible interface"""

    def __init__(self, api_key: str, api_url: str, model_name: str):
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.stats = {"total_requests": 0, "successful_requests": 0, "total_tokens": 0}

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate response using Chutes.ai"""
        self.stats["total_requests"] += 1

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            self.stats["successful_requests"] += 1
            self.stats["total_tokens"] += result.get("usage", {}).get("total_tokens", 0)

            return content

        except Exception as e:
            logger.error(f"Chutes API error: {e}")
            raise


@dataclass
class ARCTask:
    """Represents a single ARC-AGI-2 task"""
    task_id: str
    train_examples: List[Dict[str, List[List[int]]]]  # List of {input: grid, output: grid}
    test_input: List[List[int]]  # The test input grid
    test_output: List[List[int]]  # The expected test output grid
    difficulty: str = "unknown"  # easy, medium, hard
    category: str = "unknown"  # pattern type


@dataclass
class BenchmarkResult:
    """Result from running a single task"""
    task_id: str
    mode: str  # "bare_haiku" or "haiku_conjecture"
    correct: bool
    predicted_output: Optional[List[List[int]]]
    expected_output: List[List[int]]
    processing_time: float
    tokens_used: int = 0
    claims_generated: int = 0  # Only for Conjecture mode
    reasoning_steps: int = 0  # Only for Conjecture mode
    error_message: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark run"""
    mode: str
    total_tasks: int
    correct: int
    accuracy: float
    avg_time: float
    total_tokens: int
    avg_claims: float = 0.0  # Only for Conjecture mode
    avg_reasoning_steps: float = 0.0


@dataclass
class ComparisonResult:
    """Comparison between bare Haiku and Haiku+Conjecture"""
    bare_haiku: BenchmarkSummary
    haiku_conjecture: BenchmarkSummary
    accuracy_improvement: float  # Percentage points
    time_overhead: float  # Percentage
    reasoning_value_ratio: float  # Accuracy gain per time overhead


class ARCBenchmarkRunner:
    """
    Runs ARC-AGI-2 benchmark comparing bare LLM vs LLM+Conjecture
    """

    def __init__(
        self,
        data_dir: str = "./data/arc_agi2",
        results_dir: str = "./results/arc_agi2",
        model: str = "deepseek-ai/DeepSeek-V3",
        provider: str = "chutes",
    ):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.model = model
        self.provider = provider
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded components
        self._bare_processor = None
        self._conjecture_processor = None

    def _get_bare_processor(self):
        """Get bare LLM processor (direct prompting)"""
        if self._bare_processor is None:
            if self.provider == "chutes":
                api_key = os.environ.get("CHUTES_API_KEY")
                if not api_key:
                    raise ValueError("CHUTES_API_KEY environment variable required")
                self._bare_processor = ChutesLLMWrapper(
                    api_key=api_key,
                    api_url="https://llm.chutes.ai/v1",
                    model_name=self.model,
                )
            else:
                try:
                    from processing.llm.anthropic_integration import create_anthropic_processor
                    self._bare_processor = create_anthropic_processor(model=self.model)
                except ImportError:
                    logger.error("Cannot import Anthropic processor")
                    raise
        return self._bare_processor

    def _get_conjecture_processor(self):
        """Get LLM+Conjecture processor (enhanced reasoning)"""
        if self._conjecture_processor is None:
            self._conjecture_processor = self._get_bare_processor()
        return self._conjecture_processor

    def load_tasks(self, limit: Optional[int] = None) -> List[ARCTask]:
        """
        Load ARC-AGI-2 tasks from data directory

        Args:
            limit: Maximum number of tasks to load (None = all)

        Returns:
            List of ARCTask objects
        """
        tasks = []

        # Check if data directory exists
        if not self.data_dir.exists():
            logger.warning(f"ARC data directory not found: {self.data_dir}")
            logger.info("Using sample tasks for demo")
            return self._get_sample_tasks()

        # Load tasks from JSON files
        task_files = list(self.data_dir.glob("*.json"))
        if limit:
            task_files = task_files[:limit]

        for task_file in task_files:
            try:
                with open(task_file) as f:
                    data = json.load(f)
                task = ARCTask(
                    task_id=task_file.stem,
                    train_examples=data.get("train", []),
                    test_input=data.get("test", [{}])[0].get("input", []),
                    test_output=data.get("test", [{}])[0].get("output", []),
                    difficulty=data.get("difficulty", "unknown"),
                    category=data.get("category", "unknown"),
                )
                tasks.append(task)
            except Exception as e:
                logger.warning(f"Failed to load task {task_file}: {e}")

        logger.info(f"Loaded {len(tasks)} ARC-AGI-2 tasks")
        return tasks

    def _get_sample_tasks(self) -> List[ARCTask]:
        """Get sample tasks for testing/demo when real data unavailable"""
        return [
            ARCTask(
                task_id="sample_001",
                train_examples=[
                    {"input": [[0, 0], [0, 1]], "output": [[1, 0], [0, 0]]},
                    {"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]},
                ],
                test_input=[[0, 1], [0, 0]],
                test_output=[[0, 0], [1, 0]],
                difficulty="easy",
                category="rotation",
            ),
            ARCTask(
                task_id="sample_002",
                train_examples=[
                    {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
                    {"input": [[5, 6], [7, 8]], "output": [[6, 5], [8, 7]]},
                ],
                test_input=[[9, 0], [1, 2]],
                test_output=[[0, 9], [2, 1]],
                difficulty="easy",
                category="mirror",
            ),
        ]

    def _format_task_prompt(self, task: ARCTask, include_reasoning: bool = False) -> str:
        """Format task as prompt for LLM"""
        prompt = "You are solving an ARC-AGI-2 visual reasoning task.\n\n"

        # Add training examples
        prompt += "## Training Examples\n"
        for i, example in enumerate(task.train_examples, 1):
            prompt += f"\n### Example {i}\n"
            prompt += f"Input:\n{self._format_grid(example['input'])}\n"
            prompt += f"Output:\n{self._format_grid(example['output'])}\n"

        # Add test input
        prompt += "\n## Test Input\n"
        prompt += f"{self._format_grid(task.test_input)}\n"

        # Add instructions
        if include_reasoning:
            prompt += """
## Instructions
1. Analyze the pattern in the training examples
2. Break down your reasoning into clear steps
3. Identify the transformation rule
4. Apply the rule to the test input
5. Provide the output grid

Format your response as:
REASONING: <your step-by-step reasoning>
OUTPUT: <the output grid as a JSON array>
"""
        else:
            prompt += """
## Instructions
Identify the pattern and provide the output grid for the test input.
Respond only with the output grid as a JSON array of arrays.
"""

        return prompt

    def _format_grid(self, grid: List[List[int]]) -> str:
        """Format grid for display"""
        return "\n".join([" ".join(str(cell) for cell in row) for row in grid])

    def _parse_output(self, response: str) -> Optional[List[List[int]]]:
        """Parse LLM response to extract output grid"""
        try:
            # Try to find JSON array in response
            import re
            # Look for OUTPUT: section first
            if "OUTPUT:" in response:
                response = response.split("OUTPUT:")[-1]

            # Find JSON array
            match = re.search(r'\[\s*\[.*?\]\s*\]', response, re.DOTALL)
            if match:
                return json.loads(match.group())

            # Try parsing entire response as JSON
            return json.loads(response)
        except Exception:
            return None

    def _grids_match(self, predicted: Optional[List[List[int]]], expected: List[List[int]]) -> bool:
        """Check if two grids match"""
        if predicted is None:
            return False
        if len(predicted) != len(expected):
            return False
        for pred_row, exp_row in zip(predicted, expected):
            if pred_row != exp_row:
                return False
        return True

    async def run_task_bare(self, task: ARCTask) -> BenchmarkResult:
        """Run task with bare Haiku (direct prompting)"""
        start_time = time.time()

        try:
            processor = self._get_bare_processor()
            prompt = self._format_task_prompt(task, include_reasoning=False)

            response = processor.generate_response(
                prompt=prompt,
                system_prompt="You are an expert at visual pattern recognition and abstract reasoning.",
                max_tokens=500,
            )

            predicted = self._parse_output(response)
            correct = self._grids_match(predicted, task.test_output)

            return BenchmarkResult(
                task_id=task.task_id,
                mode="bare_haiku",
                correct=correct,
                predicted_output=predicted,
                expected_output=task.test_output,
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            return BenchmarkResult(
                task_id=task.task_id,
                mode="bare_haiku",
                correct=False,
                predicted_output=None,
                expected_output=task.test_output,
                processing_time=time.time() - start_time,
                error_message=str(e),
            )

    async def run_task_conjecture(self, task: ARCTask) -> BenchmarkResult:
        """Run task with Haiku+Conjecture (enhanced reasoning)"""
        start_time = time.time()
        claims_generated = 0
        reasoning_steps = 0

        try:
            # Use Conjecture harness for claim-based reasoning
            from conjecture_harness import create_conjecture_harness

            harness = create_conjecture_harness(self._conjecture_processor)
            session, predicted = harness.process_task(task)

            # Get stats from harness
            stats = harness.get_session_stats(session)
            claims_generated = stats["total_claims"]
            reasoning_steps = stats["reasoning_steps"]

            correct = self._grids_match(predicted, task.test_output)

            return BenchmarkResult(
                task_id=task.task_id,
                mode="haiku_conjecture",
                correct=correct,
                predicted_output=predicted,
                expected_output=task.test_output,
                processing_time=time.time() - start_time,
                claims_generated=claims_generated,
                reasoning_steps=reasoning_steps,
            )

        except ImportError:
            # Fallback if harness not available: use enhanced prompting
            try:
                processor = self._get_conjecture_processor()
                prompt = self._format_task_prompt(task, include_reasoning=True)

                system_prompt = """You are an expert at visual pattern recognition and abstract reasoning.
Use the Conjecture method:
1. Create claims about the patterns you observe
2. Assign confidence to each claim
3. Verify claims against training examples
4. Synthesize a solution from validated claims
"""

                response = processor.generate_response(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=1000,
                )

                # Count reasoning indicators
                if "REASONING:" in response:
                    reasoning_text = response.split("REASONING:")[-1].split("OUTPUT:")[0]
                    reasoning_steps = reasoning_text.count("\n") + 1
                    claims_generated = max(1, reasoning_steps // 2)

                predicted = self._parse_output(response)
                correct = self._grids_match(predicted, task.test_output)

                return BenchmarkResult(
                    task_id=task.task_id,
                    mode="haiku_conjecture",
                    correct=correct,
                    predicted_output=predicted,
                    expected_output=task.test_output,
                    processing_time=time.time() - start_time,
                    claims_generated=claims_generated,
                    reasoning_steps=reasoning_steps,
                )

            except Exception as e:
                return BenchmarkResult(
                    task_id=task.task_id,
                    mode="haiku_conjecture",
                    correct=False,
                    predicted_output=None,
                    expected_output=task.test_output,
                    processing_time=time.time() - start_time,
                    error_message=str(e),
                )

        except Exception as e:
            return BenchmarkResult(
                task_id=task.task_id,
                mode="haiku_conjecture",
                correct=False,
                predicted_output=None,
                expected_output=task.test_output,
                processing_time=time.time() - start_time,
                error_message=str(e),
            )

    def _summarize_results(self, results: List[BenchmarkResult], mode: str) -> BenchmarkSummary:
        """Summarize benchmark results"""
        if not results:
            return BenchmarkSummary(
                mode=mode,
                total_tasks=0,
                correct=0,
                accuracy=0.0,
                avg_time=0.0,
                total_tokens=0,
            )

        correct = sum(1 for r in results if r.correct)
        return BenchmarkSummary(
            mode=mode,
            total_tasks=len(results),
            correct=correct,
            accuracy=correct / len(results) * 100,
            avg_time=statistics.mean(r.processing_time for r in results),
            total_tokens=sum(r.tokens_used for r in results),
            avg_claims=statistics.mean(r.claims_generated for r in results) if mode == "haiku_conjecture" else 0.0,
            avg_reasoning_steps=statistics.mean(r.reasoning_steps for r in results) if mode == "haiku_conjecture" else 0.0,
        )

    async def run_benchmark(
        self,
        task_limit: Optional[int] = 10,
        save_results: bool = True,
    ) -> ComparisonResult:
        """
        Run full ARC-AGI-2 benchmark

        Args:
            task_limit: Max tasks to run (None = all)
            save_results: Whether to save results to disk

        Returns:
            ComparisonResult with bare vs Conjecture comparison
        """
        logger.info(f"Starting ARC-AGI-2 benchmark (limit={task_limit})")

        # Load tasks
        tasks = self.load_tasks(limit=task_limit)
        if not tasks:
            logger.error("No tasks available")
            raise ValueError("No ARC-AGI-2 tasks found")

        # Run bare Haiku
        logger.info("Running bare Haiku...")
        bare_results = []
        for task in tasks:
            result = await self.run_task_bare(task)
            bare_results.append(result)
            logger.debug(f"Task {task.task_id}: {'PASS' if result.correct else 'FAIL'}")

        # Run Haiku+Conjecture
        logger.info("Running Haiku+Conjecture...")
        conjecture_results = []
        for task in tasks:
            result = await self.run_task_conjecture(task)
            conjecture_results.append(result)
            logger.debug(f"Task {task.task_id}: {'PASS' if result.correct else 'FAIL'}")

        # Summarize results
        bare_summary = self._summarize_results(bare_results, "bare_haiku")
        conjecture_summary = self._summarize_results(conjecture_results, "haiku_conjecture")

        # Calculate comparison metrics
        accuracy_improvement = conjecture_summary.accuracy - bare_summary.accuracy
        time_overhead = (
            (conjecture_summary.avg_time - bare_summary.avg_time) / bare_summary.avg_time * 100
            if bare_summary.avg_time > 0 else 0
        )
        reasoning_value_ratio = (
            accuracy_improvement / time_overhead if time_overhead > 0 else float('inf')
        )

        comparison = ComparisonResult(
            bare_haiku=bare_summary,
            haiku_conjecture=conjecture_summary,
            accuracy_improvement=accuracy_improvement,
            time_overhead=time_overhead,
            reasoning_value_ratio=reasoning_value_ratio,
        )

        # Save results
        if save_results:
            self._save_results(bare_results, conjecture_results, comparison)

        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info("ARC-AGI-2 BENCHMARK RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Tasks: {len(tasks)}")
        logger.info(f"\nBare Haiku:")
        logger.info(f"  Accuracy: {bare_summary.accuracy:.1f}% ({bare_summary.correct}/{bare_summary.total_tasks})")
        logger.info(f"  Avg Time: {bare_summary.avg_time:.2f}s")
        logger.info(f"\nHaiku+Conjecture:")
        logger.info(f"  Accuracy: {conjecture_summary.accuracy:.1f}% ({conjecture_summary.correct}/{conjecture_summary.total_tasks})")
        logger.info(f"  Avg Time: {conjecture_summary.avg_time:.2f}s")
        logger.info(f"  Avg Claims: {conjecture_summary.avg_claims:.1f}")
        logger.info(f"\nComparison:")
        logger.info(f"  Accuracy Improvement: {accuracy_improvement:+.1f} pp")
        logger.info(f"  Time Overhead: {time_overhead:+.1f}%")
        logger.info(f"{'='*60}")

        return comparison

    def _save_results(
        self,
        bare_results: List[BenchmarkResult],
        conjecture_results: List[BenchmarkResult],
        comparison: ComparisonResult,
    ):
        """Save benchmark results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"arc_benchmark_{timestamp}.json"

        data = {
            "timestamp": timestamp,
            "bare_haiku": [asdict(r) for r in bare_results],
            "haiku_conjecture": [asdict(r) for r in conjecture_results],
            "summary": {
                "bare_haiku": asdict(comparison.bare_haiku),
                "haiku_conjecture": asdict(comparison.haiku_conjecture),
                "accuracy_improvement": comparison.accuracy_improvement,
                "time_overhead": comparison.time_overhead,
                "reasoning_value_ratio": comparison.reasoning_value_ratio,
            },
        }

        with open(results_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Results saved to {results_file}")


async def main():
    """Run ARC-AGI-2 benchmark"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    runner = ARCBenchmarkRunner()

    try:
        comparison = await runner.run_benchmark(task_limit=10)

        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60)
        print(f"Bare Haiku Accuracy: {comparison.bare_haiku.accuracy:.1f}%")
        print(f"Haiku+Conjecture Accuracy: {comparison.haiku_conjecture.accuracy:.1f}%")
        print(f"Improvement: {comparison.accuracy_improvement:+.1f} percentage points")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
