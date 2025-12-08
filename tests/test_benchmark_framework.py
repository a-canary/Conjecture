"""
Tests for the benchmark framework
"""

import pytest
import asyncio
from src.benchmarking.benchmark_framework import (
    AIME25Benchmark, GPQABenchmark, SWEVerifiedBenchmark,
    LiveCodeBenchBenchmark, BenchmarkRunner, BenchmarkTask
)

class TestAIME25Benchmark:
    """Test AIME25 benchmark"""

    @pytest.fixture
    def benchmark(self):
        return AIME25Benchmark()

    @pytest.mark.asyncio
    async def test_load_tasks(self, benchmark):
        tasks = await benchmark.load_tasks()
        assert len(tasks) > 0
        assert all(isinstance(task, BenchmarkTask) for task in tasks)

    def test_evaluate_response_correct(self, benchmark):
        task = BenchmarkTask("test", "question", "42")
        response = "The answer is 42, which is the solution."
        assert benchmark.evaluate_response(task, response) == True

    def test_evaluate_response_incorrect(self, benchmark):
        task = BenchmarkTask("test", "question", "42")
        response = "The answer is 24."
        assert benchmark.evaluate_response(task, response) == False

class TestGPQABenchmark:
    """Test GPQA benchmark"""

    @pytest.fixture
    def benchmark(self):
        return GPQABenchmark()

    @pytest.mark.asyncio
    async def test_load_tasks(self, benchmark):
        tasks = await benchmark.load_tasks()
        assert len(tasks) > 0

    def test_evaluate_response_key_concepts(self, benchmark):
        task = BenchmarkTask("gpqa_1", "CRISPR question")
        response = "CRISPR-Cas9 uses guide RNA to direct the Cas9 protein to cut DNA at specific locations."
        assert benchmark.evaluate_response(task, response) == True

    def test_evaluate_response_insufficient(self, benchmark):
        task = BenchmarkTask("gpqa_1", "CRISPR question")
        response = "CRISPR is a gene editing tool."
        assert benchmark.evaluate_response(task, response) == False

class TestSWEVerifiedBenchmark:
    """Test SWE-Bench Verified benchmark"""

    @pytest.fixture
    def benchmark(self):
        return SWEVerifiedBenchmark()

    @pytest.mark.asyncio
    async def test_load_tasks(self, benchmark):
        tasks = await benchmark.load_tasks()
        assert len(tasks) > 0

    def test_evaluate_response_correct_fix(self, benchmark):
        task = BenchmarkTask("swe_1", "fix reverse function")
        response = """
        def reverse_list(lst):
            return lst[::-1]
        """
        assert benchmark.evaluate_response(task, response) == True

    def test_evaluate_response_incomplete_fix(self, benchmark):
        task = BenchmarkTask("swe_1", "fix reverse function")
        response = "The function should use slicing."
        assert benchmark.evaluate_response(task, response) == False

class TestLiveCodeBenchBenchmark:
    """Test LiveCodeBench benchmark"""

    @pytest.fixture
    def benchmark(self):
        return LiveCodeBenchBenchmark()

    @pytest.mark.asyncio
    async def test_load_tasks(self, benchmark):
        tasks = await benchmark.load_tasks()
        assert len(tasks) > 0

    def test_evaluate_response_complete_code(self, benchmark):
        task = BenchmarkTask("lcb_1", "BST implementation")
        response = """
        class BSTNode:
            def __init__(self, val):
                self.val = val
                self.left = None
                self.right = None

        class BST:
            def __init__(self):
                self.root = None

            def insert(self, val):
                pass

            def search(self, val):
                pass
        """
        assert benchmark.evaluate_response(task, response) == True

    def test_evaluate_response_incomplete_code(self, benchmark):
        task = BenchmarkTask("lcb_1", "BST implementation")
        response = "You need to implement a BST with insert and search methods."
        assert benchmark.evaluate_response(task, response) == False

class TestBenchmarkRunner:
    """Test benchmark runner"""

    @pytest.fixture
    def runner(self):
        return BenchmarkRunner()

    def test_initialization(self, runner):
        assert len(runner.benchmarks) > 0
        assert "AIME25" in runner.benchmarks
        assert "GPQA" in runner.benchmarks

    @pytest.mark.asyncio
    async def test_run_single_benchmark(self, runner):
        """Test running a single benchmark with mock model"""
        async def mock_model(prompt):
            return f"Response to: {prompt[:20]}..."

        benchmark = runner.benchmarks["AIME25"]
        result = await benchmark.run_benchmark(mock_model, "test-model")

        assert result.benchmark_name == "AIME25"
        assert result.model_name == "test-model"
        assert result.total_tasks > 0
        assert result.accuracy >= 0.0
        assert result.accuracy <= 1.0

    def test_generate_results_chart(self, runner):
        """Test results chart generation"""
        from src.benchmarking.benchmark_framework import BenchmarkSummary

        # Create mock results
        mock_results = {
            "test-model": [
                BenchmarkSummary(
                    benchmark_name="AIME25",
                    model_name="test-model",
                    using_conjecture=False,
                    total_tasks=5,
                    correct_answers=3,
                    accuracy=0.6,
                    average_time=2.5,
                    total_time=12.5
                )
            ]
        }

        chart = runner.generate_results_chart(mock_results)
        assert "AIME25" in chart
        assert "test-model" in chart
        assert "60.0%" in chart