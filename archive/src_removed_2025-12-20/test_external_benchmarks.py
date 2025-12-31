#!/usr/bin/env python3
"""
Comprehensive tests for external benchmarks system
Tests HellaSwag, MMLU, GSM8K, ARC, Big-Bench Hard integration
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from benchmarks.benchmarking.external_benchmarks import (
    ExternalBenchmarks,
    ExternalBenchmarkTask,
    BenchmarkResult,
)


class TestExternalBenchmarks:
    """Test the external benchmarks system"""

    @pytest.fixture
    def benchmarks(self):
        """Create ExternalBenchmarks instance for testing"""
        return ExternalBenchmarks()

    class TestHellaSwagSamples:
        """Test HellaSwag sample generation"""

        def test_hellaswag_sample_structure(self, benchmarks):
            """Test HellaSwag sample structure"""
            samples = benchmarks.get_hellaswag_samples(num_samples=2)

            assert len(samples) == 2
            for sample in samples:
                assert isinstance(sample, ExternalBenchmarkTask)
                assert sample.task_id.startswith("hellaswag_")
                assert sample.benchmark_name == "HellaSwag"
                assert sample.domain == "commonsense_reasoning"
                assert len(sample.choices) == 4  # Multiple choice
                assert sample.correct_answer is not None
                assert isinstance(sample.metadata, dict)

        def test_hellaswag_sample_content(self, benchmarks):
            """Test HellaSwag sample content"""
            samples = benchmarks.get_hellaswag_samples(num_samples=1)
            sample = samples[0]

            # Should contain everyday activity scenario
            assert "woman" in sample.question or "bucket" in sample.question
            assert "dog" in sample.question
            assert len(sample.choices) == 4
            # Check that correct answer is one of the choices
            assert sample.correct_answer in sample.choices

        def test_hellaswag_sample_limit(self, benchmarks):
            """Test HellaSwag sample limiting"""
            # Request more than available unique samples
            samples = benchmarks.get_hellaswag_samples(num_samples=10)

            # Should limit to available unique samples (4 in mock data)
            assert len(samples) <= 4

    class TestMMLUSamples:
        """Test MMLU sample generation"""

        def test_mmlu_sample_structure(self, benchmarks):
            """Test MMLU sample structure"""
            samples = benchmarks.get_mmlu_samples(num_samples=3)

            assert len(samples) == 3
            for sample in samples:
                assert isinstance(sample, ExternalBenchmarkTask)
                assert sample.task_id.startswith("mmlu_")
                assert sample.benchmark_name == "MMLU"
                assert len(sample.choices) == 4  # Multiple choice
                assert sample.correct_answer == "Paris"
                assert isinstance(sample.metadata, dict)

        def test_mmlu_domain_coverage(self, benchmarks):
            """Test MMLU domain coverage"""
            samples = benchmarks.get_mmlu_samples(num_samples=8)
            domains = set(sample.domain for sample in samples)

            # Should cover multiple domains
            expected_domains = {"mathematics", "history", "science", "literature"}
            assert domains.intersection(expected_domains)

        def test_mmlu_metadata_structure(self, benchmarks):
            """Test MMLU metadata structure"""
            samples = benchmarks.get_mmlu_samples(num_samples=1)
            sample = samples[0]

            assert "subject" in sample.metadata
            assert "difficulty" in sample.metadata
            assert sample.metadata["difficulty"] == "medium"

    class TestGSM8KSamples:
        """Test GSM8K sample generation"""

        def test_gsm8k_sample_structure(self, benchmarks):
            """Test GSM8K sample structure"""
            samples = benchmarks.get_gsm8k_samples(num_samples=2)

            assert len(samples) == 2
            for sample in samples:
                assert isinstance(sample, ExternalBenchmarkTask)
                assert sample.task_id.startswith("gsm8k_")
                assert sample.benchmark_name == "GSM8K"
                assert sample.domain == "mathematics"
                assert len(sample.choices) == 0  # Open-ended
                assert isinstance(sample.correct_answer, str)
                assert isinstance(sample.metadata, dict)

        def test_gsm8k_math_problems(self, benchmarks):
            """Test GSM8K math problem content"""
            samples = benchmarks.get_gmm8k_samples(num_samples=1)
            sample = samples[0]

            # Should be a word problem
            assert "ducks" in sample.question
            assert "eggs" in sample.question
            assert "sell" in sample.question
            # Check for mathematical calculation
            assert any(op in sample.question for op in ["+", "-", "×", "*"])

        def test_gsm8k_metadata_grade_level(self, benchmarks):
            """Test GSM8K grade level metadata"""
            samples = benchmarks.get_gmm8k_samples(num_samples=1)
            sample = samples[0]

            assert sample.metadata["grade_level"] == "8th"
            assert "operation" in sample.metadata
            assert "steps" in sample.metadata

    class TestARCSamples:
        """Test ARC sample generation"""

        def test_arc_sample_structure(self, benchmarks):
            """Test ARC sample structure"""
            samples = benchmarks.get_arc_samples(num_samples=2)

            assert len(samples) == 2
            for sample in samples:
                assert isinstance(sample, ExternalBenchmarkTask)
                assert sample.task_id.startswith("arc_")
                assert sample.benchmark_name == "ARC"
                assert sample.domain == "science_reasoning"
                assert len(sample.choices) == 4
                assert sample.correct_answer in sample.choices
                assert isinstance(sample.metadata, dict)

        def test_arc_science_content(self, benchmarks):
            """Test ARC science question content"""
            samples = benchmarks.get_arc_samples(num_samples=1)
            sample = samples[0]

            # Should be about properties of materials
            assert "metals" in sample.question
            assert "property" in sample.question
            # Multiple choice science question
            assert len(sample.choices) == 4
            assert "high melting points" in sample.correct_answer

        def test_arc_metadata_subject(self, benchmarks):
            """Test ARC subject metadata"""
            samples = benchmarks.get_arc_samples(num_samples=1)
            sample = samples[0]

            assert sample.metadata["subject"] == "chemistry"
            assert sample.metadata["difficulty"] == "medium"

    class TestBigBenchHardSamples:
        """Test Big-Bench Hard sample generation"""

        def test_bbh_sample_structure(self, benchmarks):
            """Test Big-Bench Hard sample structure"""
            samples = benchmarks.get_bbh_samples(num_samples=2)

            assert len(samples) == 2
            for sample in samples:
                assert isinstance(sample, ExternalBenchmarkTask)
                assert sample.task_id.startswith("bbh_")
                assert sample.benchmark_name == "BigBench_Hard"
                assert len(sample.choices) == 4
                assert sample.correct_answer == "Yes"
                assert isinstance(sample.metadata, dict)

        def test_bbh_logical_reasoning(self, benchmarks):
            """Test Big-Bench Hard logical reasoning"""
            samples = benchmarks.get_bbh_samples(num_samples=1)
            sample = samples[0]

            # Should be logical reasoning
            assert "Bloops" in sample.question
            assert "Razzies" in sample.question
            assert "Lazzies" in sample.question
            assert sample.correct_answer == "Yes"

        def test_bbh_task_type_metadata(self, benchmarks):
            """Test Big-Bench Hard task type metadata"""
            samples = benchmarks.get_bbh_samples(num_samples=1)
            sample = samples[0]

            assert "task_type" in sample.metadata
            assert sample.metadata["task_type"] in [
                "logical_deduction",
                "word_sorting",
                "formal_fallacies",
            ]

    class TestTaskEvaluation:
        """Test individual task evaluation"""

        @pytest.mark.asyncio
        async def test_evaluate_task_structure(self, benchmarks):
            """Test task evaluation returns proper structure"""
            task = ExternalBenchmarkTask(
                task_id="test_001",
                benchmark_name="Test",
                domain="test",
                question="What is 2+2?",
                choices=["3", "4", "5"],
                correct_answer="4",
            )

            # Mock the LLM bridge
            with patch.object(benchmarks.llm_bridge, "generate") as mock_generate:
                mock_generate.return_value = "The answer is 4"

                result = await benchmarks.evaluate_task(task, using_conjecture=False)

                assert isinstance(result, BenchmarkResult)
                assert result.task_id == "test_001"
                assert result.benchmark_name == "Test"
                assert result.using_conjecture is False
                assert result.response is not None
                assert isinstance(result.correct_answer, str)
                assert isinstance(result.is_correct, bool)
                assert isinstance(result.confidence, float)
                assert result.response_time > 0

        @pytest.mark.asyncio
        async def test_evaluate_task_answer_extraction(self, benchmarks):
            """Test answer extraction from responses"""
            task = ExternalBenchmarkTask(
                task_id="test_002",
                benchmark_name="Test",
                domain="test",
                question="What is 2+2?",
                choices=["3", "4", "5"],
                correct_answer="4",
            )

            # Test multiple choice answer extraction
            test_cases = [
                ("The answer is B. 4", "4"),
                ("I choose B: 4", "4"),
                ("Answer: 4", "4"),
                ("4", "4"),
                ("The correct answer is the second option, which is 4", "4"),
            ]

            for response_text in test_cases:
                with patch.object(benchmarks.llm_bridge, "generate") as mock_generate:
                    mock_generate.return_value = response_text

                    result = await benchmarks.evaluate_task(
                        task, using_conjecture=False
                    )
                    assert result.predicted_answer == "4"

        @pytest.mark.asyncio
        async def test_evaluate_task_confidence_calculation(self, benchmarks):
            """Test confidence calculation"""
            task = ExternalBenchmarkTask(
                task_id="test_003",
                benchmark_name="Test",
                domain="test",
                question="What is 2+2?",
                choices=["3", "4", "5"],
                correct_answer="4",
            )

            confidence_indicators = [
                ("definitely correct", 0.9),
                ("probably correct", 0.7),
                ("maybe correct", 0.5),
                ("uncertain", 0.3),
            ]

            for indicator, expected_confidence in confidence_indicators:
                with patch.object(benchmarks.llm_bridge, "generate") as mock_generate:
                    mock_generate.return_value = f"I think the answer is {indicator}: 4"

                    result = await benchmarks.evaluate_task(
                        task, using_conjecture=False
                    )
                    assert abs(result.confidence - expected_confidence) < 0.1

        @pytest.mark.asyncio
        async def test_evaluate_task_correctness_evaluation(self, benchmarks):
            """Test correctness evaluation"""
            task = ExternalBenchmarkTask(
                task_id="test_004",
                benchmark_name="Test",
                domain="test",
                question="What is 2+2?",
                choices=["3", "4", "5"],
                correct_answer="4",
            )

            test_cases = [
                ("4", True),  # Exact match
                ("The answer is 4", True),  # Contains match
                ("5", False),  # Wrong answer
                ("Three", False),  # Wrong answer
            ]

            for response_text, expected_correct in test_cases:
                with patch.object(benchmarks.llm_bridge, "generate") as mock_generate:
                    mock_generate.return_value = response_text

                    result = await benchmarks.evaluate_task(
                        task, using_conjecture=False
                    )
                    assert result.is_correct == expected_correct

        @pytest.mark.asyncio
        async def test_evaluate_task_error_handling(self, benchmarks):
            """Test error handling in task evaluation"""
            task = ExternalBenchmarkTask(
                task_id="test_005",
                benchmark_name="Test",
                domain="test",
                question="What is 2+2?",
                choices=["3", "4", "5"],
                correct_answer="4",
            )

            # Mock LLM bridge error
            with patch.object(benchmarks.llm_bridge, "generate") as mock_generate:
                mock_generate.side_effect = Exception("LLM Error")

                result = await benchmarks.evaluate_task(task, using_conjecture=False)

                assert result.task_id == "test_005"
                assert result.is_correct is False
                assert result.confidence == 0.0
                assert "Error" in result.response
                assert "error" in result.metadata

    class TestBenchmarkSuite:
        """Test complete benchmark suite execution"""

        @pytest.mark.asyncio
        async def test_run_all_benchmarks_structure(self, benchmarks):
            """Test all benchmarks execution structure"""
            # Mock LLM responses
            with patch.object(benchmarks.llm_bridge, "generate") as mock_generate:
                mock_generate.return_value = "Sample response"

                with patch.object(
                    benchmarks.prompt_system, "process_with_context"
                ) as mock_conjecture:
                    mock_conjecture.return_value = Mock(
                        response="Enhanced response", confidence=0.8
                    )

                summary = await benchmarks.run_benchmark_suite("all", num_samples=1)

                # Check summary structure
                assert isinstance(summary, dict)
                assert "benchmark_name" in summary
                assert "timestamp" in summary
                assert "total_tasks" in summary
                assert "baseline_accuracy" in summary
                assert "conjecture_accuracy" in summary
                assert "improvement" in summary
                assert "improvement_percentage" in summary
                assert "results_by_benchmark" in summary
                assert "baseline_results" in summary
                assert "conjecture_results" in summary
                assert "success" in summary

        @pytest.mark.asyncio
        async def test_run_single_benchmark(self, benchmarks):
            """Test running single benchmark"""
            # Mock LLM responses
            with patch.object(benchmarks.llm_bridge, "generate") as mock_generate:
                mock_generate.return_value = "Sample response"

                summary = await benchmarks.run_benchmark_suite(
                    "hellaswag", num_samples=1
                )

                assert summary["benchmark_name"] == "hellaswag"
                assert summary["total_tasks"] == 1
                assert len(summary["baseline_results"]) == 1
                assert len(summary["conjecture_results"]) == 1

        @pytest.mark.asyncio
        async def test_benchmark_calculation_accuracy(self, benchmarks):
            """Test benchmark calculation accuracy"""
            # Mock different accuracy levels
            with patch.object(benchmarks.llm_bridge, "generate") as mock_generate:
                # Return correct answer for baseline, incorrect for conjecture
                mock_generate.return_value = "4"

                with patch.object(
                    benchmarks.prompt_system, "process_with_context"
                ) as mock_conjecture:
                    mock_conjecture.return_value = Mock(response="5", confidence=0.6)

                    summary = await benchmarks.run_benchmark_suite(
                        "mmlu", num_samples=2
                    )

                    # Check calculations
                    assert summary["baseline_accuracy"] == 0.5  # 1/2 correct
                    assert summary["conjecture_accuracy"] == 0.0  # 0/2 correct
                    assert summary["improvement"] == -0.5
                    # Improvement percentage should handle division by zero
                    assert isinstance(summary["improvement_percentage"], float)

        @pytest.mark.asyncio
        async def test_benchmark_results_by_type(self, benchmarks):
            """Test benchmark results grouped by benchmark type"""
            with patch.object(benchmarks.llm_bridge, "generate") as mock_generate:
                mock_generate.return_value = "Sample response"

                summary = await benchmarks.run_benchmark_suite("all", num_samples=1)

                results_by_benchmark = summary["results_by_benchmark"]

                # Should have results for all 5 benchmark types
                expected_benchmarks = [
                    "HellaSwag",
                    "MMLU",
                    "GSM8K",
                    "ARC",
                    "BigBench_Hard",
                ]
                for benchmark in expected_benchmarks:
                    assert benchmark in results_by_benchmark
                    assert "baseline" in results_by_benchmark[benchmark]
                    assert "conjecture" in results_by_benchmark[benchmark]
                    assert len(results_by_benchmark[benchmark]["baseline"]) == 1
                    assert len(results_by_benchmark[benchmark]["conjecture"]) == 1

    class TestIntegration:
        """Test integration with other systems"""

        @pytest.mark.asyncio
        async def test_config_integration(self, benchmarks):
            """Test configuration integration"""
            # Should initialize without errors
            assert benchmarks.config is not None
            assert benchmarks.llm_bridge is not None
            assert benchmarks.prompt_system is not None

        @pytest.mark.asyncio
        async def test_prompt_system_integration(self, benchmarks):
            """Test prompt system integration for conjecture evaluation"""
            task = ExternalBenchmarkTask(
                task_id="integration_test",
                benchmark_name="Test",
                domain="mathematical",
                question="What is 2+2?",
                choices=["3", "4", "5"],
                correct_answer="4",
            )

            # Mock enhanced prompt system response
            with patch.object(
                benchmarks.prompt_system, "process_with_context"
            ) as mock_conjecture:
                mock_response = Mock()
                mock_response.response = "4"
                mock_response.confidence = 0.85
                mock_response.metadata = {
                    "problem_type": "mathematical",
                    "enhancements_applied": 2,
                    "enhancement_types": [
                        "mathematical_reasoning",
                        "self_verification",
                    ],
                }
                mock_conjecture.return_value = mock_response

                result = await benchmarks.evaluate_task(task, using_conjecture=True)

                assert result.using_conjecture is True
                assert result.predicted_answer == "4"
                assert result.confidence == 0.85  # Should use enhanced confidence


class TestExternalBenchmarksPerformance:
    """Test external benchmarks performance"""

    @pytest.mark.asyncio
    async def test_suite_execution_performance(self, benchmarks):
        """Test benchmark suite execution performance"""
        import time

        # Mock fast responses
        with patch.object(benchmarks.llm_bridge, "generate") as mock_generate:
            mock_generate.return_value = "Fast response"

            with patch.object(
                benchmarks.prompt_system, "process_with_context"
            ) as mock_conjecture:
                mock_conjecture.return_value = Mock(
                    response="Fast enhanced response", confidence=0.8
                )

                start_time = time.time()
                summary = await benchmarks.run_benchmark_suite("all", num_samples=1)
                end_time = time.time()

                # Should complete reasonably quickly even with mocking
                assert end_time - start_time < 10.0, (
                    f"Suite too slow: {end_time - start_time}s"
                )
                assert summary["total_tasks"] == 5  # 5 benchmarks × 1 sample each

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, benchmarks):
        """Test memory efficiency during benchmark execution"""
        import gc
        import sys

        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Run benchmarks
        with patch.object(benchmarks.llm_bridge, "generate") as mock_generate:
            mock_generate.return_value = "Memory test response"

            summary = await benchmarks.run_benchmark_suite("gsm8k", num_samples=2)

        # Check memory usage
        gc.collect()
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects

        # Should not create excessive objects during execution
        assert object_increase < 2000, f"Too many objects created: {object_increase}"
        assert summary is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
