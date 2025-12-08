"""
Comprehensive tests for DeepEval integration with Conjecture
Tests the enhanced benchmarking system with DeepEval metrics
"""

import asyncio
import pytest
import sys
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarking.deepeval_integration import (
    ConjectureModelWrapper,
    DeepEvalBenchmarkRunner,
    AdvancedBenchmarkEvaluator
)


class TestConjectureModelWrapper:
    """Test the Conjecture model wrapper for DeepEval integration"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            "providers": [
                {
                    "name": "test-provider",
                    "url": "http://test.com",
                    "api": "test-key",
                    "model": "test-model"
                }
            ]
        }

    @pytest.fixture
    def model_wrapper(self, mock_config):
        """Create model wrapper instance for testing"""
        return ConjectureModelWrapper("test-model", use_conjecture=False)

    def test_model_wrapper_initialization(self, model_wrapper):
        """Test model wrapper initialization"""
        assert model_wrapper.model_name == "test-model"
        assert model_wrapper.use_conjecture is False
        assert model_wrapper.get_model_name() == "Conjecture-test-model"

    def test_model_wrapper_with_conjecture(self, mock_config):
        """Test model wrapper with Conjecture enhancement"""
        wrapper = ConjectureModelWrapper("test-model", use_conjecture=True)
        assert wrapper.use_conjecture is True
        assert "Conjecture" in wrapper.get_model_name()

    @pytest.mark.asyncio
    async def test_a_generate_success(self, model_wrapper):
        """Test successful async generation"""
        # Mock the wrapper's a_generate method
        with patch.object(model_wrapper, '_wrapper') as mock_wrapper:
            mock_wrapper.a_generate = AsyncMock(return_value="Test response")
            
            response = await model_wrapper.a_generate("Test prompt")
            assert response == "Test response"
            mock_wrapper.a_generate.assert_called_once_with("Test prompt")

    @pytest.mark.asyncio
    async def test_a_generate_fallback(self, model_wrapper):
        """Test fallback response on generation error"""
        # Remove wrapper to force fallback
        if hasattr(model_wrapper, '_wrapper'):
            delattr(model_wrapper, '_wrapper')
        
        # Mock legacy integration
        with patch.object(model_wrapper, '_get_model_integration') as mock_get_integration:
            mock_integration = AsyncMock()
            mock_integration.get_model_response = AsyncMock(return_value="Legacy response")
            mock_get_integration.return_value = mock_integration
            
            response = await model_wrapper.a_generate("Test prompt")
            assert response == "Legacy response"

    @pytest.mark.asyncio
    async def test_a_generate_error_handling(self, model_wrapper):
        """Test error handling in generation"""
        # Mock wrapper to raise exception
        with patch.object(model_wrapper, '_wrapper') as mock_wrapper:
            mock_wrapper.a_generate = AsyncMock(side_effect=Exception("Test error"))
            
            response = await model_wrapper.a_generate("Test prompt")
            assert "Evaluation fallback response" in response
            assert "test-model" in response


class TestDeepEvalBenchmarkRunner:
    """Test the DeepEval benchmark runner"""

    @pytest.fixture
    def benchmark_runner(self):
        """Create benchmark runner instance"""
        return DeepEvalBenchmarkRunner()

    @pytest.fixture
    def sample_tasks(self):
        """Sample tasks for testing"""
        return [
            {
                "prompt": "What is 2+2?",
                "expected_answer": "4",
                "metadata": {"category": "mathematics"}
            },
            {
                "prompt": "Write a Python function to sort a list",
                "expected_answer": "def sort_list(lst): return sorted(lst)",
                "metadata": {"category": "algorithms"}
            }
        ]

    def test_benchmark_runner_initialization(self, benchmark_runner):
        """Test benchmark runner initialization"""
        assert len(benchmark_runner.metrics) == 6
        assert "relevancy" in benchmark_runner.metrics
        assert "faithfulness" in benchmark_runner.metrics
        assert "exact_match" in benchmark_runner.metrics
        assert "summarization" in benchmark_runner.metrics
        assert "bias" in benchmark_runner.metrics
        assert "toxicity" in benchmark_runner.metrics

    @pytest.mark.asyncio
    async def test_evaluate_with_deepeval_success(self, benchmark_runner, sample_tasks):
        """Test successful DeepEval evaluation"""
        # Mock the evaluate function
        with patch('src.benchmarking.deepeval_integration.evaluate') as mock_evaluate:
            # Mock evaluation result
            mock_result = Mock()
            mock_result.score = 0.8
            mock_result.reason = "Good response"
            mock_evaluate.return_value = [mock_result]
            
            result = await benchmark_runner.evaluate_with_deepeval(
                sample_tasks, "test-model", use_conjecture=False
            )
            
            assert "relevancy" in result
            assert "faithfulness" in result
            assert result["relevancy"]["score"] == 0.8
            assert result["relevancy"]["success"] is True

    @pytest.mark.asyncio
    async def test_evaluate_with_deepeval_no_tasks(self, benchmark_runner):
        """Test evaluation with no tasks"""
        result = await benchmark_runner.evaluate_with_deepeval([], "test-model")
        assert result == {}

    @pytest.mark.asyncio
    async def test_evaluate_with_deepeval_error(self, benchmark_runner, sample_tasks):
        """Test error handling in DeepEval evaluation"""
        # Mock the evaluate function to raise exception
        with patch('src.benchmarking.deepeval_integration.evaluate') as mock_evaluate:
            mock_evaluate.side_effect = Exception("Evaluation error")
            
            result = await benchmark_runner.evaluate_with_deepeval(
                sample_tasks, "test-model", use_conjecture=False
            )
            
            # Should return error results for each metric
            assert "relevancy" in result
            assert result["relevancy"]["success"] is False
            assert "Evaluation error" in result["relevancy"]["reason"]


class TestAdvancedBenchmarkEvaluator:
    """Test the advanced benchmark evaluator"""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance"""
        return AdvancedBenchmarkEvaluator()

    @pytest.fixture
    def sample_task(self):
        """Sample task for testing"""
        return {
            "prompt": "What is the capital of France?",
            "expected_answer": "Paris",
            "metadata": {"category": "general"}
        }

    @pytest.fixture
    def sample_response(self):
        """Sample model response"""
        return "The capital of France is Paris, which is known for the Eiffel Tower."

    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization"""
        assert evaluator is not None
        # Should have either new framework or legacy runner
        assert hasattr(evaluator, 'use_new_framework')
        if evaluator.use_new_framework:
            assert hasattr(evaluator, 'evaluation_framework')
        else:
            assert hasattr(evaluator, 'deepeval_runner')

    def test_custom_evaluation_mathematics(self, evaluator):
        """Test custom evaluation for mathematics tasks"""
        task = {
            "metadata": {"category": "mathematics"},
            "expected_answer": "proof by contradiction"
        }
        response = "Let's assume the opposite and reach a contradiction. Therefore, the statement is proven by contradiction."
        
        score = evaluator._custom_evaluation(task, response)
        assert score > 0.5  # Should score well for mathematics

    def test_custom_evaluation_algorithms(self, evaluator):
        """Test custom evaluation for algorithms tasks"""
        task = {
            "metadata": {"category": "algorithms"}
        }
        response = "def quick_sort(arr): if len(arr) <= 1: return arr else: pivot = arr[0]; left = [x for x in arr[1:] if x <= pivot]; right = [x for x in arr[1:] if x > pivot]; return quick_sort(left) + [pivot] + quick_sort(right)"
        
        score = evaluator._custom_evaluation(task, response)
        assert score > 0.5  # Should score well for algorithms

    def test_custom_evaluation_debugging(self, evaluator):
        """Test custom evaluation for debugging tasks"""
        task = {
            "metadata": {"category": "debugging"}
        }
        response = "The error occurs because of a null pointer exception. The fix is to add proper null checking before accessing the object."
        
        score = evaluator._custom_evaluation(task, response)
        assert score > 0.5  # Should score well for debugging

    def test_custom_evaluation_empty_response(self, evaluator, sample_task):
        """Test custom evaluation with empty response"""
        score = evaluator._custom_evaluation(sample_task, "")
        assert score == 0.0

    def test_custom_evaluation_short_response(self, evaluator, sample_task):
        """Test custom evaluation with very short response"""
        score = evaluator._custom_evaluation(sample_task, "Hi")
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_evaluate_benchmark_response_success(self, evaluator, sample_task, sample_response):
        """Test successful benchmark response evaluation"""
        # Mock the evaluation framework if available
        if evaluator.use_new_framework and hasattr(evaluator, 'evaluation_framework'):
            mock_result = {
                "success": True,
                "metrics_results": {"relevancy": {"score": 0.8}, "faithfulness": {"score": 0.9}}
            }
            evaluator.evaluation_framework.evaluate_provider = AsyncMock(return_value=mock_result)
        
        result = await evaluator.evaluate_benchmark_response(
            sample_task, sample_response, "test-model", use_conjecture=False
        )
        
        assert "overall_score" in result
        assert "custom_score" in result
        assert "success" in result
        assert result["overall_score"] > 0

    @pytest.mark.asyncio
    async def test_evaluate_benchmark_response_error(self, evaluator, sample_task):
        """Test error handling in benchmark response evaluation"""
        # Mock to raise exception
        if evaluator.use_new_framework and hasattr(evaluator, 'evaluation_framework'):
            evaluator.evaluation_framework.evaluate_provider = AsyncMock(side_effect=Exception("Test error"))
        
        result = await evaluator.evaluate_benchmark_response(
            sample_task, "response", "test-model", use_conjecture=False
        )
        
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_benchmark_models_comparison(self, evaluator):
        """Test benchmark models comparison"""
        sample_tasks = [
            {
                "prompt": "Test task 1",
                "expected_answer": "Answer 1",
                "metadata": {"category": "general"}
            },
            {
                "prompt": "Test task 2", 
                "expected_answer": "Answer 2",
                "metadata": {"category": "mathematics"}
            }
        ]
        models = ["model1", "model2"]
        
        # Mock the evaluation framework if available
        if evaluator.use_new_framework and hasattr(evaluator, 'evaluation_framework'):
            mock_comparison_result = {
                "providers": {
                    "model1_direct": {"overall_score": 0.7, "success": True},
                    "model1_conjecture": {"overall_score": 0.8, "success": True},
                    "model2_direct": {"overall_score": 0.6, "success": True},
                    "model2_conjecture": {"overall_score": 0.75, "success": True}
                }
            }
            evaluator.evaluation_framework.evaluate_multiple_providers = AsyncMock(return_value=mock_comparison_result)
        
        result = await evaluator.benchmark_models_comparison(sample_tasks, models)
        
        assert "model1" in result
        assert "model2" in result
        assert result["model1"]["direct_avg"] > 0
        assert result["model1"]["conjecture_avg"] > 0
        assert "improvement" in result["model1"]

    def test_generate_advanced_chart(self, evaluator):
        """Test advanced chart generation"""
        results = {
            "model1": {
                "direct_avg": 0.7,
                "conjecture_avg": 0.8,
                "improvement": 14.3,
                "all_results": {
                    "conjecture": [{
                        "deepeval_metrics": {
                            "relevancy": {"score": 0.8},
                            "faithfulness": {"score": 0.9}
                        }
                    }]
                }
            },
            "model2": {
                "direct_avg": 0.6,
                "conjecture_avg": 0.75,
                "improvement": 25.0,
                "all_results": {
                    "conjecture": [{
                        "deepeval_metrics": {
                            "relevancy": {"score": 0.7},
                            "faithfulness": {"score": 0.8}
                        }
                    }]
                }
            }
        }
        
        chart = evaluator.generate_advanced_chart(results)
        
        assert "DEEPVAL BENCHMARK RESULTS CHART" in chart
        assert "model1" in chart
        assert "model2" in chart
        assert "DEEPVAL METRICS BREAKDOWN" in chart
        assert "relevancy" in chart
        assert "faithfulness" in chart


class TestIntegration:
    """Integration tests for the complete DeepEval system"""

    @pytest.mark.asyncio
    async def test_full_evaluation_pipeline(self):
        """Test the full evaluation pipeline with mocked components"""
        # Create evaluator
        evaluator = AdvancedBenchmarkEvaluator()
        
        # Sample data
        tasks = [
            {
                "prompt": "Solve: √16 = ?",
                "expected_answer": "4",
                "metadata": {"category": "mathematics"}
            }
        ]
        
        models = ["test-model"]
        
        # Mock the framework if available
        if evaluator.use_new_framework and hasattr(evaluator, 'evaluation_framework'):
            # Mock comprehensive evaluation result
            mock_result = {
                "success": True,
                "metrics_results": {
                    "relevancy": {"score": 0.9},
                    "faithfulness": {"score": 0.85},
                    "exact_match": {"score": 1.0}
                }
            }
            evaluator.evaluation_framework.evaluate_provider = AsyncMock(return_value=mock_result)
            evaluator.evaluation_framework.evaluate_multiple_providers = AsyncMock(return_value={
                "providers": {
                    "test-model_direct": {"overall_score": 0.8, "success": True},
                    "test-model_conjecture": {"overall_score": 0.9, "success": True}
                }
            })
        
        # Test single evaluation
        result = await evaluator.evaluate_benchmark_response(
            tasks[0], "The square root of 16 is 4.", "test-model", use_conjecture=False
        )
        assert result["success"] is True
        assert result["overall_score"] > 0
        
        # Test comparison
        comparison = await evaluator.benchmark_models_comparison(tasks, models)
        assert "test-model" in comparison
        assert comparison["test-model"]["improvement"] >= 0
        
        # Test chart generation
        chart = evaluator.generate_advanced_chart(comparison)
        assert "test-model" in chart


class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.mark.asyncio
    async def test_missing_expected_answer(self):
        """Test evaluation with missing expected answer"""
        evaluator = AdvancedBenchmarkEvaluator()
        task = {"prompt": "Test question", "metadata": {"category": "general"}}
        
        result = await evaluator.evaluate_benchmark_response(
            task, "Test response", "test-model", use_conjecture=False
        )
        
        # Should still work but with lower score
        assert "overall_score" in result

    @pytest.mark.asyncio
    async def test_invalid_model_name(self):
        """Test evaluation with invalid model name"""
        evaluator = AdvancedBenchmarkEvaluator()
        task = {"prompt": "Test", "expected_answer": "Answer"}
        
        # Should handle gracefully
        result = await evaluator.evaluate_benchmark_response(
            task, "Response", "invalid-model-name", use_conjecture=False
        )
        
        assert "overall_score" in result

    def test_custom_evaluation_unknown_category(self):
        """Test custom evaluation with unknown category"""
        evaluator = AdvancedBenchmarkEvaluator()
        task = {"metadata": {"category": "unknown"}}
        response = "Some response text that should be evaluated."
        
        score = evaluator._custom_evaluation(task, response)
        assert 0 <= score <= 1.0


# Performance and stress tests
class TestPerformance:
    """Test performance and scalability"""

    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self):
        """Test concurrent evaluation of multiple tasks"""
        evaluator = AdvancedBenchmarkEvaluator()
        
        # Create multiple tasks
        tasks = [
            {"prompt": f"Test question {i}", "expected_answer": f"Answer {i}"}
            for i in range(5)
        ]
        
        # Mock framework if available
        if evaluator.use_new_framework and hasattr(evaluator, 'evaluation_framework'):
            mock_result = {"success": True, "metrics_results": {"relevancy": {"score": 0.8}}}
            evaluator.evaluation_framework.evaluate_provider = AsyncMock(return_value=mock_result)
        
        # Run evaluations concurrently
        tasks_to_run = [
            evaluator.evaluate_benchmark_response(task, f"Response {i}", "test-model", False)
            for i, task in enumerate(tasks)
        ]
        
        results = await asyncio.gather(*tasks_to_run)
        
        # All should succeed
        assert len(results) == len(tasks)
        assert all(result["success"] for result in results)

    @pytest.mark.asyncio
    async def test_large_dataset_evaluation(self):
        """Test evaluation with larger dataset"""
        evaluator = AdvancedBenchmarkEvaluator()
        
        # Create larger dataset
        tasks = [
            {"prompt": f"Question {i}", "expected_answer": f"Answer {i}"}
            for i in range(10)
        ]
        
        models = ["model1", "model2"]
        
        # Mock framework if available
        if evaluator.use_new_framework and hasattr(evaluator, 'evaluation_framework'):
            mock_results = {}
            for model in models:
                for suffix in ["direct", "conjecture"]:
                    key = f"{model}_{suffix}"
                    mock_results[key] = {"overall_score": 0.7 + (0.1 if suffix == "conjecture" else 0), "success": True}
            
            evaluator.evaluation_framework.evaluate_multiple_providers = AsyncMock(return_value={
                "providers": mock_results
            })
        
        result = await evaluator.benchmark_models_comparison(tasks, models)
        
        assert len(result) == len(models)
        for model in models:
            assert model in result
            assert result[model]["tasks_evaluated"] == len(tasks)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])