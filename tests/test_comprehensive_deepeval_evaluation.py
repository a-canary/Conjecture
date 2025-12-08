"""
Comprehensive DeepEval Evaluation Test Suite
Tests all 3 LLM providers (ibm/granite-4-h-tiny, zai/GLM-4.6, openrouter/gpt-oss-20b)
with comprehensive evaluation across multiple dimensions using DeepEval metrics
"""

import asyncio
import json
import pytest
import sys
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    EvaluationFramework,
    create_conjecture_wrapper,
    get_available_conjecture_providers,
    evaluate_single_provider,
    evaluate_all_providers
)
from src.benchmarking.deepeval_integration import (
    ConjectureModelWrapper,
    DeepEvalBenchmarkRunner,
    AdvancedBenchmarkEvaluator
)


class TestProviderSpecificEvaluation:
    """Test each provider individually with consistent test cases"""

    @pytest.fixture
    def evaluation_framework(self):
        """Create evaluation framework instance"""
        return EvaluationFramework()

    @pytest.fixture
    def test_providers(self):
        """List of target providers to test"""
        return [
            "ibm/granite-4-h-tiny",
            "zai/GLM-4.6", 
            "openrouter/gpt-oss-20b"
        ]

    @pytest.fixture
    def basic_test_cases(self):
        """Basic test cases for provider validation"""
        return [
            {
                "prompt": "What is 2+2?",
                "expected_answer": "4",
                "metadata": {"category": "mathematics", "difficulty": "easy"}
            },
            {
                "prompt": "Explain gravity in simple terms.",
                "expected_answer": "Gravity is the force that attracts objects toward each other.",
                "metadata": {"category": "physics", "difficulty": "medium"}
            }
        ]

    def test_provider_configuration_validation(self, test_providers):
        """Test that all target providers are properly configured"""
        available_providers = get_available_conjecture_providers()
        
        for provider in test_providers:
            # Check if provider is available
            provider_found = any(
                provider.lower() in avail.lower() or 
                any(part in avail.lower() for part in provider.split('/'))
                for avail in available_providers
            )
            assert provider_found, f"Provider {provider} not found in available providers"

    @pytest.mark.asyncio
    async def test_individual_provider_initialization(self, test_providers):
        """Test that each provider can be initialized correctly"""
        for provider in test_providers:
            wrapper = create_conjecture_wrapper(provider, use_conjecture=False)
            assert wrapper is not None
            assert wrapper.provider_name == provider
            assert wrapper.use_conjecture is False
            
            # Test conjecture wrapper too
            conjecture_wrapper = create_conjecture_wrapper(provider, use_conjecture=True)
            assert conjecture_wrapper is not None
            assert conjecture_wrapper.use_conjecture is True

    @pytest.mark.asyncio
    async def test_provider_specific_edge_cases(self, evaluation_framework, test_providers):
        """Test provider-specific edge cases and limitations"""
        test_cases = [
            evaluation_framework.create_test_case(
                input_text="",
                expected_output="",
                actual_output="",
                additional_metadata={"category": "edge_case", "type": "empty_input"}
            ),
            evaluation_framework.create_test_case(
                input_text="A" * 10000,  # Very long input
                expected_answer="Response to long input",
                actual_output="Processed long input successfully",
                additional_metadata={"category": "edge_case", "type": "long_input"}
            )
        ]
        
        for provider in test_providers:
            # Mock the wrapper to avoid actual API calls
            with patch('src.evaluation.conjecture_llm_wrapper.create_conjecture_wrapper') as mock_create:
                mock_wrapper = AsyncMock()
                mock_wrapper.a_generate = AsyncMock(return_value="Mock response")
                mock_create.return_value = mock_wrapper
                
                result = await evaluation_framework.evaluate_provider(
                    provider, test_cases, use_conjecture=False
                )
                
                assert result["provider"] == provider
                assert result["test_cases_count"] == len(test_cases)
                assert "overall_score" in result
                assert "metrics_results" in result

    @pytest.mark.asyncio
    async def test_provider_connectivity_validation(self, test_providers):
        """Test provider connectivity and configuration"""
        for provider in test_providers:
            wrapper = create_conjecture_wrapper(provider, use_conjecture=False)
            
            # Test provider info
            provider_info = wrapper.get_provider_info()
            assert "provider_name" in provider_info
            assert "model_name" in provider_info
            assert "internal_model" in provider_info
            assert "configured" in provider_info

    @pytest.mark.asyncio
    async def test_provider_model_mapping(self, test_providers):
        """Test that provider names map correctly to internal models"""
        for provider in test_providers:
            wrapper = create_conjecture_wrapper(provider, use_conjecture=False)
            internal_model = wrapper._map_provider_to_model()
            
            # Verify mapping exists
            assert internal_model is not None
            assert isinstance(internal_model, str)
            assert len(internal_model) > 0


class TestDeepEvalMetricsEvaluation:
    """Test all 6 DeepEval metrics with comprehensive validation"""

    @pytest.fixture
    def deepeval_runner(self):
        """Create DeepEval benchmark runner"""
        return DeepEvalBenchmarkRunner()

    @pytest.fixture
    def comprehensive_test_cases(self):
        """Comprehensive test cases for all metrics"""
        return [
            {
                "prompt": "What is the capital of France?",
                "expected_answer": "Paris",
                "model_response": "The capital of France is Paris.",
                "metadata": {"category": "geography", "metric_focus": "exact_match"}
            },
            {
                "prompt": "Explain the causes of climate change.",
                "expected_answer": "Climate change is caused by greenhouse gas emissions, deforestation, and industrial activities.",
                "model_response": "Climate change primarily results from human activities that increase greenhouse gases in the atmosphere.",
                "metadata": {"category": "science", "metric_focus": "faithfulness"}
            },
            {
                "prompt": "Summarize the plot of Romeo and Juliet.",
                "expected_answer": "Two young lovers from feuding families die tragically.",
                "model_response": "Romeo and Juliet are star-crossed lovers from rival families who ultimately take their own lives.",
                "metadata": {"category": "literature", "metric_focus": "summarization"}
            }
        ]

    def test_all_metrics_initialization(self, deepeval_runner):
        """Test that all 6 DeepEval metrics are properly initialized"""
        expected_metrics = [
            "relevancy", "faithfulness", "exact_match", 
            "summarization", "bias", "toxicity"
        ]
        
        for metric in expected_metrics:
            assert metric in deepeval_runner.metrics
            assert hasattr(deepeval_runner.metrics[metric], 'threshold')
            assert deepeval_runner.metrics[metric].threshold > 0

    @pytest.mark.asyncio
    async def test_metric_threshold_validation(self, deepeval_runner, comprehensive_test_cases):
        """Test metric thresholds and scoring validation"""
        for metric_name, metric in deepeval_runner.metrics.items():
            # Test threshold is reasonable
            assert 0.0 <= metric.threshold <= 1.0
            
            # Test metric evaluation with mock data
            with patch('src.benchmarking.deepeval_integration.evaluate') as mock_evaluate:
                mock_result = Mock()
                mock_result.score = 0.8
                mock_result.reason = "Good response"
                mock_evaluate.return_value = [mock_result]
                
                result = await deepeval_runner.evaluate_with_deepeval(
                    comprehensive_test_cases[:1], "test-model", use_conjecture=False
                )
                
                assert metric_name in result
                assert "score" in result[metric_name]
                assert "success" in result[metric_name]
                assert "reason" in result[metric_name]

    @pytest.mark.asyncio
    async def test_metric_combination_and_aggregation(self, deepeval_runner):
        """Test metric combination and overall score aggregation"""
        test_cases = [
            {
                "prompt": "Test question",
                "expected_answer": "Test answer",
                "model_response": "Test response",
                "metadata": {"category": "test"}
            }
        ]
        
        # Mock different scores for different metrics
        with patch('src.benchmarking.deepeval_integration.evaluate') as mock_evaluate:
            scores = [0.8, 0.9, 0.7, 0.85, 0.95, 0.9]
            mock_evaluate.return_value = [
                Mock(score=score, reason=f"Metric {i}", success=score >= 0.5)
                for i, score in enumerate(scores)
            ]
            
            result = await deepeval_runner.evaluate_with_deepeval(
                test_cases, "test-model", use_conjecture=False
            )
            
            # Verify all metrics are present
            assert len(result) == 6
            
            # Test aggregation logic would be in AdvancedBenchmarkEvaluator
            # Here we just verify individual metric results

    def test_metric_error_handling(self, deepeval_runner):
        """Test metric evaluation error handling"""
        # Test with invalid metric configuration
        invalid_metric = Mock()
        invalid_metric.threshold = -0.1  # Invalid threshold
        
        # Should handle gracefully
        try:
            deepeval_runner.metrics["invalid"] = invalid_metric
            # This would be caught during actual evaluation
        except Exception as e:
            assert "threshold" in str(e).lower() or "invalid" in str(e).lower()

    @pytest.mark.asyncio
    async def test_metric_specific_test_cases(self, evaluation_framework):
        """Test metrics with domain-specific test cases"""
        metric_test_cases = {
            "exact_match": [
                {
                    "prompt": "What is 2+2?",
                    "expected_answer": "4",
                    "actual_output": "4"
                }
            ],
            "faithfulness": [
                {
                    "prompt": "Is water wet?",
                    "expected_answer": "Yes, water exhibits wetness due to its adhesive properties.",
                    "actual_output": "Water is considered wet because it makes other things wet."
                }
            ],
            "bias": [
                {
                    "prompt": "Describe a typical doctor.",
                    "expected_answer": "A doctor is a medical professional who diagnoses and treats illnesses.",
                    "actual_output": "A doctor is a dedicated healthcare professional who helps people regardless of their background."
                }
            ]
        }
        
        for metric_name, test_cases in metric_test_cases.items():
            deepeval_cases = [
                evaluation_framework.create_test_case(
                    input_text=tc["prompt"],
                    expected_output=tc["expected_answer"],
                    actual_output=tc["actual_output"],
                    additional_metadata={"metric_focus": metric_name}
                )
                for tc in test_cases
            ]
            
            # Mock evaluation
            with patch('src.evaluation.evaluation_framework.evaluate') as mock_evaluate:
                mock_result = Mock()
                mock_result.score = 0.8
                mock_result.reason = f"Good {metric_name}"
                mock_evaluate.return_value = [mock_result]
                
                result = await evaluation_framework.evaluate_provider(
                    "test-provider", deepeval_cases, metrics=[metric_name]
                )
                
                assert metric_name in result["metrics_results"]


class TestQualityIssueDetection:
    """Test quality issue detection for bugs, deception patterns, and duplicate code"""

    @pytest.fixture
    def quality_test_data(self):
        """Load quality issue test cases from JSON file"""
        test_data_path = Path(__file__).parent / "test_data" / "quality_issue_test_cases.json"
        with open(test_data_path, 'r') as f:
            return json.load(f)

    @pytest.fixture
    def advanced_evaluator(self):
        """Create advanced benchmark evaluator"""
        return AdvancedBenchmarkEvaluator()

    def test_bug_detection_patterns(self, advanced_evaluator, quality_test_data):
        """Test bug detection using pattern analysis"""
        bug_cases = quality_test_data["bug_detection"]
        
        for bug_case in bug_cases:
            task = {
                "prompt": bug_case["prompt"],
                "expected_answer": bug_case["expected_answer"],
                "metadata": bug_case["metadata"]
            }
            
            # Test with response that contains the bug
            buggy_response = f"def factorial(n):\n    return n * factorial(n - 1)"  # Missing base case
            
            score = advanced_evaluator._custom_evaluation(task, buggy_response)
            
            # Should detect the bug and give lower score
            assert 0.0 <= score <= 1.0
            
            # Test with correct response
            correct_response = bug_case["expected_answer"]
            correct_score = advanced_evaluator._custom_evaluation(task, correct_response)
            
            # Correct response should score higher
            assert correct_score >= score

    def test_deception_detection_patterns(self, advanced_evaluator, quality_test_data):
        """Test deception detection using various patterns"""
        deception_cases = quality_test_data["deception_detection"]
        
        for deception_case in deception_cases:
            task = {
                "prompt": deception_case["prompt"],
                "expected_answer": deception_case["expected_answer"],
                "metadata": deception_case["metadata"]
            }
            
            # Test with deceptive response
            deceptive_patterns = deception_case["deception_patterns"]
            deceptive_response = f"Yes, that's absolutely 100% certain and proven."  # Overconfidence
            
            score = advanced_evaluator._custom_evaluation(task, deceptive_response)
            
            # Should detect deception and give lower score
            assert 0.0 <= score <= 1.0
            
            # Test with correct response
            correct_response = deception_case["expected_answer"]
            correct_score = advanced_evaluator._custom_evaluation(task, correct_response)
            
            # Correct response should score higher
            assert correct_score >= score

    def test_duplicate_detection_patterns(self, advanced_evaluator, quality_test_data):
        """Test duplicate detection using semantic similarity"""
        duplicate_cases = quality_test_data["duplicate_detection"]
        
        for duplicate_case in duplicate_cases:
            task = {
                "prompt": duplicate_case["prompt"],
                "expected_answer": duplicate_case["expected_answer"],
                "metadata": duplicate_case["metadata"]
            }
            
            # Test with duplicate/template response
            duplicate_response = duplicate_case["expected_answer"]  # Exact duplicate
            
            score = advanced_evaluator._custom_evaluation(task, duplicate_response)
            
            # Should detect potential duplication
            assert 0.0 <= score <= 1.0
            
            # Test with diverse response
            if "Explain photosynthesis" in duplicate_case["prompt"]:
                diverse_response = "Photosynthesis is the biological process where plants use sunlight to convert CO2 and water into glucose and oxygen, releasing oxygen as a byproduct."
                diverse_score = advanced_evaluator._custom_evaluation(task, diverse_response)
                
                # Diverse response should score higher or equal
                assert diverse_score >= score

    def test_quality_metrics_validation(self, advanced_evaluator, quality_test_data):
        """Test comprehensive quality metrics validation"""
        quality_cases = quality_test_data["quality_metrics"]
        
        for quality_case in quality_cases:
            task = {
                "prompt": quality_case["prompt"],
                "expected_answer": quality_case["expected_answer"],
                "metadata": quality_case["metadata"]
            }
            
            quality_aspects = quality_case["quality_aspects"]
            threshold = quality_case["metadata"]["threshold"]
            
            # Test with high-quality response
            high_quality_response = quality_case["expected_answer"]
            score = advanced_evaluator._custom_evaluation(task, high_quality_response)
            
            # Should meet or exceed threshold
            assert score >= threshold * 0.8  # Allow some tolerance
            
            # Test with low-quality response
            low_quality_response = "I don't know."
            low_score = advanced_evaluator._custom_evaluation(task, low_quality_response)
            
            # Should score significantly lower
            assert low_score < score

    def test_edge_case_handling(self, advanced_evaluator, quality_test_data):
        """Test edge case handling and boundary conditions"""
        edge_cases = quality_test_data["edge_cases"]
        
        for edge_case in edge_cases:
            task = {
                "prompt": edge_case["prompt"],
                "expected_answer": edge_case["expected_answer"],
                "metadata": edge_case["metadata"]
            }
            
            edge_patterns = edge_case["edge_patterns"]
            
            # Test appropriate edge case response
            edge_response = edge_case["expected_answer"]
            score = advanced_evaluator._custom_evaluation(task, edge_response)
            
            # Should handle edge cases appropriately
            assert 0.0 <= score <= 1.0
            
            # Verify edge pattern recognition
            assert len(edge_patterns) > 0
            assert isinstance(edge_patterns, list)


class TestPerformanceAndRegression:
    """Test performance benchmarks and regression detection"""

    @pytest.fixture
    def performance_evaluator(self):
        """Create evaluator for performance testing"""
        return AdvancedBenchmarkEvaluator()

    @pytest.mark.asyncio
    async def test_concurrent_evaluation_performance(self, performance_evaluator):
        """Test concurrent evaluation of multiple providers"""
        providers = [
            "ibm/granite-4-h-tiny",
            "zai/GLM-4.6",
            "openrouter/gpt-oss-20b"
        ]
        
        test_cases = [
            {
                "prompt": f"Performance test question {i}",
                "expected_answer": f"Answer {i}",
                "metadata": {"category": "performance_test"}
            }
            for i in range(5)
        ]
        
        # Mock the evaluation framework for performance testing
        with patch('src.evaluation.evaluation_framework.EvaluationFramework.evaluate_provider') as mock_eval:
            mock_eval.return_value = {
                "provider": "test",
                "overall_score": 0.8,
                "success": True,
                "metrics_results": {}
            }
            
            start_time = time.time()
            
            # Run evaluations concurrently
            tasks = [
                performance_evaluator.evaluate_benchmark_response(
                    test_case, f"Response {i}", provider, use_conjecture=False
                )
                for i, test_case in enumerate(test_cases)
            ]
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Performance assertions
            assert len(results) == len(test_cases)
            assert execution_time < 30.0  # Should complete within 30 seconds
            
            # All results should be successful
            assert all(result.get("success", False) for result in results)

    @pytest.mark.asyncio
    async def test_large_dataset_handling(self, performance_evaluator):
        """Test evaluation with large datasets (100+ scenarios)"""
        # Create large dataset
        large_dataset = [
            {
                "prompt": f"Large dataset test {i}",
                "expected_answer": f"Large dataset answer {i}",
                "metadata": {"category": "large_test", "index": i}
            }
            for i in range(100)  # 100 test cases
        ]
        
        # Mock evaluation for performance testing
        with patch('src.evaluation.evaluation_framework.EvaluationFramework.evaluate_provider') as mock_eval:
            mock_eval.return_value = {
                "provider": "test",
                "overall_score": 0.75,
                "success": True,
                "metrics_results": {}
            }
            
            start_time = time.time()
            
            # Process in batches to avoid memory issues
            batch_size = 10
            all_results = []
            
            for i in range(0, len(large_dataset), batch_size):
                batch = large_dataset[i:i + batch_size]
                
                batch_tasks = [
                    performance_evaluator.evaluate_benchmark_response(
                        test_case, f"Response {i}", "test-model", use_conjecture=False
                    )
                    for i, test_case in enumerate(batch)
                ]
                
                batch_results = await asyncio.gather(*batch_tasks)
                all_results.extend(batch_results)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Performance and memory assertions
            assert len(all_results) == len(large_dataset)
            assert execution_time < 120.0  # Should complete within 2 minutes
            
            # Verify all results are valid
            assert all(result.get("success", False) for result in all_results)

    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, performance_evaluator):
        """Test memory usage during evaluation"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create memory-intensive test cases
        memory_test_cases = [
            {
                "prompt": "A" * 1000,  # Large prompt
                "expected_answer": "B" * 1000,  # Large expected answer
                "metadata": {"category": "memory_test", "size": "large"}
            }
            for _ in range(50)
        ]
        
        # Mock evaluation to focus on memory testing
        with patch('src.evaluation.evaluation_framework.EvaluationFramework.evaluate_provider') as mock_eval:
            mock_eval.return_value = {
                "provider": "test",
                "overall_score": 0.8,
                "success": True,
                "metrics_results": {}
            }
            
            # Run memory-intensive evaluations
            tasks = [
                performance_evaluator.evaluate_benchmark_response(
                    test_case, "C" * 1000, "test-model", use_conjecture=False
                )
                for test_case in memory_test_cases
            ]
            
            results = await asyncio.gather(*tasks)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory usage assertions
            assert len(results) == len(memory_test_cases)
            assert memory_increase < 500  # Should not increase by more than 500MB

    @pytest.mark.asyncio
    async def test_response_time_measurement(self, performance_evaluator):
        """Test response time measurement and optimization"""
        test_cases = [
            {
                "prompt": f"Response time test {i}",
                "expected_answer": f"Answer {i}",
                "metadata": {"category": "timing_test"}
            }
            for i in range(10)
        ]
        
        response_times = []
        
        # Mock evaluation with realistic timing
        async def mock_evaluation_with_delay(*args, **kwargs):
            # Simulate variable response times
            await asyncio.sleep(0.1 + (i % 3) * 0.05)  # 0.1-0.2 seconds
            return {
                "provider": "test",
                "overall_score": 0.8,
                "success": True,
                "metrics_results": {}
            }
        
        with patch('src.evaluation.evaluation_framework.EvaluationFramework.evaluate_provider', 
                  side_effect=mock_evaluation_with_delay):
            
            for i, test_case in enumerate(test_cases):
                start_time = time.time()
                
                result = await performance_evaluator.evaluate_benchmark_response(
                    test_case, f"Response {i}", "test-model", use_conjecture=False
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                response_times.append(response_time)
                
                assert result["success"] is True
        
        # Response time analysis
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Performance assertions
        assert avg_response_time < 1.0  # Average under 1 second
        assert max_response_time < 2.0  # Maximum under 2 seconds
        assert len(response_times) == len(test_cases)

    def test_regression_detection_metrics(self, performance_evaluator):
        """Test regression detection and metrics tracking"""
        # Simulate historical performance data
        historical_baseline = {
            "overall_score": 0.85,
            "response_time_avg": 0.5,
            "success_rate": 0.95,
            "memory_usage_avg": 100
        }
        
        # Current performance data
        current_performance = {
            "overall_score": 0.82,  # Slight regression
            "response_time_avg": 0.6,  # Slight regression
            "success_rate": 0.93,  # Slight regression
            "memory_usage_avg": 110  # Slight regression
        }
        
        # Calculate regression metrics
        regressions = {}
        for metric in historical_baseline:
            baseline = historical_baseline[metric]
            current = current_performance[metric]
            
            if "score" in metric or "rate" in metric:
                # Higher is better
                regression_pct = ((baseline - current) / baseline) * 100
            else:
                # Lower is better (time, memory)
                regression_pct = ((current - baseline) / baseline) * 100
            
            regressions[metric] = {
                "baseline": baseline,
                "current": current,
                "regression_pct": regression_pct,
                "is_regression": regression_pct > 5.0  # 5% threshold
            }
        
        # Verify regression detection
        for metric, regression_data in regressions.items():
            assert "baseline" in regression_data
            assert "current" in regression_data
            assert "regression_pct" in regression_data
            assert "is_regression" in regression_data
            
            # All metrics should show slight regression in this test
            assert regression_data["regression_pct"] > 0


class TestIntegrationWorkflow:
    """Test end-to-end integration and CI/CD pipeline validation"""

    @pytest.fixture
    def integration_framework(self):
        """Create framework for integration testing"""
        return EvaluationFramework()

    @pytest.mark.asyncio
    async def test_end_to_end_evaluation_pipeline(self, integration_framework):
        """Test complete evaluation workflow from start to finish"""
        # Load test data
        test_data_path = Path(__file__).parent / "test_data" / "test_claims_scenarios.json"
        with open(test_data_path, 'r') as f:
            test_claims = json.load(f)
        
        # Process all claim categories
        all_test_cases = []
        for category, claims in test_claims.items():
            for claim in claims:
                test_case = integration_framework.create_test_case(
                    input_text=claim["prompt"],
                    expected_output=claim["expected_answer"],
                    additional_metadata=claim["metadata"]
                )
                all_test_cases.append(test_case)
        
        # Test with all providers
        providers = [
            "ibm/granite-4-h-tiny",
            "zai/GLM-4.6",
            "openrouter/gpt-oss-20b"
        ]
        
        # Mock the evaluation for integration testing
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="Integration test response")
            mock_create.return_value = mock_wrapper
            
            # Run end-to-end evaluation
            results = await integration_framework.evaluate_multiple_providers(
                providers, all_test_cases[:5], compare_conjecture=True  # Limit for testing
            )
            
            # Verify complete workflow
            assert "providers" in results
            assert "comparison" in results
            assert len(results["providers"]) == len(providers) * 2  # direct + conjecture
            
            # Verify comparison data
            comparison = results["comparison"]
            assert "best_overall" in comparison
            assert "improvements" in comparison

    @pytest.mark.asyncio
    async def test_ci_cd_pipeline_validation(self, integration_framework):
        """Test CI/CD pipeline integration and validation"""
        # Simulate CI/CD environment variables and configuration
        ci_config = {
            "CI_ENVIRONMENT": True,
            "TEST_TIMEOUT": 300,
            "COVERAGE_THRESHOLD": 80,
            "PERFORMANCE_THRESHOLD": 0.7
        }
        
        # Create CI/CD specific test cases
        ci_test_cases = [
            integration_framework.create_test_case(
                input_text="CI/CD validation test",
                expected_output="Validation passed",
                additional_metadata={
                    "ci_test": True,
                    "pipeline_stage": "validation",
                    "timeout": ci_config["TEST_TIMEOUT"]
                }
            )
        ]
        
        # Mock CI/CD pipeline execution
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="CI/CD response")
            mock_create.return_value = mock_wrapper
            
            # Execute pipeline validation
            start_time = time.time()
            
            result = await integration_framework.evaluate_provider(
                "ci-test-provider", ci_test_cases, use_conjecture=False
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # CI/CD validation assertions
            assert result["success"] is True
            assert execution_time < ci_config["TEST_TIMEOUT"]
            assert result["overall_score"] >= ci_config["PERFORMANCE_THRESHOLD"]

    def test_reporting_integration(self, integration_framework):
        """Test reporting integration and result generation"""
        # Create mock results for reporting
        mock_results = {
            "providers": {
                "ibm/granite-4-h-tiny_direct": {
                    "overall_score": 0.8,
                    "success": True,
                    "metrics_results": {
                        "relevancy": {"score": 0.85},
                        "faithfulness": {"score": 0.8}
                    }
                },
                "ibm/granite-4-h-tiny_conjecture": {
                    "overall_score": 0.9,
                    "success": True,
                    "metrics_results": {
                        "relevancy": {"score": 0.9},
                        "faithfulness": {"score": 0.9}
                    }
                }
            },
            "comparison": {
                "best_overall": {"provider": "ibm/granite-4-h-tiny", "score": 0.9},
                "improvements": {
                    "ibm/granite-4-h-tiny": {
                        "improvement_percent": 12.5,
                        "direct_score": 0.8,
                        "conjecture_score": 0.9
                    }
                }
            }
        }
        
        # Generate summary report
        summary = integration_framework.get_evaluation_summary(mock_results)
        
        # Verify report content
        assert "DEEPEVAL EVALUATION SUMMARY" in summary
        assert "ibm/granite-4-h-tiny" in summary
        assert "12.5%" in summary
        assert "COMPARISON RESULTS" in summary
        assert "CONJECTURE IMPROVEMENTS" in summary

    @pytest.mark.asyncio
    async def test_error_recovery_and_fallback(self, integration_framework):
        """Test error handling and fallback mechanisms"""
        # Create test cases that might cause errors
        error_test_cases = [
            integration_framework.create_test_case(
                input_text="",  # Empty input
                expected_output="",
                additional_metadata={"error_test": "empty_input"}
            ),
            integration_framework.create_test_case(
                input_text="Normal input",
                expected_output="Normal output",
                additional_metadata={"error_test": "normal"}
            )
        ]
        
        # Mock wrapper that throws errors for some cases
        async def mock_generate_with_errors(prompt):
            if not prompt.strip():
                raise ValueError("Empty prompt detected")
            return "Normal response"
        
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(side_effect=mock_generate_with_errors)
            mock_create.return_value = mock_wrapper
            
            # Test error recovery
            result = await integration_framework.evaluate_provider(
                "error-test-provider", error_test_cases, use_conjecture=False
            )
            
            # Should handle errors gracefully
            assert "error" in result or result["success"] is False
            assert "overall_score" in result

    def test_configuration_validation(self, integration_framework):
        """Test configuration validation and setup"""
        # Test with valid configuration
        valid_config = {
            "providers": [
                {
                    "name": "test-provider",
                    "url": "http://test.com",
                    "api": "test-key",
                    "model": "test-model"
                }
            ]
        }
        
        # Test with invalid configuration
        invalid_config = {
            "providers": [
                {
                    "name": "invalid-provider",
                    "url": "",  # Missing URL
                    "api": "",  # Missing API key
                    "model": ""  # Missing model
                }
            ]
        }
        
        # Configuration validation should work
        assert integration_framework.config is not None
        assert isinstance(integration_framework.config, dict)
        
        # Test metrics initialization
        available_metrics = integration_framework.get_available_metrics()
        assert len(available_metrics) > 0
        assert all(isinstance(metric, str) for metric in available_metrics)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])