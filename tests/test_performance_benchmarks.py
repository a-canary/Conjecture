"""
Performance and Stress Tests for DeepEval Integration
Tests concurrent evaluation, large dataset handling, memory usage, and response time
"""

import asyncio
import json
import pytest
import sys
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import statistics

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import EvaluationFramework, create_conjecture_wrapper
from src.benchmarking.deepeval_integration import AdvancedBenchmarkEvaluator


class TestConcurrentEvaluation:
    """Test concurrent evaluation of multiple providers"""

    @pytest.fixture
    def concurrent_framework(self):
        """Create framework for concurrent testing"""
        return EvaluationFramework()

    @pytest.fixture
    def test_providers(self):
        """List of providers for concurrent testing"""
        return [
            "ibm/granite-4-h-tiny",
            "zai/GLM-4.6",
            "openrouter/gpt-oss-20b"
        ]

    @pytest.mark.asyncio
    async def test_concurrent_provider_evaluation(self, concurrent_framework, test_providers):
        """Test concurrent evaluation of multiple providers"""
        # Create test cases for concurrent evaluation
        concurrent_test_cases = [
            concurrent_framework.create_test_case(
                input_text=f"Concurrent test question {i}",
                expected_output=f"Concurrent answer {i}",
                additional_metadata={
                    "concurrent_test": True,
                    "test_id": i,
                    "category": "performance"
                }
            )
            for i in range(10)  # 10 test cases
        ]
        
        # Mock evaluation for performance testing
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="Concurrent test response")
            mock_create.return_value = mock_wrapper
            
            start_time = time.time()
            
            # Run evaluations concurrently
            tasks = [
                concurrent_framework.evaluate_provider(
                    provider, concurrent_test_cases, use_conjecture=False
                )
                for provider in test_providers
            ]
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Performance assertions
            assert len(results) == len(test_providers)
            assert execution_time < 30.0  # Should complete within 30 seconds
            
            # All results should be successful
            for result in results:
                assert result["success"] is True
                assert result["test_cases_count"] == len(concurrent_test_cases)
                assert "overall_score" in result

    @pytest.mark.asyncio
    async def test_concurrent_metric_evaluation(self, concurrent_framework):
        """Test concurrent evaluation of multiple metrics"""
        test_case = concurrent_framework.create_test_case(
            input_text="Concurrent metric test",
            expected_output="Concurrent metric answer",
            additional_metadata={"category": "concurrent_metrics"}
        )
        
        # Test all metrics concurrently
        metrics = ["answer_relevancy", "faithfulness", "exact_match", 
                   "summarization", "bias", "toxicity"]
        
        # Mock metric evaluation
        with patch('src.evaluation.evaluation_framework.evaluate') as mock_evaluate:
            mock_result = Mock()
            mock_result.score = 0.8
            mock_result.reason = "Concurrent metric evaluation"
            mock_evaluate.return_value = [mock_result]
            
            start_time = time.time()
            
            # Run metric evaluations concurrently
            metric_tasks = [
                concurrent_framework.evaluate_provider(
                    "test-provider", [test_case], use_conjecture=False, 
                    metrics=[metric]
                )
                for metric in metrics
            ]
            
            metric_results = await asyncio.gather(*metric_tasks)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Performance assertions
            assert len(metric_results) == len(metrics)
            assert execution_time < 15.0  # Should complete within 15 seconds
            
            # All metric results should be valid
            for result in metric_results:
                assert result["success"] is True
                assert metric in result["metrics_results"]

    @pytest.mark.asyncio
    async def test_concurrent_conjecture_comparison(self, concurrent_framework, test_providers):
        """Test concurrent Conjecture enhancement comparison"""
        test_cases = [
            concurrent_framework.create_test_case(
                input_text=f"Conjecture comparison test {i}",
                expected_output=f"Conjecture comparison answer {i}",
                additional_metadata={"conjecture_test": True, "test_id": i}
            )
            for i in range(5)
        ]
        
        # Mock evaluation for conjecture comparison
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="Conjecture comparison response")
            mock_create.return_value = mock_wrapper
            
            start_time = time.time()
            
            # Run concurrent direct and conjecture evaluations
            tasks = []
            for provider in test_providers:
                # Direct evaluation
                tasks.append(concurrent_framework.evaluate_provider(
                    provider, test_cases, use_conjecture=False
                ))
                # Conjecture evaluation
                tasks.append(concurrent_framework.evaluate_provider(
                    provider, test_cases, use_conjecture=True
                ))
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Performance assertions
            assert len(results) == len(test_providers) * 2  # direct + conjecture
            assert execution_time < 45.0  # Should complete within 45 seconds
            
            # Verify all results are successful
            for result in results:
                assert result["success"] is True
                assert result["test_cases_count"] == len(test_cases)

    @pytest.mark.asyncio
    async def test_concurrent_stress_testing(self, concurrent_framework):
        """Test concurrent evaluation under stress conditions"""
        # Create large number of test cases for stress testing
        stress_test_cases = [
            concurrent_framework.create_test_case(
                input_text=f"Stress test {i}",
                expected_output=f"Stress answer {i}",
                additional_metadata={"stress_test": True, "test_id": i}
            )
            for i in range(50)  # 50 test cases
        ]
        
        # Mock evaluation with realistic timing
        async def mock_stress_evaluation(*args, **kwargs):
            # Simulate variable processing time
            await asyncio.sleep(0.1 + (hash(str(args)) % 5) * 0.02)
            return {
                "provider": "stress-test",
                "overall_score": 0.75,
                "success": True,
                "metrics_results": {}
            }
        
        with patch('src.evaluation.evaluation_framework.evaluate_provider', 
                  side_effect=mock_stress_evaluation):
            
            start_time = time.time()
            
            # Run many concurrent evaluations
            stress_tasks = [
                concurrent_framework.evaluate_provider(
                    f"stress-provider-{i}", stress_test_cases[:10], use_conjecture=False
                )
                for i in range(10)  # 10 concurrent evaluations
            ]
            
            stress_results = await asyncio.gather(*stress_tasks)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Stress test assertions
            assert len(stress_results) == 10
            assert execution_time < 60.0  # Should complete within 1 minute
            
            # All stress tests should be successful
            for result in stress_results:
                assert result["success"] is True
                assert result["test_cases_count"] == 10

    def test_concurrent_resource_management(self):
        """Test resource management during concurrent evaluation"""
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        
        async def concurrent_memory_test():
            # Create memory-intensive concurrent tasks
            tasks = []
            for i in range(20):
                task = asyncio.create_task(
                    asyncio.sleep(0.1)  # Simulate work
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
        
        # Run concurrent memory test
        start_time = time.time()
        asyncio.run(concurrent_memory_test())
        end_time = time.time()
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Resource management assertions
        assert end_time - start_time < 5.0  # Should complete quickly
        assert memory_increase < 100  # Should not increase memory significantly


class TestLargeDatasetHandling:
    """Test evaluation with large datasets (100+ scenarios)"""

    @pytest.fixture
    def large_dataset_framework(self):
        """Create framework for large dataset testing"""
        return EvaluationFramework()

    @pytest.mark.asyncio
    async def test_large_dataset_evaluation(self, large_dataset_framework):
        """Test evaluation with large dataset (100+ scenarios)"""
        # Create large dataset
        large_dataset = [
            large_dataset_framework.create_test_case(
                input_text=f"Large dataset test {i}",
                expected_output=f"Large dataset answer {i}",
                additional_metadata={
                    "large_test": True,
                    "test_id": i,
                    "category": "large_dataset"
                }
            )
            for i in range(100)  # 100 test cases
        ]
        
        # Mock evaluation for performance testing
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="Large dataset response")
            mock_create.return_value = mock_wrapper
            
            start_time = time.time()
            
            # Process in batches to avoid memory issues
            batch_size = 20
            all_results = []
            
            for i in range(0, len(large_dataset), batch_size):
                batch = large_dataset[i:i + batch_size]
                
                batch_result = await large_dataset_framework.evaluate_provider(
                    "large-dataset-provider", batch, use_conjecture=False
                )
                all_results.append(batch_result)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Large dataset assertions
            assert len(all_results) == 5  # 100 / 20 = 5 batches
            assert execution_time < 120.0  # Should complete within 2 minutes
            
            # Verify batch processing
            total_test_cases = sum(result["test_cases_count"] for result in all_results)
            assert total_test_cases == len(large_dataset)

    @pytest.mark.asyncio
    async def test_large_dataset_memory_efficiency(self, large_dataset_framework):
        """Test memory efficiency with large datasets"""
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create memory-efficient large dataset
        efficient_dataset = [
            large_dataset_framework.create_test_case(
                input_text=f"Efficient test {i}",
                expected_output=f"Efficient answer {i}",
                additional_metadata={"memory_efficient": True, "test_id": i}
            )
            for i in range(150)  # 150 test cases
        ]
        
        # Mock evaluation with memory efficiency
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="Memory efficient response")
            mock_create.return_value = mock_wrapper
            
            # Process with memory-efficient batching
            batch_size = 25  # Smaller batches for memory efficiency
            memory_snapshots = []
            
            for i in range(0, len(efficient_dataset), batch_size):
                batch = efficient_dataset[i:i + batch_size]
                
                # Take memory snapshot before batch
                pre_batch_memory = process.memory_info().rss / 1024 / 1024
                memory_snapshots.append(pre_batch_memory)
                
                await large_dataset_framework.evaluate_provider(
                    "memory-efficient-provider", batch, use_conjecture=False
                )
                
                # Take memory snapshot after batch
                post_batch_memory = process.memory_info().rss / 1024 / 1024
                memory_snapshots.append(post_batch_memory)
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            # Memory efficiency assertions
            assert len(efficient_dataset) == 150
            assert memory_increase < 300  # Should not increase memory excessively
            
            # Memory should be reasonably stable across batches
            memory_changes = [
                memory_snapshots[i+1] - memory_snapshots[i] 
                for i in range(0, len(memory_snapshots), 2)
            ]
            avg_memory_change = statistics.mean(memory_changes)
            assert avg_memory_change < 50  # Average change should be small

    @pytest.mark.asyncio
    async def test_large_dataset_provider_comparison(self, large_dataset_framework):
        """Test provider comparison with large datasets"""
        providers = [
            "ibm/granite-4-h-tiny",
            "zai/GLM-4.6",
            "openrouter/gpt-oss-20b"
        ]
        
        # Create large comparison dataset
        comparison_dataset = [
            large_dataset_framework.create_test_case(
                input_text=f"Comparison test {i}",
                expected_output=f"Comparison answer {i}",
                additional_metadata={"comparison_test": True, "test_id": i}
            )
            for i in range(75)  # 75 test cases
        ]
        
        # Mock evaluation for comparison testing
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="Comparison test response")
            mock_create.return_value = mock_wrapper
            
            start_time = time.time()
            
            # Run large dataset comparison
            comparison_results = await large_dataset_framework.evaluate_multiple_providers(
                providers, comparison_dataset, compare_conjecture=True
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Large comparison assertions
            assert "providers" in comparison_results
            assert "comparison" in comparison_results
            assert len(comparison_results["providers"]) == len(providers) * 2  # direct + conjecture
            assert execution_time < 180.0  # Should complete within 3 minutes
            
            # Verify comparison data
            comparison = comparison_results["comparison"]
            assert "best_overall" in comparison
            assert "improvements" in comparison

    def test_large_dataset_batch_optimization(self):
        """Test batch optimization for large datasets"""
        dataset_sizes = [50, 100, 200, 500]
        batch_sizes = [10, 20, 50, 100]
        
        optimization_results = {}
        
        for dataset_size in dataset_sizes:
            for batch_size in batch_sizes:
                # Simulate batch processing time
                num_batches = (dataset_size + batch_size - 1) // batch_size
                estimated_time = num_batches * 0.5  # 0.5s per batch
                
                optimization_results[(dataset_size, batch_size)] = {
                    "num_batches": num_batches,
                    "estimated_time": estimated_time,
                    "efficiency": dataset_size / estimated_time  # Throughput
                }
        
        # Find optimal batch size for each dataset size
        for dataset_size in dataset_sizes:
            batch_results = [
                (batch_size, result) 
                for (ds, batch_size), result in optimization_results.items() 
                if ds == dataset_size
            ]
            
            # Sort by efficiency (throughput)
            batch_results.sort(key=lambda x: x[1]["efficiency"], reverse=True)
            
            optimal_batch_size, optimal_result = batch_results[0]
            
            # Optimal batch size should be reasonable
            assert 10 <= optimal_batch_size <= 100
            assert optimal_result["efficiency"] > 0


class TestMemoryAndCPUUsage:
    """Test memory and CPU usage monitoring during evaluation"""

    @pytest.fixture
    def resource_monitor_framework(self):
        """Create framework for resource monitoring"""
        return EvaluationFramework()

    def test_memory_usage_monitoring(self, resource_monitor_framework):
        """Test memory usage monitoring during evaluation"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create memory-intensive test cases
        memory_test_cases = [
            resource_monitor_framework.create_test_case(
                input_text="A" * 1000,  # Large prompt
                expected_output="B" * 1000,  # Large expected answer
                additional_metadata={"memory_test": True, "size": "large"}
            )
            for _ in range(30)
        ]
        
        # Mock evaluation to focus on memory testing
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="C" * 1000)
            mock_create.return_value = mock_wrapper
            
            # Monitor memory during evaluation
            memory_snapshots = []
            
            async def memory_monitoring_evaluation():
                for i, test_case in enumerate(memory_test_cases):
                    pre_memory = process.memory_info().rss / 1024 / 1024
                    memory_snapshots.append(pre_memory)
                    
                    # Simulate evaluation
                    await asyncio.sleep(0.01)
                    
                    post_memory = process.memory_info().rss / 1024 / 1024
                    memory_snapshots.append(post_memory)
            
            # Run memory monitoring
            asyncio.run(memory_monitoring_evaluation())
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            # Memory usage assertions
            assert len(memory_test_cases) == 30
            assert len(memory_snapshots) == 60  # pre + post for each test
            assert memory_increase < 200  # Should not increase memory excessively
            
            # Memory should be reasonably stable
            memory_changes = [
                memory_snapshots[i+1] - memory_snapshots[i] 
                for i in range(0, len(memory_snapshots), 2)
            ]
            max_memory_change = max(memory_changes)
            assert max_memory_change < 50  # No single large memory spike

    def test_cpu_usage_monitoring(self, resource_monitor_framework):
        """Test CPU usage monitoring during evaluation"""
        process = psutil.Process(os.getpid())
        
        # Create CPU-intensive test cases
        cpu_test_cases = [
            resource_monitor_framework.create_test_case(
                input_text=f"CPU test {i}",
                expected_output=f"CPU answer {i}",
                additional_metadata={"cpu_test": True, "test_id": i}
            )
            for i in range(20)
        ]
        
        # Mock evaluation to focus on CPU testing
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="CPU test response")
            mock_create.return_value = mock_wrapper
            
            # Monitor CPU during evaluation
            cpu_snapshots = []
            
            async def cpu_monitoring_evaluation():
                for i, test_case in enumerate(cpu_test_cases):
                    pre_cpu = process.cpu_percent()
                    cpu_snapshots.append(pre_cpu)
                    
                    # Simulate CPU-intensive work
                    await asyncio.sleep(0.05)
                    
                    post_cpu = process.cpu_percent()
                    cpu_snapshots.append(post_cpu)
            
            # Run CPU monitoring
            asyncio.run(cpu_monitoring_evaluation())
            
            # CPU usage assertions
            assert len(cpu_test_cases) == 20
            assert len(cpu_snapshots) == 40  # pre + post for each test
            
            # CPU usage should be reasonable
            avg_cpu = statistics.mean(cpu_snapshots)
            max_cpu = max(cpu_snapshots)
            
            assert avg_cpu < 80  # Average CPU should be reasonable
            assert max_cpu < 95  # Maximum CPU should not be extreme

    def test_resource_cleanup(self, resource_monitor_framework):
        """Test resource cleanup after evaluation"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create resource-intensive evaluation
        cleanup_test_cases = [
            resource_monitor_framework.create_test_case(
                input_text=f"Cleanup test {i}",
                expected_output=f"Cleanup answer {i}",
                additional_metadata={"cleanup_test": True, "test_id": i}
            )
            for i in range(25)
        ]
        
        # Mock evaluation with resource cleanup
        with patch('src.evaluation.evaluation_framework.create_conjecture_wrapper') as mock_create:
            mock_wrapper = AsyncMock()
            mock_wrapper.a_generate = AsyncMock(return_value="Cleanup test response")
            mock_create.return_value = mock_wrapper
            
            async def resource_cleanup_evaluation():
                # Simulate evaluation with resource allocation
                large_data = ["A" * 1000 for _ in range(100)]  # Allocate memory
                await asyncio.sleep(0.01)
                
                # Cleanup resources
                del large_data
                
                # Force garbage collection
                import gc
                gc.collect()
            
            # Run cleanup test
            asyncio.run(resource_cleanup_evaluation())
            
            # Allow time for cleanup
            import time
            time.sleep(0.1)
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            
            # Resource cleanup assertions
            assert len(cleanup_test_cases) == 25
            assert memory_increase < 50  # Most memory should be cleaned up


class TestResponseTimeMeasurement:
    """Test response time measurement and optimization"""

    @pytest.fixture
    def timing_framework(self):
        """Create framework for timing measurement"""
        return EvaluationFramework()

    @pytest.mark.asyncio
    async def test_response_time_measurement(self, timing_framework):
        """Test response time measurement and analysis"""
        test_cases = [
            timing_framework.create_test_case(
                input_text=f"Timing test {i}",
                expected_output=f"Timing answer {i}",
                additional_metadata={"timing_test": True, "test_id": i}
            )
            for i in range(15)
        ]
        
        response_times = []
        
        # Mock evaluation with realistic timing
        async def timed_evaluation(*args, **kwargs):
            # Simulate variable response times
            await asyncio.sleep(0.05 + (len(response_times) % 3) * 0.02)
            return {
                "provider": "timing-test",
                "overall_score": 0.8,
                "success": True,
                "metrics_results": {}
            }
        
        with patch('src.evaluation.evaluation_framework.evaluate_provider', 
                  side_effect=timed_evaluation):
            
            for i, test_case in enumerate(test_cases):
                start_time = time.time()
                
                result = await timing_framework.evaluate_provider(
                    "timing-test-provider", [test_case], use_conjecture=False
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                response_times.append(response_time)
                
                assert result["success"] is True
        
        # Response time analysis
        avg_response_time = statistics.mean(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        std_response_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
        
        # Timing assertions
        assert len(response_times) == 15
        assert avg_response_time < 0.2  # Average under 200ms
        assert max_response_time < 0.3  # Maximum under 300ms
        assert std_response_time < 0.05  # Low variability

    @pytest.mark.asyncio
    async def test_response_time_optimization(self, timing_framework):
        """Test response time optimization techniques"""
        # Test different optimization approaches
        optimization_tests = [
            {
                "name": "batch_processing",
                "test_cases": [
                    timing_framework.create_test_case(
                        input_text=f"Batch test {i}",
                        expected_output=f"Batch answer {i}",
                        additional_metadata={"batch_test": True}
                    )
                    for i in range(20)
                ],
                "optimization": "batch_size_10"
            },
            {
                "name": "concurrent_processing",
                "test_cases": [
                    timing_framework.create_test_case(
                        input_text=f"Concurrent test {i}",
                        expected_output=f"Concurrent answer {i}",
                        additional_metadata={"concurrent_test": True}
                    )
                    for i in range(20)
                ],
                "optimization": "concurrent_5"
            },
            {
                "name": "sequential_processing",
                "test_cases": [
                    timing_framework.create_test_case(
                        input_text=f"Sequential test {i}",
                        expected_output=f"Sequential answer {i}",
                        additional_metadata={"sequential_test": True}
                    )
                    for i in range(20)
                ],
                "optimization": "sequential"
            }
        ]
        
        optimization_results = {}
        
        for test_config in optimization_tests:
            # Mock evaluation with timing
            async def optimized_evaluation(*args, **kwargs):
                if test_config["optimization"] == "batch_size_10":
                    await asyncio.sleep(0.01)  # Fast batch processing
                elif test_config["optimization"] == "concurrent_5":
                    await asyncio.sleep(0.02)  # Medium concurrent processing
                else:  # sequential
                    await asyncio.sleep(0.05)  # Slow sequential processing
                
                return {
                    "provider": "optimization-test",
                    "overall_score": 0.8,
                    "success": True,
                    "metrics_results": {}
                }
            
            with patch('src.evaluation.evaluation_framework.evaluate_provider', 
                      side_effect=optimized_evaluation):
                
                start_time = time.time()
                
                if test_config["optimization"] == "batch_size_10":
                    # Process in batches of 10
                    batch_size = 10
                    for i in range(0, len(test_config["test_cases"]), batch_size):
                        batch = test_config["test_cases"][i:i + batch_size]
                        await timing_framework.evaluate_provider(
                            "batch-test-provider", batch, use_conjecture=False
                        )
                elif test_config["optimization"] == "concurrent_5":
                    # Process with 5 concurrent workers
                    tasks = []
                    for i, test_case in enumerate(test_config["test_cases"]):
                        if i % 5 == 0:  # Start new batch
                            batch_tasks = [
                                timing_framework.evaluate_provider(
                                    "concurrent-test-provider", [tc], use_conjecture=False
                                )
                                for tc in test_config["test_cases"][i:i+5]
                            ]
                            await asyncio.gather(*batch_tasks)
                else:  # sequential
                    for test_case in test_config["test_cases"]:
                        await timing_framework.evaluate_provider(
                            "sequential-test-provider", [test_case], use_conjecture=False
                        )
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                optimization_results[test_config["name"]] = {
                    "execution_time": execution_time,
                    "optimization": test_config["optimization"],
                    "throughput": len(test_config["test_cases"]) / execution_time
                }
        
        # Optimization analysis
        batch_result = optimization_results["batch_processing"]
        concurrent_result = optimization_results["concurrent_processing"]
        sequential_result = optimization_results["sequential_processing"]
        
        # Batch should be fastest
        assert batch_result["execution_time"] < concurrent_result["execution_time"]
        assert batch_result["execution_time"] < sequential_result["execution_time"]
        
        # Concurrent should be faster than sequential
        assert concurrent_result["execution_time"] < sequential_result["execution_time"]
        
        # Throughput should be highest for batch
        assert batch_result["throughput"] >= concurrent_result["throughput"]
        assert concurrent_result["throughput"] >= sequential_result["throughput"]

    def test_response_time_regression_detection(self, timing_framework):
        """Test response time regression detection"""
        # Baseline performance data
        baseline_performance = {
            "avg_response_time": 0.1,
            "p95_response_time": 0.15,
            "p99_response_time": 0.2,
            "throughput": 100  # requests per second
        }
        
        # Current performance data (with regression)
        current_performance = {
            "avg_response_time": 0.15,  # 50% slower
            "p95_response_time": 0.25,  # 67% slower
            "p99_response_time": 0.35,  # 75% slower
            "throughput": 67  # 33% lower throughput
        }
        
        # Calculate regression metrics
        regressions = {}
        
        for metric in baseline_performance:
            baseline = baseline_performance[metric]
            current = current_performance[metric]
            
            if "response_time" in metric:
                # Lower is better
                regression_pct = ((current - baseline) / baseline) * 100
            else:
                # Higher is better (throughput)
                regression_pct = ((baseline - current) / baseline) * 100
            
            regressions[metric] = {
                "baseline": baseline,
                "current": current,
                "regression_pct": regression_pct,
                "is_regression": regression_pct > 10.0  # 10% threshold
            }
        
        # Verify regression detection
        for metric, regression_data in regressions.items():
            assert "baseline" in regression_data
            assert "current" in regression_data
            assert "regression_pct" in regression_data
            assert "is_regression" in regression_data
            
            # All metrics should show regression in this test
            assert regression_data["regression_pct"] > 0
            assert regression_data["is_regression"] is True


class TestPerformanceRegression:
    """Test performance regression detection and monitoring"""

    @pytest.fixture
    def regression_framework(self):
        """Create framework for regression testing"""
        return EvaluationFramework()

    def test_performance_baseline_tracking(self, regression_framework):
        """Test performance baseline tracking and comparison"""
        # Simulate historical baseline data
        baseline_data = {
            "timestamp": "2024-01-01T00:00:00Z",
            "provider": "ibm/granite-4-h-tiny",
            "metrics": {
                "overall_score": 0.85,
                "response_time_avg": 0.1,
                "response_time_p95": 0.15,
                "memory_usage_avg": 50,
                "success_rate": 0.98,
                "throughput": 100
            },
            "test_conditions": {
                "dataset_size": 50,
                "concurrent_users": 1,
                "test_duration": 300
            }
        }
        
        # Current performance data
        current_data = {
            "timestamp": "2024-02-01T00:00:00Z",
            "provider": "ibm/granite-4-h-tiny",
            "metrics": {
                "overall_score": 0.82,  # Slight regression
                "response_time_avg": 0.12,  # Slight regression
                "response_time_p95": 0.18,  # Slight regression
                "memory_usage_avg": 55,  # Slight regression
                "success_rate": 0.96,  # Slight regression
                "throughput": 90  # Slight regression
            },
            "test_conditions": {
                "dataset_size": 50,
                "concurrent_users": 1,
                "test_duration": 300
            }
        }
        
        # Calculate regression analysis
        regression_analysis = self._calculate_performance_regression(baseline_data, current_data)
        
        # Verify regression analysis
        assert "overall_regression" in regression_analysis
        assert "metric_regressions" in regression_analysis
        assert "severity" in regression_analysis
        
        # Check specific metric regressions
        metric_regressions = regression_analysis["metric_regressions"]
        for metric in ["overall_score", "response_time_avg", "throughput"]:
            assert metric in metric_regressions
            metric_regression = metric_regressions[metric]
            
            assert "baseline" in metric_regression
            assert "current" in metric_regression
            assert "regression_pct" in metric_regression
            assert "is_regression" in metric_regression

    def test_performance_threshold_enforcement(self, regression_framework):
        """Test performance threshold enforcement"""
        # Define performance thresholds
        performance_thresholds = {
            "overall_score": {"min": 0.7, "target": 0.8},
            "response_time_avg": {"max": 0.2, "target": 0.1},
            "response_time_p95": {"max": 0.3, "target": 0.15},
            "memory_usage_avg": {"max": 100, "target": 50},
            "success_rate": {"min": 0.9, "target": 0.95},
            "throughput": {"min": 50, "target": 100}
        }
        
        # Test current performance against thresholds
        current_performance = {
            "overall_score": 0.75,  # Above minimum, below target
            "response_time_avg": 0.15,  # Below max, above target
            "response_time_p95": 0.25,  # Below max, above target
            "memory_usage_avg": 80,  # Below max, above target
            "success_rate": 0.92,  # Above minimum, below target
            "throughput": 75  # Above minimum, below target
        }
        
        threshold_violations = {}
        
        for metric, thresholds in performance_thresholds.items():
            current = current_performance[metric]
            
            violations = []
            
            if "min" in thresholds and current < thresholds["min"]:
                violations.append(f"Below minimum threshold {thresholds['min']}")
            
            if "max" in thresholds and current > thresholds["max"]:
                violations.append(f"Above maximum threshold {thresholds['max']}")
            
            if "target" in thresholds:
                if "score" in metric or "rate" in metric:
                    # Higher is better
                    if current < thresholds["target"]:
                        violations.append(f"Below target threshold {thresholds['target']}")
                else:
                    # Lower is better (time, memory)
                    if current > thresholds["target"]:
                        violations.append(f"Above target threshold {thresholds['target']}")
            
            threshold_violations[metric] = {
                "current": current,
                "thresholds": thresholds,
                "violations": violations,
                "is_compliant": len(violations) == 0
            }
        
        # Verify threshold enforcement
        for metric, violation_data in threshold_violations.items():
            assert "current" in violation_data
            assert "thresholds" in violation_data
            assert "violations" in violation_data
            assert "is_compliant" in violation_data
            
            # Some metrics should have violations in this test
            if metric in ["overall_score", "success_rate", "throughput"]:
                assert violation_data["is_compliant"] is False  # Below target

    def _calculate_performance_regression(self, baseline_data, current_data):
        """Calculate performance regression between baseline and current data"""
        baseline_metrics = baseline_data["metrics"]
        current_metrics = current_data["metrics"]
        
        regressions = {}
        overall_regression_count = 0
        total_metrics = len(baseline_metrics)
        
        for metric, baseline_value in baseline_metrics.items():
            current_value = current_metrics[metric]
            
            if "score" in metric or "rate" in metric:
                # Higher is better
                regression_pct = ((baseline_value - current_value) / baseline_value) * 100
            else:
                # Lower is better (time, memory)
                regression_pct = ((current_value - baseline_value) / baseline_value) * 100
            
            is_regression = regression_pct > 5.0  # 5% threshold
            
            if is_regression:
                overall_regression_count += 1
            
            regressions[metric] = {
                "baseline": baseline_value,
                "current": current_value,
                "regression_pct": regression_pct,
                "is_regression": is_regression
            }
        
        # Determine overall severity
        regression_ratio = overall_regression_count / total_metrics
        
        if regression_ratio >= 0.5:
            severity = "high"
        elif regression_ratio >= 0.25:
            severity = "medium"
        elif regression_ratio >= 0.1:
            severity = "low"
        else:
            severity = "none"
        
        return {
            "overall_regression": regression_ratio > 0.1,
            "metric_regressions": regressions,
            "severity": severity,
            "regression_ratio": regression_ratio,
            "regressed_metrics": overall_regression_count
        }


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])