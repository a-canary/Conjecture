"""
OPTIMIZED: Performance Regression Tests for Conjecture
Validates that optimizations maintain or improve performance over time
"""

import asyncio
import time
import pytest
import statistics
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os

from src.conjecture import Conjecture
from src.monitoring import get_performance_monitor
from src.core.models import ClaimState
from src.config.config import Config


class PerformanceRegressionTest:
    """
    OPTIMIZED: Comprehensive performance regression testing framework
    """
    
    def __init__(self):
        self.results = {}
        self.baseline_results = {}
        self.performance_monitor = get_performance_monitor()
        
        # Performance targets based on optimization requirements
        self.performance_targets = {
            "max_explore_time": 70.0,  # Target: <70s for True Conjecture
            "max_overhead_ratio": 1.5,  # Target: <1.5x overhead
            "min_cache_hit_rate": 80.0,  # Target: 80%+ cache hit rate
            "max_context_collection_time": 5.0,  # Target: <5s for context
            "max_claim_generation_time": 10.0,  # Target: <10s for generation
            "max_batch_operation_time": 2.0,  # Target: <2s for batch ops
        }
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance regression tests"""
        print("🧪 Running Performance Regression Tests")
        print("=" * 50)
        
        test_results = {
            "test_run_timestamp": datetime.utcnow().isoformat(),
            "tests": {},
            "summary": {},
            "regression_detected": False
        }
        
        # Run individual tests
        tests = [
            ("explore_performance", self.test_explore_performance),
            ("context_collection_performance", self.test_context_collection_performance),
            ("cache_performance", self.test_cache_performance),
            ("database_performance", self.test_database_performance),
            ("parallel_processing_performance", self.test_parallel_processing_performance),
            ("memory_usage", self.test_memory_usage),
            ("concurrent_load", self.test_concurrent_load),
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running {test_name}...")
            try:
                result = await test_func()
                test_results["tests"][test_name] = result
                
                # Check against targets
                regression = self._check_performance_regression(test_name, result)
                if regression:
                    test_results["regression_detected"] = True
                    
                status = "❌ REGRESSION" if regression else "✅ PASS"
                print(f"   {status}: {test_name}")
                
            except Exception as e:
                test_results["tests"][test_name] = {
                    "error": str(e),
                    "status": "ERROR"
                }
                print(f"   ❌ ERROR: {test_name} - {e}")
        
        # Generate summary
        test_results["summary"] = self._generate_summary(test_results["tests"])
        
        # Print summary
        self._print_summary(test_results)
        
        # Save results
        self._save_results(test_results)
        
        return test_results
        
    async def test_explore_performance(self) -> Dict[str, Any]:
        """Test explore method performance"""
        config = Config()
        async with Conjecture(config) as cf:
            # Test with different query complexities
            test_queries = [
                "simple query",
                "machine learning basics",
                "quantum computing applications in cryptography",
                "complex multi-domain research topic with multiple subtopics"
            ]
            
            results = []
            for query in test_queries:
                start_time = time.time()
                result = await cf.explore(query, max_claims=5)
                end_time = time.time()
                
                results.append({
                    "query": query,
                    "duration": end_time - start_time,
                    "claims_returned": len(result.claims),
                    "search_time": result.search_time
                })
                
            # Calculate statistics
            durations = [r["duration"] for r in results]
            return {
                "test_name": "explore_performance",
                "results": results,
                "statistics": {
                    "average_duration": statistics.mean(durations),
                    "median_duration": statistics.median(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "std_deviation": statistics.stdev(durations) if len(durations) > 1 else 0,
                    "total_queries": len(results),
                    "queries_per_second": len(results) / sum(durations) if sum(durations) > 0 else 0
                },
                "targets_met": self._check_explore_targets(durations)
            }
            
    async def test_context_collection_performance(self) -> Dict[str, Any]:
        """Test context collection performance"""
        config = Config()
        async with Conjecture(config) as cf:
            # Test context collection with different claim types
            test_claims = [
                "simple factual claim",
                "complex technical concept with multiple components",
                "multi-domain claim requiring diverse context",
                "claim requiring extensive skill and sample matching"
            ]
            
            results = []
            for claim_content in test_claims:
                start_time = time.time()
                context = await cf.context_collector.collect_context_for_claim(
                    claim_content, {}, max_skills=5, max_samples=10
                )
                end_time = time.time()
                
                results.append({
                    "claim_content": claim_content[:50] + "...",
                    "duration": end_time - start_time,
                    "skills_found": len(context.get("skills", [])),
                    "samples_found": len(context.get("samples", [])),
                    "total_context_items": len(context.get("skills", [])) + len(context.get("samples", []))
                })
                
            # Calculate statistics
            durations = [r["duration"] for r in results]
            return {
                "test_name": "context_collection_performance",
                "results": results,
                "statistics": {
                    "average_duration": statistics.mean(durations),
                    "median_duration": statistics.median(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "std_deviation": statistics.stdev(durations) if len(durations) > 1 else 0,
                    "total_claims": len(results)
                },
                "targets_met": all(d <= self.performance_targets["max_context_collection_time"] for d in durations)
            }
            
    async def test_cache_performance(self) -> Dict[str, Any]:
        """Test cache performance across multiple operations"""
        config = Config()
        async with Conjecture(config) as cf:
            # Test cache hit rates with repeated operations
            test_query = "machine learning optimization techniques"
            
            # First run (cache miss)
            start_time = time.time()
            result1 = await cf.explore(test_query, max_claims=5)
            first_run_time = time.time() - start_time
            
            # Second run (should hit cache)
            start_time = time.time()
            result2 = await cf.explore(test_query, max_claims=5)
            second_run_time = time.time() - start_time
            
            # Multiple runs to test cache consistency
            run_times = [first_run_time, second_run_time]
            for i in range(3):  # 3 more runs
                start_time = time.time()
                await cf.explore(test_query, max_claims=5)
                run_times.append(time.time() - start_time)
                
            # Calculate cache performance
            cache_stats = cf.performance_monitor.get_cache_performance("context_collection")
            
            return {
                "test_name": "cache_performance",
                "results": {
                    "first_run_time": first_run_time,
                    "second_run_time": second_run_time,
                    "all_run_times": run_times,
                    "speedup_ratio": first_run_time / second_run_time if second_run_time > 0 else 1
                },
                "cache_stats": cache_stats,
                "targets_met": cache_stats.get("hit_rate", 0) >= self.performance_targets["min_cache_hit_rate"]
            }
            
    async def test_database_performance(self) -> Dict[str, Any]:
        """Test database batch operation performance"""
        config = Config()
        async with Conjecture(config) as cf:
            # Test batch claim creation
            batch_sizes = [1, 5, 10, 25, 50]
            results = []
            
            for batch_size in batch_sizes:
                # Prepare batch data
                claims_data = []
                for i in range(batch_size):
                    claims_data.append({
                        "content": f"Test claim {i} for batch performance testing",
                        "confidence": 0.8,
                        "tags": ["test", "batch"],
                        "state": ClaimState.EXPLORE
                    })
                
                # Time batch operation
                start_time = time.time()
                batch_result = await cf.data_manager.batch_create_claims(claims_data)
                end_time = time.time()
                
                results.append({
                    "batch_size": batch_size,
                    "duration": end_time - start_time,
                    "claims_per_second": batch_size / (end_time - start_time) if end_time > start_time else 0,
                    "success": batch_result.success if hasattr(batch_result, 'success') else True
                })
                
            # Calculate statistics
            durations = [r["duration"] for r in results]
            return {
                "test_name": "database_performance",
                "results": results,
                "statistics": {
                    "average_duration": statistics.mean(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "throughput_claims_per_second": statistics.mean([r["claims_per_second"] for r in results])
                },
                "targets_met": all(d <= self.performance_targets["max_batch_operation_time"] for d in durations)
            }
            
    async def test_parallel_processing_performance(self) -> Dict[str, Any]:
        """Test parallel processing performance"""
        config = Config()
        async with Conjecture(config) as cf:
            # Test concurrent operations
            concurrent_tasks = 5
            query = "parallel processing test query"
            
            start_time = time.time()
            tasks = [
                cf.explore(f"{query} {i}", max_claims=3) 
                for i in range(concurrent_tasks)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Filter successful results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            
            return {
                "test_name": "parallel_processing_performance",
                "results": {
                    "concurrent_tasks": concurrent_tasks,
                    "successful_tasks": len(successful_results),
                    "failed_tasks": len([r for r in results if isinstance(r, Exception)]),
                    "total_duration": end_time - start_time,
                    "tasks_per_second": concurrent_tasks / (end_time - start_time) if end_time > start_time else 0
                },
                "targets_met": len(successful_results) == concurrent_tasks
            }
            
    async def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage during operations"""
        try:
            import psutil
            process = psutil.Process()
            
            # Get baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform memory-intensive operations
            config = Config()
            async with Conjecture(config) as cf:
                # Multiple explore operations
                tasks = [
                    cf.explore(f"memory test query {i}", max_claims=10)
                    for i in range(5)
                ]
                await asyncio.gather(*tasks)
                
            # Get peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                "test_name": "memory_usage",
                "results": {
                    "baseline_memory_mb": baseline_memory,
                    "peak_memory_mb": peak_memory,
                    "memory_increase_mb": peak_memory - baseline_memory,
                    "memory_efficiency": "GOOD" if peak_memory - baseline_memory < 100 else "NEEDS_ATTENTION"
                },
                "targets_met": (peak_memory - baseline_memory) < 100  # Less than 100MB increase
            }
            
        except ImportError:
            return {
                "test_name": "memory_usage",
                "results": {"error": "psutil not available"},
                "targets_met": False
            }
            
    async def test_concurrent_load(self) -> Dict[str, Any]:
        """Test system under concurrent load"""
        config = Config()
        async with Conjecture(config) as cf:
            # Simulate high concurrent load
            concurrent_requests = 20
            requests_per_second_target = 10
            
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = []
            for i in range(concurrent_requests):
                task = cf.explore(f"concurrent load test {i}", max_claims=2)
                tasks.append(task)
                
            # Execute with semaphore to limit concurrency
            semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent operations
            
            async def limited_task(task):
                async with semaphore:
                    return await task
                    
            limited_tasks = [limited_task(t) for t in tasks]
            
            results = await asyncio.gather(*limited_tasks, return_exceptions=True)
            end_time = time.time()
            
            # Analyze results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            total_duration = end_time - start_time
            actual_rps = len(successful_results) / total_duration if total_duration > 0 else 0
            
            return {
                "test_name": "concurrent_load",
                "results": {
                    "concurrent_requests": concurrent_requests,
                    "successful_requests": len(successful_results),
                    "failed_requests": len([r for r in results if isinstance(r, Exception)]),
                    "total_duration": total_duration,
                    "actual_rps": actual_rps,
                    "target_rps": requests_per_second_target
                },
                "targets_met": actual_rps >= requests_per_second_target * 0.5  # At least 50% of target
            }
            
    def _check_performance_regression(self, test_name: str, result: Dict[str, Any]) -> bool:
        """Check if performance regression is detected"""
        if "statistics" not in result:
            return False
            
        stats = result["statistics"]
        regression_detected = False
        
        # Check specific regression conditions based on test type
        if test_name == "explore_performance":
            avg_duration = stats.get("average_duration", 0)
            if avg_duration > self.performance_targets["max_explore_time"]:
                regression_detected = True
                
        elif test_name == "context_collection_performance":
            avg_duration = stats.get("average_duration", 0)
            if avg_duration > self.performance_targets["max_context_collection_time"]:
                regression_detected = True
                
        elif test_name == "database_performance":
            avg_duration = stats.get("average_duration", 0)
            if avg_duration > self.performance_targets["max_batch_operation_time"]:
                regression_detected = True
                
        return regression_detected
        
    def _check_explore_targets(self, durations: List[float]) -> bool:
        """Check if explore performance targets are met"""
        if not durations:
            return False
            
        avg_duration = statistics.mean(durations)
        max_duration = max(durations)
        
        # Check both average and maximum against targets
        return (avg_duration <= self.performance_targets["max_explore_time"] and
                max_duration <= self.performance_targets["max_explore_time"] * 1.5)
                
    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary"""
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() 
                          if result.get("targets_met", False))
        failed_tests = total_tests - passed_tests
        regression_detected = any(result.get("targets_met", False) == False 
                              for result in test_results.values())
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "pass_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "regression_detected": regression_detected,
            "overall_status": "PASS" if failed_tests == 0 else "REGRESSION DETECTED"
        }
        
    def _print_summary(self, test_results: Dict[str, Any]):
        """Print test summary"""
        summary = test_results["summary"]
        
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE REGRESSION TEST SUMMARY")
        print("=" * 50)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Pass Rate: {summary['pass_rate']:.1f}%")
        print(f"Overall Status: {summary['overall_status']}")
        
        if summary['regression_detected']:
            print("\n⚠️  PERFORMANCE REGRESSION DETECTED!")
            print("Review failed tests and consider rollback of changes")
        else:
            print("\n✅ All performance targets met!")
            
        print("=" * 50)
        
    def _save_results(self, test_results: Dict[str, Any]):
        """Save test results to file"""
        results_dir = "test_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_dir}/performance_regression_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(test_results, f, indent=2)
            print(f"\n📁 Results saved to: {filename}")
        except Exception as e:
            print(f"Failed to save results: {e}")


# pytest fixtures and test functions
@pytest.fixture
async def performance_tester():
    """Pytest fixture for performance regression testing"""
    return PerformanceRegressionTest()


@pytest.mark.asyncio
async def test_performance_regression_full(performance_tester):
    """Full performance regression test suite"""
    results = await performance_tester.run_all_tests()
    
    # Assert no regression detected
    assert not results["regression_detected"], (
        f"Performance regression detected: {results['summary']['overall_status']}"
    )
    
    # Assert minimum pass rate
    assert results["summary"]["pass_rate"] >= 80, (
        f"Pass rate too low: {results['summary']['pass_rate']:.1f}%"
    )


@pytest.mark.asyncio
async def test_explore_performance_target(performance_tester):
    """Test explore performance against targets"""
    result = await performance_tester.test_explore_performance()
    
    # Check against targets
    assert result["targets_met"], (
        f"Explore performance targets not met: {result['statistics']}"
    )


@pytest.mark.asyncio
async def test_cache_performance_target(performance_tester):
    """Test cache performance against targets"""
    result = await performance_tester.test_cache_performance()
    
    # Check cache hit rate target
    assert result["targets_met"], (
        f"Cache performance targets not met: {result['cache_stats']}"
    )


if __name__ == "__main__":
    """Run performance regression tests directly"""
    async def main():
        tester = PerformanceRegressionTest()
        await tester.run_all_tests()
        
    asyncio.run(main())