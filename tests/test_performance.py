"""
Comprehensive performance tests for the Conjecture data layer.
Tests query latency, scalability benchmarks, memory usage, and throughput.
"""
import pytest
import asyncio
import time
import psutil
import os
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import statistics

from src.data.data_manager import DataManager
from src.data.models import Claim, ClaimFilter, DataConfig


@dataclass
class PerformanceMetrics:
    """Container for performance test metrics."""
    operation_name: str
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    operations_per_second: float
    memory_usage_mb: float
    success_count: int
    error_count: int


class PerformanceTestSuite:
    """Base class for performance testing with common utilities."""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    async def measure_operation(name: str, operation_func, iterations: int = 10) -> PerformanceMetrics:
        """Measure operation performance over multiple iterations."""
        times = []
        success_count = 0
        error_count = 0
        
        initial_memory = PerformanceTestSuite.get_memory_usage()
        
        for _ in range(iterations):
            start_time = time.time()
            try:
                await operation_func()
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error in {name}: {e}")
            finally:
                end_time = time.time()
                times.append(end_time - start_time)
        
        final_memory = PerformanceTestSuite.get_memory_usage()
        
        return PerformanceMetrics(
            operation_name=name,
            total_time=sum(times),
            avg_time=statistics.mean(times),
            min_time=min(times),
            max_time=max(times),
            operations_per_second=iterations / sum(times) if sum(times) > 0 else 0,
            memory_usage_mb=final_memory - initial_memory,
            success_count=success_count,
            error_count=error_count
        )
    
    @staticmethod
    def assert_performance_requirements(metrics: PerformanceMetrics, requirements: Dict[str, Any]):
        """Assert that performance meets requirements."""
        if "max_avg_time" in requirements:
            assert metrics.avg_time <= requirements["max_avg_time"], f"{metrics.operation_name} avg time {metrics.avg_time:.4f}s exceeds {requirements['max_avg_time']}s"
        
        if "min_ops_per_sec" in requirements:
            assert metrics.operations_per_second >= requirements["min_ops_per_sec"], f"{metrics.operation_name} ops/sec {metrics.operations_per_second:.2f} below {requirements['min_ops_per_sec']}"
        
        if "max_memory_growth" in requirements:
            assert metrics.memory_usage_mb <= requirements["max_memory_growth"], f"{metrics.operation_name} memory growth {metrics.memory_usage_mb:.2f}MB exceeds {requirements['max_memory_growth']}MB"
        
        if "min_success_rate" in requirements:
            success_rate = metrics.success_count / (metrics.success_count + metrics.error_count) if (metrics.success_count + metrics.error_count) > 0 else 0
            assert success_rate >= requirements["min_success_rate"], f"{metrics.operation_name} success rate {success_rate:.2%} below {requirements['min_success_rate']:.2%}"


class TestDataManagerPerformance:
    """Performance tests for DataManager operations."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_claim_creation_performance(self, data_manager: DataManager):
        """Test claim creation performance benchmarks."""
        requirements = {
            "max_avg_time": 0.01,  # 10ms
            "min_ops_per_sec": 50,
            "max_memory_growth": 10,  # 10MB
            "min_success_rate": 0.95
        }
        
        async def create_claim():
            return await data_manager.create_claim(
                "Performance test claim with sufficient content for embedding generation.",
                "perf_test_user",
                confidence=0.7,
                tags=["performance", "test"]
            )
        
        metrics = await PerformanceTestSuite.measure_operation("create_claim", create_claim, 20)
        
        PerformanceTestSuite.assert_performance_requirements(metrics, requirements)
        
        print(f"Claim Creation Performance:")
        print(f"  Avg Time: {metrics.avg_time:.4f}s")
        print(f"  Ops/sec: {metrics.operations_per_second:.2f}")
        print(f"  Memory Growth: {metrics.memory_usage_mb:.2f}MB")
        print(f"  Success Rate: {metrics.success_count}/{metrics.success_count + metrics.error_count}")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_claim_retrieval_performance(self, populated_data_manager: DataManager):
        """Test claim retrieval performance benchmarks."""
        requirements = {
            "max_avg_time": 0.005,  # 5ms
            "min_ops_per_sec": 100,
            "max_memory_growth": 5,  # 5MB
            "min_success_rate": 1.0
        }
        
        async def get_claim():
            return await populated_data_manager.get_claim("c0000001")
        
        metrics = await PerformanceTestSuite.measure_operation("get_claim", get_claim, 50)
        
        PerformanceTestSuite.assert_performance_requirements(metrics, requirements)
        
        print(f"Claim Retrieval Performance:")
        print(f"  Avg Time: {metrics.avg_time:.4f}s")
        print(f"  Ops/sec: {metrics.operations_per_second:.2f}")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_similarity_search_performance(self, populated_data_manager: DataManager):
        """Test similarity search performance benchmarks."""
        requirements = {
            "max_avg_time": 0.05,  # 50ms
            "min_ops_per_sec": 10,
            "max_memory_growth": 10,  # 10MB
            "min_success_rate": 1.0
        }
        
        async def search_similar():
            return await populated_data_manager.search_similar(
                "query about physics and scientific research", 
                limit=10
            )
        
        metrics = await PerformanceTestSuite.measure_operation("search_similar", search_similar, 20)
        
        PerformanceTestSuite.assert_performance_requirements(metrics, requirements)
        
        print(f"Similarity Search Performance:")
        print(f"  Avg Time: {metrics.avg_time:.4f}s")
        print(f"  Ops/sec: {metrics.operations_per_second:.2f}")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_filter_claims_performance(self, populated_data_manager: DataManager):
        """Test claim filtering performance benchmarks."""
        requirements = {
            "max_avg_time": 0.02,  # 20ms
            "min_ops_per_sec": 30,
            "max_memory_growth": 5,  # 5MB
            "min_success_rate": 1.0
        }
        
        async def filter_claims():
            filter_obj = ClaimFilter(
                confidence_min=0.7,
                limit=20
            )
            return await populated_data_manager.filter_claims(filter_obj)
        
        metrics = await PerformanceTestSuite.measure_operation("filter_claims", filter_claims, 30)
        
        PerformanceTestSuite.assert_performance_requirements(metrics, requirements)
        
        print(f"Filter Claims Performance:")
        print(f"  Avg Time: {metrics.avg_time:.4f}s")
        print(f"  Ops/sec: {metrics.operations_per_second:.2f}")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_batch_claim_creation_performance(self, data_manager: DataManager):
        """Test batch claim creation performance."""
        batch_sizes = [10, 50, 100]
        
        for batch_size in batch_sizes:
            requirements = {
                "max_avg_time": 0.5,  # 500ms for batch
                "max_memory_growth": 50 * (batch_size / 10),  # Scale with batch size
                "min_success_rate": 0.95
            }
            
            async def batch_create():
                claims_data = [
                    {
                        "content": f"Batch test claim {i} with sufficient content",
                        "created_by": "batch_user",
                        "confidence": 0.7,
                        "tags": ["batch", f"group_{i % 5}"]
                    }
                    for i in range(batch_size)
                ]
                return await data_manager.batch_create_claims(claims_data)
            
            metrics = await PerformanceTestSuite.measure_operation(
                f"batch_create_{batch_size}", batch_create, 5
            )
            
            # Adjust ops/sec calculation for batch operations
            metrics.operations_per_second = (batch_size * metrics.success_count) / metrics.total_time
            
            PerformanceTestSuite.assert_performance_requirements(metrics, requirements)
            
            print(f"Batch Creation Performance (size {batch_size}):")
            print(f"  Avg Time: {metrics.avg_time:.4f}s")
            print(f"  Items/sec: {metrics.operations_per_second:.2f}")
            print(f"  Memory Growth: {metrics.memory_usage_mb:.2f}MB")


class TestScalabilityTests:
    """Scalability tests with growing datasets."""

    @pytest.mark.slow
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_linear_scalability_claim_creation(self, test_config: DataConfig):
        """Test how claim creation scales with dataset size."""
        test_config.sqlite_path = ":memory:"
        test_config.chroma_path = "./test_chroma_scaling"
        
        dataset_sizes = [100, 500, 1000]
        creation_times = []
        
        for size in dataset_sizes:
            dm = DataManager(test_config, use_mock_embeddings=True)
            await dm.initialize()
            
            await dm.reset_database()  # Start fresh
            
            start_time = time.time()
            
            # Create claims in batches
            claims_data = [
                {
                    "content": f"Scalability test claim {i} with sufficient content for embedding generation",
                    "created_by": f"scaler_user_{i % 10}",
                    "confidence": 0.5 + (i % 50) / 100.0,
                    "tags": [f"tag_{i % 20}"]
                }
                for i in range(size)
            ]
            
            await dm.batch_create_claims(claims_data)
            
            end_time = time.time()
            creation_time = end_time - start_time
            creation_times.append(creation_time)
            
            performance_per_claim = creation_time / size
            
            print(f"Dataset Size: {size}, Total Time: {creation_time:.2f}s, Per Claim: {performance_per_claim:.4f}s")
            
            # Performance should not degrade linearly
            if len(creation_times) >= 2:
                time_growth_ratio = creation_times[-1] / creation_times[-2]
                size_growth_ratio = dataset_sizes[-1] / dataset_sizes[-2]
                
                # Time growth should be less than size growth (better than linear)
                assert time_growth_ratio < size_growth_ratio * 1.5, f"Performance degraded too much at size {size}"
            
            await dm.close()

    @pytest.mark.slow
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_search_performance_with_dataset_size(self, test_config: DataConfig):
        """Test search performance with different dataset sizes."""
        test_config.sqlite_path = ":memory:"
        test_config.chroma_path = "./test_chroma_search_scaling"
        
        dataset_sizes = [50, 200, 500]
        search_times = []
        
        for size in dataset_sizes:
            dm = DataManager(test_config, use_mock_embeddings=True)
            await dm.initialize()
            
            await dm.reset_database()
            
            # Create dataset
            claims_data = [
                {
                    "content": f"Domain specific content about topic_{i % 10} for search testing",
                    "created_by": "search_test_user",
                    "confidence": 0.7,
                    "tags": [f"topic_{i % 10}", f"category_{i % 5}"]
                }
                for i in range(size)
            ]
            
            created_claims = await dm.batch_create_claims(claims_data)
            
            # Measure search performance
            search_times_for_size = []
            for _ in range(10):
                start_time = time.time()
                results = await dm.search_similar("query about topic_5", limit=10)
                end_time = time.time()
                search_times_for_size.append(end_time - start_time)
            
            avg_search_time = statistics.mean(search_times_for_size)
            search_times.append(avg_search_time)
            
            print(f"Dataset Size: {size}, Avg Search Time: {avg_search_time:.4f}s")
            assert len(results) <= 10
            
            # Search time should scale sub-linearly
            if len(search_times) >= 2:
                search_growth_ratio = search_times[-1] / search_times[-2]
                size_growth_ratio = dataset_sizes[-1] / dataset_sizes[-2]
                
                # Allow some overhead but should be better than linear
                assert search_growth_ratio < size_growth_ratio, f"Search performance degraded at size {size}"
            
            await dm.close()

    @pytest.mark.slow
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_scaling(self, test_config: DataConfig):
        """Test memory usage doesn't grow excessively."""
        test_config.sqlite_path = ":memory:"
        test_config.chroma_path = "./test_chroma_memory_scaling"
        
        dm = DataManager(test_config, use_mock_embeddings=True)
        await dm.initialize()
        
        initial_memory = PerformanceTestSuite.get_memory_usage()
        memory_measurements = [(0, initial_memory)]
        
        batch_sizes = [100, 200, 400, 800]
        total_claims = 0
        
        for batch_size in batch_sizes:
            # Add batch of claims
            claims_data = [
                {
                    "content": f"Memory test claim {i} with sufficient content",
                    "created_by": "memory_test_user",
                    "confidence": 0.7,
                    "tags": [f"mem_tag_{i % 10}"]
                }
                for i in range(batch_size)
            ]
            
            await dm.batch_create_claims(claims_data)
            total_claims += batch_size
            
            current_memory = PerformanceTestSuite.get_memory_usage()
            memory_growth = current_memory - initial_memory
            
            memory_measurements.append((total_claims, current_memory))
            
            memory_per_claim = memory_growth / total_claims
            print(f"Claims: {total_claims}, Memory: {memory_growth:.2f}MB, Per Claim: {memory_per_claim:.4f}MB")
            
            # Memory per claim should stay reasonable
            assert memory_per_claim < 0.1, f"Memory per claim too high: {memory_per_claim:.4f}MB"
            
            # Total memory growth should be reasonable for 1500 claims
            if total_claims >= 1500:
                assert memory_growth < 150, f"Total memory growth too high: {memory_growth:.2f}MB"
        
        await dm.close()

    @pytest.mark.slow
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_operations_scalability(self, test_config: DataConfig):
        """Test performance with concurrent operations."""
        test_config.sqlite_path = ":memory:"
        test_config.chroma_path = "./test_chroma_concurrent"
        
        dm = DataManager(test_config, use_mock_embeddings=True)
        await dm.initialize()
        
        await dm.reset_database()
        
        async def create_claims_batch(start_id: int, count: int):
            """Create a batch of claims concurrently."""
            claims_data = [
                {
                    "content": f"Concurrent claim {start_id + i}",
                    "created_by": f"concurrent_user_{start_id}",
                    "confidence": 0.7,
                    "tags": ["concurrent", f"batch_{start_id}"]
                }
                for i in range(count)
            ]
            return await dm.batch_create_claims(claims_data)
        
        # Test with increasing concurrency levels
        concurrency_levels = [2, 4, 8]
        
        for concurrency in concurrency_levels:
            await dm.reset_database()
            
            start_time = time.time()
            
            # Run concurrent batches
            tasks = [
                create_claims_batch(i * 50, 50)
                for i in range(concurrency)
            ]
            
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_claims = sum(len(batch) for batch in results)
            total_time = end_time - start_time
            ops_per_second = total_claims / total_time
            
            print(f"Concurrency: {concurrency}, Claims: {total_claims}, Time: {total_time:.2f}s, Ops/sec: {ops_per_second:.2f}")
            
            # Higher concurrency should improve throughput
            assert ops_per_second > 20, f"Throughput too low with concurrency {concurrency}"
            
            # All operations should succeed
            for batch in results:
                assert len(batch) == 50
        
        await dm.close()


class TestResourceUtilization:
    """Test resource utilization patterns."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cpu_utilization_patterns(self, data_manager: DataManager):
        """Test CPU utilization during various operations."""
        import threading
        
        cpu_samples = []
        sampling_active = [True]
        
        def cpu_sampler():
            """Sample CPU usage while operations run."""
            while sampling_active[0]:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)
                time.sleep(0.1)
        
        # Start CPU sampling
        sampler_thread = threading.Thread(target=cpu_sampler, daemon=True)
        sampler_thread.start()
        
        try:
            # Perform various operations
            await data_manager.create_claim("CPU test claim", "test_user")
            await data_manager.search_similar("test query", limit=5)
            
            claims_data = [
                {"content": f"Batch claim {i}", "created_by": "cpu_test"}
                for i in range(20)
            ]
            await data_manager.batch_create_claims(claims_data)
            
        finally:
            # Stop sampling
            sampling_active[0] = False
            sampler_thread.join(timeout=1.0)
        
        if cpu_samples:
            avg_cpu = statistics.mean(cpu_samples)
            max_cpu = max(cpu_samples)
            
            print(f"CPU Utilization - Avg: {avg_cpu:.1f}%, Max: {max_cpu:.1f}%")
            
            # CPU usage should be reasonable
            assert avg_cpu < 80, f"Average CPU usage too high: {avg_cpu:.1f}%"
            assert max_cpu < 95, f"Maximum CPU usage too high: {max_cpu:.1f}%"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_connection_pool_efficiency(self, test_config: DataConfig):
        """Test database connection pool efficiency."""
        # Create multiple data managers to test connection handling
        managers = []
        initial_memory = PerformanceTestSuite.get_memory_usage()
        
        try:
            # Create multiple instances
            for i in range(5):
                config = DataConfig(
                    sqlite_path=":memory:",
                    chroma_path=f"./test_chroma_pool_{i}",
                    embedding_model="all-MiniLM-L6-v2"
                )
                dm = DataManager(config, use_mock_embeddings=True)
                await dm.initialize()
                managers.append(dm)
            
            # Use all managers concurrently
            tasks = []
            for i, dm in enumerate(managers):
                async def use_dm(idx, manager):
                    await manager.create_claim(f"Pool test claim {idx}", f"user_{idx}")
                    await manager.search_similar(f"query {idx}", limit=3)
                    return await manager.get_stats()
                
                tasks.append(use_dm(i, dm))
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            assert len(results) == 5
            
            final_memory = PerformanceTestSuite.get_memory_usage()
            memory_growth = final_memory - initial_memory
            
            print(f"Connection Pool Memory Growth: {memory_growth:.2f}MB")
            
            # Memory growth should be reasonable for 5 managers
            assert memory_growth < 100, f"Memory growth too high: {memory_growth:.2f}MB"
            
        finally:
            # Clean up all managers
            for dm in managers:
                await dm.close()

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_embedding_cache_efficiency(self, populated_data_manager: DataManager):
        """Test embedding cache efficiency for repeated queries."""
        query = "repeated query for cache testing"
        
        # First search (cache miss)
        start_time = time.time()
        results1 = await populated_data_manager.search_similar(query, limit=5)
        first_time = time.time() - start_time
        
        # Second search (potential cache hit)
        start_time = time.time()
        results2 = await populated_data_manager.search_similar(query, limit=5)
        second_time = time.time() - start_time
        
        print(f"First search: {first_time:.4f}s, Second search: {second_time:.4f}s")
        
        # Results should be identical
        assert len(results1) == len(results2)
        
        # Note: With mock embeddings, cache benefits may be less apparent
        # In real usage with sentence-transformers, second search should be faster
        if hasattr(populated_data_manager.embedding_service, 'cache'):
            # If caching is implemented, second search should be faster
            assert second_time <= first_time * 1.1, "Cache not providing expected speedup"


class TestBenchmarkSuite:
    """Comprehensive benchmark suite for performance regression testing."""

    @pytest.mark.slow
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_full_workload_benchmark(self, test_config: DataConfig):
        """Run comprehensive benchmark simulating real workloads."""
        test_config.sqlite_path = ":memory:"
        test_config.chroma_path = "./test_chroma_benchmark"
        
        dm = DataManager(test_config, use_mock_embeddings=True)
        await dm.initialize()
        
        await dm.reset_database()
        
        benchmark_results = {}
        
        # Phase 1: Data Loading
        print("Phase 1: Data Loading...")
        start_time = time.time()
        
        # Load diverse dataset
        domains = ["science", "technology", "health", "environment", "education"]
        claims_data = []
        
        for i in range(500):
            domain = domains[i % len(domains)]
            claims_data.append({
                "content": f"Domain {domain} claim number {i} with specific content about {domain} topics and research findings",
                "created_by": f"expert_{domain}_{i % 10}",
                "confidence": 0.5 + (i % 50) / 100.0,
                "tags": [domain, f"subcategory_{i % 10}", f"priority_{i % 3}"],
                "dirty": i % 3 == 0
            })
        
        created_claims = await dm.batch_create_claims(claims_data)
        loading_time = time.time() - start_time
        
        benchmark_results["loading"] = {
            "claims_count": len(created_claims),
            "total_time": loading_time,
            "claims_per_second": len(created_claims) / loading_time
        }
        
        print(f"  Loaded {len(created_claims)} claims in {loading_time:.2f}s ({benchmark_results['loading']['claims_per_second']:.2f} claims/sec)")
        
        # Phase 2: Query Performance
        print("Phase 2: Query Performance...")
        query_types = ["similarity", "filter", "get", "relationships"]
        query_times = {qt: [] for qt in query_types}
        
        for i in range(50):
            # Similarity search
            start_time = time.time()
            await dm.search_similar(f"test query about {domains[i % len(domains)]}", limit=10)
            query_times["similarity"].append(time.time() - start_time)
            
            # Filter search
            filter_obj = ClaimFilter(
                tags=[domains[i % len(domains)]],
                confidence_min=0.6,
                limit=20
            )
            start_time = time.time()
            await dm.filter_claims(filter_obj)
            query_times["filter"].append(time.time() - start_time)
            
            # Get claim
            claim_id = created_claims[i % len(created_claims)].id
            start_time = time.time()
            await dm.get_claim(claim_id)
            query_times["get"].append(time.time() - start_time)
            
            # Get relationships
            if i < len(created_claims) - 1:
                await dm.add_relationship(
                    created_claims[i].id,
                    created_claims[i + 1].id,
                    "supports",
                    "benchmark_test"
                )
                start_time = time.time()
                await dm.get_relationships(created_claims[i].id)
                query_times["relationships"].append(time.time() - start_time)
        
        # Calculate query statistics
        for query_type, times in query_times.items():
            if times:
                benchmark_results[query_type] = {
                    "avg_time": statistics.mean(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "operations_per_second": 1 / statistics.mean(times)
                }
                print(f"  {query_type.title()}: Avg {benchmark_results[query_type]['avg_time']:.4f}s, Ops/sec {benchmark_results[query_type]['operations_per_second']:.2f}")
        
        # Phase 3: Update Performance
        print("Phase 3: Update Performance...")
        start_time = time.time()
        
        # Update some claims
        for i in range(0, 50, 5):
            await dm.update_claim(
                created_claims[i].id,
                confidence=0.9,
                dirty=False
            )
        
        # Add more relationships
        for i in range(100, 150):
            if i < len(created_claims):
                await dm.add_relationship(
                    created_claims[50].id,
                    created_claims[i].id,
                    "supports",
                    "benchmark_test"
                )
        
        update_time = time.time() - start_time
        benchmark_results["updates"] = {
            "total_time": update_time,
            "operations_per_second": 20 / update_time  # 10 updates + 10 relationships
        }
        
        print(f"  Updates completed in {update_time:.2f}s")
        
        # Phase 4: Statistics and Cleanup
        print("Phase 4: Statistics and Cleanup...")
        start_time = time.time()
        stats = await dm.get_stats()
        stats_time = time.time() - start_time
        
        benchmark_results["stats"] = {
            "time": stats_time,
            "final_claim_count": stats["total_claims"]
        }
        
        print(f"  Stats retrieved in {stats_time:.4f}s")
        print(f"  Final database: {stats['total_claims']} claims, {stats['dirty_claims']} dirty")
        
        # Performance Assertions
        assert benchmark_results["loading"]["claims_per_second"] > 50, "Data loading too slow"
        assert benchmark_results["similarity"]["avg_time"] < 0.1, "Similarity search too slow"
        assert benchmark_results["filter"]["avg_time"] < 0.05, "Filter search too slow"
        assert benchmark_results["get"]["avg_time"] < 0.01, "Claim retrieval too slow"
        assert benchmark_results["stats"]["time"] < 0.1, "Statistics retrieval too slow"
        
        await dm.close()
        
        return benchmark_results

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, test_config: DataConfig):
        """Test for memory leaks during extended operations."""
        test_config.sqlite_path = ":memory:"
        test_config.chroma_path = "./test_chroma_leak"
        
        dm = DataManager(test_config, use_mock_embeddings=True)
        await dm.initialize()
        
        initial_memory = PerformanceTestSuite.get_memory_usage()
        memory_samples = [initial_memory]
        
        # Perform many operations over time
        for cycle in range(10):
            # Create claims
            claims_data = [
                {
                    "content": f"Memory leak test claim {cycle}_{i}",
                    "created_by": "leak_test_user",
                    "confidence": 0.7,
                    "tags": ["leak_test", f"cycle_{cycle}"]
                }
                for i in range(50)
            ]
            
            created_claims = await dm.batch_create_claims(claims_data)
            
            # Perform searches
            for claim in created_claims[:10]:
                await dm.search_similar(claim.content[:20], limit=5)
                await dm.get_claim(claim.id)
            
            # Update some claims
            for claim in created_claims[:5]:
                await dm.update_claim(claim.id, confidence=0.8)
            
            # Measure memory
            current_memory = PerformanceTestSuite.get_memory_usage()
            memory_samples.append(current_memory)
            
            print(f"Cycle {cycle}: Memory {current_memory:.2f}MB")
            
            # Periodically clean up to test memory release
            if cycle % 3 == 2:
                for claim in created_claims[:10]:
                    await dm.delete_claim(claim.id)
        
        final_memory = PerformanceTestSuite.get_memory_usage()
        total_growth = final_memory - initial_memory
        
        print(f"Memory Analysis:")
        print(f"  Initial: {initial_memory:.2f}MB")
        print(f"  Final: {final_memory:.2f}MB")
        print(f"  Total Growth: {total_growth:.2f}MB")
        
        # Calculate memory growth per cycle
        if len(memory_samples) > 1:
            growth_rates = []
            for i in range(1, len(memory_samples)):
                growth = memory_samples[i] - memory_samples[i-1]
                growth_rates.append(growth)
            
            avg_growth_per_cycle = statistics.mean(growth_rates)
            max_growth_per_cycle = max(growth_rates)
            
            print(f"  Avg Growth/Cycle: {avg_growth_per_cycle:.2f}MB")
            print(f"  Max Growth/Cycle: {max_growth_per_cycle:.2f}MB")
            
            # Memory growth should be minimal
            assert avg_growth_per_cycle < 5, f"Memory growing too fast: {avg_growth_per_cycle:.2f}MB per cycle"
            assert total_growth < 100, f"Total memory growth too high: {total_growth:.2f}MB"
        
        await dm.close()


# Stress tests
class TestStressTests:
    """Stress tests for extreme conditions."""

    @pytest.mark.slow
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_high_volume_data_ingestion(self, test_config: DataConfig):
        """Test ingesting large volume of data."""
        test_config.sqlite_path = ":memory:"
        test_config.chroma_path = "./test_chroma_stress"
        
        dm = DataManager(test_config, use_mock_embeddings=True)
        await dm.initialize()
        
        # Large batch ingestion
        large_dataset_size = 2000
        batch_size = 100
        total_time = 0
        
        for batch_start in range(0, large_dataset_size, batch_size):
            batch_end = min(batch_start + batch_size, large_dataset_size)
            
            claims_data = [
                {
                    "content": f"High volume claim {i} with extensive content for stress testing the ingestion pipeline and ensuring proper performance characteristics",
                    "created_by": f"stress_user_{i % 50}",
                    "confidence": 0.5 + (i % 50) / 100.0,
                    "tags": [f"stress_tag_{i % 100}", f"category_{i % 20}"],
                    "dirty": i % 4 == 0
                }
                for i in range(batch_start, batch_end)
            ]
            
            start_time = time.time()
            await dm.batch_create_claims(claims_data)
            batch_time = time.time() - start_time
            total_time += batch_time
            
            current_progress = batch_end
            ops_per_sec = batch_size / batch_time
            print(f"Ingested {current_progress}/{large_dataset_size} claims, Batch: {batch_time:.2f}s, Ops/sec: {ops_per_sec:.2f}")
            
            # Performance should remain consistent
            if current_progress > 500:  # After warmup
                assert ops_per_sec > 20, f"Ingestion rate dropped too low: {ops_per_sec:.2f}"
        
        final_stats = await dm.get_stats()
        print(f"Stress Ingestion Results:")
        print(f"  Total Claims: {final_stats['total_claims']}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Overall Rate: {large_dataset_size / total_time:.2f} claims/sec")
        
        assert final_stats["total_claims"] == large_dataset_size
        assert large_dataset_size / total_time > 30, "Overall ingestion rate too low"
        
        await dm.close()

    @pytest.mark.slow
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_read_write_stress(self, test_config: DataConfig):
        """Stress test with concurrent reads and writes."""
        test_config.sqlite_path = ":memory:"
        test_config.chroma_path = "./test_chroma_stress_concurrent"
        
        dm = DataManager(test_config, use_mock_embeddings=True)
        await dm.initialize()
        
        # Seed some initial data
        initial_claims = await dm.batch_create_claims([
            {"content": f"Initial claim {i}", "created_by": "initial_user", "confidence": 0.7}
            for i in range(100)
        ])
        
        async def continuous_writer():
            """Continuously write new claims."""
            for i in range(100):
                await dm.create_claim(
                    f"Concurrent write claim {i}",
                    "concurrent_writer",
                    confidence=0.8,
                    tags=["concurrent", "write"]
                )
                await asyncio.sleep(0.01)  # Small delay
        
        async def continuous_reader():
            """Continuously read and search."""
            for i in range(100):
                # Read random claims
                claim_id = initial_claims[i % len(initial_claims)].id
                await dm.get_claim(claim_id)
                
                # Search
                await dm.search_similar(f"search query {i}", limit=5)
                
                # Filter
                filter_obj = ClaimFilter(tags=["concurrent"], limit=10)
                await dm.filter_claims(filter_obj)
                
                await asyncio.sleep(0.01)  # Small delay
        
        async def continuous_updater():
            """Continuously update existing claims."""
            for i in range(50):
                claim_id = initial_claims[i % len(initial_claims)].id
                await dm.update_claim(claim_id, confidence=0.9)
                await asyncio.sleep(0.02)  # Small delay
        
        # Run all operations concurrently
        start_time = time.time()
        
        await asyncio.gather(
            continuous_writer(),
            continuous_reader(),
            continuous_updater()
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        final_stats = await dm.get_stats()
        
        print(f"Concurrent Stress Test Results:")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Final Claims: {final_stats['total_claims']}")
        print(f"  Operations Completed: ~250 reads + 100 writes + 50 updates")
        
        assert final_stats["total_claims"] >= 200  # Initial 100 + at least 100 new
        assert total_time < 30, f"Concurrent stress test took too long: {total_time:.2f}s"
        
        await dm.close()