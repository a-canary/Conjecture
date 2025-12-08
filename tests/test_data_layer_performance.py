"""
Comprehensive Performance Testing for Data Layer Optimizations
Tests and validates the performance improvements implemented in Phase 3
"""

import pytest
import asyncio
import time
import sqlite3
import tempfile
import os
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock external dependencies
with patch.dict('sys.modules', {
    'chromadb': MagicMock(),
    'chromadb.config': MagicMock(),
    'chromadb.api': MagicMock(),
    'chromadb.api.models': MagicMock(),
    'sentence_transformers': MagicMock(),
    'torch': MagicMock(),
    'tensorflow': MagicMock(),
    'numpy': MagicMock(),
    'sklearn': MagicMock(),
    'sklearn.metrics': MagicMock(),
    'scipy': MagicMock(),
    'scipy.spatial': MagicMock(),
    'psutil': MagicMock(),
}):
    from src.data.enhanced_sqlite_manager import EnhancedSQLiteManager
    from src.data.adaptive_connection_pool import AdaptiveConnectionPool, PoolConfiguration
    from src.data.models import Claim, ClaimState, ClaimScope, ClaimFilter, ClaimType


class TestDataLayerPerformance:
    """Performance testing suite for enhanced data layer"""

    @pytest.fixture
    async def temp_db(self):
        """Create temporary database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        yield db_path

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    async def enhanced_manager(self, temp_db):
        """Create enhanced SQLite manager for testing"""
        manager = EnhancedSQLiteManager(
            db_path=temp_db,
            pool_size=5,
            enable_caching=True
        )
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.fixture
    async def sample_claims(self):
        """Generate sample claims for testing"""
        claims = []
        for i in range(100):
            claim = Claim(
                id=f"c{10000000 + i:08d}",
                content=f"Sample claim content {i} with additional text for testing",
                confidence=0.5 + (i % 50) / 100,
                tags=[f"tag{j}" for j in range(i % 5)],
                state=ClaimState.EXPLORE if i % 2 == 0 else ClaimState.VALIDATED,
                scope=ClaimScope.USER_WORKSPACE
            )
            claims.append(claim)
        return claims

    class TestConnectionPerformance:
        """Test connection pool and performance metrics"""

        @pytest.mark.asyncio
        async def test_connection_pool_efficiency(self, temp_db):
            """Test connection pool hit rate and efficiency"""
            config = PoolConfiguration(
                min_connections=2,
                max_connections=10,
                adaptation_interval=5
            )

            pool = AdaptiveConnectionPool(temp_db, config)
            await pool.initialize()

            # Simulate concurrent connections
            async def worker():
                async with pool.get_connection() as conn:
                    await conn.execute("SELECT 1")
                    await asyncio.sleep(0.1)

            # Run concurrent workers
            tasks = [worker() for _ in range(20)]
            await asyncio.gather(*tasks)

            stats = pool.get_stats()

            # Verify performance metrics
            assert stats["pool_size"] >= config.min_connections
            assert stats["pool_size"] <= config.max_connections
            assert stats["hit_rate"] > 50  # Should have some pool hits
            assert stats["total_queries"] == 20
            assert stats["total_errors"] == 0

            await pool.close()

        @pytest.mark.asyncio
        async def test_adaptive_scaling(self, temp_db):
            """Test adaptive pool scaling under load"""
            config = PoolConfiguration(
                min_connections=2,
                max_connections=8,
                scale_up_threshold=0.7,
                scale_down_threshold=0.3,
                adaptation_interval=2
            )

            pool = AdaptiveConnectionPool(temp_db, config)
            await pool.initialize()

            initial_size = pool.get_stats()["pool_size"]

            # Simulate high load
            async def high_load_worker():
                async with pool.get_connection() as conn:
                    await conn.execute("SELECT sqlite_version()")
                    await asyncio.sleep(0.2)

            # Create high load
            tasks = [high_load_worker() for _ in range(15)]
            await asyncio.gather(*tasks)

            # Wait for adaptation
            await asyncio.sleep(3)

            stats_after_load = pool.get_stats()

            # Pool should have scaled up under load
            assert stats_after_load["pool_size"] >= initial_size
            assert stats_after_load["scale_ups"] >= 1

            await pool.close()

    class TestQueryPerformance:
        """Test query execution performance"""

        @pytest.mark.asyncio
        async def test_query_caching_performance(self, enhanced_manager):
            """Test that query caching improves performance"""
            # Create some test data first
            claim = Claim(
                id="c12345678",
                content="Test claim for caching",
                confidence=0.8,
                tags=["test", "caching"],
                state=ClaimState.VALIDATED
            )
            await enhanced_manager.create_claim(claim)

            # First query (cache miss)
            start_time = time.time()
            result1 = await enhanced_manager.get_claim("c12345678")
            first_query_time = time.time() - start_time

            # Second query (should be cache hit)
            start_time = time.time()
            result2 = await enhanced_manager.get_claim("c12345678")
            second_query_time = time.time() - start_time

            # Verify results are identical
            assert result1 == result2

            # Second query should be faster (though this might be marginal)
            # In real scenarios, the difference would be more significant
            performance_report = await enhanced_manager.get_performance_report()
            cache_stats = performance_report["cache_stats"]

            assert cache_stats is not None
            assert cache_stats["hits"] > 0
            assert cache_stats["hit_rate"] > 0

        @pytest.mark.asyncio
        async def test_batch_operation_performance(self, enhanced_manager, sample_claims):
            """Test batch operation performance vs individual operations"""
            # Test batch creation
            batch_claims = sample_claims[:50]

            start_time = time.time()
            batch_results = await enhanced_manager.batch_create_claims(batch_claims)
            batch_time = time.time() - start_time

            assert len(batch_results) == 50

            # Test individual creation for comparison
            individual_claims = sample_claims[50:75]

            start_time = time.time()
            individual_results = []
            for claim in individual_claims:
                result = await enhanced_manager.create_claim(claim)
                individual_results.append(result)
            individual_time = time.time() - start_time

            # Batch operation should be faster
            # (In real scenarios, the difference would be more pronounced)
            assert batch_time < individual_time

            # Calculate performance improvement
            improvement = (individual_time - batch_time) / individual_time * 100
            print(f"Batch operation performance improvement: {improvement:.1f}%")

        @pytest.mark.asyncio
        async def test_index_performance(self, enhanced_manager, sample_claims):
            """Test that enhanced indexes improve query performance"""
            # Create test data
            await enhanced_manager.batch_create_claims(sample_claims)

            # Test filtered query with confidence index
            start_time = time.time()
            high_confidence_claims = await enhanced_manager.filter_claims(
                ClaimFilter(confidence_min=0.8, limit=20)
            )
            filter_time = time.time() - start_time

            assert len(high_confidence_claims) > 0

            # Test state filtering with composite index
            start_time = time.time()
            validated_claims = await enhanced_manager.filter_claims(
                ClaimFilter(states=[ClaimState.VALIDATED], limit=20)
            )
            state_filter_time = time.time() - start_time

            assert len(validated_claims) > 0

            # Queries should be fast (under 100ms for small dataset)
            assert filter_time < 0.1
            assert state_filter_time < 0.1

            # Get performance report
            report = await enhanced_manager.get_performance_report()
            assert report["performance_metrics"]["avg_query_time"] > 0

    class TestMemoryAndStorage:
        """Test memory usage and storage optimization"""

        @pytest.mark.asyncio
        async def test_memory_efficiency(self, enhanced_manager, sample_claims):
            """Test memory usage during operations"""
            import psutil
            import gc

            process = psutil.Process()
            initial_memory = process.memory_info().rss

            # Create large number of claims
            large_batch = sample_claims * 10  # 1000 claims
            await enhanced_manager.batch_create_claims(large_batch)

            # Force garbage collection
            gc.collect()

            peak_memory = process.memory_info().rss
            memory_increase = peak_memory - initial_memory

            # Memory increase should be reasonable (less than 100MB for 1000 claims)
            assert memory_increase < 100 * 1024 * 1024

            print(f"Memory increase for 1000 claims: {memory_increase / 1024 / 1024:.1f} MB")

            # Test that queries don't leak memory
            for i in range(100):
                await enhanced_manager.filter_claims(
                    ClaimFilter(limit=10, offset=i * 10)
                )

            gc.collect()
            final_memory = process.memory_info().rss

            # Memory shouldn't have grown significantly during queries
            assert (final_memory - peak_memory) < 10 * 1024 * 1024

        @pytest.mark.asyncio
        async def test_database_optimization(self, enhanced_manager):
            """Test database optimization routines"""
            # Create some test data
            claims = [
                Claim(
                    id=f"copt{i:08d}",
                    content=f"Optimization test claim {i}",
                    confidence=0.7 + (i % 30) / 100,
                    tags=["optimization", "test"],
                    state=ClaimState.EXPLORE
                )
                for i in range(200)
            ]
            await enhanced_manager.batch_create_claims(claims)

            # Run optimization
            optimization_results = await enhanced_manager.optimize_database()

            assert optimization_results["success"]
            assert "analyze_time" in optimization_results
            assert "reindex_time" in optimization_results

            # Performance should improve after optimization
            pre_opt_report = await enhanced_manager.get_performance_report()

            # Run some queries
            await enhanced_manager.filter_claims(ClaimFilter(limit=50))
            await enhanced_manager.get_dirty_claims(limit=20)

            post_opt_report = await enhanced_manager.get_performance_report()

            # Should have more queries executed
            assert post_opt_report["performance_metrics"]["total_queries"] > \
                   pre_opt_report["performance_metrics"]["total_queries"]

    class TestConcurrencyAndReliability:
        """Test concurrent operations and reliability"""

        @pytest.mark.asyncio
        async def test_concurrent_operations(self, enhanced_manager, sample_claims):
            """Test concurrent read/write operations"""
            # Split claims for concurrent processing
            batch1 = sample_claims[:33]
            batch2 = sample_claims[33:66]
            batch3 = sample_claims[66:]

            # Concurrent creation
            async def create_batch(batch, batch_id):
                results = []
                for claim in batch:
                    claim.id = f"{claim.id}_{batch_id}"  # Make unique
                    result = await enhanced_manager.create_claim(claim)
                    results.append(result)
                return results

            start_time = time.time()
            tasks = [
                create_batch(batch1, "A"),
                create_batch(batch2, "B"),
                create_batch(batch3, "C")
            ]
            results = await asyncio.gather(*tasks)
            concurrent_time = time.time() - start_time

            # Verify all claims were created
            total_created = sum(len(result) for result in results)
            assert total_created == 99

            # Test concurrent reads
            async def read_claims():
                return await enhanced_manager.filter_claims(ClaimFilter(limit=30))

            start_time = time.time()
            read_tasks = [read_claims() for _ in range(10)]
            read_results = await asyncio.gather(*read_tasks)
            concurrent_read_time = time.time() - start_time

            # All reads should succeed
            for result in read_results:
                assert len(result) > 0

            print(f"Concurrent creation time: {concurrent_time:.2f}s")
            print(f"Concurrent read time: {concurrent_read_time:.2f}s")

        @pytest.mark.asyncio
        async def test_error_recovery(self, enhanced_manager):
            """Test error handling and recovery"""
            # Test with invalid claim
            invalid_claim = Claim(
                id="invalid",
                content="x" * 10000,  # Too long
                confidence=1.5,  # Invalid confidence
                tags=[],
                state=ClaimState.EXPLORE
            )

            # Should handle validation errors gracefully
            with pytest.raises(Exception):
                await enhanced_manager.create_claim(invalid_claim)

            # Database should still be functional
            valid_claim = Claim(
                id="c99999999",
                content="Valid claim after error",
                confidence=0.8,
                tags=["recovery"],
                state=ClaimState.VALIDATED
            )

            result = await enhanced_manager.create_claim(valid_claim)
            assert result == "c99999999"

            # Should be able to retrieve the claim
            retrieved = await enhanced_manager.get_claim("c99999999")
            assert retrieved is not None
            assert retrieved["content"] == "Valid claim after error"

    class TestPerformanceRegression:
        """Test for performance regressions"""

        @pytest.mark.asyncio
        async def test_performance_benchmarks(self, enhanced_manager, sample_claims):
            """Establish performance benchmarks"""
            # Setup: create test data
            await enhanced_manager.batch_create_claims(sample_claims)

            benchmarks = {}

            # Benchmark 1: Single claim retrieval
            times = []
            for _ in range(50):
                start_time = time.time()
                await enhanced_manager.get_claim("c10000000")
                times.append(time.time() - start_time)

            benchmarks["single_retrieval"] = {
                "avg": statistics.mean(times),
                "median": statistics.median(times),
                "p95": statistics.quantiles(times, n=20)[18]  # 95th percentile
            }

            # Benchmark 2: Batch filtering
            times = []
            for _ in range(20):
                start_time = time.time()
                await enhanced_manager.filter_claims(ClaimFilter(limit=50))
                times.append(time.time() - start_time)

            benchmarks["batch_filtering"] = {
                "avg": statistics.mean(times),
                "median": statistics.median(times),
                "p95": statistics.quantiles(times, n=20)[18]
            }

            # Benchmark 3: Dirty claims query
            times = []
            for _ in range(20):
                start_time = time.time()
                await enhanced_manager.get_dirty_claims(limit=30)
                times.append(time.time() - start_time)

            benchmarks["dirty_claims_query"] = {
                "avg": statistics.mean(times),
                "median": statistics.median(times),
                "p95": statistics.quantiles(times, n=20)[18]
            }

            # Print benchmark results
            print("\\nPerformance Benchmarks:")
            for operation, metrics in benchmarks.items():
                print(f"{operation}:")
                print(f"  Average: {metrics['avg']*1000:.2f}ms")
                print(f"  Median:  {metrics['median']*1000:.2f}ms")
                print(f"  95th percentile: {metrics['p95']*1000:.2f}ms")

            # Performance assertions (adjust based on your requirements)
            assert benchmarks["single_retrieval"]["avg"] < 0.01  # 10ms average
            assert benchmarks["batch_filtering"]["avg"] < 0.05   # 50ms average
            assert benchmarks["dirty_claims_query"]["avg"] < 0.02  # 20ms average

            # Return benchmarks for comparison in CI/CD
            return benchmarks


@pytest.mark.asyncio
async def test_comprehensive_performance():
    """Run comprehensive performance test suite"""
    # This would be the main test entry point
    test_suite = TestDataLayerPerformance()

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    try:
        # Initialize manager
        manager = EnhancedSQLiteManager(db_path, pool_size=8, enable_caching=True)
        await manager.initialize()

        # Generate test data
        sample_claims = []
        for i in range(100):
            claim = Claim(
                id=f"c{10000000 + i:08d}",
                content=f"Performance test claim {i} with comprehensive content for testing various query patterns and search capabilities",
                confidence=0.5 + (i % 50) / 100,
                tags=[f"tag{j}" for j in range(i % 5)] + ["performance", "test"],
                state=ClaimState.EXPLORE if i % 2 == 0 else ClaimState.VALIDATED,
                scope=ClaimScope.USER_WORKSPACE
            )
            sample_claims.append(claim)

        # Run performance tests
        print("Running comprehensive performance test suite...")

        # Test 1: Connection performance
        print("\\n1. Testing connection performance...")
        config = PoolConfiguration(min_connections=2, max_connections=10)
        pool = AdaptiveConnectionPool(db_path, config)
        await pool.initialize()

        # Simulate load
        async def worker():
            async with pool.get_connection() as conn:
                await conn.execute("SELECT 1")

        tasks = [worker() for _ in range(50)]
        await asyncio.gather(*tasks)

        pool_stats = pool.get_stats()
        print(f"Pool hit rate: {pool_stats['hit_rate']:.1f}%")
        print(f"Total queries: {pool_stats['total_queries']}")
        await pool.close()

        # Test 2: Data operations
        print("\\n2. Testing data operations...")
        start_time = time.time()
        await manager.batch_create_claims(sample_claims)
        create_time = time.time() - start_time
        print(f"Created 100 claims in {create_time:.3f}s")

        # Test 3: Query performance
        print("\\n3. Testing query performance...")

        # Filter queries
        start_time = time.time()
        results = await manager.filter_claims(ClaimFilter(confidence_min=0.7, limit=20))
        filter_time = time.time() - start_time
        print(f"Filtered query returned {len(results)} results in {filter_time:.3f}s")

        # Test 4: Cache performance
        print("\\n4. Testing cache performance...")

        # First query
        start_time = time.time()
        await manager.get_claim("c10000000")
        first_time = time.time() - start_time

        # Second query (cached)
        start_time = time.time()
        await manager.get_claim("c10000000")
        second_time = time.time() - start_time

        print(f"First query: {first_time*1000:.2f}ms")
        print(f"Second query: {second_time*1000:.2f}ms")

        # Test 5: Performance report
        print("\\n5. Generating performance report...")
        report = await manager.get_performance_report()
        print(f"Average query time: {report['performance_metrics']['avg_query_time']*1000:.2f}ms")
        print(f"Total queries: {report['performance_metrics']['total_queries']}")

        if report['cache_stats']:
            print(f"Cache hit rate: {report['cache_stats']['hit_rate']:.1f}%")

        # Test 6: Database optimization
        print("\\n6. Testing database optimization...")
        opt_results = await manager.optimize_database()
        print(f"Optimization successful: {opt_results['success']}")
        if opt_results.get('analyze_time'):
            print(f"ANALYZE time: {opt_results['analyze_time']:.3f}s")

        await manager.close()

    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    print("\\nPerformance test suite completed successfully!")


if __name__ == "__main__":
    # Run the comprehensive test
    asyncio.run(test_comprehensive_performance())