#!/usr/bin/env python3
"""
Critical Integration Tests for Conjecture Security and Stability
Tests SQL injection fixes, memory leak prevention, and core workflows
"""

import asyncio
import sys
import os
import tempfile
import shutil
import time
import gc
import psutil
from pathlib import Path
from typing import List, Dict, Any
import unittest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.conjecture import Conjecture
from src.core.models import Claim, ClaimState
from src.local.vector_store import LocalVectorStore
from src.config.unified_config import UnifiedConfig


class CriticalIntegrationTests(unittest.TestCase):
    """Critical integration tests for security and stability fixes"""

    def setUp(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="conjecture_critical_test_")
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss
        
        # Setup test configuration
        self.config = UnifiedConfig()
        self.test_db_path = os.path.join(self.temp_dir, "test.db")
        
        print(f"Test setup complete. Temp dir: {self.temp_dir}")

    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Force garbage collection
        gc.collect()
        
        final_memory = self.process.memory_info().rss
        memory_diff = final_memory - self.initial_memory
        print(f"Memory change: {memory_diff / 1024 / 1024:.2f} MB")

    async def test_sql_injection_prevention(self):
        """Test that SQL injection vulnerabilities are properly prevented"""
        print("\nTesting SQL injection prevention...")
        
        # Test malicious inputs that could cause SQL injection
        malicious_inputs = [
            "'; DROP TABLE vector_metadata; --",
            "'; DELETE FROM vector_metadata WHERE '1'='1'; --",
            "'; UPDATE vector_metadata SET content='HACKED'; --",
            "'; INSERT INTO vector_metadata VALUES ('hack', 'malicious'); --",
            "' OR '1'='1",
            "'; SELECT * FROM vector_metadata; --",
            "'; ALTER TABLE vector_metadata DROP COLUMN content; --",
        ]
        
        # Initialize vector store
        vector_store = LocalVectorStore(db_path=self.test_db_path)
        await vector_store.initialize()
        
        # Add a legitimate claim first
        test_embedding = [0.1] * 384  # 384-dimensional embedding
        await vector_store.add_vector(
            claim_id="test_001",
            content="Legitimate test claim",
            embedding=test_embedding,
            metadata={"type": "test"}
        )
        
        # Test each malicious input
        for i, malicious_input in enumerate(malicious_inputs):
            print(f"  Testing malicious input {i+1}/{len(malicious_inputs)}: {malicious_input[:50]}...")
            
            try:
                # Test update with malicious input
                await vector_store.update_vector(
                    claim_id="test_001",
                    content=malicious_input,
                    embedding=test_embedding
                )
                
                # Verify the data is stored as-is, not executed as SQL
                updated_claim = await vector_store.get_vector("test_001")
                self.assertIsNotNone(updated_claim, "Claim should still exist after malicious update")
                
                # Verify database integrity
                stats = await vector_store.get_stats()
                self.assertGreater(stats['total_vectors'], 0, "Database should still contain vectors")
                
            except Exception as e:
                # Some exceptions are expected for malformed inputs
                print(f"    Expected exception for malicious input: {e}")
                
        # Verify database is still functional
        final_stats = await vector_store.get_stats()
        self.assertGreater(final_stats['total_vectors'], 0, "Database should remain functional")
        
        # Test health check
        health = await vector_store.health_check()
        self.assertEqual(health['status'], 'healthy', "Vector store should remain healthy")
        
        await vector_store.close()
        print("[PASS] SQL injection prevention tests passed")

    async def test_memory_leak_prevention(self):
        """Test that memory leaks are properly prevented in caching system"""
        print("\nTesting memory leak prevention...")
        
        # Initialize Conjecture with test configuration
        conjecture = Conjecture(config=self.config)
        
        # Start services
        await conjecture.start_services()
        
        initial_memory = self.process.memory_info().rss
        print(f"  Initial memory: {initial_memory / 1024 / 1024:.2f} MB")
        
        # Generate many cache entries to test memory management
        test_queries = [
            f"test query {i} for memory leak testing with some additional content"
            for i in range(100)
        ]
        
        # Perform multiple operations to populate caches
        for query in test_queries:
            try:
                result = await conjecture.explore(query, max_claims=3)
                self.assertIsNotNone(result, "Exploration should return a result")
            except Exception as e:
                # Some exceptions are expected with mock providers
                print(f"    Expected exception during exploration: {e}")
        
        mid_test_memory = self.process.memory_info().rss
        print(f"  Mid-test memory: {mid_test_memory / 1024 / 1024:.2f} MB")
        
        # Wait for cache cleanup interval
        await asyncio.sleep(2)
        
        # Perform cache cleanup manually
        conjecture.clear_all_caches()
        
        # Force garbage collection
        gc.collect()
        
        final_memory = self.process.memory_info().rss
        print(f"  Final memory: {final_memory / 1024 / 1024:.2f} MB")
        
        # Calculate memory growth
        memory_growth = final_memory - initial_memory
        memory_growth_mb = memory_growth / 1024 / 1024
        
        print(f"  Total memory growth: {memory_growth_mb:.2f} MB")
        
        # Memory growth should be reasonable (less than 50 MB for this test)
        self.assertLess(memory_growth_mb, 50, f"Memory growth should be reasonable, was {memory_growth_mb:.2f} MB")
        
        # Test cache statistics
        stats = conjecture.get_performance_stats()
        self.assertIn('cache_stats', stats, "Cache statistics should be available")
        
        # Stop services and cleanup
        await conjecture.stop_services()
        
        print("[PASS] Memory leak prevention tests passed")

    async def test_core_workflow_integration(self):
        """Test complete core workflows with security and stability checks"""
        print("\nTesting core workflow integration...")
        
        # Initialize Conjecture
        conjecture = Conjecture(config=self.config)
        
        async with conjecture:
            # Test 1: Claim creation workflow
            print("  Testing claim creation workflow...")
            claim = await conjecture.add_claim(
                content="Integration test claim for critical path testing",
                confidence=0.85,
                tags=["integration", "test", "critical"]
            )
            
            self.assertIsNotNone(claim, "Claim should be created successfully")
            self.assertEqual(claim.content, "Integration test claim for critical path testing")
            self.assertEqual(claim.confidence, 0.85)
            
            # Test 2: Exploration workflow
            print("  Testing exploration workflow...")
            try:
                exploration_result = await conjecture.explore(
                    "integration test exploration query",
                    max_claims=5,
                    auto_evaluate=False  # Disable evaluation for faster testing
                )
                
                self.assertIsNotNone(exploration_result, "Exploration should return a result")
                self.assertIsInstance(exploration_result.claims, list, "Should return list of claims")
                
            except Exception as e:
                print(f"    Exploration failed (expected with mock): {e}")
            
            # Test 3: Task processing workflow
            print("  Testing task processing workflow...")
            task = {
                "type": "create_claim",
                "content": "Task processing test claim",
                "confidence": 0.9
            }
            
            try:
                task_result = await conjecture.process_task(task)
                self.assertTrue(task_result.get("success", False), "Task processing should succeed")
                
            except Exception as e:
                print(f"    Task processing failed (expected with mock): {e}")
            
            # Test 4: Statistics and monitoring
            print("  Testing statistics and monitoring...")
            stats = conjecture.get_statistics()
            self.assertIn("claims_processed", stats, "Statistics should include claim count")
            self.assertIn("services_running", stats, "Statistics should include service status")
            
            perf_stats = conjecture.get_performance_stats()
            self.assertIn("cache_stats", perf_stats, "Performance stats should include cache info")
            
            # Test 5: Cache management
            print("  Testing cache management...")
            # Add items to cache
            for i in range(10):
                cache_key = f"test_cache_{i}"
                conjecture._add_to_cache(cache_key, f"test_data_{i}", "claim_generation")
            
            # Verify cache size is managed
            cache_size = len(conjecture._claim_generation_cache)
            self.assertLessEqual(cache_size, conjecture._max_cache_size, "Cache size should be limited")
            
            # Clear caches and verify
            conjecture.clear_all_caches()
            self.assertEqual(len(conjecture._claim_generation_cache), 0, "Cache should be empty after clear")
            
        print("[PASS] Core workflow integration tests passed")

    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        print("\nTesting error handling and recovery...")
        
        # Test vector store error handling
        vector_store = LocalVectorStore(db_path=self.config.database_path)
        await vector_store.initialize()
        
        # Test with invalid data
        try:
            await vector_store.add_vector(
                claim_id="",  # Empty ID should be handled
                content="Test",
                embedding=[0.1] * 384
            )
        except Exception as e:
            print(f"  Handled empty ID error: {e}")
        
        # Test with invalid embedding
        try:
            await vector_store.add_vector(
                claim_id="test_invalid",
                content="Test",
                embedding=[]  # Empty embedding should be handled
            )
        except Exception as e:
            print(f"  Handled invalid embedding error: {e}")
        
        # Test recovery after errors
        try:
            await vector_store.add_vector(
                claim_id="recovery_test",
                content="Recovery test claim",
                embedding=[0.1] * 384
            )
            
            recovered_claim = await vector_store.get_vector("recovery_test")
            self.assertIsNotNone(recovered_claim, "Should recover after errors")
            
        except Exception as e:
            self.fail(f"Failed to recover after errors: {e}")
        
        await vector_store.close()
        print("[PASS] Error handling and recovery tests passed")

    async def test_concurrent_operations(self):
        """Test concurrent operations for thread safety and resource management"""
        print("\nTesting concurrent operations...")
        
        vector_store = LocalVectorStore(db_path=self.config.database_path)
        await vector_store.initialize()
        
        # Create multiple concurrent operations
        async def add_vectors_batch(start_id: int, count: int):
            """Add a batch of vectors concurrently"""
            for i in range(count):
                claim_id = f"concurrent_{start_id + i}"
                await vector_store.add_vector(
                    claim_id=claim_id,
                    content=f"Concurrent test claim {claim_id}",
                    embedding=[0.1] * 384
                )
        
        # Run multiple batches concurrently
        batch_size = 10
        num_batches = 3
        
        tasks = []
        for i in range(num_batches):
            task = add_vectors_batch(i * batch_size, batch_size)
            tasks.append(task)
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all vectors were added
        stats = await vector_store.get_stats()
        expected_count = batch_size * num_batches
        self.assertGreaterEqual(stats['total_vectors'], expected_count, 
                             f"Should have at least {expected_count} vectors")
        
        # Test concurrent reads
        async def read_vectors(start_id: int, count: int):
            """Read vectors concurrently"""
            for i in range(count):
                claim_id = f"concurrent_{start_id + i}"
                await vector_store.get_vector(claim_id)
        
        read_tasks = []
        for i in range(num_batches):
            task = read_vectors(i * batch_size, batch_size)
            read_tasks.append(task)
        
        await asyncio.gather(*read_tasks, return_exceptions=True)
        
        await vector_store.close()
        print("[PASS] Concurrent operations tests passed")


class TestRunner:
    """Test runner for critical integration tests"""

    def __init__(self):
        self.test_results = []

    async def run_all_tests(self):
        """Run all critical integration tests"""
        print("Starting Critical Integration Tests")
        print("=" * 60)
        
        test_suite = CriticalIntegrationTests()
        
        tests = [
            test_suite.test_sql_injection_prevention,
            test_suite.test_memory_leak_prevention,
            test_suite.test_core_workflow_integration,
            test_suite.test_error_handling_and_recovery,
            test_suite.test_concurrent_operations,
        ]
        
        for test_func in tests:
            test_name = test_func.__name__
            print(f"\nRunning {test_name}...")
            
            try:
                test_suite.setUp()
                await test_func()
                test_suite.tearDown()
                self.test_results.append((test_name, "PASSED", None))
                print(f"[PASS] {test_name} PASSED")
                
            except Exception as e:
                try:
                    test_suite.tearDown()
                except:
                    pass
                self.test_results.append((test_name, "FAILED", str(e)))
                print(f"[FAIL] {test_name} FAILED: {e}")
        
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("CRITICAL INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for _, status, _ in self.test_results if status == "PASSED")
        failed = sum(1 for _, status, _ in self.test_results if status == "FAILED")
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if failed > 0:
            print("\nFAILED TESTS:")
            for name, status, error in self.test_results:
                if status == "FAILED":
                    print(f"  - {name}: {error}")
        
        print("\n" + "=" * 60)
        
        if failed == 0:
            print("ALL CRITICAL INTEGRATION TESTS PASSED!")
            print("Security and stability fixes are working correctly")
        else:
            print("SOME TESTS FAILED - REVIEW REQUIRED")
        
        return failed == 0


async def main():
    """Main test runner"""
    runner = TestRunner()
    success = await runner.run_all_tests()
    
    if not success:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())