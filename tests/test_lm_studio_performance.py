"""
Performance and Load Tests for LM Studio Integration
Tests the stability and performance of the LM Studio integration
"""

import unittest
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "./src")

from conjecture import Conjecture


class TestLMStudioPerformance(unittest.TestCase):
    """Performance tests for LM Studio integration"""

    @classmethod
    def setUpClass(cls):
        """Set up environment for performance tests"""
        # Set environment to use LM Studio
        os.environ["Conjecture_LLM_PROVIDER"] = "lm_studio"
        os.environ["Conjecture_LLM_API_URL"] = "http://127.0.0.1:1234"
        os.environ["Conjecture_LLM_MODEL"] = "ibm/granite-4-h-tiny"

        cls.conjecture = Conjecture()

    def test_response_time(self):
        """Test that response times are within acceptable limits"""
        start_time = time.time()

        # Perform exploration
        result = self.conjecture.explore("simple topic", max_claims=1)

        end_time = time.time()
        response_time = end_time - start_time

        # Check that response time is reasonable (under 30 seconds for simple query)
        # Note: LM Studio response time can vary based on hardware
        self.assertLess(
            response_time,
            30.0,
            f"Response time {response_time:.2f}s exceeded 30 seconds",
        )

        print(f"Response time: {response_time:.2f}s for exploration")

    def test_multiple_requests(self):
        """Test handling multiple sequential requests"""
        start_time = time.time()

        # Make multiple exploration requests
        for i in range(3):
            result = self.conjecture.explore(f"topic {i}", max_claims=1)
            self.assertIsNotNone(result)

        total_time = time.time() - start_time

        # Verify all requests completed successfully
        print(f"Completed 3 requests in {total_time:.2f}s")
        self.assertLess(total_time, 60.0)  # Should complete within 60 seconds

    def test_concurrent_requests(self):
        """Test handling concurrent requests"""

        def make_request(query):
            """Helper function to make a request"""
            return self.conjecture.explore(query, max_claims=1)

        # Create multiple concurrent requests
        queries = [f"concurrent topic {i}" for i in range(3)]

        start_time = time.time()

        # Execute requests concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request, query) for query in queries]
            results = [future.result() for future in futures]

        total_time = time.time() - start_time

        # Verify all requests completed successfully
        self.assertEqual(len(results), len(queries))
        for result in results:
            self.assertIsNotNone(result)

        print(f"Completed {len(queries)} concurrent requests in {total_time:.2f}s")
        self.assertLess(total_time, 60.0)  # Should complete within 60 seconds

    def test_memory_usage_stability(self):
        """Test that memory usage remains stable over multiple operations"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform multiple operations
        for i in range(5):
            result = self.conjecture.explore(f"memory test {i}", max_claims=1)
            self.assertIsNotNone(result)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(
            memory_increase,
            100.0,
            f"Memory increase {memory_increase:.2f}MB exceeded 100MB threshold",
        )

        print(
            f"Memory usage: initial={initial_memory:.2f}MB, final={final_memory:.2f}MB, "
            f"increase={memory_increase:.2f}MB"
        )


def run_performance_tests():
    """Run the performance tests for LM Studio integration"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLMStudioPerformance)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("Running performance tests for LM Studio integration...")
    print(
        "Ensure LM Studio is running at http://127.0.0.1:1234 with ibm/granite-4-h-tiny model"
    )
    run_performance_tests()
