"""
Performance Benchmarks for Conjecture System
Tests performance characteristics and optimization opportunities
"""

import asyncio
import time
import sys
import os
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import psutil
import tracemalloc

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from core.unified_models import Claim, ClaimType, ClaimState


class PerformanceBenchmark:
    """Performance testing framework for Conjecture components"""

    def __init__(self):
        self.results = {}
        self.process = psutil.Process()

    def benchmark_claim_creation(self, num_claims: int = 1000) -> Dict[str, Any]:
        """Benchmark claim creation performance"""
        print(f"Benchmarking claim creation ({num_claims} claims)...")

        # Memory tracking
        tracemalloc.start()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Time tracking
        start_time = time.time()

        claims = []
        for i in range(num_claims):
            claim = Claim(
                id=f"perf_test_{i}",
                content=f"Performance test claim number {i} with sufficient content length",
                confidence=0.8 + (i % 20) / 100,  # Vary confidence
                type=[ClaimType.CONCEPT],
                tags=[f"tag_{i % 10}", "performance", "test"],
            )
            claims.append(claim)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        claims_per_second = num_claims / execution_time

        result = {
            "operation": "claim_creation",
            "num_claims": num_claims,
            "execution_time": execution_time,
            "claims_per_second": claims_per_second,
            "memory_used_mb": memory_used,
            "peak_memory_kb": peak / 1024,
            "avg_claim_size_bytes": current / num_claims if num_claims > 0 else 0,
        }

        self.results["claim_creation"] = result
        print(
            f"Created {num_claims} claims in {execution_time:.3f}s ({claims_per_second:.1f} claims/sec)"
        )
        return result

    def benchmark_claim_relationships(self, num_claims: int = 500) -> Dict[str, Any]:
        """Benchmark claim relationship operations"""
        print(f"Benchmarking claim relationships ({num_claims} claims)...")

        # Create claims
        claims = []
        for i in range(num_claims):
            claim = Claim(
                id=f"rel_test_{i}",
                content=f"Relationship test claim {i}",
                confidence=0.8,
                type=[ClaimType.CONCEPT],
                tags=["relationship", "test"],
            )
            claims.append(claim)

        # Benchmark relationship operations
        start_time = time.time()

        # Add supports relationships
        for i, claim in enumerate(claims):
            if i > 0:
                claim.add_support(claims[i - 1].id)
            if i < len(claims) - 1:
                claim.add_supports(claims[i + 1].id)

        # Update confidence scores
        for claim in claims:
            claim.update_confidence(0.9)

        end_time = time.time()
        execution_time = end_time - start_time
        operations_per_second = (
            (num_claims * 3) / execution_time if execution_time > 0 else 0
        )  # 3 operations per claim

        result = {
            "operation": "claim_relationships",
            "num_claims": num_claims,
            "execution_time": execution_time,
            "operations_per_second": operations_per_second,
            "total_relationships": num_claims * 2,  # supports + supported_by
        }

        self.results["claim_relationships"] = result
        print(
            f"Processed {num_claims * 3} relationship operations in {execution_time:.3f}s ({operations_per_second:.1f} ops/sec)"
        )
        return result

    def benchmark_context_building(self, context_size: int = 100) -> Dict[str, Any]:
        """Benchmark context building performance"""
        print(f"Benchmarking context building ({context_size} claims)...")

        # Create context claims
        context_claims = []
        for i in range(context_size):
            claim = Claim(
                id=f"context_{i}",
                content=f"Context claim {i} about machine learning algorithms and data processing",
                confidence=0.7 + (i % 30) / 100,
                type=[ClaimType.CONCEPT if i % 2 == 0 else ClaimType.REFERENCE],
                tags=["context", "ml", "data", f"topic_{i % 5}"],
            )
            context_claims.append(claim)

        # Benchmark context formatting
        start_time = time.time()

        formatted_contexts = []
        for claim in context_claims:
            formatted = claim.format_for_context()
            formatted_contexts.append(formatted)

        end_time = time.time()
        execution_time = end_time - start_time
        contexts_per_second = context_size / execution_time if execution_time > 0 else 0

        # Calculate context size
        total_chars = sum(len(ctx) for ctx in formatted_contexts)
        avg_context_size = total_chars / context_size if context_size > 0 else 0

        result = {
            "operation": "context_building",
            "context_size": context_size,
            "execution_time": execution_time,
            "contexts_per_second": contexts_per_second,
            "total_context_chars": total_chars,
            "avg_context_size": avg_context_size,
        }

        self.results["context_building"] = result
        print(
            f"Built {context_size} contexts in {execution_time:.3f}s ({contexts_per_second:.1f} contexts/sec)"
        )
        return result

    def benchmark_concurrent_operations(
        self, num_threads: int = 4, operations_per_thread: int = 250
    ) -> Dict[str, Any]:
        """Benchmark concurrent claim operations"""
        print(
            f"Benchmarking concurrent operations ({num_threads} threads, {operations_per_thread} ops/thread)..."
        )

        def worker_thread(thread_id: int, num_ops: int) -> Dict[str, Any]:
            """Worker thread function"""
            thread_start = time.time()
            claims = []

            for i in range(num_ops):
                claim = Claim(
                    id=f"thread_{thread_id}_claim_{i}",
                    content=f"Concurrent operation claim from thread {thread_id}, operation {i}",
                    confidence=0.8,
                    type=[ClaimType.CONCEPT],
                    tags=["concurrent", f"thread_{thread_id}"],
                )
                claims.append(claim)

                # Simulate some processing
                claim.update_confidence(0.85)

            thread_end = time.time()
            return {
                "thread_id": thread_id,
                "operations": num_ops,
                "time": thread_end - thread_start,
                "claims": claims,
            }

        # Run concurrent operations
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(worker_thread, i, operations_per_thread)
                for i in range(num_threads)
            ]
            results = [future.result() for future in futures]

        end_time = time.time()
        total_time = end_time - start_time
        total_operations = num_threads * operations_per_thread

        # Calculate statistics
        thread_times = [result["time"] for result in results]
        avg_thread_time = statistics.mean(thread_times)
        max_thread_time = max(thread_times)
        min_thread_time = min(thread_times)

        ops_per_second = total_operations / total_time if total_time > 0 else 0

        result = {
            "operation": "concurrent_operations",
            "num_threads": num_threads,
            "operations_per_thread": operations_per_thread,
            "total_operations": total_operations,
            "total_time": total_time,
            "ops_per_second": ops_per_second,
            "avg_thread_time": avg_thread_time,
            "max_thread_time": max_thread_time,
            "min_thread_time": min_thread_time,
            "thread_efficiency": avg_thread_time / max_thread_time
            if max_thread_time > 0
            else 1.0,
        }

        self.results["concurrent_operations"] = result
        print(
            f"Completed {total_operations} concurrent operations in {total_time:.3f}s ({ops_per_second:.1f} ops/sec)"
        )
        return result

    def benchmark_memory_usage(
        self, claim_batches: List[int] = [100, 500, 1000, 5000]
    ) -> Dict[str, Any]:
        """Benchmark memory usage scaling"""
        print("Benchmarking memory usage scaling...")

        memory_results = []

        for batch_size in claim_batches:
            # Force garbage collection
            import gc

            gc.collect()

            # Measure baseline memory
            baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Create claims
            start_time = time.time()
            claims = []

            for i in range(batch_size):
                claim = Claim(
                    id=f"memory_test_{batch_size}_{i}",
                    content=f"Memory test claim {i} with additional content to increase memory usage per claim",
                    confidence=0.8,
                    type=[ClaimType.CONCEPT],
                    tags=["memory", "test", f"batch_{batch_size}"] * 3,  # More tags
                )
                claims.append(claim)

            end_time = time.time()
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024

            memory_used = peak_memory - baseline_memory
            memory_per_claim = memory_used / batch_size if batch_size > 0 else 0
            creation_time = end_time - start_time

            memory_results.append(
                {
                    "batch_size": batch_size,
                    "memory_used_mb": memory_used,
                    "memory_per_claim_kb": memory_per_claim * 1024,
                    "creation_time": creation_time,
                    "claims_per_second": batch_size / creation_time
                    if creation_time > 0
                    else 0,
                }
            )

            print(
                f"  Batch {batch_size}: {memory_used:.2f}MB total, {memory_per_claim * 1024:.2f}KB per claim"
            )

        # Calculate scaling efficiency
        scaling_efficiency = []
        for i in range(1, len(memory_results)):
            prev = memory_results[i - 1]
            curr = memory_results[i]
            size_ratio = curr["batch_size"] / prev["batch_size"]
            if prev["memory_used_mb"] > 0:
            if prev["memory_used_mb"] > 0:
            if prev["memory_used_mb"] > 0:
            if prev["memory_used_mb"] > 0:
            if prev["memory_used_mb"] > 0:
            if prev["memory_used_mb"] > 0:
            if prev["memory_used_mb"] > 0:
                memory_ratio = curr["memory_used_mb"] / prev["memory_used_mb"]
                efficiency = memory_ratio / size_ratio
                scaling_efficiency.append(efficiency)
            else:
                # If previous memory was zero, assume linear scaling
                scaling_efficiency.append(1.0)
            else:
                # If previous memory was zero, assume linear scaling
                scaling_efficiency.append(1.0)
            else:
                # If previous memory was zero, assume linear scaling
                scaling_efficiency.append(1.0)
            else:
                # If previous memory was zero, assume linear scaling
                scaling_efficiency.append(1.0)
            else:
                # If previous memory was zero, assume linear scaling
                scaling_efficiency.append(1.0)
            else:
                # If previous memory was zero, assume linear scaling
                scaling_efficiency.append(1.0)
            else:
                # If previous memory was zero, assume linear scaling
                scaling_efficiency.append(1.0)

        avg_scaling_efficiency = (
            statistics.mean(scaling_efficiency) if scaling_efficiency else 1.0
        )

        result = {
            "operation": "memory_usage",
            "results": memory_results,
            "avg_scaling_efficiency": avg_scaling_efficiency,
            "total_claims_tested": sum(batch_size for batch_size in claim_batches),
        }

        self.results["memory_usage"] = result
        print(f"Memory scaling efficiency: {avg_scaling_efficiency:.2f} (1.0 = linear)")
        return result

    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("=" * 60)
        report.append("CONJECTURE PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 60)
        report.append("")

        for operation, result in self.results.items():
            report.append(f"OPERATION: {operation.upper()}")
            report.append("-" * 40)

            if operation == "claim_creation":
                report.append(f"Claims Created: {result['num_claims']:,}")
                report.append(f"Execution Time: {result['execution_time']:.3f}s")
                report.append(
                    f"Throughput: {result['claims_per_second']:.1f} claims/sec"
                )
                report.append(f"Memory Used: {result['memory_used_mb']:.2f} MB")
                report.append(f"Peak Memory: {result['peak_memory_kb']:.1f} KB")
                report.append(
                    f"Avg Claim Size: {result['avg_claim_size_bytes']:.0f} bytes"
                )

            elif operation == "claim_relationships":
                report.append(f"Claims Processed: {result['num_claims']:,}")
                report.append(f"Execution Time: {result['execution_time']:.3f}s")
                report.append(f"Operations/sec: {result['operations_per_second']:.1f}")
                report.append(f"Total Relationships: {result['total_relationships']:,}")

            elif operation == "context_building":
                report.append(f"Context Size: {result['context_size']:,}")
                report.append(f"Execution Time: {result['execution_time']:.3f}s")
                report.append(f"Contexts/sec: {result['contexts_per_second']:.1f}")
                report.append(f"Total Characters: {result['total_context_chars']:,}")
                report.append(
                    f"Avg Context Size: {result['avg_context_size']:.0f} chars"
                )

            elif operation == "concurrent_operations":
                report.append(f"Threads: {result['num_threads']}")
                report.append(f"Total Operations: {result['total_operations']:,}")
                report.append(f"Total Time: {result['total_time']:.3f}s")
                report.append(f"Throughput: {result['ops_per_second']:.1f} ops/sec")
                report.append(f"Thread Efficiency: {result['thread_efficiency']:.2f}")
                report.append(f"Avg Thread Time: {result['avg_thread_time']:.3f}s")

            elif operation == "memory_usage":
                report.append(f"Total Claims Tested: {result['total_claims_tested']:,}")
                report.append(
                    f"Scaling Efficiency: {result['avg_scaling_efficiency']:.2f}"
                )
                report.append("")
                report.append("Batch Results:")
                for batch_result in result["results"]:
                    report.append(
                        f"  {batch_result['batch_size']:5d} claims: "
                        f"{batch_result['memory_used_mb']:6.2f}MB, "
                        f"{batch_result['memory_per_claim_kb']:6.1f}KB/claim"
                    )

            report.append("")

        # System information
        report.append("SYSTEM INFORMATION")
        report.append("-" * 20)
        report.append(f"CPU Count: {psutil.cpu_count()}")
        report.append(
            f"Memory Total: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB"
        )
        report.append(
            f"Memory Available: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB"
        )
        report.append(f"Python Version: {sys.version.split()[0]}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)


def run_performance_tests():
    """Run complete performance test suite"""
    print("Starting Conjecture Performance Benchmarks")
    print("=" * 50)

    benchmark = PerformanceBenchmark()

    # Run individual benchmarks
    benchmark.benchmark_claim_creation(1000)
    benchmark.benchmark_claim_relationships(500)
    benchmark.benchmark_context_building(100)
    benchmark.benchmark_concurrent_operations(4, 250)
    benchmark.benchmark_memory_usage([100, 500, 1000, 5000])

    # Generate report
    report = benchmark.generate_report()
    print(report)

    # Save report to file
    with open("performance_report.txt", "w") as f:
        f.write(report)

    print(f"\nPerformance report saved to: performance_report.txt")
    return benchmark.results


if __name__ == "__main__":
    run_performance_tests()
