"""
Scaling Metrics and Benchmarks - Phase 2
Comprehensive benchmarking system for measuring scaling improvements and capacity planning
"""

import asyncio
import time
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    """Types of benchmarks"""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    CONCURRENCY = "concurrency"
    SCALABILITY = "scalability"
    STRESS = "stress"
    ENDURANCE = "endurance"
    RESOURCE_UTILIZATION = "resource_utilization"

class BenchmarkStatus(Enum):
    """Benchmark execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BenchmarkMetric:
    """Individual benchmark metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """Result of a benchmark execution"""
    benchmark_id: str
    benchmark_type: BenchmarkType
    status: BenchmarkStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    metrics: List[BenchmarkMetric] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    system_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CapacityPlan:
    """Capacity planning recommendation"""
    scenario: str
    current_capacity: Dict[str, Any]
    recommended_capacity: Dict[str, Any]
    scaling_factors: Dict[str, float]
    cost_estimate: Optional[float] = None
    implementation_timeline: Optional[str] = None
    risks: List[str] = field(default_factory=list)

class ScalingBenchmark:
    """
    Comprehensive scaling benchmark system
    """

    def __init__(
        self,
        output_directory: str = "benchmark_results",
        max_concurrent_benchmarks: int = 3,
        enable_visualization: bool = True
    ):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.max_concurrent_benchmarks = max_concurrent_benchmarks
        self.enable_visualization = enable_visualization

        # Benchmark execution
        self._active_benchmarks: Dict[str, BenchmarkResult] = {}
        self._benchmark_history: List[BenchmarkResult] = []
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_benchmarks)

        # Performance baselines
        self._baselines: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self._statistics = {
            "total_benchmarks": 0,
            "successful_benchmarks": 0,
            "failed_benchmarks": 0,
            "total_execution_time": 0.0
        }

        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info(f"ScalingBenchmark initialized with output directory: {output_directory}")

    async def run_throughput_benchmark(
        self,
        operation: Callable,
        operations_count: int = 1000,
        concurrency: int = 10,
        duration_seconds: Optional[float] = None,
        warmup_operations: int = 100,
        **operation_kwargs
    ) -> BenchmarkResult:
        """
        Run throughput benchmark measuring operations per second

        Args:
            operation: The operation to benchmark
            operations_count: Number of operations to execute
            concurrency: Number of concurrent operations
            duration_seconds: Alternative to operations_count - run for specified time
            warmup_operations: Number of warmup operations
            **operation_kwargs: Additional arguments for the operation
        """
        benchmark_id = f"throughput_{int(time.time())}"
        logger.info(f"Starting throughput benchmark {benchmark_id}")

        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.THROUGHPUT,
            status=BenchmarkStatus.RUNNING,
            start_time=datetime.utcnow(),
            configuration={
                "operations_count": operations_count,
                "concurrency": concurrency,
                "duration_seconds": duration_seconds,
                "warmup_operations": warmup_operations
            }
        )

        self._active_benchmarks[benchmark_id] = result

        try:
            async with self._execution_semaphore:
                # Warmup phase
                if warmup_operations > 0:
                    logger.info(f"Warming up with {warmup_operations} operations...")
                    await self._execute_warmup(operation, warmup_operations, **operation_kwargs)

                # Main benchmark phase
                start_time = time.time()
                completed_operations = 0
                errors = []
                operation_times = []

                if duration_seconds:
                    # Time-based benchmark
                    end_time = start_time + duration_seconds

                    async def continuous_operation():
                        nonlocal completed_operations
                        while time.time() < end_time:
                            try:
                                op_start = time.time()
                                await operation(**operation_kwargs)
                                operation_times.append(time.time() - op_start)
                                completed_operations += 1
                            except Exception as e:
                                errors.append(str(e))

                    # Run continuous operations with concurrency
                    tasks = [
                        asyncio.create_task(continuous_operation())
                        for _ in range(concurrency)
                    ]

                    await asyncio.gather(*tasks, return_exceptions=True)

                else:
                    # Count-based benchmark
                    semaphore = asyncio.Semaphore(concurrency)

                    async def limited_operation():
                        async with semaphore:
                            try:
                                op_start = time.time()
                                await operation(**operation_kwargs)
                                operation_times.append(time.time() - op_start)
                                return True
                            except Exception as e:
                                errors.append(str(e))
                                return False

                    # Execute operations in batches
                    batch_size = concurrency * 10
                    for i in range(0, operations_count, batch_size):
                        batch_end = min(i + batch_size, operations_count)
                        batch_tasks = [
                            asyncio.create_task(limited_operation())
                            for _ in range(batch_end - i)
                        ]

                        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                        completed_operations += sum(1 for r in batch_results if r is True)

                end_time = time.time()
                total_duration = end_time - start_time

                # Calculate metrics
                throughput = completed_operations / total_duration if total_duration > 0 else 0
                error_rate = len(errors) / max(1, completed_operations + len(errors))

                result.metrics = [
                    BenchmarkMetric("operations_completed", completed_operations, "count"),
                    BenchmarkMetric("total_duration", total_duration, "seconds"),
                    BenchmarkMetric("throughput", throughput, "operations/second"),
                    BenchmarkMetric("error_rate", error_rate, "percentage"),
                    BenchmarkMetric("average_operation_time", statistics.mean(operation_times) if operation_times else 0, "seconds"),
                    BenchmarkMetric("median_operation_time", statistics.median(operation_times) if operation_times else 0, "seconds"),
                    BenchmarkMetric("p95_operation_time", np.percentile(operation_times, 95) if operation_times else 0, "seconds"),
                    BenchmarkMetric("p99_operation_time", np.percentile(operation_times, 99) if operation_times else 0, "seconds")
                ]

                result.errors = errors
                result.status = BenchmarkStatus.COMPLETED
                result.end_time = datetime.utcnow()
                result.duration = total_duration

                logger.info(f"Throughput benchmark {benchmark_id} completed: {throughput:.2f} ops/sec")

        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.errors.append(str(e))
            result.end_time = datetime.utcnow()
            result.duration = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"Throughput benchmark {benchmark_id} failed: {e}")

        finally:
            # Move to history
            if benchmark_id in self._active_benchmarks:
                del self._active_benchmarks[benchmark_id]
            self._benchmark_history.append(result)
            self._update_statistics(result)

            # Save results
            await self._save_benchmark_result(result)

        return result

    async def run_latency_benchmark(
        self,
        operation: Callable,
        samples: int = 1000,
        percentiles: List[float] = None,
        **operation_kwargs
    ) -> BenchmarkResult:
        """
        Run latency benchmark measuring operation response times

        Args:
            operation: The operation to benchmark
            samples: Number of samples to collect
            percentiles: Percentiles to calculate (default: [50, 90, 95, 99, 99.9])
            **operation_kwargs: Additional arguments for the operation
        """
        if percentiles is None:
            percentiles = [50, 90, 95, 99, 99.9]

        benchmark_id = f"latency_{int(time.time())}"
        logger.info(f"Starting latency benchmark {benchmark_id}")

        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.LATENCY,
            status=BenchmarkStatus.RUNNING,
            start_time=datetime.utcnow(),
            configuration={
                "samples": samples,
                "percentiles": percentiles
            }
        )

        self._active_benchmarks[benchmark_id] = result

        try:
            async with self._execution_semaphore:
                latencies = []
                errors = []

                # Collect latency samples
                for i in range(samples):
                    try:
                        start_time = time.time()
                        await operation(**operation_kwargs)
                        latency = time.time() - start_time
                        latencies.append(latency)
                    except Exception as e:
                        errors.append(str(e))

                # Calculate latency metrics
                if latencies:
                    result.metrics = [
                        BenchmarkMetric("samples", len(latencies), "count"),
                        BenchmarkMetric("mean_latency", statistics.mean(latencies), "seconds"),
                        BenchmarkMetric("median_latency", statistics.median(latencies), "seconds"),
                        BenchmarkMetric("min_latency", min(latencies), "seconds"),
                        BenchmarkMetric("max_latency", max(latencies), "seconds"),
                        BenchmarkMetric("std_deviation", statistics.stdev(latencies), "seconds"),
                        BenchmarkMetric("error_count", len(errors), "count")
                    ]

                    # Add percentile metrics
                    for p in percentiles:
                        percentile_value = np.percentile(latencies, p)
                        result.metrics.append(
                            BenchmarkMetric(f"p{p}_latency", percentile_value, "seconds")
                        )
                else:
                    result.metrics = [
                        BenchmarkMetric("samples", 0, "count"),
                        BenchmarkMetric("error_count", len(errors), "count")
                    ]

                result.errors = errors
                result.status = BenchmarkStatus.COMPLETED
                result.end_time = datetime.utcnow()
                result.duration = (result.end_time - result.start_time).total_seconds()

                logger.info(f"Latency benchmark {benchmark_id} completed: {len(latencies)} samples")

        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.errors.append(str(e))
            result.end_time = datetime.utcnow()
            result.duration = (result.end_time - result.start_time).total_seconds()
            logger.error(f"Latency benchmark {benchmark_id} failed: {e}")

        finally:
            # Move to history
            if benchmark_id in self._active_benchmarks:
                del self._active_benchmarks[benchmark_id]
            self._benchmark_history.append(result)
            self._update_statistics(result)

            # Save results
            await self._save_benchmark_result(result)

        return result

    async def run_concurrency_benchmark(
        self,
        operation: Callable,
        max_concurrency: int = 50,
        step_size: int = 5,
        operations_per_step: int = 100,
        **operation_kwargs
    ) -> BenchmarkResult:
        """
        Run concurrency benchmark testing system behavior under different concurrency levels

        Args:
            operation: The operation to benchmark
            max_concurrency: Maximum concurrency level to test
            step_size: Concurrency increment per step
            operations_per_step: Number of operations per concurrency level
            **operation_kwargs: Additional arguments for the operation
        """
        benchmark_id = f"concurrency_{int(time.time())}"
        logger.info(f"Starting concurrency benchmark {benchmark_id}")

        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.CONCURRENCY,
            status=BenchmarkStatus.RUNNING,
            start_time=datetime.utcnow(),
            configuration={
                "max_concurrency": max_concurrency,
                "step_size": step_size,
                "operations_per_step": operations_per_step
            }
        )

        self._active_benchmarks[benchmark_id] = result

        try:
            async with self._execution_semaphore:
                concurrency_results = []

                # Test different concurrency levels
                for concurrency in range(step_size, max_concurrency + 1, step_size):
                    logger.info(f"Testing concurrency level: {concurrency}")

                    step_start = time.time()
                    step_errors = []
                    step_times = []

                    # Execute operations with current concurrency
                    semaphore = asyncio.Semaphore(concurrency)

                    async def limited_operation():
                        async with semaphore:
                            try:
                                op_start = time.time()
                                await operation(**operation_kwargs)
                                step_times.append(time.time() - op_start)
                                return True
                            except Exception as e:
                                step_errors.append(str(e))
                                return False

                    # Create tasks for this step
                    tasks = [
                        asyncio.create_task(limited_operation())
                        for _ in range(operations_per_step)
                    ]

                    step_results = await asyncio.gather(*tasks, return_exceptions=True)
                    step_duration = time.time() - step_start

                    # Calculate step metrics
                    successful_operations = sum(1 for r in step_results if r is True)
                    step_throughput = successful_operations / step_duration if step_duration > 0 else 0

                    concurrency_data = {
                        "concurrency": concurrency,
                        "successful_operations": successful_operations,
                        "total_operations": operations_per_step,
                        "duration": step_duration,
                        "throughput": step_throughput,
                        "error_count": len(step_errors),
                        "average_response_time": statistics.mean(step_times) if step_times else 0,
                        "median_response_time": statistics.median(step_times) if step_times else 0
                    }

                    concurrency_results.append(concurrency_data)

                # Analyze concurrency results
                max_throughput = max(r["throughput"] for r in concurrency_results)
                optimal_concurrency = max(
                    concurrency_results,
                    key=lambda r: r["throughput"]
                )["concurrency"]

                result.metrics = [
                    BenchmarkMetric("max_throughput", max_throughput, "operations/second"),
                    BenchmarkMetric("optimal_concurrency", optimal_concurrency, "count"),
                    BenchmarkMetric("concurrency_levels_tested", len(concurrency_results), "count")
                ]

                # Store detailed concurrency data in metadata
                result.metadata["concurrency_results"] = concurrency_results

                result.status = BenchmarkStatus.COMPLETED
                result.end_time = datetime.utcnow()
                result.duration = (result.end_time - result.start_time).total_seconds()

                logger.info(f"Concurrency benchmark {benchmark_id} completed: optimal concurrency = {optimal_concurrency}")

                # Generate visualization if enabled
                if self.enable_visualization:
                    await self._generate_concurrency_chart(benchmark_id, concurrency_results)

        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.errors.append(str(e))
            result.end_time = datetime.utcnow()
            result.duration = (result.end_time - result.start_time).total_seconds()
            logger.error(f"Concurrency benchmark {benchmark_id} failed: {e}")

        finally:
            # Move to history
            if benchmark_id in self._active_benchmarks:
                del self._active_benchmarks[benchmark_id]
            self._benchmark_history.append(result)
            self._update_statistics(result)

            # Save results
            await self._save_benchmark_result(result)

        return result

    async def run_scalability_benchmark(
        self,
        operation: Callable,
        load_factors: List[float],
        baseline_load: float = 1.0,
        **operation_kwargs
    ) -> BenchmarkResult:
        """
        Run scalability benchmark testing system behavior under different load factors

        Args:
            operation: The operation to benchmark
            load_factors: List of load factors to test (multiplier of baseline)
            baseline_load: Baseline load factor
            **operation_kwargs: Additional arguments for the operation
        """
        benchmark_id = f"scalability_{int(time.time())}"
        logger.info(f"Starting scalability benchmark {benchmark_id}")

        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_type=BenchmarkType.SCALABILITY,
            status=BenchmarkStatus.RUNNING,
            start_time=datetime.utcnow(),
            configuration={
                "load_factors": load_factors,
                "baseline_load": baseline_load
            }
        )

        self._active_benchmarks[benchmark_id] = result

        try:
            async with self._execution_semaphore:
                scalability_results = []

                # Test different load factors
                for load_factor in load_factors:
                    logger.info(f"Testing load factor: {load_factor}x")

                    # Adjust operation parameters based on load factor
                    scaled_kwargs = {
                        k: v * load_factor if isinstance(v, (int, float)) and k != 'timeout' else v
                        for k, v in operation_kwargs.items()
                    }

                    # Run throughput test for this load factor
                    start_time = time.time()
                    errors = []
                    completed_operations = 0

                    # Measure operations over a fixed time period
                    measurement_time = 10.0  # 10 seconds per load factor
                    end_time = start_time + measurement_time

                    async def continuous_operations():
                        nonlocal completed_operations
                        while time.time() < end_time:
                            try:
                                await operation(**scaled_kwargs)
                                completed_operations += 1
                            except Exception as e:
                                errors.append(str(e))

                    # Run with appropriate concurrency
                    concurrency = min(10, int(5 * load_factor))  # Scale concurrency with load
                    tasks = [
                        asyncio.create_task(continuous_operations())
                        for _ in range(concurrency)
                    ]

                    await asyncio.gather(*tasks, return_exceptions=True)

                    actual_duration = time.time() - start_time
                    throughput = completed_operations / actual_duration if actual_duration > 0 else 0

                    scalability_data = {
                        "load_factor": load_factor,
                        "completed_operations": completed_operations,
                        "duration": actual_duration,
                        "throughput": throughput,
                        "error_count": len(errors),
                        "concurrency": concurrency
                    }

                    scalability_results.append(scalability_data)

                # Analyze scalability
                baseline_throughput = next(
                    (r["throughput"] for r in scalability_results if r["load_factor"] == baseline_load),
                    0
                )

                if baseline_throughput > 0:
                    scaling_efficiency = []
                    for r in scalability_results:
                        expected_throughput = baseline_throughput * r["load_factor"]
                        actual_throughput = r["throughput"]
                        efficiency = actual_throughput / expected_throughput if expected_throughput > 0 else 0
                        scaling_efficiency.append(efficiency)

                    avg_efficiency = statistics.mean(scaling_efficiency)
                else:
                    avg_efficiency = 0

                result.metrics = [
                    BenchmarkMetric("baseline_throughput", baseline_throughput, "operations/second"),
                    BenchmarkMetric("average_scaling_efficiency", avg_efficiency, "percentage"),
                    BenchmarkMetric("max_load_factor", max(load_factors), "multiplier")
                ]

                # Store detailed scalability data
                result.metadata["scalability_results"] = scalability_results

                result.status = BenchmarkStatus.COMPLETED
                result.end_time = datetime.utcnow()
                result.duration = (result.end_time - result.start_time).total_seconds()

                logger.info(f"Scalability benchmark {benchmark_id} completed: efficiency = {avg_efficiency:.1%}")

                # Generate visualization if enabled
                if self.enable_visualization:
                    await self._generate_scalability_chart(benchmark_id, scalability_results)

        except Exception as e:
            result.status = BenchmarkStatus.FAILED
            result.errors.append(str(e))
            result.end_time = datetime.utcnow()
            result.duration = (result.end_time - result.start_time).total_seconds()
            logger.error(f"Scalability benchmark {benchmark_id} failed: {e}")

        finally:
            # Move to history
            if benchmark_id in self._active_benchmarks:
                del self._active_benchmarks[benchmark_id]
            self._benchmark_history.append(result)
            self._update_statistics(result)

            # Save results
            await self._save_benchmark_result(result)

        return result

    async def _execute_warmup(self, operation: Callable, warmup_count: int, **kwargs):
        """Execute warmup operations"""
        for i in range(warmup_count):
            try:
                await operation(**kwargs)
            except Exception as e:
                logger.warning(f"Warmup operation {i+1} failed: {e}")

    async def _save_benchmark_result(self, result: BenchmarkResult):
        """Save benchmark result to file"""
        filename = self.output_directory / f"{result.benchmark_id}.json"

        # Convert result to serializable format
        serializable_result = {
            "benchmark_id": result.benchmark_id,
            "benchmark_type": result.benchmark_type.value,
            "status": result.status.value,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat() if result.end_time else None,
            "duration": result.duration,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp.isoformat(),
                    "tags": m.tags,
                    "metadata": m.metadata
                }
                for m in result.metrics
            ],
            "errors": result.errors,
            "configuration": result.configuration,
            "system_info": result.system_info,
            "metadata": result.metadata
        }

        with open(filename, 'w') as f:
            json.dump(serializable_result, f, indent=2)

    async def _generate_concurrency_chart(self, benchmark_id: str, data: List[Dict[str, Any]]):
        """Generate concurrency visualization chart"""
        if not self.enable_visualization:
            return

        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            concurrencies = [d["concurrency"] for d in data]
            throughputs = [d["throughput"] for d in data]
            response_times = [d["average_response_time"] * 1000 for d in data]  # Convert to ms

            # Throughput chart
            ax1.plot(concurrencies, throughputs, 'b-o', label='Throughput')
            ax1.set_xlabel('Concurrency Level')
            ax1.set_ylabel('Throughput (ops/sec)')
            ax1.set_title('Throughput vs Concurrency')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Response time chart
            ax2.plot(concurrencies, response_times, 'r-o', label='Avg Response Time')
            ax2.set_xlabel('Concurrency Level')
            ax2.set_ylabel('Response Time (ms)')
            ax2.set_title('Response Time vs Concurrency')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.savefig(self.output_directory / f"{benchmark_id}_concurrency.png", dpi=300, bbox_inches='tight')
            plt.close()

        except ImportError:
            logger.warning("Matplotlib not available for visualization")
        except Exception as e:
            logger.error(f"Error generating concurrency chart: {e}")

    async def _generate_scalability_chart(self, benchmark_id: str, data: List[Dict[str, Any]]):
        """Generate scalability visualization chart"""
        if not self.enable_visualization:
            return

        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            load_factors = [d["load_factor"] for d in data]
            throughputs = [d["throughput"] for d in data]

            # Throughput scaling
            ax1.plot(load_factors, throughputs, 'g-o', label='Actual Throughput')

            # Ideal scaling line
            baseline_throughput = throughputs[0] if throughputs else 1
            ideal_throughputs = [baseline_throughput * lf for lf in load_factors]
            ax1.plot(load_factors, ideal_throughputs, 'r--', label='Ideal Scaling')

            ax1.set_xlabel('Load Factor')
            ax1.set_ylabel('Throughput (ops/sec)')
            ax1.set_title('Scalability Analysis')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Scaling efficiency
            efficiencies = [t / (baseline_throughput * lf) * 100 for t, lf in zip(throughputs, load_factors)]
            ax2.plot(load_factors, efficiencies, 'b-o', label='Scaling Efficiency')
            ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100% Efficiency')
            ax2.set_xlabel('Load Factor')
            ax2.set_ylabel('Efficiency (%)')
            ax2.set_title('Scaling Efficiency')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.savefig(self.output_directory / f"{benchmark_id}_scalability.png", dpi=300, bbox_inches='tight')
            plt.close()

        except ImportError:
            logger.warning("Matplotlib not available for visualization")
        except Exception as e:
            logger.error(f"Error generating scalability chart: {e}")

    def _update_statistics(self, result: BenchmarkResult):
        """Update benchmark statistics"""
        self._statistics["total_benchmarks"] += 1

        if result.status == BenchmarkStatus.COMPLETED:
            self._statistics["successful_benchmarks"] += 1
        else:
            self._statistics["failed_benchmarks"] += 1

        if result.duration:
            self._statistics["total_execution_time"] += result.duration

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results"""
        return {
            "statistics": self._statistics,
            "recent_benchmarks": [
                {
                    "benchmark_id": b.benchmark_id,
                    "type": b.benchmark_type.value,
                    "status": b.status.value,
                    "duration": b.duration,
                    "metrics_count": len(b.metrics)
                }
                for b in self._benchmark_history[-10:]
            ],
            "active_benchmarks": len(self._active_benchmarks),
            "performance_trends": self._calculate_performance_trends()
        }

    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends from benchmark history"""
        if not self._benchmark_history:
            return {}

        # Group by benchmark type
        by_type = {}
        for benchmark in self._benchmark_history:
            if benchmark.status == BenchmarkStatus.COMPLETED:
                benchmark_type = benchmark.benchmark_type.value
                if benchmark_type not in by_type:
                    by_type[benchmark_type] = []
                by_type[benchmark_type].append(benchmark)

        trends = {}
        for benchmark_type, benchmarks in by_type.items():
            if len(benchmarks) >= 2:
                # Calculate trend for key metrics
                if benchmark_type == "throughput":
                    throughputs = [
                        next((m.value for m in b.metrics if m.name == "throughput"), 0)
                        for b in benchmarks
                    ]
                    if throughputs:
                        trend = (throughputs[-1] - throughputs[0]) / throughputs[0] * 100
                        trends[benchmark_type] = {
                            "trend_percent": trend,
                            "latest_value": throughputs[-1],
                            "baseline_value": throughputs[0]
                        }

        return trends

    def create_capacity_plan(
        self,
        current_metrics: Dict[str, Any],
        growth_scenarios: List[Dict[str, Any]],
        performance_targets: Dict[str, Any]
    ) -> CapacityPlan:
        """
        Create capacity planning recommendations based on benchmark results

        Args:
            current_metrics: Current system performance metrics
            growth_scenarios: List of growth scenarios to plan for
            performance_targets: Target performance metrics

        Returns:
            CapacityPlan with recommendations
        """
        # Analyze current capacity from benchmarks
        recent_throughput_benchmarks = [
            b for b in self._benchmark_history
            if b.benchmark_type == BenchmarkType.THROUGHPUT and b.status == BenchmarkStatus.COMPLETED
        ]

        current_capacity = {
            "max_throughput": 0,
            "optimal_concurrency": 0,
            "average_latency": 0
        }

        if recent_throughput_benchmarks:
            latest_benchmark = recent_throughput_benchmarks[-1]
            for metric in latest_benchmark.metrics:
                if metric.name == "throughput":
                    current_capacity["max_throughput"] = metric.value
                elif metric.name == "optimal_concurrency":
                    current_capacity["optimal_concurrency"] = metric.value
                elif metric.name == "average_operation_time":
                    current_capacity["average_latency"] = metric.value

        # Analyze scalability from benchmarks
        scalability_benchmarks = [
            b for b in self._benchmark_history
            if b.benchmark_type == BenchmarkType.SCALABILITY and b.status == BenchmarkStatus.COMPLETED
        ]

        scaling_factors = {}
        if scalability_benchmarks:
            latest_scalability = scalability_benchmarks[-1]
            efficiency_metric = next(
                (m for m in latest_scalability.metrics if m.name == "average_scaling_efficiency"),
                None
            )
            if efficiency_metric:
                scaling_factors["efficiency"] = efficiency_metric.value

        # Generate recommendations
        recommended_capacity = {}
        risks = []

        for scenario in growth_scenarios:
            scenario_name = scenario.get("name", "unnamed")
            growth_factor = scenario.get("growth_factor", 1.0)

            # Calculate required capacity
            required_throughput = current_capacity["max_throughput"] * growth_factor

            # Apply scaling efficiency
            efficiency = scaling_factors.get("efficiency", 1.0)
            adjusted_throughput = required_throughput / efficiency

            recommended_capacity[scenario_name] = {
                "required_throughput": adjusted_throughput,
                "recommended_concurrency": int(current_capacity["optimal_concurrency"] * growth_factor),
                "additional_resources_needed": growth_factor > 1.5
            }

            # Assess risks
            if growth_factor > 2.0:
                risks.append(f"High growth scenario ({scenario_name}) may require architectural changes")

            if efficiency < 0.7:
                risks.append(f"Poor scaling efficiency ({efficiency:.1%}) may impact performance under load")

        return CapacityPlan(
            scenario="capacity_planning",
            current_capacity=current_capacity,
            recommended_capacity=recommended_capacity,
            scaling_factors=scaling_factors,
            implementation_timeline="3-6 months",
            risks=risks
        )

    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        logger.info("ScalingBenchmark cleanup completed")