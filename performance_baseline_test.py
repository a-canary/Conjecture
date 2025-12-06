#!/usr/bin/env python3
"""
Performance Baseline Test for Conjecture System
Measures startup time, memory usage, and API call patterns
"""

import asyncio
import time
import psutil
import os
import sys
import tracemalloc
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

class PerformanceProfiler:
    """Comprehensive performance profiler for Conjecture"""

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.snapshots = []
        self.metrics = {
            "startup_time": 0.0,
            "peak_memory": 0,
            "api_call_times": [],
            "database_operations": [],
            "cache_performance": {},
            "provider_performance": {}
        }

    def start_profiling(self):
        """Start performance profiling"""
        tracemalloc.start()
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        print(f"🔬 Performance profiling started at {datetime.now()}")
        print(f"📊 Initial memory: {self.start_memory:.2f} MB")

    def take_snapshot(self, label: str):
        """Take a performance snapshot"""
        current_time = time.time()
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        elapsed = current_time - self.start_time if self.start_time else 0
        memory_diff = current_memory - self.start_memory if self.start_memory else 0

        snapshot = {
            "label": label,
            "timestamp": current_time,
            "elapsed_time": elapsed,
            "memory_mb": current_memory,
            "memory_diff_mb": memory_diff
        }

        self.snapshots.append(snapshot)
        print(f"📸 Snapshot [{label}]: {elapsed:.2f}s, {current_memory:.2f} MB ({memory_diff:+.2f})")

    def record_api_call(self, provider: str, operation: str, duration: float, success: bool):
        """Record API call performance"""
        self.metrics["api_call_times"].append({
            "provider": provider,
            "operation": operation,
            "duration": duration,
            "success": success,
            "timestamp": time.time()
        })

    def record_database_operation(self, operation: str, duration: float, record_count: int = 0):
        """Record database operation performance"""
        self.metrics["database_operations"].append({
            "operation": operation,
            "duration": duration,
            "record_count": record_count,
            "timestamp": time.time()
        })

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        peak_memory = max(s["memory_mb"] for s in self.snapshots) if self.snapshots else current_memory

        # Analyze API call performance
        api_stats = {}
        if self.metrics["api_call_times"]:
            for call in self.metrics["api_call_times"]:
                provider = call["provider"]
                if provider not in api_stats:
                    api_stats[provider] = {"calls": 0, "total_time": 0, "successes": 0}
                api_stats[provider]["calls"] += 1
                api_stats[provider]["total_time"] += call["duration"]
                if call["success"]:
                    api_stats[provider]["successes"] += 1

            # Calculate averages
            for provider, stats in api_stats.items():
                stats["average_time"] = stats["total_time"] / stats["calls"]
                stats["success_rate"] = stats["successes"] / stats["calls"]

        report = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_duration": time.time() - self.start_time if self.start_time else 0,
                "start_memory_mb": self.start_memory,
                "current_memory_mb": current_memory,
                "peak_memory_mb": peak_memory,
                "memory_growth_mb": current_memory - self.start_memory if self.start_memory else 0
            },
            "snapshots": self.snapshots,
            "api_performance": api_stats,
            "database_operations": self.metrics["database_operations"],
            "total_api_calls": len(self.metrics["api_call_times"]),
            "successful_api_calls": sum(1 for call in self.metrics["api_call_times"] if call["success"])
        }

        return report

async def test_startup_performance(profiler: PerformanceProfiler) -> Dict[str, Any]:
    """Test Conjecture startup performance"""
    print("\n🚀 Testing Conjecture Startup Performance...")
    profiler.take_snapshot("import_start")

    # Import Conjecture
    from conjecture import Conjecture
    profiler.take_snapshot("import_complete")

    # Initialize Conjecture
    init_start = time.time()
    conjecture = Conjecture()
    init_time = time.time() - init_start

    profiler.metrics["startup_time"] = init_time
    profiler.take_snapshot("initialization_complete")

    # Start services
    services_start = time.time()
    await conjecture.start_services()
    services_time = time.time() - services_start

    profiler.take_snapshot("services_started")

    startup_results = {
        "import_time": profiler.snapshots[1]["elapsed_time"] - profiler.snapshots[0]["elapsed_time"],
        "initialization_time": init_time,
        "services_start_time": services_time,
        "total_startup_time": init_time + services_time
    }

    print(f"📈 Startup Performance:")
    print(f"  Import time: {startup_results['import_time']:.3f}s")
    print(f"  Initialization time: {startup_results['initialization_time']:.3f}s")
    print(f"  Services start time: {startup_results['services_start_time']:.3f}s")
    print(f"  Total startup: {startup_results['total_startup_time']:.3f}s")

    return conjecture, startup_results

async def test_llm_provider_performance(conjecture, profiler: PerformanceProfiler) -> Dict[str, Any]:
    """Test LLM provider performance and Model Matrix patterns"""
    print("\n🧠 Testing LLM Provider Performance...")

    # Test provider availability and switching
    provider_results = {}

    # Get available providers
    try:
        providers = conjecture.llm_bridge.get_available_providers()
        print(f"📋 Available providers: {providers}")

        # Test each provider with a simple request
        test_prompt = "Generate a simple test claim about performance optimization."

        for provider in providers[:3]:  # Test up to 3 providers
            print(f"🔍 Testing provider: {provider}")

            try:
                # Switch provider if needed
                if hasattr(conjecture.llm_bridge, 'switch_provider'):
                    conjecture.llm_bridge.switch_provider(provider)

                # Make API call
                api_start = time.time()
                from conjecture import LLMRequest
                request = LLMRequest(
                    prompt=test_prompt,
                    max_tokens=100,
                    temperature=0.5,
                    task_type="performance_test"
                )

                response = conjecture.llm_bridge.process(request)
                api_time = time.time() - api_start

                profiler.record_api_call(provider, "test_call", api_time, response.success)
                profiler.take_snapshot(f"provider_{provider}_test")

                provider_results[provider] = {
                    "response_time": api_time,
                    "success": response.success,
                    "tokens_used": response.tokens_used,
                    "content_length": len(response.content) if response.success else 0
                }

                print(f"  ✅ {provider}: {api_time:.2f}s, success: {response.success}")

            except Exception as e:
                print(f"  ❌ {provider}: Error - {e}")
                provider_results[provider] = {
                    "response_time": 0,
                    "success": False,
                    "error": str(e)
                }
                profiler.record_api_call(provider, "test_call", 0, False)

    except Exception as e:
        print(f"❌ Provider testing failed: {e}")

    return provider_results

async def test_exploration_performance(conjecture, profiler: PerformanceProfiler) -> Dict[str, Any]:
    """Test exploration performance with different claim loads"""
    print("\n🔍 Testing Exploration Performance...")

    exploration_results = {}
    test_queries = [
        "simple performance test",
        "complex system optimization with multiple factors",
        "database query optimization and indexing strategies"
    ]

    for i, query in enumerate(test_queries):
        print(f"🧪 Testing exploration {i+1}/3: '{query[:30]}...'")

        try:
            exploration_start = time.time()

            # Test different max_claims values
            max_claims = [3, 5, 10][i] if i < 3 else 5

            result = await conjecture.explore(
                query=query,
                max_claims=max_claims,
                auto_evaluate=False  # Disable evaluation for this test
            )

            exploration_time = time.time() - exploration_start

            profiler.record_database_operation(
                "explore_claims_creation",
                exploration_time,
                len(result.claims)
            )
            profiler.take_snapshot(f"exploration_{i+1}")

            exploration_results[f"test_{i+1}"] = {
                "query": query,
                "max_claims": max_claims,
                "actual_claims": len(result.claims),
                "processing_time": exploration_time,
                "claims_per_second": len(result.claims) / exploration_time if exploration_time > 0 else 0
            }

            print(f"  ✅ Exploration {i+1}: {exploration_time:.2f}s, {len(result.claims)} claims")

        except Exception as e:
            print(f"  ❌ Exploration {i+1}: Error - {e}")
            exploration_results[f"test_{i+1}"] = {
                "query": query,
                "error": str(e)
            }

    return exploration_results

async def test_memory_usage_patterns(conjecture, profiler: PerformanceProfiler) -> Dict[str, Any]:
    """Test memory usage patterns and identify leaks"""
    print("\n🧠 Testing Memory Usage Patterns...")

    memory_results = {}

    # Test multiple consecutive explorations to check for memory leaks
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_results["initial_memory_mb"] = initial_memory

    for i in range(5):
        print(f"🔄 Memory test iteration {i+1}/5...")

        try:
            # Perform exploration
            result = await conjecture.explore(
                query=f"memory test iteration {i+1}",
                max_claims=3,
                auto_evaluate=False
            )

            # Measure memory after operation
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory

            profiler.take_snapshot(f"memory_test_{i+1}")

            print(f"  Memory: {current_memory:.2f} MB (+{memory_growth:.2f} MB)")

        except Exception as e:
            print(f"  ❌ Memory test {i+1}: Error - {e}")

    # Test cache clearing
    print("🧹 Testing cache clearing...")
    try:
        if hasattr(conjecture, 'clear_all_caches'):
            conjecture.clear_all_caches()
            profiler.take_snapshot("cache_cleared")
            print("  ✅ Caches cleared")
    except Exception as e:
        print(f"  ❌ Cache clearing failed: {e}")

    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    memory_results["final_memory_mb"] = final_memory
    memory_results["total_growth_mb"] = final_memory - initial_memory

    return memory_results

async def main():
    """Main performance test execution"""
    print("🔬 Conjecture Performance Baseline Test")
    print("=" * 50)

    # Initialize profiler
    profiler = PerformanceProfiler()
    profiler.start_profiling()

    try:
        # Phase 1: Startup Performance
        conjecture, startup_results = await test_startup_performance(profiler)

        # Phase 2: LLM Provider Performance
        provider_results = await test_llm_provider_performance(conjecture, profiler)

        # Phase 3: Exploration Performance
        exploration_results = await test_exploration_performance(conjecture, profiler)

        # Phase 4: Memory Usage Patterns
        memory_results = await test_memory_usage_patterns(conjecture, profiler)

        # Generate comprehensive report
        profiler.take_snapshot("test_complete")
        final_report = profiler.generate_report()

        # Add test-specific results
        final_report["startup_performance"] = startup_results
        final_report["provider_performance"] = provider_results
        final_report["exploration_performance"] = exploration_results
        final_report["memory_usage_patterns"] = memory_results

        # Save report to file
        report_path = "performance_baseline_report.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)

        print(f"\n📊 Performance Baseline Report saved to: {report_path}")

        # Print summary
        print("\n🎯 Performance Summary:")
        print(f"  Total test duration: {final_report['test_metadata']['total_duration']:.2f}s")
        print(f"  Startup time: {startup_results['total_startup_time']:.3f}s")
        print(f"  Peak memory: {final_report['test_metadata']['peak_memory_mb']:.2f} MB")
        print(f"  Memory growth: {final_report['test_metadata']['memory_growth_mb']:+.2f} MB")
        print(f"  Total API calls: {final_report['total_api_calls']}")
        print(f"  Successful API calls: {final_report['successful_api_calls']}")

        # Performance bottlenecks identification
        print("\n🔍 Performance Bottlenecks Identified:")

        if startup_results['total_startup_time'] > 5.0:
            print(f"  ⚠️  Slow startup: {startup_results['total_startup_time']:.3f}s > 5s")

        if final_report['test_metadata']['memory_growth_mb'] > 100:
            print(f"  ⚠️  High memory growth: {final_report['test_metadata']['memory_growth_mb']:.2f} MB")

        avg_api_time = 0
        if final_report['api_performance']:
            avg_times = [stats['average_time'] for stats in final_report['api_performance'].values()]
            avg_api_time = sum(avg_times) / len(avg_times)

        if avg_api_time > 10.0:
            print(f"  ⚠️  Slow API responses: {avg_api_time:.2f}s average")

        # Cleanup
        await conjecture.stop_services()

    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())