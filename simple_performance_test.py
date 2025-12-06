#!/usr/bin/env python3
"""
Simple Performance Baseline Test for Conjecture System
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

class SimpleProfiler:
    """Simple performance profiler"""

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.snapshots = []
        self.metrics = {
            "startup_time": 0.0,
            "api_calls": [],
            "memory_snapshots": []
        }

    def start(self):
        """Start profiling"""
        tracemalloc.start()
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        print(f"Profiling started - Memory: {self.start_memory:.2f} MB")

    def snapshot(self, label: str):
        """Take snapshot"""
        current_time = time.time()
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        elapsed = current_time - self.start_time if self.start_time else 0

        data = {
            "label": label,
            "elapsed": elapsed,
            "memory_mb": current_memory,
            "memory_diff": current_memory - self.start_memory
        }

        self.snapshots.append(data)
        print(f"Snapshot [{label}]: {elapsed:.2f}s, {current_memory:.2f} MB")

    def record_api(self, provider: str, operation: str, duration: float, success: bool):
        """Record API call"""
        self.metrics["api_calls"].append({
            "provider": provider,
            "operation": operation,
            "duration": duration,
            "success": success
        })

    def get_report(self) -> Dict[str, Any]:
        """Generate report"""
        return {
            "snapshots": self.snapshots,
            "api_calls": self.metrics["api_calls"],
            "total_duration": time.time() - self.start_time if self.start_time else 0
        }

async def main():
    """Main test"""
    print("Conjecture Performance Baseline Test")
    print("=" * 40)

    profiler = SimpleProfiler()
    profiler.start()

    try:
        # Test 1: Import and initialization
        print("\n1. Testing startup performance...")
        profiler.snapshot("import_start")

        from conjecture import Conjecture
        profiler.snapshot("import_done")

        init_start = time.time()
        cf = Conjecture()
        init_time = time.time() - init_start
        profiler.metrics["startup_time"] = init_time

        profiler.snapshot("initialization_done")
        print(f"Initialization time: {init_time:.3f}s")

        # Test 2: Services startup
        print("\n2. Testing services startup...")
        services_start = time.time()
        await cf.start_services()
        services_time = time.time() - services_start
        profiler.snapshot("services_started")
        print(f"Services startup time: {services_time:.3f}s")

        # Test 3: Check provider availability
        print("\n3. Testing provider availability...")
        try:
            providers = cf.llm_bridge.get_available_providers()
            print(f"Available providers: {providers}")

            # Test simple API call if providers available
            if providers:
                from conjecture import LLMRequest
                test_prompt = "Generate a simple test claim."

                api_start = time.time()
                request = LLMRequest(
                    prompt=test_prompt,
                    max_tokens=50,
                    temperature=0.5,
                    task_type="test"
                )

                response = cf.llm_bridge.process(request)
                api_time = time.time() - api_start

                profiler.record_api(providers[0], "test_call", api_time, response.success)
                profiler.snapshot("api_test_done")

                print(f"API call time: {api_time:.2f}s, success: {response.success}")
                if response.success:
                    print(f"Response length: {len(response.content)} chars")

        except Exception as e:
            print(f"Provider test failed: {e}")

        # Test 4: Simple exploration
        print("\n4. Testing exploration performance...")
        try:
            explore_start = time.time()
            result = await cf.explore(
                query="simple performance test",
                max_claims=3,
                auto_evaluate=False
            )
            explore_time = time.time() - explore_start
            profiler.snapshot("exploration_done")

            print(f"Exploration time: {explore_time:.2f}s")
            print(f"Claims generated: {len(result.claims)}")

            if result.claims:
                for i, claim in enumerate(result.claims[:3]):
                    print(f"  Claim {i+1}: {claim.content[:60]}...")

        except Exception as e:
            print(f"Exploration test failed: {e}")

        # Test 5: Memory usage
        print("\n5. Testing memory usage...")
        try:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_growth = current_memory - profiler.start_memory
            print(f"Current memory: {current_memory:.2f} MB")
            print(f"Memory growth: {memory_growth:+.2f} MB")

            # Test cache clearing if available
            if hasattr(cf, 'clear_all_caches'):
                cf.clear_all_caches()
                after_cache_memory = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"Memory after cache clear: {after_cache_memory:.2f} MB")

            profiler.snapshot("memory_test_done")

        except Exception as e:
            print(f"Memory test failed: {e}")

        # Generate final report
        profiler.snapshot("test_complete")
        report = profiler.get_report()

        # Add summary metrics
        report["summary"] = {
            "startup_time": profiler.metrics["startup_time"],
            "services_time": services_time,
            "total_api_calls": len(profiler.metrics["api_calls"]),
            "successful_api_calls": sum(1 for call in profiler.metrics["api_calls"] if call["success"]),
            "memory_growth_mb": psutil.Process().memory_info().rss / 1024 / 1024 - profiler.start_memory
        }

        # Save report
        with open("performance_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nPerformance report saved to: performance_report.json")

        # Print summary
        print(f"\nPerformance Summary:")
        print(f"  Startup time: {report['summary']['startup_time']:.3f}s")
        print(f"  Services time: {report['summary']['services_time']:.3f}s")
        print(f"  API calls: {report['summary']['total_api_calls']}/{report['summary']['successful_api_calls']} successful")
        print(f"  Memory growth: {report['summary']['memory_growth_mb']:+.2f} MB")
        print(f"  Total test time: {report['total_duration']:.2f}s")

        # Cleanup
        await cf.stop_services()
        print("\nTest completed successfully!")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())