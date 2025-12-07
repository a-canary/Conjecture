#!/usr/bin/env python3
"""
Test script for optimized Conjecture performance
"""

import asyncio
import time
import psutil
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

async def main():
    print("Testing Optimized Conjecture Performance")
    print("=" * 50)

    # Measure initial memory
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"Initial memory: {initial_memory:.2f} MB")

    # Test 1: Import and initialization time
    print("\n1. Testing optimized initialization...")
    import_start = time.time()

    from conjecture_optimized import OptimizedConjecture
    import_time = time.time() - import_start

    print(f"Import time: {import_time:.3f}s")

    # Test initialization
    init_start = time.time()
    cf = OptimizedConjecture()
    init_time = time.time() - init_start

    init_memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"Initialization time: {init_time:.3f}s")
    print(f"Memory after init: {init_memory:.2f} MB (+{init_memory - initial_memory:.2f} MB)")

    # Test 2: Services startup
    print("\n2. Testing services startup...")
    services_start = time.time()
    await cf.start_services()
    services_time = time.time() - services_start

    services_memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"Services startup time: {services_time:.3f}s")
    print(f"Memory after services: {services_memory:.2f} MB (+{services_memory - init_memory:.2f} MB)")

    # Test 3: First exploration (cold start)
    print("\n3. Testing first exploration (cold start)...")
    explore_start = time.time()
    try:
        result1 = await cf.explore(
            query="performance optimization techniques",
            max_claims=3,
            auto_evaluate=False
        )
        explore_time = time.time() - explore_start
        print(f"First exploration time: {explore_time:.2f}s")
        print(f"Claims generated: {len(result1.claims)}")
    except Exception as e:
        print(f"First exploration failed: {e}")
        explore_time = time.time() - explore_start

    # Test 4: Second exploration (warm cache)
    print("\n4. Testing second exploration (warm cache)...")
    explore2_start = time.time()
    try:
        result2 = await cf.explore(
            query="performance optimization techniques",
            max_claims=3,
            auto_evaluate=False
        )
        explore2_time = time.time() - explore2_start
        print(f"Second exploration time: {explore2_time:.2f}s")
        print(f"Cache hit: {hasattr(result2, 'optimization_stats') and result2.optimization_stats.get('cache_hit', False)}")
    except Exception as e:
        print(f"Second exploration failed: {e}")
        explore2_time = time.time() - explore2_start

    # Test 5: Performance stats
    print("\n5. Getting performance statistics...")
    try:
        stats = cf.get_performance_stats()
        print(f"Cache hit rate: {stats['cache_stats']['hit_rate']:.1f}%")
        print(f"Components initialized: {stats['initialized_components']}")
        print(f"Component init times: {stats['performance']['component_init_times']}")
    except Exception as e:
        print(f"Stats failed: {e}")

    # Test 6: Memory after operations
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"\n6. Final memory: {final_memory:.2f} MB (+{final_memory - initial_memory:.2f} MB total)")

    # Cleanup
    await cf.stop_services()

    # Performance summary
    total_time = import_time + init_time + services_time + explore_time
    print(f"\nPERFORMANCE SUMMARY:")
    print(f"  Import time: {import_time:.3f}s")
    print(f"  Initialization time: {init_time:.3f}s")
    print(f"  Services time: {services_time:.3f}s")
    print(f"  First exploration: {explore_time:.2f}s")
    print(f"  Second exploration: {explore2_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total memory growth: {final_memory - initial_memory:+.2f} MB")

    # Performance targets
    print(f"\nPERFORMANCE TARGETS:")
    print(f"  {'✅' if import_time < 2.0 else '❌'} Import < 2s: {import_time:.3f}s")
    print(f"  {'✅' if init_time < 1.0 else '❌'} Init < 1s: {init_time:.3f}s")
    print(f"  {'✅' if services_time < 2.0 else '❌'} Services < 2s: {services_time:.3f}s")
    print(f"  {'✅' if explore_time < 15.0 else '❌'} Exploration < 15s: {explore_time:.2f}s")
    print(f"  {'✅' if final_memory - initial_memory < 100 else '❌'} Memory growth < 100MB: {final_memory - initial_memory:.2f} MB")

if __name__ == "__main__":
    asyncio.run(main())