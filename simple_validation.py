#!/usr/bin/env python3
"""
Simple validation script for testing optimization components
Tests core functionality without complex database dependencies.
"""
import asyncio
import sys
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    """Simple validation of optimization framework components."""
    print("🧪 Simple Testing Optimization Validation")
    print("=" * 50)

    results = []

    # Test 1: Test Optimizer Import
    print("\n1️⃣ Testing Test Optimizer Import...")
    try:
        from testing_optimization.test_optimizer import TestDependencyAnalyzer
        print("✅ Test Optimizer import successful")
        results.append({"test": "optimizer_import", "success": True})
    except Exception as e:
        print(f"❌ Test Optimizer import failed: {e}")
        results.append({"test": "optimizer_import", "success": False, "error": str(e)})

    # Test 2: Test Monitor Import
    print("\n2️⃣ Testing Test Monitor Import...")
    try:
        from testing_optimization.test_monitor import ComprehensiveTestMonitor
        print("✅ Test Monitor import successful")
        results.append({"test": "monitor_import", "success": True})
    except Exception as e:
        print(f"❌ Test Monitor import failed: {e}")
        results.append({"test": "monitor_import", "success": False, "error": str(e)})

    # Test 3: Basic Dependency Analysis
    print("\n3️⃣ Testing Basic Dependency Analysis...")
    try:
        from testing_optimization.test_optimizer import TestDependencyAnalyzer

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            (project_root / "src").mkdir()
            (project_root / "tests").mkdir()

            # Create test files
            (project_root / "src" / "test_module.py").write_text("""
def hello():
    return "world"
""", encoding='utf-8')

            (project_root / "tests" / "test_test_module.py").write_text("""
from src.test_module import hello

def test_hello():
    assert hello() == "world"
""", encoding='utf-8')

            analyzer = TestDependencyAnalyzer(project_root)
            deps = analyzer.analyze_file_dependencies(project_root / "tests" / "test_test_module.py")

            if len(deps) > 0:
                print("✅ Dependency analysis working")
                results.append({"test": "dependency_analysis", "success": True, "dependencies": len(deps)})
            else:
                print("⚠️  Dependency analysis returned no dependencies")
                results.append({"test": "dependency_analysis", "success": False, "error": "No dependencies found"})

    except Exception as e:
        print(f"❌ Dependency analysis failed: {e}")
        results.append({"test": "dependency_analysis", "success": False, "error": str(e)})

    # Test 4: Test Monitor Basic Functionality
    print("\n4️⃣ Testing Test Monitor Basic Functionality...")
    try:
        from testing_optimization.test_monitor import (
            ComprehensiveTestMonitor, TestExecutionMetric, get_test_monitor
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = get_test_monitor(Path(temp_dir))

            # Start session
            session_id = monitor.start_session("test_validation")

            # Record test metric
            test_metric = TestExecutionMetric(
                test_name="test_example",
                test_file="test_file.py",
                start_time=time.time(),
                end_time=time.time() + 0.5,
                duration=0.5,
                status="passed",
                memory_usage_mb=5.0,
                cpu_usage_percent=10.0
            )
            monitor.record_test_execution(test_metric)

            # End session
            session_results = monitor.end_session()

            if session_results and "session_id" in session_results:
                print("✅ Test monitor working")
                results.append({"test": "test_monitor", "success": True})
            else:
                print("❌ Test monitor failed to produce results")
                results.append({"test": "test_monitor", "success": False, "error": "No session results"})

    except Exception as e:
        print(f"❌ Test monitor failed: {e}")
        results.append({"test": "test_monitor", "success": False, "error": str(e)})

    # Test 5: UTF-8 Validation
    print("\n5️⃣ Testing UTF-8 Compliance...")
    try:
        test_strings = [
            "ASCII only",
            "Café Résumé",
            "北京测试",
            "🚀 Test Emoji 🧪"
        ]

        utf8_compliant = True
        for test_string in test_strings:
            try:
                test_string.encode('utf-8')
                json.dumps({"text": test_string}, ensure_ascii=False)
            except (UnicodeEncodeError, UnicodeDecodeError):
                utf8_compliant = False
                break

        if utf8_compliant:
            print("✅ UTF-8 compliance validated")
            results.append({"test": "utf8_compliance", "success": True})
        else:
            print("❌ UTF-8 compliance failed")
            results.append({"test": "utf8_compliance", "success": False, "error": "UTF-8 validation failed"})

    except Exception as e:
        print(f"❌ UTF-8 compliance test failed: {e}")
        results.append({"test": "utf8_compliance", "success": False, "error": str(e)})

    # Test 6: Performance Metrics
    print("\n6️⃣ Testing Performance Metrics...")
    try:
        from testing_optimization.test_optimizer import TestPerformanceProfiler

        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_file = Path(temp_dir) / "test_metrics.json"
            profiler = TestPerformanceProfiler(metrics_file)

            # Add test metrics
            profiler.update_test_metrics("fast_test", execution_time=0.1, coverage_score=0.9)
            profiler.update_test_metrics("slow_test", execution_time=2.5, coverage_score=0.7)
            profiler.save_metrics()

            # Retrieve slow tests
            slow_tests = profiler.get_slow_tests(1.0)

            if len(slow_tests) > 0:
                print("✅ Performance profiler working")
                results.append({"test": "performance_profiler", "success": True, "slow_tests": len(slow_tests)})
            else:
                print("⚠️  Performance profiler found no slow tests")
                results.append({"test": "performance_profiler", "success": False, "error": "No slow tests detected"})

    except Exception as e:
        print(f"❌ Performance profiler failed: {e}")
        results.append({"test": "performance_profiler", "success": False, "error": str(e)})

    # Generate Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)

    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["success"])
    failed_tests = total_tests - passed_tests

    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📈 Success Rate: {passed_tests/total_tests:.1%}")

    print("\n📋 Detailed Results:")
    for result in results:
        status = "✅" if result["success"] else "❌"
        test_name = result["test"].replace("_", " ").title()
        print(f"{status} {test_name}")
        if "error" in result:
            print(f"   Error: {result['error']}")

    # Save results
    summary = {
        "timestamp": time.time(),
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "success_rate": passed_tests / total_tests,
        "results": results
    }

    with open("simple_validation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n💾 Results saved to: simple_validation_results.json")

    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)