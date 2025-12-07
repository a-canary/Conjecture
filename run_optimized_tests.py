#!/usr/bin/env python3
"""
Optimized Test Runner - Main entry point for optimized test execution
Integrates test selection, parallel execution, monitoring, and reporting.
"""
import asyncio
import json
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
import uuid

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from testing_optimization.test_optimizer import TestingOptimizationFramework
from testing_optimization.test_monitor import (
    ComprehensiveTestMonitor, TestExecutionMetric, get_test_monitor
)


class OptimizedTestRunner:
    """Main optimized test runner that coordinates all optimization components."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.optimization_framework = TestingOptimizationFramework(project_root)
        self.monitor = get_test_monitor(project_root / "test_reports")
        self.session_id = None

    async def run_tests(self,
                       strategy: str = "auto",
                       max_time: Optional[float] = None,
                       base_commit: Optional[str] = None,
                       generate_report: bool = True,
                       parallel: bool = True) -> Dict[str, Any]:
        """Run optimized tests with full monitoring and reporting."""

        print("🚀 Starting Optimized Test Execution")
        print(f"📊 Strategy: {strategy}")
        print(f"⏱️  Max Time: {max_time or 'unlimited'}s")
        print(f"🔍 Base Commit: {base_commit or 'none'}")
        print("=" * 60)

        # Start monitoring session
        self.session_id = f"opt_test_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.monitor.start_session(self.session_id)

        try:
            # Run optimization and test selection
            optimization_result = self.optimization_framework.optimize_and_run(
                strategy=strategy,
                max_time=max_time,
                base_commit=base_commit
            )

            # Record test metrics
            await self._record_test_metrics(optimization_result)

            # End monitoring session
            monitoring_results = self.monitor.end_session()

            # Combine results
            final_results = {
                "session_id": self.session_id,
                "optimization": optimization_result,
                "monitoring": monitoring_results,
                "strategy": strategy,
                "timestamp": time.time()
            }

            # Display results
            self._display_results(final_results)

            # Save comprehensive report
            if generate_report:
                await self._save_comprehensive_report(final_results)

            return final_results

        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            # Still try to end monitoring to get partial results
            try:
                monitoring_results = self.monitor.end_session()
                return {
                    "session_id": self.session_id,
                    "error": str(e),
                    "monitoring": monitoring_results,
                    "strategy": strategy
                }
            except Exception:
                return {"session_id": self.session_id, "error": str(e)}

    async def _record_test_metrics(self, optimization_result: Dict[str, Any]):
        """Record individual test metrics for monitoring."""
        execution_result = optimization_result.get("execution", {})
        selected_tests = optimization_result.get("optimization", {}).get("selected_tests", [])

        # Parse pytest output to extract individual test results
        output = execution_result.get("output", "")
        if output:
            test_results = self._parse_pytest_output(output)
            for test_name, test_data in test_results.items():
                metric = TestExecutionMetric(
                    test_name=test_name,
                    test_file=next((f for f in selected_tests if test_name in f), "unknown"),
                    start_time=test_data.get("start_time", time.time()),
                    end_time=test_data.get("end_time", time.time()),
                    duration=test_data.get("duration", 0.0),
                    status=test_data.get("status", "unknown"),
                    memory_usage_mb=test_data.get("memory_mb", 0.0),
                    cpu_usage_percent=test_data.get("cpu_percent", 0.0),
                    error_message=test_data.get("error"),
                    coverage_lines=test_data.get("coverage_lines", 0),
                    coverage_percentage=test_data.get("coverage_percentage", 0.0)
                )
                self.monitor.record_test_execution(metric)

    def _parse_pytest_output(self, output: str) -> Dict[str, Dict[str, Any]]:
        """Parse pytest output to extract test results."""
        test_results = {}
        lines = output.split('\n')

        for line in lines:
            if '::test_' in line and any(status in line for status in ['PASSED', 'FAILED', 'SKIPPED', 'ERROR']):
                parts = line.split()
                if len(parts) >= 2:
                    test_path = parts[0]
                    status = parts[1]

                    # Extract test name from path
                    test_name = test_path.split('::')[-1] if '::' in test_path else test_path

                    test_results[test_name] = {
                        "status": status.lower().replace('.', ''),
                        "duration": 0.0,  # Would need more detailed parsing
                        "memory_mb": 0.0,
                        "cpu_percent": 0.0
                    }

        return test_results

    def _display_results(self, results: Dict[str, Any]):
        """Display formatted results to console."""
        print("\n" + "=" * 60)
        print("📊 OPTIMIZED TEST EXECUTION RESULTS")
        print("=" * 60)

        # Optimization results
        optimization = results.get("optimization", {}).get("optimization", {})
        execution = results.get("optimization", {}).get("execution", {})
        monitoring = results.get("monitoring", {})

        print(f"\n🎯 Optimization Strategy: {optimization.get('optimization_strategy', 'unknown')}")
        print(f"📝 Reasoning: {optimization.get('reasoning', 'N/A')}")
        print(f"🧪 Tests Selected: {results.get('optimization', {}).get('tests_selected', 0)}")
        print(f"⏱️  Estimated Time Saved: {optimization.get('estimated_time_saved', 0):.2f}s")
        print(f"📈 Coverage Retained: {optimization.get('coverage_retained', 0):.1%}")

        print(f"\n✅ Execution Status: {execution.get('status', 'unknown')}")
        print(f"⏱️  Actual Time: {execution.get('time', 0):.2f}s")
        print(f"🧪 Tests Run: {execution.get('tests_run', 0)}")

        # Monitoring results
        session_metrics = monitoring.get("session_metrics", {})
        if session_metrics:
            print(f"\n📊 Session Metrics:")
            print(f"  ✅ Passed: {session_metrics.get('passed_tests', 0)}")
            print(f"  ❌ Failed: {session_metrics.get('failed_tests', 0)}")
            print(f"  ⏭️  Skipped: {session_metrics.get('skipped_tests', 0)}")
            print(f"  💥 Errors: {session_metrics.get('error_tests', 0)}")
            print(f"  📈 Total: {session_metrics.get('total_tests', 0)}")
            print(f"  💾 Peak Memory: {session_metrics.get('peak_memory_usage_mb', 0):.1f}MB")
            print(f"  📊 Coverage: {session_metrics.get('coverage_overall', 0):.1%}")
            print(f"  🎯 Performance Score: {session_metrics.get('performance_score', 0):.1f}")
            print(f"  ⚡ Optimization Score: {session_metrics.get('optimization_score', 0):.1f}")

        # Performance analysis
        performance_analysis = monitoring.get("performance_analysis", {})
        optimization_suggestions = performance_analysis.get("optimization_suggestions", [])

        if optimization_suggestions:
            print(f"\n💡 Optimization Suggestions:")
            for suggestion in optimization_suggestions:
                print(f"  • {suggestion}")

        # Report files
        report_files = [monitoring.get("html_report"), monitoring.get("json_report")]
        available_reports = [f for f in report_files if f]

        if available_reports:
            print(f"\n📄 Generated Reports:")
            for report_file in available_reports:
                print(f"  📊 {Path(report_file).name}")

        print("\n" + "=" * 60)

    async def _save_comprehensive_report(self, results: Dict[str, Any]):
        """Save comprehensive report with all metrics and analysis."""
        report_dir = self.project_root / "test_reports"
        report_dir.mkdir(exist_ok=True)

        report_file = report_dir / f"comprehensive_report_{self.session_id}.json"

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Comprehensive report saved: {report_file}")
        except Exception as e:
            print(f"⚠️  Could not save comprehensive report: {e}")

    def generate_baseline_comparison(self, baseline_file: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comparison with baseline performance."""
        if not baseline_file or not baseline_file.exists():
            return {"error": "Baseline file not found"}

        try:
            with open(baseline_file, 'r', encoding='utf-8') as f:
                baseline = json.load(f)

            # Get latest session metrics
            if not self.monitor.performance_analyzer.performance_history:
                return {"error": "No current session data available"}

            current_session = self.monitor.performance_analyzer.performance_history[-1]

            # Generate comparison
            comparison = self.monitor.performance_analyzer.compare_sessions(
                baseline.get("session_metrics", {}),  # This would need proper format conversion
                current_session
            )

            return comparison

        except Exception as e:
            return {"error": f"Comparison failed: {e}"}


async def main():
    """Main entry point for optimized test runner."""
    parser = argparse.ArgumentParser(
        description="Optimized Test Runner with intelligent selection and monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Auto-optimized run
  %(prog)s --strategy changes         # Optimize based on code changes
  %(prog)s --strategy performance --max-time 300  # Time-constrained optimization
  %(prog)s --base-commit HEAD~1      # Compare with previous commit
  %(prog)s --baseline baseline.json  # Compare with baseline performance
        """
    )

    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory)"
    )

    parser.add_argument(
        "--strategy",
        choices=["auto", "changes", "performance", "priority"],
        default="auto",
        help="Optimization strategy (default: auto)"
    )

    parser.add_argument(
        "--max-time",
        type=float,
        help="Maximum execution time in seconds"
    )

    parser.add_argument(
        "--base-commit",
        help="Base commit for change detection"
    )

    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel execution"
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable detailed report generation"
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline file for performance comparison"
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (JSON format)"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    # Validate project root
    if not (project_root / "pyproject.toml").exists():
        print(f"❌ Invalid project root: {project_root}")
        print("   No pyproject.toml found")
        return 1

    try:
        # Create and run optimized test runner
        runner = OptimizedTestRunner(project_root)

        results = await runner.run_tests(
            strategy=args.strategy,
            max_time=args.max_time,
            base_commit=args.base_commit,
            generate_report=not args.no_report,
            parallel=not args.no_parallel
        )

        # Baseline comparison
        if args.baseline:
            print("\n🔍 Generating baseline comparison...")
            comparison = runner.generate_baseline_comparison(args.baseline)
            if "error" not in comparison:
                print("📊 Baseline Comparison:")
                print(f"  Overall improvement: {'✅' if comparison.get('summary', {}).get('overall_improvement') else '❌'}")
                for improvement in comparison.get('improvements', []):
                    print(f"  ✅ {improvement}")
                for regression in comparison.get('regressions', []):
                    print(f"  ❌ {regression}")
                results["baseline_comparison"] = comparison
            else:
                print(f"⚠️  Baseline comparison failed: {comparison['error']}")

        # Save results to file if requested
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"\n💾 Results saved to: {args.output}")
            except Exception as e:
                print(f"\n⚠️  Could not save results: {e}")

        # Return appropriate exit code
        execution_status = results.get("optimization", {}).get("execution", {}).get("status", "unknown")
        return 0 if execution_status == "success" else 1

    except KeyboardInterrupt:
        print("\n⏹️  Test execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)